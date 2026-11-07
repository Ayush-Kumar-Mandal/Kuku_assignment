"""LLM client tests with scripted fake models: no network, every code path."""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agents.schemas import Segment
from config.loader import load_parser_config, load_settings
from llm.checks import ScreenplayIndex, check_promo_segments, promo_checks
from llm.client import (LLMClient, LLMConfigError, RateLimitExhausted, StructuredOutputError,
                        extract_json)
from llm.parser_fallback import make_parser_fallback
from llm.prompts import load_chat_prompt
from parser.screenplay_parser import parse_episode

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ep.txt"


# ------------------------------------------------------------------ fake chat model
class FakeError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class FakeChat:
    """Scripted model. `script` items: dict (structured result / JSON body), str (raw text),
    or an Exception to raise."""

    def __init__(self, script, profile=None):
        self.script = list(script)
        self.profile = profile
        self.calls = []          # (mode, messages)
        self.bound = None

    def _next(self):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def with_structured_output(self, schema, method=None, include_raw=False):
        outer = self

        class R:
            def invoke(self, messages):
                outer.calls.append(("native", messages))
                item = outer._next()
                raw = AIMessage(content=json.dumps(item), usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
                return {"raw": raw, "parsed": item, "parsing_error": None}
        return R()

    def bind(self, **kw):
        self.bound = kw
        return self

    def invoke(self, messages):
        self.calls.append(("fallback", messages))
        item = self._next()
        text = item if isinstance(item, str) else json.dumps(item)
        return AIMessage(content=text)   # no usage -> client estimates


class Pick(BaseModel):
    start_timestamp: str
    end_timestamp: str
    strength: int = Field(ge=1, le=10)


GOOD = {"start_timestamp": "00:03.60", "end_timestamp": "00:05.21", "strength": 8}


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.logs_dir = str(tmp_path / "logs")
    return s


def make_client(settings, fake, agent_mode=None):
    # Single-model behaviour; backup-model switching has its own tests below.
    settings.models["default"] = settings.models["default"].model_copy(update={"fallbacks": []})
    if agent_mode:
        settings.models["default"].structured_output = agent_mode
    waits = []
    client = LLMClient(settings, run_id="t", model_factory=lambda spec, st: fake, sleep=waits.append)
    return client, waits


@pytest.fixture
def prompt(settings):
    return load_chat_prompt(settings.resolve(settings.paths.prompts_dir), "smoke_test")


def call(client, prompt, checks=()):
    return client.structured("default", prompt, {"excerpt": "x"}, Pick, checks=checks)


# ------------------------------------------------------------------------- client
def test_native_revalidates_and_retries_with_error(settings, prompt):
    fake = FakeChat([{**GOOD, "strength": 15}, GOOD], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.mode == "native" and r.meta.retries == 1 and r.output.strength == 8
    assert "strength" in r.meta.errors[0][0]
    retry_msg = fake.calls[1][1][-1].content
    assert "Problems to fix" in retry_msg and "strength" in retry_msg
    assert r.meta.input_tokens == 200 and not r.meta.tokens_estimated


def test_fallback_when_profile_lacks_structured_output(settings, prompt):
    fake = FakeChat(["```json\n" + json.dumps(GOOD) + "\n```"], profile={"tool_calling": True})
    client, _ = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.mode == "fallback" and r.meta.retries == 0
    assert fake.bound == settings.providers["groq"].json_mode_kwargs
    system = fake.calls[0][1][0].content
    assert "JSON Schema" in system and "start_timestamp" in system      # schema merged into the one system msg
    assert r.meta.tokens_estimated and r.meta.input_tokens > 0


def test_auto_switches_to_fallback_when_native_errors(settings, prompt):
    fake = FakeChat([FakeError("json_schema not supported", 400), json.dumps(GOOD)], profile=None)
    client, _ = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.switched_to_fallback and r.meta.mode == "fallback" and r.meta.retries == 0
    assert client.resolve_mode("default") == "fallback"   # remembered for the run


def test_custom_check_failure_retries(settings, prompt):
    fake = FakeChat([{**GOOD, "start_timestamp": "00:04.00"}, GOOD], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    check = lambda o: [] if o.start_timestamp == "00:03.60" else ["start 00:04.00 invented; nearest 00:03.60"]
    r = call(client, prompt, checks=[check])
    assert r.meta.retries == 1 and "00:03.60" in fake.calls[1][1][-1].content


def test_rate_limit_waits_dont_use_retries(settings, prompt):
    fake = FakeChat([FakeError("429 Too Many", 429), FakeError("RESOURCE_EXHAUSTED"), GOOD], profile={"structured_output": True})
    client, waits = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.retries == 0 and r.meta.rate_limit_waits == 2
    assert waits == settings.llm.rate_limit_backoff_s[:2]


@pytest.mark.parametrize("err", [
    FakeError("503 UNAVAILABLE. This model is currently experiencing high demand.", 503),
    FakeError("500 INTERNAL"), FakeError("Service overloaded"),
])
def test_server_overload_is_waited_out_not_a_mode_switch(settings, prompt, err):
    fake = FakeChat([err, GOOD], profile={"structured_output": True})
    client, waits = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.mode == "native" and not r.meta.switched_to_fallback      # still native
    assert r.meta.retries == 0 and r.meta.rate_limit_waits == 1 and len(waits) == 1
    assert client.resolve_mode("default") == "native"                       # not marked broken


def test_daily_quota_stops_at_once_with_resume_hint(settings, prompt):
    err = FakeError("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
                    "limit: 20 ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... "
                    "'retryDelay': '81005s'", 429)
    fake = FakeChat([err, GOOD], profile={"structured_output": True})
    client, waits = make_client(settings, fake)
    with pytest.raises(RateLimitExhausted, match=r"daily quota.*22\.5h") as ei:
        call(client, prompt)
    assert waits == [] and ei.value.daily is True                       # no pointless backoff


def test_rate_limit_exhausted(settings, prompt):
    n = len(settings.llm.rate_limit_backoff_s) + 1
    fake = FakeChat([FakeError("rate limit", 429)] * n, profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    with pytest.raises(RateLimitExhausted):
        call(client, prompt)


def test_gives_up_after_max_retries(settings, prompt):
    bad = {**GOOD, "strength": 0}
    fake = FakeChat([bad] * (settings.llm.max_retries + 1), profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    with pytest.raises(StructuredOutputError) as ei:
        call(client, prompt)
    assert ei.value.meta.attempts == settings.llm.max_retries + 1
    assert client.usage.to_dict()["by_agent"]["default"]["failures"] == 1


def test_auth_error_is_not_retried(settings, prompt):
    fake = FakeChat([FakeError("Invalid API Key", 401)], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    with pytest.raises(LLMConfigError):
        call(client, prompt)


def test_unknown_model_is_not_retried(settings, prompt):
    fake = FakeChat([FakeError("model_not_found: does not exist", 404)], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    with pytest.raises(LLMConfigError, match="cannot be used"):
        call(client, prompt)
    assert len(fake.calls) == 1


def test_missing_key_message(settings, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("llm.client.load_dotenv", lambda: None)
    client = LLMClient(settings, run_id="t")
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        client.model("default")


def test_logs_written(settings, prompt):
    fake = FakeChat([{**GOOD, "strength": 15}, GOOD], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    r = call(client, prompt)
    assert len(r.meta.log_files) == 2 and all(Path(f).exists() for f in r.meta.log_files)
    line = json.loads((client.log_dir / "calls.jsonl").read_text().splitlines()[-1])
    assert line["mode"] == "native" and line["retries"] == 1
    usage = json.loads((client.log_dir / "usage.json").read_text())
    assert usage["by_agent"]["default"]["native_calls"] == 1 and usage["retry_rate"] == 0.5


def test_extract_json():
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


# ------------------------------------------------------------------------- checks
CHECK_EP = """EPISODE NUMBER: Episode 2
SCENE 01
Scene start: 00:00.00
Scene end: 01:00.00
**ACTION:**
Start: (00:00.00) | End: (00:10.00)
A door bursts open.

**MARA (Angry):**
Start: (00:10.00) | End: (00:14.00)
"You lied to me."

**CAMERA:**
Start: (00:12.00) | End: (00:30.00)
Slow push in.

**TOMAS:**
Start: (00:30.00) | End: (00:45.00)
"I had to."
"""


@pytest.fixture
def index(tmp_path, settings):
    f = tmp_path / "ep2.txt"
    f.write_text(CHECK_EP, encoding="utf-8")
    cfg = load_parser_config(settings=settings)
    return ScreenplayIndex([parse_episode(f, cfg)], cfg, settings.promo.boundary_tolerance_s)


def seg(s, e, ep=2):
    return Segment(episode=ep, start_timestamp=s, end_timestamp=e)


def test_valid_segment_passes(index, settings):
    assert check_promo_segments(index, [seg("00:00.00", "00:30.00")], settings, "p") == []


def test_invented_timestamp_rejected_with_suggestions(index, settings):
    probs = check_promo_segments(index, [seg("00:00.00", "00:31.00")], settings, "p")
    assert any("not a timestamp in episode 2" in p and "00:30.00" in p for p in probs)


def test_real_timestamp_next_to_another_boundary_is_accepted(settings):
    """Regression (full-50-v1, episode 10): 00:09.15 is a real line start, but 00:09.10 is a boundary exactly
    one tolerance (0.05 s) earlier. Float rounding put 00:09.10 just outside the tolerance and the exact
    match right after it was never looked at, so a real timestamp was rejected as invented."""
    from parser.screenplay_parser import parse_text
    cfg = load_parser_config(settings=settings)
    ep = parse_text("""EPISODE NUMBER: Episode 10
SCENE 01
Scene start: 00:00.00
Scene end: 00:40.00
**ACTION:**
Start: (00:07.50) | End: (00:09.10)
She turns.

**MARA (Cold):**
Start: (00:09.15) | End: (00:12.00)
"Get out."
""", cfg)
    idx = ScreenplayIndex([ep], cfg, settings.promo.boundary_tolerance_s)
    assert idx.exists(10, 9.15) and idx.exists(10, 9.10)
    probs = check_promo_segments(idx, [seg("00:09.15", "00:31.00", ep=10)], settings, "p")
    assert not any("00:09.15 is not a timestamp" in p for p in probs)


def test_cut_inside_spoken_line_rejected(index, settings):
    # 00:12.00 exists (camera start) but falls inside MARA's line 10-14
    probs = check_promo_segments(index, [seg("00:12.00", "00:45.00")], settings, "p")
    assert any("cuts into a line" in p and "00:10.00" in p for p in probs)


def test_duration_limits(index, settings):
    assert any("under the 20s minimum" in p for p in check_promo_segments(index, [seg("00:00.00", "00:14.00")], settings, "p"))
    settings.promo.max_duration_s = 40
    assert any("over the 40s maximum" in p for p in check_promo_segments(index, [seg("00:00.00", "00:45.00")], settings, "p"))


def test_multi_segment_overlap_and_unknown_episode(index, settings):
    probs = check_promo_segments(index, [seg("00:00.00", "00:14.00"), seg("00:10.00", "00:30.00")], settings, "p")
    assert any("overlaps" in p for p in probs)
    assert any("episode 9 does not exist" in p for p in check_promo_segments(index, [seg("00:00.00", "00:30.00", ep=9)], settings, "p"))


def test_promo_checks_factory(index, settings):
    class P(BaseModel):
        promo_id: str
        segments: list[Segment]

    class Out(BaseModel):
        candidates: list[P]

    check = promo_checks(index, settings, lambda o: o.candidates)
    out = Out(candidates=[P(promo_id="c1", segments=[seg("00:00.00", "00:30.00")]),
                          P(promo_id="c2", segments=[seg("00:00.00", "00:31.00")])])
    probs = check(out)
    assert probs and all(p.startswith("promo c2") for p in probs)


# ---------------------------------------------------------------- parser fallback
def test_llm_parser_fallback_recovers_line(settings):
    garbage_line = next(u.line_no for u in parse_episode(FIXTURE).unparsed)
    fake = FakeChat([{"items": [{"line_no": garbage_line, "kind": "action", "text": "this line matches nothing at all",
                                 "start_timestamp": "00:09.50", "end_timestamp": "00:09.50"}]}],
                    profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    cfg = load_parser_config(settings=settings)
    ep = parse_episode(FIXTURE, cfg, fallback=make_parser_fallback(client, cfg))
    assert ep.coverage.recovered_by_fallback == 1
    assert any(it.source == "llm_fallback" for it in ep.all_items())
    assert client.usage.to_dict()["by_agent"]["parser_fallback"]["calls"] == 1


# ------------------------------------------------------------------ model switching
DAILY = "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel 'retryDelay': '81005s'"


def two_model_client(settings, primary_script, backup_script):
    from config.loader import ModelSpec
    settings.models["default"] = ModelSpec(provider="groq", model="main-model",
                                           fallbacks=[ModelSpec(provider="groq", model="backup-model")])
    fakes = {"main-model": FakeChat(primary_script, profile={"structured_output": True}),
             "backup-model": FakeChat(backup_script, profile={"structured_output": True})}
    waits = []
    client = LLMClient(settings, run_id="t", model_factory=lambda spec, st: fakes[spec.model], sleep=waits.append)
    return client, fakes, waits


def test_daily_quota_switches_to_backup_model_for_rest_of_run(settings, prompt):
    client, fakes, waits = two_model_client(settings, [FakeError(DAILY, 429)], [GOOD, GOOD])
    r = call(client, prompt)
    assert r.meta.model == "backup-model" and r.meta.retries == 0 and waits == []
    assert r.meta.model_switches == [{"from": "groq:main-model", "to": "groq:backup-model", "reason": "daily quota"}]
    r2 = call(client, prompt)                                   # next call goes straight to the backup
    assert r2.meta.model == "backup-model" and len(fakes["main-model"].calls) == 1
    assert client.usage.to_dict()["totals"]["model_switches"] == 1


def test_overload_that_outlasts_backoff_also_switches(settings, prompt):
    n = len(settings.llm.rate_limit_backoff_s) + 1
    client, fakes, waits = two_model_client(settings, [FakeError("503 UNAVAILABLE high demand", 503)] * n, [GOOD])
    r = call(client, prompt)
    assert r.meta.model == "backup-model" and len(waits) == n - 1
    assert r.meta.model_switches[0]["reason"] == "rate-limited/overloaded"


def test_no_backup_left_raises(settings, prompt):
    client, _, _ = two_model_client(settings, [FakeError(DAILY, 429)], [FakeError(DAILY, 429)])
    with pytest.raises(RateLimitExhausted, match="no backup model left"):
        call(client, prompt)


def test_fallback_providers_are_validated():
    from config.loader import Settings
    raw = load_settings().model_dump()
    raw["models"]["reader"]["fallbacks"] = [{"provider": "nope", "model": "x"}]
    with pytest.raises(ValueError, match="nope"):
        Settings.model_validate(raw)


def test_provider_schema_rejection_is_a_retry_not_a_mode_switch(settings, prompt):
    # Groq returns 400 json_validate_failed when the model's answer misses the schema: a model
    # mistake, not "native structured output unsupported".
    err = FakeError("Error code: 400 - {'error': {'message': \"Failed to validate JSON. Please adjust your "
                    "prompt. See 'failed_generation' for more details.\", 'code': 'json_validate_failed'}}", 400)
    fake = FakeChat([err, GOOD], profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.mode == "native" and not r.meta.switched_to_fallback and r.meta.retries == 1
    assert client.resolve_mode("default") == "native"


def test_structured_error_carries_last_valid_output(settings, prompt):
    fake = FakeChat([GOOD] * (settings.llm.max_retries + 1), profile={"structured_output": True})
    client, _ = make_client(settings, fake)
    with pytest.raises(StructuredOutputError) as ei:
        call(client, prompt, checks=[lambda o: ["always wrong"]])
    assert ei.value.last_output is not None and ei.value.last_output.start_timestamp == "00:03.60"


def test_per_minute_limit_waits_as_long_as_the_provider_asks(settings, prompt):
    err = FakeError("Error code: 429 - Rate limit reached for model `qwen` on input tokens per minute (ITPM): "
                    "Limit 7000, Used 6665, Requested 2022. Please try again in 14.46s.", 429)
    fake = FakeChat([err, GOOD], profile={"structured_output": True})
    client, waits = make_client(settings, fake)
    r = call(client, prompt)
    assert r.meta.retries == 0 and len(waits) == 1
    assert waits[0] >= 14.46 and waits[0] >= settings.llm.rate_limit_backoff_s[0]   # not the fixed 10s


def test_short_rate_limits_never_stop_the_run(settings, prompt):
    n = len(settings.llm.rate_limit_backoff_s) + 3                     # more 429s than backoff steps
    err = FakeError("429 Rate limit reached ... tokens per minute ... Please try again in 2s.", 429)
    fake = FakeChat([err] * n + [GOOD], profile={"structured_output": True})
    client, waits = make_client(settings, fake)
    assert call(client, prompt).meta.rate_limit_waits == n
    assert len(waits) == n and all(w <= max(settings.llm.rate_limit_backoff_s) for w in waits)
