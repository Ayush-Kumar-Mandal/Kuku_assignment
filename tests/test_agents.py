"""LLM agents with scripted fake models: prompts, schemas, custom checks, retries. No network."""
import json

import pytest
from langchain_core.messages import AIMessage

from agents.base import AgentError
from agents.llm_agents import LLMAgents
from agents.render import render_promo_text
from config.loader import load_moment_types, load_parser_config, load_settings
from llm.client import LLMClient
from parser.screenplay_parser import parse_show
from synthetic import write_show


class RouterChat:
    """Fake chat model: answers by schema name from `script[schema_name]` (list, popped)."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.profile = {"structured_output": True}
        self.seen: dict[str, list] = {}          # schema name -> list of message lists

    def with_structured_output(self, schema, method=None, include_raw=False):
        outer = self

        class R:
            def invoke(self, messages):
                name = schema.__name__
                outer.seen.setdefault(name, []).append(messages)
                item = outer.script[name].pop(0)
                raw = AIMessage(content=json.dumps(item),
                                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                return {"raw": raw, "parsed": item, "parsing_error": None}
        return R()

    def text(self, name, call=0):
        return "\n".join(m.content for m in self.seen[name][call])


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.logs_dir = str(tmp_path / "logs")
    return s


@pytest.fixture
def show(tmp_path, settings):
    return parse_show(write_show(tmp_path / "show", (1, 2)), load_parser_config(settings=settings))


def make(settings, script):
    fake = RouterChat(script)
    client = LLMClient(settings, run_id="t", model_factory=lambda spec, st: fake, sleep=lambda s: None)
    agents = LLMAgents(client, settings, load_parser_config(settings=settings), load_moment_types(settings=settings),
                       show="Test Show", genre="romance drama")
    return agents, fake


def moment(kind, ts, mtype="confrontation", strength=8):
    return {"kind": kind, "timestamp": ts, "quote": "Line", "description": f"{kind} at {ts}",
            "strength": strength, "moment_type": mtype}


READ_OK = {"moments": [moment("hook", "00:04.00"), moment("cliffhanger", "00:28.00", mtype="slapfest")],
           "new_viewer_context": "Mara confronts Tomas.", "story_summary": "Ep1: Mara confronts Tomas."}


# ---------------------------------------------------------------------------- reader
def test_reader_assigns_ids_normalises_types_and_carries_summary(settings, show):
    agents, fake = make(settings, {"ReaderOutput": [READ_OK]})
    out = agents.read_episode(show.episode(1), "", {})
    ids = [m["moment_id"] for m in out["moments"]]
    assert ids == ["m01_h1", "m01_c1"]
    assert all(m["episode"] == 1 for m in out["moments"])
    assert out["moments"][1]["moment_type"] == "other"            # unknown type -> default_type
    assert out["story_summary"] == "Ep1: Mara confronts Tomas."
    assert out["llm"]["mode"] == "native" and out["llm"]["retries"] == 0
    prompt = fake.text("ReaderOutput")
    assert "Test Show" in prompt and "romance drama" in prompt       # the Reader may know the show
    assert "00:04.00 | 00:07.50 | MARA (Angry) | Line 1 of episode 1." in prompt


def test_reader_rejects_invented_timestamp_and_retries_with_nearest(settings, show):
    bad = {**READ_OK, "moments": [moment("hook", "00:05.00")]}     # not a line start
    agents, fake = make(settings, {"ReaderOutput": [bad, READ_OK]})
    out = agents.read_episode(show.episode(1), "Before.", {})
    assert out["llm"]["retries"] == 1
    retry = fake.seen["ReaderOutput"][1][-1].content
    assert "00:05.00" in retry and "00:04.00" in retry                # nearest real start offered
    assert "Before." in fake.text("ReaderOutput")                     # story so far is in the prompt


def test_reader_enforces_moment_limits(settings, show):
    settings.agents.reader.max_hooks = 1
    two = {**READ_OK, "moments": [moment("hook", "00:04.00"), moment("hook", "00:08.00")]}
    agents, fake = make(settings, {"ReaderOutput": [two, READ_OK]})
    agents.read_episode(show.episode(1), "", {})
    assert "max is 1" in fake.seen["ReaderOutput"][1][-1].content


# ---------------------------------------------------------------------------- cutter
MOMENTS = [{"moment_id": "m01_h1", "episode": 1, "kind": "hook", "timestamp": "00:04.00", "description": "clash",
            "strength": 8, "moment_type": "confrontation"},
           {"moment_id": "m01_c1", "episode": 1, "kind": "cliffhanger", "timestamp": "00:28.00",
            "description": "threat", "strength": 9, "moment_type": "danger"}]


def cut(**kw):
    base = {"start_timestamp": "00:04.00", "end_timestamp": "00:31.50", "hook_timestamp": "00:04.00",
            "cliffhanger_timestamp": "00:28.00", "hook_moment_id": "m01_h1", "cliffhanger_moment_id": "m01_c1",
            "hook": "Mara slaps Tomas", "cliffhanger": "He threatens her", "moment_type": "confrontation",
            "why_it_works": "Opens on violence, ends on a threat."}
    return {**base, **kw}


def test_cutter_builds_promo_with_duration_from_screenplay(settings, show):
    agents, _ = make(settings, {"CutterOutput": [{"promos": [cut()]}]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    assert p["promo_id"] == "c01_1" and p["maker"] == "cutter" and p["version"] == 1
    assert p["segments"] == [{"episode": 1, "start_timestamp": "00:04.00", "end_timestamp": "00:31.50"}]
    assert p["total_duration_seconds"] == 27.5
    assert p["source_moment_ids"] == ["m01_h1", "m01_c1"]


@pytest.mark.parametrize("bad, expect", [
    (cut(hook_timestamp="00:12.00"), "first 3s"),                       # hook lands too late
    (cut(cliffhanger_timestamp="00:08.00"), "within"),                  # cliffhanger far from the end
    (cut(hook_moment_id="m09_h1"), "unknown moment"),
    (cut(end_timestamp="00:15.50"), "under the 20s minimum"),
    (cut(end_timestamp="00:30.00"), "cuts off a line"),
])
def test_cutter_custom_checks_trigger_retry(settings, show, bad, expect):
    agents, fake = make(settings, {"CutterOutput": [{"promos": [bad]}, {"promos": [cut()]}]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    assert expect in fake.seen["CutterOutput"][1][-1].content
    assert p["total_duration_seconds"] == 27.5


def test_cutter_skips_llm_when_no_moments(settings, show):
    agents, fake = make(settings, {})
    assert agents.cut_episode(show.episode(1), [], {}) == [] and not fake.seen


# ---------------------------------------------------------------------------- joiner
def join(**kw):
    base = {"segments": [{"episode": 1, "start_timestamp": "00:04.00", "end_timestamp": "00:15.50"},
                         {"episode": 2, "start_timestamp": "00:40.00", "end_timestamp": "00:55.50"}],
            "hook_timestamp": "00:04.00", "cliffhanger_timestamp": "00:52.00",
            "hook_moment_id": "m01_h1", "cliffhanger_moment_id": "m02_c1", "hook": "slap", "cliffhanger": "gun",
            "moment_type": "danger", "why_it_works": "Escalates.", "join_rationale": "Ep1 sets up, ep2 pays off."}
    return {**base, **kw}


J_MOMENTS = MOMENTS + [{"moment_id": "m02_c1", "episode": 2, "kind": "cliffhanger", "timestamp": "00:52.00",
                        "description": "gun", "strength": 9, "moment_type": "danger"}]


def test_joiner_needs_two_segments_and_cliffhanger_in_last(settings, show):
    one = join(segments=join()["segments"][:1])
    agents, fake = make(settings, {"JoinerOutput": [{"promos": [one]}, {"promos": [join()]}]})
    eps = {e.number: e for e in show.episodes}
    [p] = agents.join(eps, J_MOMENTS, [], {})
    assert "at least 2 segments" in fake.seen["JoinerOutput"][1][-1].content
    assert p["promo_id"] == "j01" and p["maker"] == "joiner" and p["total_duration_seconds"] == 27.0
    assert p["join_rationale"].startswith("Ep1")
    prompt = fake.text("JoinerOutput")
    assert "m02_c1" in prompt and "Line 13 of episode 2." in prompt      # moments come with excerpts


def test_joiner_rejects_cliffhanger_outside_last_segment(settings, show):
    agents, fake = make(settings, {"JoinerOutput": [{"promos": [join(cliffhanger_timestamp="00:08.00")]},
                                                     {"promos": [join()]}]})
    agents.join({e.number: e for e in show.episodes}, J_MOMENTS, [], {})
    assert "last segment" in fake.seen["JoinerOutput"][1][-1].content


# ---------------------------------------------------------------------------- revise
def test_revise_keeps_identity_bumps_version_and_sees_instruction(settings, show):
    agents, fake = make(settings, {"CutterOutput": [{"promos": [cut()]}],
                                   "RevisedCut": [cut(start_timestamp="00:00.00", hook_timestamp="00:00.00",
                                                      revision_note="Opened on the door slam.")]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    new = agents.revise("cutter", p, "Open on the door slam", {1: show.episode(1)}, {})
    assert (new["promo_id"], new["maker"], new["version"]) == ("c01_1", "cutter", 2)
    assert new["total_duration_seconds"] == 31.5 and new["revision_note"] == "Opened on the door slam."
    assert "Open on the door slam" in fake.text("RevisedCut")


# ---------------------------------------------------------------------------- evaluator
EVAL_OK = {"hook_reason": "slap", "hook": 8, "cliffhanger_reason": "threat", "cliffhanger": 9,
           "standalone_reason": "clear", "standalone": 7, "passed": True, "fix_instruction": ""}


def test_evaluator_sees_only_promo_text(settings, show):
    agents, fake = make(settings, {"CutterOutput": [{"promos": [cut()]}], "EvalOutput": [EVAL_OK]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    text = render_promo_text(p, {1: show.episode(1)}, agents.pats)
    ev = agents.evaluate(text, p, {})
    assert ev["scores"] == {"hook": 8, "cliffhanger": 9, "standalone": 7} and ev["passed"] is True
    assert ev["fix_instruction"] is None and ev["reasons"]["hook"] == "slap"
    prompt = fake.text("EvalOutput")
    assert text in prompt
    for leak in ("Test Show", "romance drama", "Synthetic episode", "Mara slaps Tomas", "Opens on violence"):
        assert leak not in prompt                                        # no show info, no maker's pitch


def test_evaluator_fail_needs_fix_instruction(settings, show):
    fail = {**EVAL_OK, "hook": 4, "passed": False, "fix_instruction": ""}
    agents, fake = make(settings, {"EvalOutput": [fail, {**fail, "fix_instruction": "Start on the slap."}]})
    ev = agents.evaluate("[00:00.00] MARA: hi\n[00:21.00] END OF PROMO", {"promo_id": "c01_1"}, {})
    assert ev["passed"] is False and ev["fix_instruction"] == "Start on the slap."
    assert "fix_instruction" in fake.seen["EvalOutput"][1][-1].content


# ---------------------------------------------------------------------------- failures
def test_exhausted_retries_raise_agent_error(settings, show):
    bad = {**READ_OK, "moments": [moment("hook", "00:05.00")]}
    agents, _ = make(settings, {"ReaderOutput": [bad] * (settings.llm.max_retries + 1)})
    with pytest.raises(AgentError, match="reader"):
        agents.read_episode(show.episode(1), "", {})


def test_memory_rules_reach_the_prompt(settings, show):
    agents, fake = make(settings, {"ReaderOutput": [READ_OK]})
    agents.read_episode(show.episode(1), "", {"enabled": True, "rules": [{"text": "Never open on a camera move."}]})
    assert "Never open on a camera move." in fake.text("ReaderOutput")


def test_duration_error_names_valid_end_times(settings, show):
    settings.promo.max_duration_s = 30                          # 00:04.00 + 30s -> ends up to 00:34.00
    long = cut(end_timestamp="00:47.50", cliffhanger_timestamp="00:44.00")
    agents, fake = make(settings, {"CutterOutput": [{"promos": [long]}, {"promos": [cut()]}]})
    agents.cut_episode(show.episode(1), MOMENTS, {})
    retry = fake.seen["CutterOutput"][1][-1].content
    assert "over by 13.5s" in retry
    assert "00:27.50" in retry and "00:31.50" in retry          # real line ends that fit 20-30s


def test_joined_duration_error_says_how_much_to_cut(settings, show):
    settings.promo.max_duration_s = 25
    agents, fake = make(settings, {"JoinerOutput": [{"promos": [join()]}, {"promos": [join(segments=[
        {"episode": 1, "start_timestamp": "00:04.00", "end_timestamp": "00:11.50"},
        {"episode": 2, "start_timestamp": "00:40.00", "end_timestamp": "00:55.50"}])]}]})
    agents.join({e.number: e for e in show.episodes}, J_MOMENTS, [], {})
    assert "over by 2.0s" in fake.seen["JoinerOutput"][1][-1].content


def test_cutter_prompt_gives_end_window_for_each_hook(settings, show):
    agents, fake = make(settings, {"CutterOutput": [{"promos": [cut()]}]})
    agents.cut_episode(show.episode(1), MOMENTS, {})
    prompt = fake.text("CutterOutput")
    # m01_h1 at 00:04.00 with 20-60s limits: a clip starting there must end 00:24.00-01:04.00
    assert "m01_h1 | hook | 00:04.00" in prompt and "end between 00:24.00 and 01:04.00" in prompt


# ---------------------------------------------------------------- salvage partial answers
def test_cutter_keeps_valid_promos_when_one_never_passes(settings, show):
    bad = cut(end_timestamp="00:15.50")                       # always too short
    good = cut()
    agents, _ = make(settings, {"CutterOutput": [{"promos": [good, bad]}] * (settings.llm.max_retries + 1)})
    promos = agents.cut_episode(show.episode(1), MOMENTS, {})
    assert [p["promo_id"] for p in promos] == ["c01_1"] and promos[0]["total_duration_seconds"] == 27.5
    assert promos[0]["llm"]["salvaged"] == {"kept": 1, "dropped": 1}


def test_joiner_keeps_valid_promos_when_one_never_passes(settings, show):
    short = join(segments=[{"episode": 1, "start_timestamp": "00:04.00", "end_timestamp": "00:07.50"},
                           {"episode": 2, "start_timestamp": "00:48.00", "end_timestamp": "00:55.50"}])
    agents, _ = make(settings, {"JoinerOutput": [{"promos": [short, join()]}] * (settings.llm.max_retries + 1)})
    promos = agents.join({e.number: e for e in show.episodes}, J_MOMENTS, [], {})
    assert [p["promo_id"] for p in promos] == ["j01"] and promos[0]["total_duration_seconds"] == 27.0


def test_nothing_salvageable_is_still_an_agent_error(settings, show):
    bad = cut(end_timestamp="00:15.50")
    agents, _ = make(settings, {"CutterOutput": [{"promos": [bad]}] * (settings.llm.max_retries + 1)})
    with pytest.raises(AgentError):
        agents.cut_episode(show.episode(1), MOMENTS, {})


# ---------------------------------------------------------------- FeedbackAgent (Phase 2)
def test_feedback_agent_reads_comment_with_the_cut_and_screenplay(settings, show):
    plan = {"interpretation": "The editor wants the clip to stop right after the threat lands.",
            "instruction": "End at 00:31.50, right after line 7; keep the start at 00:04.00.",
            "keep": "the opening slap"}
    agents, fake = make(settings, {"CutterOutput": [{"promos": [cut()]}], "FixPlan": [plan]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    out = agents.interpret_fix(p, "end on the threat, not after it", {1: show.episode(1)}, {})
    assert out["interpretation"].startswith("The editor wants") and "00:31.50" in out["instruction"]
    assert out["keep"] == "the opening slap" and out["llm"]["mode"] == "native"
    prompt = fake.text("FixPlan")
    assert "end on the threat, not after it" in prompt                # the editor's words
    assert "END OF PROMO" in prompt                                   # the cut as the viewer sees it
    assert "00:28.00 | 00:31.50 | MARA (Angry) | Line 7 of episode 1." in prompt   # screenplay to point at


def test_feedback_agent_may_only_cite_real_timestamps(settings, show):
    bad = {"interpretation": "x", "instruction": "End at 00:30.00 instead.", "keep": ""}   # mid-line, not a boundary
    good = {"interpretation": "x", "instruction": "End at 00:31.50 instead.", "keep": ""}
    agents, fake = make(settings, {"CutterOutput": [{"promos": [cut()]}], "FixPlan": [bad, good]})
    [p] = agents.cut_episode(show.episode(1), MOMENTS, {})
    out = agents.interpret_fix(p, "end a bit later", {1: show.episode(1)}, {})
    retry = fake.seen["FixPlan"][1][-1].content
    assert "00:30.00" in retry and "00:31.50" in retry and out["instruction"] == "End at 00:31.50 instead."


# ---------------------------------------------------------------- FeedbackAgent learn (Phase 3)
FEEDBACK = [{"id": 11, "show": "S", "genre": "romance", "promo_id": "c01_1", "decision": "reject",
             "comment": "starts on scenery, no hook", "moment_type": "reveal", "fix_interpretation": None,
             "promo": {"hook": "Wide shot of the castle", "cliffhanger": "She gasps", "total_duration_seconds": 40}},
            {"id": 12, "show": "S", "genre": "romance", "promo_id": "c02_1", "decision": "fix",
             "comment": "open on the slap, not the hallway", "moment_type": "confrontation", "fix_interpretation": None,
             "promo": {"hook": "Hallway", "cliffhanger": "Slap", "total_duration_seconds": 35}}]
RULES = [{"id": 3, "text": "End right on the unanswered question.", "status": "active", "scope": "global",
          "scope_value": None, "supporting_count": 2, "contradicting_count": 0, "confidence": 0.75}]
LEARN_OK = {"relations": [], "conflicts": [],
            "new_rules": [{"text": "Open on the first line of open conflict, never on scenery or setup.",
                           "source_feedback_ids": [11, 12], "conflicts_with": [], "why": "both say the opening is weak"}]}


def test_learn_sees_feedback_and_rules_and_returns_rules(settings):
    agents, fake = make(settings, {"LearnOutput": [LEARN_OK]})
    out = agents.learn(FEEDBACK, [], RULES, {"show": "S", "genre": "romance", "min_support": 2})
    assert out["new_rules"][0]["source_feedback_ids"] == [11, 12] and out["llm"]["mode"] == "native"
    prompt = fake.text("LearnOutput")
    assert "#11" in prompt and "open on the slap, not the hallway" in prompt and "R3" in prompt
    assert "End right on the unanswered question." in prompt


@pytest.mark.parametrize("bad, expect", [
    ({**LEARN_OK, "new_rules": [{**LEARN_OK["new_rules"][0], "source_feedback_ids": [11]}]}, "at least 2"),
    ({**LEARN_OK, "new_rules": [{**LEARN_OK["new_rules"][0], "source_feedback_ids": [11, 99]}]}, "unknown feedback"),
    ({**LEARN_OK, "relations": [{"feedback_id": 11, "rule_id": 42, "relation": "supports", "why": ""}]}, "unknown rule"),
    ({**LEARN_OK, "conflicts": [{"rule_a": 3, "rule_b": 3, "why": ""}]}, "two different rules"),
])
def test_learn_checks_ids_and_support(settings, bad, expect):
    agents, fake = make(settings, {"LearnOutput": [bad, LEARN_OK]})
    agents.learn(FEEDBACK, [], RULES, {"show": "S", "genre": "romance", "min_support": 2})
    assert expect in fake.seen["LearnOutput"][1][-1].content


def test_editor_memory_section_is_labelled_with_rules_and_examples():
    from agents.llm_agents import render_memory
    text = render_memory({"enabled": True, "guidelines_version": "v4",
                          "rules": [{"id": 3, "text": "End right on the unanswered question.", "scope": "global"}],
                          "examples": [{"label": "approved", "summary": "opens on the slap"},
                                       {"label": "rejected", "summary": "opens on scenery", "reason": "no hook"}]})
    assert text.startswith("Editor memory") and "v4" in text
    assert "End right on the unanswered question." in text
    assert "Approved by the editor" in text and "opens on the slap" in text
    assert "Rejected by the editor" in text and "opens on scenery" in text
    assert render_memory({"enabled": False}).startswith("Editor memory: none")
