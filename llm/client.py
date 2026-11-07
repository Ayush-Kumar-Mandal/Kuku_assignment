"""One structured-output LLM call path for every agent.

    result = client.structured("cutter", prompt, {"episode": ...}, CutterOutput, checks=[...])
    result.output  -> validated CutterOutput
    result.meta    -> mode (native/fallback), retries, tokens, log files

Flow per call:
  1. Pick the mode. "auto" uses native structured output when the LangChain model
     profile says the model supports it, otherwise JSON mode + Pydantic parsing.
  2. Invoke. Native output is re-validated with Pydantic anyway (score ranges etc.),
     because providers don't all enforce min/max constraints from the JSON Schema.
  3. Run the custom checks (timestamps exist, durations, line boundaries).
  4. On any failure, retry up to llm.max_retries with the errors in a new message.
  Rate limits are waited out (llm.rate_limit_backoff_s) without using up a retry;
  when the waits run out RateLimitExhausted is raised so the graph can stop and later
  resume from its checkpoint.
Every attempt's raw response is written to logs/llm/<run_id>/.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from config.loader import ModelSpec, Settings
from llm.prompts import load_fragment
from llm.usage import UsageTracker

log = logging.getLogger("llm")

T = TypeVar("T", bound=BaseModel)
Check = Callable[[Any], list[str]]          # validated output -> list of problems
ModelFactory = Callable[[ModelSpec, Settings], Any]
Mode = Literal["native", "fallback"]


# ------------------------------------------------------------------------- errors
class LLMConfigError(RuntimeError):
    """Missing API key, bad credentials, unknown model: retrying won't help."""


class RateLimitExhausted(RuntimeError):
    """Still rate-limited after every configured backoff wait, or a daily quota is used up
    (daily=True: waiting minutes can't help; resume the run once the quota resets)."""

    def __init__(self, msg: str, daily: bool = False, retry_after_s: float | None = None):
        super().__init__(msg)
        self.daily = daily
        self.retry_after_s = retry_after_s


class StructuredOutputError(RuntimeError):
    """No answer passed validation + checks. `last_output` is the last answer that passed
    schema validation (but failed a custom check), so a caller can salvage its valid parts."""

    def __init__(self, msg: str, meta: "CallMeta", last_output: Any = None):
        super().__init__(msg)
        self.meta = meta
        self.last_output = last_output


def _status(e: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return (_status(e) == 429 or "ratelimit" in type(e).__name__.lower()
            or "resource_exhausted" in s or "rate limit" in s or "quota" in s or "429" in s)


_OVERLOAD = ("unavailable", "overloaded", "high demand", "try again later", "internal error", "500 internal",
             "bad gateway", "gateway timeout")


_RETRY_DELAY = re.compile(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


_TRY_AGAIN = re.compile(r"(?:retry|try again)\s+in\s+(\d+(?:\.\d+)?)s\b", re.I)


def _short_hint(e: Exception) -> float | None:
    """Seconds from "Please try again in 14.46s" (per-minute limits). None if absent."""
    m = _TRY_AGAIN.search(str(e))
    return float(m.group(1)) if m else None


def _daily_quota(e: Exception, max_wait_s: float) -> tuple[bool, float | None]:
    """A per-day quota (or a server-requested wait longer than all our backoff): no point waiting."""
    s = str(e)
    m = _RETRY_DELAY.search(s)
    delay = float(m.group(1)) if m else None
    daily = _is_rate_limit(e) and ("perday" in s.lower().replace("_", "").replace(" ", "")
                                   or "per day" in s.lower() or (delay is not None and delay > max_wait_s))
    return daily, delay


def _is_output_rejection(e: Exception) -> bool:
    """The provider checked the MODEL's answer against the schema and it failed (Groq:
    json_validate_failed). A model mistake to retry, not a missing native feature."""
    s = str(e).lower()
    return "json_validate_failed" in s or "failed to validate json" in s or "failed_generation" in s


def _is_transient(e: Exception) -> bool:
    """Rate limit or a temporary server problem (5xx, "high demand"): wait and resend the same
    request. Not a sign the model can't do structured output, and not a model mistake."""
    s = str(e).lower()
    return _is_rate_limit(e) or _status(e) in (500, 502, 503, 504) or any(k in s for k in _OVERLOAD)


def _is_fatal(e: Exception) -> bool:
    """Bad key, no access, unknown model: retrying can never succeed."""
    s = str(e).lower()
    name = type(e).__name__.lower()
    return _status(e) in (401, 403, 404) or "api key not valid" in s or "invalid api key" in s \
        or "permission_denied" in s or "model_not_found" in s or "does not exist" in s \
        or "authentication" in name or "notfound" in name


# --------------------------------------------------------------------------- meta
@dataclass
class CallMeta:
    call_id: str
    agent: str
    provider: str
    model: str
    mode: Mode
    trace_id: str | None = None
    label: str | None = None
    switched_to_fallback: bool = False      # native was tried, failed, JSON mode used
    attempts: int = 0
    retries: int = 0                        # attempts after the first
    rate_limit_waits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False          # provider sent no usage; chars/4 estimate
    latency_s: float = 0.0
    success: bool = False
    errors: list[list[str]] = field(default_factory=list)   # per failed attempt
    model_switches: list[dict] = field(default_factory=list)  # quota ran out -> backup model
    log_files: list[str] = field(default_factory=list)


@dataclass
class CallResult(Generic[T]):
    output: T
    meta: CallMeta


# ----------------------------------------------------------------------- helpers
def _text(msg: Any) -> str:
    if msg is None:
        return ""
    t = getattr(msg, "text", None)
    if isinstance(t, str):                 # langchain-core 1.x: property (a callable str subclass)
        return str(t)
    if callable(t):                        # older versions: method
        t = t()
        if isinstance(t, str):
            return t
    c = getattr(msg, "content", "")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, default=str)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> Any:
    """Parse a JSON object from model text, tolerating ``` fences and stray prose."""
    s = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in the response")
        return json.loads(s[start:end + 1])


def _format_errors(e: Exception | list[str]) -> list[str]:
    if isinstance(e, list):
        return e
    if isinstance(e, ValidationError):
        return [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
    return [f"{type(e).__name__}: {e}"]


def _usage(msg: Any, messages: Sequence[BaseMessage]) -> tuple[int, int, bool]:
    um = getattr(msg, "usage_metadata", None) or {}
    if um.get("input_tokens") or um.get("output_tokens"):
        return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0)), False
    # No usage reported: rough chars/4 estimate so cost numbers are never silently zero.
    est_in = sum(len(_text(m)) for m in messages) // 4
    return est_in, len(_text(msg)) // 4, True


def default_model_factory(spec: ModelSpec, settings: Settings):
    from langchain.chat_models import init_chat_model

    prov = settings.providers[spec.provider]
    if not os.environ.get(prov.api_key_env):
        raise LLMConfigError(f"{prov.api_key_env} is not set (needed for {spec.provider}:{spec.model}). "
                             f"Copy .env.example to .env and add your key.")
    extra = dict(spec.kwargs)
    if spec.temperature is not None:     # None = provider default (some models ignore it)
        extra["temperature"] = spec.temperature
    return init_chat_model(
        spec.model, model_provider=spec.provider,
        timeout=settings.llm.request_timeout_s, max_retries=settings.llm.provider_max_retries,
        **extra,
    )


# ------------------------------------------------------------------------ client
class LLMClient:
    def __init__(self, settings: Settings, run_id: str | None = None,
                 model_factory: ModelFactory | None = None, sleep: Callable[[float], None] = time.sleep):
        load_dotenv()
        self.settings = settings
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = settings.resolve(settings.paths.logs_dir) / "llm" / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir = settings.resolve(settings.paths.prompts_dir)
        self._factory = model_factory or default_model_factory
        self._sleep = sleep
        self._models: dict[tuple[str, int], Any] = {}
        self._active: dict[str, int] = {}                    # agent -> index in its model chain
        self._native_broken: set[tuple[str, str]] = set()   # (provider, model) that failed native
        self._seq = 0
        self.usage = UsageTracker()

    # -- model selection --------------------------------------------------------------
    def spec(self, agent: str) -> ModelSpec:
        """The model currently in use for `agent` (its config entry, or a backup after a switch)."""
        return self.settings.model_for(agent).chain()[self._active.get(agent, 0)]

    def model(self, agent: str):
        key = (agent, self._active.get(agent, 0))
        if key not in self._models:
            self._models[key] = self._factory(self.spec(agent), self.settings)
        return self._models[key]

    def _switch_model(self, agent: str, meta: "CallMeta", e: "RateLimitExhausted") -> bool:
        """Move `agent` to its next backup model for the rest of the run. False if none left."""
        chain = self.settings.model_for(agent).chain()
        i = self._active.get(agent, 0)
        if i + 1 >= len(chain):
            return False
        old, new = chain[i], chain[i + 1]
        self._active[agent] = i + 1
        meta.model_switches.append({"from": f"{old.provider}:{old.model}", "to": f"{new.provider}:{new.model}",
                                    "reason": "daily quota" if e.daily else "rate-limited/overloaded"})
        meta.provider, meta.model = new.provider, new.model
        log.warning("%s: %s:%s out of quota (%s); switching to %s:%s for the rest of the run",
                    agent, old.provider, old.model, meta.model_switches[-1]["reason"], new.provider, new.model)
        return True

    def resolve_mode(self, agent: str) -> Mode:
        spec = self.spec(agent)
        if spec.structured_output != "auto":
            return spec.structured_output
        if (spec.provider, spec.model) in self._native_broken:
            return "fallback"
        profile = getattr(self.model(agent), "profile", None) or {}
        # Unknown profile (None): try native; a failure drops to fallback automatically.
        return "fallback" if profile.get("structured_output") is False or (
            profile and "structured_output" not in profile) else "native"

    # -- main entry point -------------------------------------------------------------
    def structured(self, agent: str, prompt: ChatPromptTemplate, variables: dict, schema: type[T],
                   checks: Sequence[Check] = (), trace_id: str | None = None,
                   label: str | None = None) -> CallResult[T]:
        spec = self.spec(agent)
        self._seq += 1
        meta = CallMeta(call_id=f"{self._seq:04d}_{agent}_{uuid.uuid4().hex[:6]}", agent=agent,
                        provider=spec.provider, model=spec.model, mode=self.resolve_mode(agent),
                        trace_id=trace_id, label=label)
        base = list(prompt.format_messages(**variables))
        followups: list[BaseMessage] = []
        last_valid: T | None = None
        max_attempts = self.settings.llm.max_retries + 1
        t0 = time.monotonic()

        try:
            while meta.attempts < max_attempts:
                meta.attempts += 1
                messages = self._with_mode_instructions(base, schema, meta.mode) + followups
                try:
                    raw, data, error = self._invoke_with_backoff(agent, meta, messages, schema)
                except LLMConfigError:
                    raise
                except RateLimitExhausted as e:
                    if self._switch_model(agent, meta, e):
                        # Same request on the backup model; not a retry. Mode is re-decided for it.
                        meta.attempts -= 1
                        meta.mode, meta.switched_to_fallback = self.resolve_mode(agent), False
                        continue
                    if len(self.settings.model_for(agent).chain()) > 1:
                        raise RateLimitExhausted(f"{e} -- no backup model left for {agent}",
                                                 daily=e.daily, retry_after_s=e.retry_after_s) from e
                    raise
                except Exception as e:  # provider rejected the request
                    if (meta.mode == "native" and spec.structured_output == "auto" and not meta.switched_to_fallback
                            and not _is_output_rejection(e)):
                        # e.g. the model/endpoint doesn't support json_schema: drop to JSON mode
                        # for this model for the rest of the run; doesn't count as a retry.
                        log.warning("%s: native structured output failed (%s); switching to JSON mode", agent, e)
                        cur = self.spec(agent)            # may be a backup model after a switch
                        self._native_broken.add((cur.provider, cur.model))
                        meta.mode, meta.switched_to_fallback = "fallback", True
                        meta.attempts -= 1
                        self._log_attempt(meta, messages, None, [f"native failed: {e}"])
                        continue
                    raw, data, error = None, None, e

                problems: list[str] = []
                output: T | None = None
                if error is not None:
                    problems = _format_errors(error)
                else:
                    try:
                        output = schema.model_validate(data)   # always re-validate
                    except ValidationError as ve:
                        problems = _format_errors(ve)
                if output is not None:
                    last_valid = output
                    for check in checks:
                        try:
                            problems += check(output)
                        except Exception as ce:  # a broken check must not crash the run
                            problems.append(f"check {getattr(check, '__name__', check)} crashed: {ce}")

                self._log_attempt(meta, messages, raw, problems)
                if not problems:
                    meta.success = True
                    return CallResult(output=output, meta=meta)

                meta.errors.append(problems)
                log.info("%s attempt %d/%d rejected: %s", agent, meta.attempts, max_attempts, "; ".join(problems)[:300])
                previous = json.dumps(data, ensure_ascii=False, indent=1) if data is not None else _text(raw)
                followups = [HumanMessage(load_fragment(self.prompts_dir, "_shared", "retry").format(
                    previous=previous[:6000] or "(empty)", errors="\n".join(f"- {p}" for p in problems)))]

            raise StructuredOutputError(
                f"{agent}: no valid output after {meta.attempts} attempts; last problems: {meta.errors[-1]}", meta,
                last_output=last_valid)
        finally:
            meta.retries = max(meta.attempts - 1, 0)
            meta.latency_s = round(time.monotonic() - t0, 2)
            self.usage.record(meta)
            self._log_call(meta)

    # -- invocation -------------------------------------------------------------------
    def _with_mode_instructions(self, base: list[BaseMessage], schema: type[BaseModel], mode: Mode) -> list[BaseMessage]:
        if mode == "native":
            return list(base)
        instr = load_fragment(self.prompts_dir, "_shared", "json_fallback").format(
            schema=json.dumps(schema.model_json_schema(), ensure_ascii=False))
        # Merge into the existing system message: some providers (Gemini) accept only one.
        out = list(base)
        for i, m in enumerate(out):
            if isinstance(m, SystemMessage):
                out[i] = SystemMessage(content=f"{_text(m)}\n\n{instr}")
                return out
        return [SystemMessage(content=instr)] + out

    def _invoke_with_backoff(self, agent, meta: CallMeta, messages, schema):
        waits = self.settings.llm.rate_limit_backoff_s
        i = 0
        while True:
            try:
                return self._invoke_once(agent, meta, messages, schema)
            except Exception as e:
                if _is_fatal(e):
                    raise LLMConfigError(f"{meta.provider}:{meta.model} cannot be used (check key / model name "
                                         f"in config/settings.yaml): {e}") from e
                if not _is_transient(e):
                    raise
                daily, delay = _daily_quota(e, max(waits, default=0))
                if daily:
                    when = f"; provider says retry in {delay / 3600:.1f}h" if delay else ""
                    raise RateLimitExhausted(
                        f"{agent}: {meta.provider}:{meta.model} daily quota used up{when}. The run is "
                        f"checkpointed: resume it later, or point this agent at another model in "
                        f"config/settings.yaml and resume now. ({str(e)[:200]})",
                        daily=True, retry_after_s=delay) from e
                hint = _short_hint(e)
                # A per-minute limit with the provider's own "try again in Xs" always clears:
                # keep waiting (up to max_rate_limit_waits) instead of stopping the run.
                limit = self.settings.llm.max_rate_limit_waits if hint is not None else len(waits)
                if i >= limit:
                    raise RateLimitExhausted(f"{agent}: still rate-limited/overloaded after {i} waits: {e}") from e
                wait = waits[min(i, len(waits) - 1)] if waits else 0.0
                if hint is not None:
                    wait = max(min(wait, max(waits, default=wait)), hint + 0.5)
                meta.rate_limit_waits += 1
                why = "rate-limited" if _is_rate_limit(e) else "provider overloaded"
                log.warning("%s %s; waiting %.1fs (%d/%d)", agent, why, wait, i + 1, limit)
                self._sleep(wait)
                i += 1

    def _invoke_once(self, agent, meta: CallMeta, messages, schema):
        """Returns (raw AIMessage, data dict | None, error | None)."""
        model = self.model(agent)
        prov = self.settings.providers[meta.provider]
        if meta.mode == "native":
            runnable = model.with_structured_output(schema, method=prov.native_method, include_raw=True)
            res = runnable.invoke(messages)
            raw, parsed, err = res.get("raw"), res.get("parsed"), res.get("parsing_error")
            self._count_tokens(meta, raw, messages)
            if parsed is None:
                return raw, None, err or ValueError("model returned no structured output")
            data = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
            return raw, data, None

        bound = model.bind(**prov.json_mode_kwargs) if prov.json_mode_kwargs else model
        raw = bound.invoke(messages)
        self._count_tokens(meta, raw, messages)
        try:
            return raw, extract_json(_text(raw)), None
        except ValueError as e:
            return raw, None, e

    def _count_tokens(self, meta: CallMeta, raw, messages):
        i, o, est = _usage(raw, messages)
        meta.input_tokens += i
        meta.output_tokens += o
        meta.tokens_estimated |= est

    # -- logging ----------------------------------------------------------------------
    def _log_attempt(self, meta: CallMeta, messages, raw, problems: list[str]):
        path = self.log_dir / f"{meta.call_id}_a{meta.attempts}{'_native' if meta.mode == 'native' else ''}.json"
        record = {
            "call_id": meta.call_id, "agent": meta.agent, "provider": meta.provider, "model": meta.model,
            "mode": meta.mode, "attempt": meta.attempts, "trace_id": meta.trace_id, "label": meta.label,
            "messages": [{"role": m.type, "content": _text(m)} for m in messages],
            "response": None if raw is None else {
                "content": _text(raw),
                "tool_calls": getattr(raw, "tool_calls", None),
                "usage_metadata": getattr(raw, "usage_metadata", None),
                "response_metadata": getattr(raw, "response_metadata", None),
            },
            "problems": problems,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        meta.log_files.append(str(path))

    def _log_call(self, meta: CallMeta):
        with (self.log_dir / "calls.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(meta), ensure_ascii=False, default=str) + "\n")
        (self.log_dir / "usage.json").write_text(json.dumps(self.usage.to_dict(), indent=2), encoding="utf-8")
        log.info("%s %s:%s mode=%s%s ok=%s retries=%d tokens=%d/%d%s %.1fs", meta.agent, meta.provider,
                 meta.model, meta.mode, " (switched)" if meta.switched_to_fallback else "", meta.success,
                 meta.retries, meta.input_tokens, meta.output_tokens, "~" if meta.tokens_estimated else "",
                 meta.latency_s)
