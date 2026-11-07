"""Per-agent counters for LLM calls and tokens (feeds the cost estimate in Step 8)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass
class AgentUsage:
    calls: int = 0
    attempts: int = 0
    retries: int = 0
    failures: int = 0
    native_calls: int = 0
    fallback_calls: int = 0
    switched_to_fallback: int = 0
    rate_limit_waits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_token_calls: int = 0
    model_switches: int = 0
    latency_s: float = 0.0


class UsageTracker:
    def __init__(self):
        self.by_agent: dict[str, AgentUsage] = defaultdict(AgentUsage)
        self.models: dict[str, str] = {}

    def record(self, meta) -> None:
        u = self.by_agent[meta.agent]
        self.models[meta.agent] = f"{meta.provider}:{meta.model}"
        u.calls += 1
        u.attempts += meta.attempts
        u.retries += meta.retries
        u.failures += 0 if meta.success else 1
        u.native_calls += meta.mode == "native"
        u.fallback_calls += meta.mode == "fallback"
        u.switched_to_fallback += meta.switched_to_fallback
        u.rate_limit_waits += meta.rate_limit_waits
        u.input_tokens += meta.input_tokens
        u.output_tokens += meta.output_tokens
        u.estimated_token_calls += meta.tokens_estimated
        u.model_switches += len(meta.model_switches)
        u.latency_s = round(u.latency_s + meta.latency_s, 2)

    def totals(self) -> AgentUsage:
        t = AgentUsage()
        for u in self.by_agent.values():
            for k, v in asdict(u).items():
                setattr(t, k, round(getattr(t, k) + v, 2))
        return t

    def retry_rate(self) -> float:
        t = self.totals()
        return round(t.retries / t.attempts, 3) if t.attempts else 0.0

    def to_dict(self) -> dict:
        return {
            "by_agent": {a: {"model": self.models.get(a), **asdict(u)} for a, u in self.by_agent.items()},
            "totals": asdict(self.totals()),
            "retry_rate": self.retry_rate(),
        }


def usage_from_log(calls_jsonl) -> dict:
    """Per-agent usage rebuilt from logs/llm/<run_id>/calls.jsonl. That file is appended by
    every process that worked on the run, so unlike the in-memory tracker it stays correct
    when a run is stopped and resumed (rate limits, review pauses, crashes)."""
    import json
    from collections import Counter
    from pathlib import Path

    by_agent: dict[str, AgentUsage] = defaultdict(AgentUsage)
    models: dict[str, Counter] = defaultdict(Counter)
    model_tokens: dict[str, dict[str, dict]] = defaultdict(dict)
    switches: list[dict] = []
    path = Path(calls_jsonl)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in lines:
        if not line.strip():
            continue
        c = json.loads(line)
        u = by_agent[c["agent"]]
        key = f"{c['provider']}:{c['model']}"
        models[c["agent"]][key] += 1
        mt = model_tokens[c["agent"]].setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        mt["calls"] += 1
        mt["input_tokens"] += c.get("input_tokens", 0)
        mt["output_tokens"] += c.get("output_tokens", 0)
        u.calls += 1
        u.attempts += c.get("attempts", 0)
        u.retries += c.get("retries", 0)
        u.failures += 0 if c.get("success") else 1
        u.native_calls += c.get("mode") == "native"
        u.fallback_calls += c.get("mode") == "fallback"
        u.switched_to_fallback += bool(c.get("switched_to_fallback"))
        u.rate_limit_waits += c.get("rate_limit_waits", 0)
        u.input_tokens += c.get("input_tokens", 0)
        u.output_tokens += c.get("output_tokens", 0)
        u.estimated_token_calls += bool(c.get("tokens_estimated"))
        u.model_switches += len(c.get("model_switches") or [])
        u.latency_s = round(u.latency_s + c.get("latency_s", 0.0), 2)
        switches += [{"agent": c["agent"], "call_id": c.get("call_id"), **s} for s in c.get("model_switches") or []]
    tracker = UsageTracker()
    tracker.by_agent = by_agent
    return {
        "by_agent": {a: {"models": dict(models[a]), "model_tokens": model_tokens[a], **asdict(u)}
                     for a, u in by_agent.items()},
        "totals": asdict(tracker.totals()),
        "retry_rate": tracker.retry_rate(),
        "model_switches": switches,
    }
