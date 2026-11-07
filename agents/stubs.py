"""Deterministic stand-ins for the LLM agents (Step 3: graph skeleton).

They use real timestamps from the parsed screenplay, so everything downstream (checks,
ranking, UI, traces) sees realistic data, but no tokens are spent. Knobs let tests force
each graph path: evaluator failures, revision rounds, crashes mid-run.

Promo shape (the LLM agents build the same dicts from agents/schemas.py outputs):
  {promo_id, maker, version, segments: [{episode, start_timestamp, end_timestamp}],
   total_duration_seconds, hook_moment_id, cliffhanger_moment_id, source_moment_ids,
   hook, cliffhanger, moment_type, why_it_works, join_rationale?, revision_note?}
"""
from __future__ import annotations

import copy
from collections import Counter

from agents.base import AgentError
from config.loader import Settings
from parser.models import Episode


class StubCrash(RuntimeError):
    """Raised on purpose to test resume-from-checkpoint."""


class StubAgents:
    def __init__(self, settings: Settings, fail_versions: set[int] | None = None,
                 always_fail: set[str] | None = None, crash_on: dict[str, int] | None = None,
                 error_on: dict[str, set[int]] | None = None):
        self.settings = settings
        self.fail_versions = {1} if fail_versions is None else fail_versions
        self.always_fail = always_fail or set()
        self.crash_on = dict(crash_on or {})     # method -> crash on the Nth call
        self.error_on = dict(error_on or {})     # method -> call numbers raising AgentError
        self.calls: Counter = Counter()
        self.memory_seen: dict[str, dict] = {}   # method -> last editor memory it was given

    def _tick(self, method: str):
        self.calls[method] += 1
        if self.crash_on.get(method) == self.calls[method]:
            raise StubCrash(f"stub crash in {method} call {self.calls[method]}")
        if self.calls[method] in self.error_on.get(method, ()):
            raise AgentError(f"stub: {method} call {self.calls[method]} gave no valid output")

    # ---------------------------------------------------------------- reader
    def read_episode(self, episode: Episode, story_summary: str, memory: dict) -> dict:
        self._tick("read_episode")
        self.memory_seen['read_episode'] = memory
        lines = [it for it in episode.all_items() if it.kind == "dialogue"] or episode.all_items()
        first, last = lines[0], lines[-1]
        n = episode.number
        mk = lambda kind, it, strength: {
            "moment_id": f"m{n:02d}_{kind[0]}1", "episode": n, "kind": kind,
            "timestamp": it.start_ts, "description": f"{it.character or it.kind}: {it.text[:80]}",
            "strength": strength, "moment_type": "confrontation"}
        return {
            "moments": [mk("hook", first, 7), mk("cliffhanger", last, 8)],
            "new_viewer_context": episode.logline or "",
            "story_summary": (story_summary + f" Ep{n}: {episode.logline or ''}").strip()[-1500:],
        }

    # ---------------------------------------------------------------- cutter
    def cut_episode(self, episode: Episode, moments: list[dict], memory: dict) -> list[dict]:
        self._tick("cut_episode")
        self.memory_seen['cut_episode'] = memory
        pc = self.settings.promo
        hook = next((m for m in moments if m["kind"] == "hook"), None)
        cliff = next((m for m in moments if m["kind"] == "cliffhanger"), None)
        if hook is None:
            return []
        start = next(it for it in episode.all_items() if it.start_ts == hook["timestamp"])
        # End on the first dialogue line that gives at least min + 5 seconds.
        ends = [it for it in episode.all_items() if it.kind == "dialogue"
                and pc.min_duration_s + 5 <= it.end - start.start <= pc.max_duration_s]
        if not ends:
            return []
        end = ends[0]
        return [{
            "promo_id": f"c{episode.number:02d}_1", "maker": "cutter", "version": 1,
            "segments": [{"episode": episode.number, "start_timestamp": start.start_ts, "end_timestamp": end.end_ts}],
            "total_duration_seconds": round(end.end - start.start, 2),
            "hook_moment_id": hook["moment_id"], "cliffhanger_moment_id": (cliff or hook)["moment_id"],
            "source_moment_ids": [hook["moment_id"]],
            "hook": hook["description"], "cliffhanger": f"{end.character}: {end.text[:80]}",
            "moment_type": hook["moment_type"], "why_it_works": "stub: opens on the first line of conflict",
        }]

    # ---------------------------------------------------------------- joiner
    def join(self, episodes: dict[int, Episode], moments: list[dict], candidates: list[dict], memory: dict) -> list[dict]:
        self._tick("join")
        self.memory_seen['join'] = memory
        pc = self.settings.promo
        short = sorted(candidates, key=lambda c: c["total_duration_seconds"])
        for a in short:
            for b in short:
                if a is b or a["segments"][0]["episode"] == b["segments"][0]["episode"]:
                    continue
                total = a["total_duration_seconds"] + b["total_duration_seconds"]
                if total <= pc.max_duration_s:
                    return [{
                        "promo_id": "j01", "maker": "joiner", "version": 1,
                        "segments": copy.deepcopy(a["segments"] + b["segments"]),
                        "total_duration_seconds": round(total, 2),
                        "hook_moment_id": a["hook_moment_id"], "cliffhanger_moment_id": b["cliffhanger_moment_id"],
                        "source_moment_ids": a["source_moment_ids"] + b["source_moment_ids"],
                        "hook": a["hook"], "cliffhanger": b["cliffhanger"], "moment_type": a["moment_type"],
                        "why_it_works": "stub: joins two short clips",
                        "join_rationale": f"stub: {a['promo_id']} sets up, {b['promo_id']} pays off",
                    }]
        return []

    # ---------------------------------------------------------------- revise
    def revise(self, maker: str, promo: dict, instruction: str, episodes: dict[int, Episode], memory: dict,
               source: str = "evaluator") -> dict:
        self._tick(f"revise_{maker}")
        new = copy.deepcopy(promo)
        new["version"] = promo["version"] + 1
        new["revision_note"] = f"applied: {instruction}"
        return new

    # ---------------------------------------------------------------- feedback
    def interpret_fix(self, promo: dict, comment: str, episodes: dict[int, Episode], memory: dict) -> dict:
        self._tick("interpret_fix")
        return {"interpretation": f"stub reading of: {comment}", "instruction": f"stub instruction for: {comment}"}

    def learn(self, new_feedback: list[dict], older_feedback: list[dict], rules: list[dict], context: dict) -> dict:
        """Deterministic stand-in for FeedbackAgent job 2: identical comments seen at least
        min_support times become a rule; a new comment found inside a rule's text supports it."""
        self._tick("learn")
        relations = [{"feedback_id": f["id"], "rule_id": r["id"], "relation": "supports", "why": "same advice"}
                     for f in new_feedback for r in rules
                     if f["comment"] and f["comment"].strip().lower() in r["text"].lower()]
        groups: dict[str, list[int]] = {}
        for f in new_feedback + older_feedback:
            if f["comment"].strip():
                groups.setdefault(f["comment"].strip().lower(), []).append(f["id"])
        taken = {r["text"].lower() for r in rules}
        new_rules = [{"text": f"Editors asked: {c}", "source_feedback_ids": ids, "conflicts_with": [], "why": "repeated"}
                     for c, ids in groups.items() if len(ids) >= context["min_support"]
                     and f"editors asked: {c}" not in taken]
        return {"relations": relations, "new_rules": new_rules, "conflicts": []}

    # ---------------------------------------------------------------- evaluator
    def evaluate(self, promo_text: str, promo: dict, memory: dict) -> dict:
        self._tick("evaluate")
        self.memory_seen['evaluate'] = memory
        fail = promo["promo_id"] in self.always_fail or promo["version"] in self.fail_versions
        base = 5 if fail else 8
        # Later versions score a little higher, so "keep best" has something to pick.
        bump = min(promo["version"] - 1, 2) * 0.5
        scores = {"hook": base + bump, "cliffhanger": base + 1, "standalone": base}
        scores = {k: int(min(10, v)) for k, v in scores.items()}
        return {"scores": scores, "passed": not fail,
                "fix_instruction": "Start one line earlier so the hook lands in the first 3 seconds." if fail else None}
