"""LLM-backed agents: Reader, Cutter, Joiner (+ revision mode) and Evaluator.

Same interface as agents/stubs.py (see agents/base.py). Each method:
  1. builds prompt variables (screenplay lines as an unambiguous table, limits from config),
  2. calls LLMClient.structured() with a Pydantic schema + custom checks (llm/checks.py),
     which validates, retries with the errors in the prompt, and logs every attempt,
  3. turns the validated output into the plain-dict shape stored in graph state. Ids,
     versions and durations are assigned/computed here, never trusted from the model.

Show name and genre go to the makers; the Evaluator gets ONLY the rendered promo text.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from agents.base import AgentError
from agents.render import render_promo_text
from agents.schemas import (CutterOutput, EvalOutput, FixPlan, JoinerOutput, JoinPromo, LearnOutput, ReaderOutput,
                            RevisedCut, RevisedJoin, Segment)
from config.loader import MomentTypes, ParserConfig, Settings
from llm.checks import ScreenplayIndex, check_promo_segments, check_timestamp
from llm.client import CallMeta, LLMClient, StructuredOutputError
from llm.prompts import load_chat_prompt
from parser.models import Episode, Item, format_ts

log = logging.getLogger("agents")
EMPTY_SUMMARY = "(nothing yet: this is the first episode)"
NO_MEMORY = "Editor memory: none (no editor guidelines or examples to apply)."


# --------------------------------------------------------------------------- rendering
def _who(it: Item) -> str:
    if not it.character:
        return it.kind.upper()
    tags = "".join(f" ({t})" for t in it.voice_tags)
    tone = f" ({', '.join(it.emotions)})" if it.emotions else ""
    return f"{it.character}{tags}{tone}"


def render_lines(items: list[Item]) -> str:
    """One line per item as separate columns, so a model can't mistake a range for one time."""
    return "\n".join(f"{it.start_ts} | {it.end_ts} | {_who(it)} | {it.text}" for it in items)


def render_memory(memory: dict | None) -> str:
    """The labelled "Editor memory" section injected into the Reader, Cutter, Joiner and
    Evaluator prompts (memory_read fills it from MemoryStore; spec section 8)."""
    memory = memory or {}
    rules = memory.get("rules") or []
    examples = [e for e in memory.get("examples") or [] if isinstance(e, dict)]
    if memory.get("enabled") is False or not (rules or examples):
        return NO_MEMORY
    v = f"guidelines {memory['guidelines_version']}, " if memory.get("guidelines_version") else ""
    out = [f"Editor memory ({v}learned from this editor's past reviews; follow it):"]
    if rules:
        out.append("Guidelines:")
        out += [f"- {r.get('text') or r.get('rule')}" if isinstance(r, dict) else f"- {r}" for r in rules]
    for label, title in (("approved", "Approved by the editor (aim for promos like these):"),
                         ("rejected", "Rejected by the editor (avoid what made these fail):")):
        names = {"approved": ("approved", "approve"), "rejected": ("rejected", "reject")}[label]
        mine = [e for e in examples if (e.get("label") or e.get("decision")) in names]
        if mine:
            out.append(title)
            for e in mine:
                text = e.get("summary") or e.get("text") or ""
                reason = e.get("reason")
                out.append(f"- {text}" + (f" (editor: {reason})" if reason and reason not in text else ""))
    return "\n".join(out)


def meta_dict(m: CallMeta) -> dict:
    return {"call_id": m.call_id, "provider": m.provider, "model": m.model, "mode": m.mode,
            "switched_to_fallback": m.switched_to_fallback, "attempts": m.attempts, "retries": m.retries,
            "rate_limit_waits": m.rate_limit_waits, "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens, "latency_s": m.latency_s, "model_switches": m.model_switches}


class LLMAgents:
    def __init__(self, client: LLMClient, settings: Settings, parser_cfg: ParserConfig,
                 moment_types: MomentTypes, show: str, genre: str):
        self.client = client
        self.settings = settings
        self.parser_cfg = parser_cfg
        self.pats = parser_cfg.compiled
        self.mt = moment_types
        self.show = show
        self.genre = genre
        self.kinds = settings.agents.screenplay_kinds
        # Timestamps cited in free text, found with parser.yaml's own pattern (no regex in code).
        self._ts_re = re.compile(re.sub(r"\(\?P<\w+>", "(?:", parser_cfg.timestamp.pattern))
        self._indexes: dict[tuple, ScreenplayIndex] = {}

    # ------------------------------------------------------------------ helpers
    def _index(self, episodes: list[Episode]) -> ScreenplayIndex:
        key = tuple((e.number, id(e)) for e in episodes)
        if key not in self._indexes:
            self._indexes[key] = ScreenplayIndex(episodes, self.parser_cfg, self.settings.promo.boundary_tolerance_s)
        return self._indexes[key]

    def _lines(self, ep: Episode, start: float | None = None, end: float | None = None) -> list[Item]:
        items = [it for it in ep.all_items() if it.kind in self.kinds]
        if start is not None:
            items = [it for it in items if it.end > start and it.start < end]
        return items

    def _mtype(self, value: str) -> str:
        v = (value or "").strip().lower()
        return v if v in self.mt.moment_types else self.mt.default_type

    def _common(self, memory: dict) -> dict:
        pc = self.settings.promo
        return {"show": self.show, "genre": self.genre, "memory": render_memory(memory),
                "moment_types": "\n".join(f"- {k}: {v}" for k, v in self.mt.moment_types.items()),
                "min_s": f"{pc.min_duration_s:g}", "max_s": f"{pc.max_duration_s:g}",
                "hook_window_s": f"{pc.hook_window_s:g}", "cliffhanger_window_s": f"{pc.cliffhanger_window_s:g}",
                "max_segments": pc.max_segments}

    def _call(self, agent: str, prompt_name: str, variables: dict, schema, checks, trace_id: str, label: str):
        prompt = load_chat_prompt(self.client.prompts_dir, prompt_name)
        try:
            return self.client.structured(agent, prompt, variables, schema, checks=checks,
                                          trace_id=trace_id, label=label)
        except StructuredOutputError as e:
            raise AgentError(f"{agent} ({label}): {e}") from e

    def _tid(self, key: str) -> str:
        return f"{self.client.run_id}:{key}"

    # ---------------------------------------------------------- shared promo checks
    def _line_start(self, index: ScreenplayIndex, ep: int, ts: str, what: str) -> tuple[list[str], float | None]:
        """ts must be the start time of a line shown to the model."""
        t = index.seconds(ts)
        if ep not in index.episodes:
            return [f"{what}: episode {ep} does not exist"], None
        if t is None:
            return [f"{what}: '{ts}' is not a timestamp in MM:SS.ff form"], None
        starts = sorted({it.start for it in self._lines(index.episodes[ep])})
        if not any(abs(s - t) <= index.tol for s in starts):
            return [f"{what}: {ts} is not the start of a line in episode {ep}; "
                    f"nearest line starts: {index.nearest(starts, t)}"], None
        return [], t

    def _promo_problems(self, index: ScreenplayIndex, segs: list[Segment], hook_ts: str, cliff_ts: str,
                        ids: list[str], known: set[str] | None, name: str) -> list[str]:
        pc = self.settings.promo
        probs = check_promo_segments(index, segs, self.settings, name)
        first, last = segs[0], segs[-1]
        s0, e_last = index.seconds(first.start_timestamp), index.seconds(last.end_timestamp)
        s_last = index.seconds(last.start_timestamp)
        p, th = self._line_start(index, first.episode, hook_ts, f"{name} hook_timestamp")
        probs += p
        if th is not None and s0 is not None:
            offset = th - s0
            if offset < -index.tol or offset > pc.hook_window_s + index.tol:
                where = "the first segment" if len(segs) > 1 else "the clip"
                probs.append(f"{name}: hook line starts {offset:.2f}s into {where}; it must start in the first "
                             f"{pc.hook_window_s:g}s (move the start to {hook_ts} or just before it)")
        p, tc = self._line_start(index, last.episode, cliff_ts, f"{name} cliffhanger_timestamp")
        probs += p
        if tc is not None and s_last is not None and e_last is not None:
            if not (s_last - index.tol <= tc <= e_last + index.tol):
                probs.append(f"{name}: cliffhanger line {cliff_ts} must be inside the last segment "
                             f"({last.start_timestamp}-{last.end_timestamp}, episode {last.episode})")
            elif e_last - tc > pc.cliffhanger_window_s + index.tol:
                probs.append(f"{name}: cliffhanger line starts {e_last - tc:.2f}s before the end; it must start "
                             f"within {pc.cliffhanger_window_s:g}s of the end (end sooner after it)")
        if known is not None:
            for mid in ids:
                if mid and mid not in known:
                    probs.append(f"{name}: unknown moment id '{mid}' (use one from the list or \"\")")
        return probs

    def _promo_dict(self, index: ScreenplayIndex, promo_id: str, maker: str, version: int,
                    segs: list[Segment], p: Any) -> dict:
        total = sum(index.seconds(s.end_timestamp) - index.seconds(s.start_timestamp) for s in segs)
        ids = [x for x in (p.hook_moment_id, p.cliffhanger_moment_id) if x]
        out = {"promo_id": promo_id, "maker": maker, "version": version,
               "segments": [s.model_dump() for s in segs], "total_duration_seconds": round(total, 2),
               "hook_timestamp": p.hook_timestamp, "cliffhanger_timestamp": p.cliffhanger_timestamp,
               "hook_moment_id": p.hook_moment_id or None, "cliffhanger_moment_id": p.cliffhanger_moment_id or None,
               "source_moment_ids": list(dict.fromkeys(ids)), "hook": p.hook, "cliffhanger": p.cliffhanger,
               "moment_type": self._mtype(p.moment_type), "why_it_works": p.why_it_works}
        if hasattr(p, "join_rationale"):
            out["join_rationale"] = p.join_rationale
        return out

    # ------------------------------------------------------------------- reader
    def read_episode(self, episode: Episode, story_summary: str, memory: dict) -> dict:
        rc = self.settings.agents.reader
        index = self._index([episode])
        n = episode.number

        def check(out: ReaderOutput) -> list[str]:
            probs: list[str] = []
            counts = Counter(m.kind for m in out.moments)
            for kind, limit in (("hook", rc.max_hooks), ("cliffhanger", rc.max_cliffhangers)):
                if counts[kind] > limit:
                    probs.append(f"{counts[kind]} {kind} moments, max is {limit}; keep the strongest")
            for i, m in enumerate(out.moments, start=1):
                probs += self._line_start(index, n, m.timestamp, f"moment {i} ({m.kind})")[0]
            return probs

        variables = {**self._common(memory), "episode": n, "logline": episode.logline or "(none)",
                     "story_summary": story_summary or EMPTY_SUMMARY, "lines": render_lines(self._lines(episode)),
                     "max_hooks": rc.max_hooks, "max_cliffhangers": rc.max_cliffhangers,
                     "summary_max_words": rc.summary_max_words}
        r = self._call("reader", "reader", variables, ReaderOutput, [check], self._tid(f"ep{n:02d}"), f"read ep{n}")
        seq: Counter = Counter()
        moments = []
        for m in r.output.moments:
            seq[m.kind] += 1
            moments.append({"moment_id": f"m{n:02d}_{m.kind[0]}{seq[m.kind]}", "episode": n, "kind": m.kind,
                            "timestamp": m.timestamp, "quote": m.quote, "description": m.description,
                            "strength": m.strength, "moment_type": self._mtype(m.moment_type)})
        return {"moments": moments, "new_viewer_context": r.output.new_viewer_context,
                "story_summary": r.output.story_summary, "llm": meta_dict(r.meta)}

    # ------------------------------------------------------------------- cutter
    def cut_episode(self, episode: Episode, moments: list[dict], memory: dict) -> list[dict]:
        if not moments:
            return []
        limit = self.settings.agents.cutter.max_promos_per_episode
        index = self._index([episode])
        n = episode.number
        known = {m["moment_id"] for m in moments}

        def per_promo(i: int, p) -> list[str]:
            seg = Segment(episode=n, start_timestamp=p.start_timestamp, end_timestamp=p.end_timestamp)
            return self._promo_problems(index, [seg], p.hook_timestamp, p.cliffhanger_timestamp,
                                        [p.hook_moment_id, p.cliffhanger_moment_id], known, f"promo {i}")

        pc = self.settings.promo

        def row(m: dict) -> str:
            base = (f"{m['moment_id']} | {m['kind']} | {m['timestamp']} | {m['strength']} | "
                    f"{m['moment_type']} | {m['description']}")
            t = index.seconds(m["timestamp"])
            if m["kind"] != "hook" or t is None:
                return base
            # Models can't do MM:SS arithmetic reliably: give the legal end window outright.
            return (f"{base} | a clip starting here must end between {format_ts(t + pc.min_duration_s)} "
                    f"and {format_ts(t + pc.max_duration_s)}")

        variables = {**self._common(memory), "episode": n, "logline": episode.logline or "(none)",
                     "max_promos": limit, "lines": render_lines(self._lines(episode)),
                     "moments": "\n".join(row(m) for m in moments)}
        items, llm = self._call_promos("cutter", "cutter", variables, CutterOutput, per_promo, limit,
                                       self._tid(f"ep{n:02d}"), f"cut ep{n}")
        promos = []
        for i, p in enumerate(items, start=1):
            seg = Segment(episode=n, start_timestamp=p.start_timestamp, end_timestamp=p.end_timestamp)
            d = self._promo_dict(index, f"c{n:02d}_{i}", "cutter", 1, [seg], p)
            promos.append({**d, "llm": llm})
        return promos

    def _call_promos(self, agent: str, prompt_name: str, variables: dict, schema, per_promo, limit: int,
                     trace_id: str, label: str) -> tuple[list, dict]:
        """Call for a LIST of promos. If retries run out but some promos in the last schema-valid
        answer pass every check, keep those and drop the rest: one bad promo must not throw
        away the good ones (a full run once lost every joined promo that way)."""
        def check(out) -> list[str]:
            probs = [f"{len(out.promos)} promos, max is {limit}; keep the best"] if len(out.promos) > limit else []
            for i, p in enumerate(out.promos, start=1):
                probs += per_promo(i, p)
            return probs

        prompt = load_chat_prompt(self.client.prompts_dir, prompt_name)
        try:
            r = self.client.structured(agent, prompt, variables, schema, checks=[check], trace_id=trace_id, label=label)
            return r.output.promos, meta_dict(r.meta)
        except StructuredOutputError as e:
            last = e.last_output
            keep = [p for i, p in enumerate(last.promos, start=1) if not per_promo(i, p)][:limit] if last else []
            if not keep:
                raise AgentError(f"{agent} ({label}): {e}") from e
            llm = {**meta_dict(e.meta), "salvaged": {"kept": len(keep), "dropped": len(last.promos) - len(keep)}}
            log.warning("%s (%s): kept %d valid promo(s) of %d after retries ran out", agent, label,
                        len(keep), len(last.promos))
            return keep, llm

    # ------------------------------------------------------------------- joiner
    def join(self, episodes: dict[int, Episode], moments: list[dict], candidates: list[dict], memory: dict) -> list[dict]:
        jc = self.settings.agents.joiner
        picked = sorted((m for m in moments if m["strength"] >= jc.min_moment_strength),
                        key=lambda m: -m["strength"])[: jc.max_moments]
        if jc.max_promos == 0 or len(picked) < 2:
            return []
        index = self._index(list(episodes.values()))
        picked.sort(key=lambda m: (m["episode"], index.seconds(m["timestamp"]) or 0))
        known = {m["moment_id"] for m in moments}

        def per_promo(i: int, p) -> list[str]:
            return self._join_problems(index, p, known, f"promo {i}")

        blocks = []
        for m in picked:
            t = index.seconds(m["timestamp"]) or 0
            lines = self._lines(episodes[m["episode"]], t - jc.context_before_s, t + jc.context_after_s)
            blocks.append(f"[{m['moment_id']}] episode {m['episode']} | {m['kind']} at {m['timestamp']} | "
                          f"strength {m['strength']} | {m['moment_type']} | {m['description']}\n{render_lines(lines)}")
        cands = "\n".join(
            f"{c['promo_id']} | " + " + ".join(f"ep {s['episode']} {s['start_timestamp']}-{s['end_timestamp']}"
                                              for s in c["segments"])
            + f" ({c['total_duration_seconds']}s) | {c.get('hook', '')} | {c.get('cliffhanger', '')}"
            for c in candidates) or "(none)"
        variables = {**self._common(memory), "max_promos": jc.max_promos, "candidates": cands,
                     "moments": "\n\n".join(blocks)}
        items, llm = self._call_promos("joiner", "joiner", variables, JoinerOutput, per_promo, jc.max_promos,
                                       self._tid("joiner"), "join")
        return [{**self._promo_dict(index, f"j{i:02d}", "joiner", 1, p.segments, p), "llm": llm}
                for i, p in enumerate(items, start=1)]

    def _join_problems(self, index, p: JoinPromo, known, name) -> list[str]:
        if len(p.segments) < 2:
            return [f"{name}: has {len(p.segments)} segment; a joined promo needs at least 2 segments"]
        return self._promo_problems(index, p.segments, p.hook_timestamp, p.cliffhanger_timestamp,
                                    [p.hook_moment_id, p.cliffhanger_moment_id], known, name)

    # ------------------------------------------------------------------- revise
    def revise(self, maker: str, promo: dict, instruction: str, episodes: dict[int, Episode], memory: dict,
               source: str = "evaluator") -> dict:
        if maker not in ("cutter", "joiner"):
            raise ValueError(f"unknown maker {maker!r}")
        segs, index, lines, current = self._revise_window(promo, episodes, label_episodes=maker != "cutter")
        variables = {**self._common(memory), "episode": segs[0].episode, "instruction": instruction,
                     "source": "human editor" if source == "editor" else "automatic evaluator",
                     "current": current, "promo_text": render_promo_text(promo, episodes, self.pats), "lines": lines}
        name = "revised promo"

        if maker == "cutter":
            ep_n = segs[0].episode

            def check(out: RevisedCut) -> list[str]:
                seg = Segment(episode=ep_n, start_timestamp=out.start_timestamp, end_timestamp=out.end_timestamp)
                return self._promo_problems(index, [seg], out.hook_timestamp, out.cliffhanger_timestamp, [], None, name)

            r = self._call("cutter", "cutter_revise", variables, RevisedCut, [check],
                           self._tid(promo["promo_id"]), f"revise {promo['promo_id']}")
            new_segs = [Segment(episode=ep_n, start_timestamp=r.output.start_timestamp,
                                end_timestamp=r.output.end_timestamp)]
        else:
            def check(out: RevisedJoin) -> list[str]:
                return self._join_problems(index, out, None, name)

            r = self._call("joiner", "joiner_revise", variables, RevisedJoin, [check],
                           self._tid(promo["promo_id"]), f"revise {promo['promo_id']}")
            new_segs = r.output.segments
        new = self._promo_dict(index, promo["promo_id"], maker, promo["version"] + 1, new_segs, r.output)
        return {**new, "revision_note": r.output.revision_note, "llm": meta_dict(r.meta)}

    def _revise_window(self, promo: dict, episodes: dict[int, Episode], label_episodes: bool = True):
        """Screenplay around the current cut (revise.context_s either side), so a reviser can
        move boundaries either way. Shared by revise() and interpret_fix(): the FeedbackAgent
        points at exactly the lines the maker will see."""
        ctx_s = self.settings.agents.revise.context_s
        segs = [Segment(**s) for s in promo["segments"]]
        used = [episodes[n] for n in dict.fromkeys(s.episode for s in segs)]
        index = self._index(used)
        blocks = []
        for ep in used:
            spans = [(index.seconds(s.start_timestamp), index.seconds(s.end_timestamp)) for s in segs if s.episode == ep.number]
            lo, hi = min(a for a, _ in spans) - ctx_s, max(b for _, b in spans) + ctx_s
            body = render_lines(self._lines(ep, lo, hi))
            blocks.append(f"Episode {ep.number}:\n{body}" if label_episodes else body)
        current = {k: promo.get(k) for k in ("segments", "total_duration_seconds", "hook_timestamp",
                                             "cliffhanger_timestamp", "hook", "cliffhanger", "moment_type",
                                             "why_it_works", "join_rationale") if promo.get(k) is not None}
        return segs, index, "\n\n".join(blocks), "\n".join(f"{k}: {v}" for k, v in current.items())

    # ---------------------------------------------------------------- feedback
    def interpret_fix(self, promo: dict, comment: str, episodes: dict[int, Episode], memory: dict) -> dict:
        """FeedbackAgent, job 1: the editor's comment -> what they want + a precise edit.
        Every time it cites must be a real cut point in the promo's episodes."""
        segs, index, lines, current = self._revise_window(promo, episodes)
        eps = sorted({s.episode for s in segs})

        def check(out: FixPlan) -> list[str]:
            probs = []
            for m in self._ts_re.finditer(out.instruction):
                ts = m.group(0)
                t = index.seconds(ts)
                if t is None or not any(index.exists(ep, t) for ep in eps):
                    probs += check_timestamp(index, eps[0], ts, "instruction")[:1] or \
                        [f"instruction: {ts} is not a cut point in episode(s) {eps}"]
            return probs

        variables = {**self._common(memory), "comment": comment, "current": current, "lines": lines,
                     "promo_text": render_promo_text(promo, episodes, self.pats)}
        pid = promo.get("promo_id", "promo")
        r = self._call("feedback", "feedback_fix", variables, FixPlan, [check], self._tid(pid), f"interpret fix {pid}")
        return {"interpretation": r.output.interpretation, "instruction": r.output.instruction,
                "keep": r.output.keep, "llm": meta_dict(r.meta)}

    # ---------------------------------------------------------------- learn (FeedbackAgent job 2)
    @staticmethod
    def _feedback_line(f: dict) -> str:
        p = f.get("promo") or {}
        extra = f" (FeedbackAgent read it as: {f['fix_interpretation']})" if f.get("fix_interpretation") else ""
        return (f"#{f['id']} [{f.get('show')} / {f.get('genre')}] {f['decision']} {f.get('promo_id')} "
                f"({f.get('moment_type')}, {p.get('total_duration_seconds', '?')}s): opens \"{p.get('hook', '')}\", "
                f"ends \"{p.get('cliffhanger', '')}\". Editor: \"{f.get('comment') or '(no comment)'}\"{extra}")

    def learn(self, new_feedback: list[dict], older_feedback: list[dict], rules: list[dict], context: dict) -> dict:
        min_support = context["min_support"]
        new_ids = {f["id"] for f in new_feedback}
        fb_ids = new_ids | {f["id"] for f in older_feedback}
        rule_ids = {r["id"] for r in rules}

        def check(out: LearnOutput) -> list[str]:
            probs = []
            for x in out.relations:
                if x.feedback_id not in new_ids:
                    probs.append(f"relation: #{x.feedback_id} is not a NEW feedback item (new: {sorted(new_ids)})")
                if x.rule_id not in rule_ids:
                    probs.append(f"relation: unknown rule R{x.rule_id} (rules: {sorted(rule_ids)})")
            for i, nr in enumerate(out.new_rules, start=1):
                src = set(nr.source_feedback_ids)
                if src - fb_ids:
                    probs.append(f"new rule {i}: unknown feedback ids {sorted(src - fb_ids)}")
                if len(src & fb_ids) < min_support:
                    probs.append(f"new rule {i}: needs at least {min_support} distinct feedback items, has {len(src & fb_ids)}")
                if not nr.text.strip():
                    probs.append(f"new rule {i}: empty text")
                if set(nr.conflicts_with) - rule_ids:
                    probs.append(f"new rule {i}: conflicts_with unknown rules {sorted(set(nr.conflicts_with) - rule_ids)}")
            for c in out.conflicts:
                if c.rule_a == c.rule_b:
                    probs.append(f"conflict R{c.rule_a}: a conflict needs two different rules")
                elif {c.rule_a, c.rule_b} - rule_ids:
                    probs.append(f"conflict: unknown rules {sorted({c.rule_a, c.rule_b} - rule_ids)}")
            return probs

        variables = {"show": context.get("show"), "genre": context.get("genre"), "min_support": min_support,
                     "new_feedback": "\n".join(self._feedback_line(f) for f in new_feedback) or "(none)",
                     "older_feedback": "\n".join(self._feedback_line(f) for f in older_feedback) or "(none)",
                     "rules": "\n".join(f"R{r['id']} [{r['status']}, {r['scope']}{':' + r['scope_value'] if r.get('scope_value') else ''}, "
                                        f"+{r['supporting_count']}/-{r['contradicting_count']}] {r['text']}" for r in rules) or "(none yet)"}
        r = self._call("feedback", "feedback_learn", variables, LearnOutput, [check], self._tid("memory"), "learn")
        o = r.output
        return {"relations": [x.model_dump() for x in o.relations], "new_rules": [x.model_dump() for x in o.new_rules],
                "conflicts": [x.model_dump() for x in o.conflicts], "llm": meta_dict(r.meta)}

    # ---------------------------------------------------------------- evaluator
    def evaluate(self, promo_text: str, promo: dict, memory: dict) -> dict:
        ev = self.settings.evaluator

        def check(out: EvalOutput) -> list[str]:
            if not out.passed and not out.fix_instruction.strip():
                return ["passed is false but fix_instruction is empty; give one concrete edit"]
            return []

        # Only the viewer-facing text goes in: no show name, genre, logline or maker's pitch.
        variables = {"promo_text": promo_text, "hook_window_s": f"{self.settings.promo.hook_window_s:g}",
                     "min_each_score": ev.min_each_score, "pass_threshold": f"{ev.pass_threshold:g}",
                     "memory": render_memory(memory)}
        pid = promo.get("promo_id", "promo")
        r = self._call("evaluator", "evaluator", variables, EvalOutput, [check], self._tid(pid),
                       f"evaluate {pid} v{promo.get('version', 1)}")
        o = r.output
        return {"scores": {"hook": o.hook, "cliffhanger": o.cliffhanger, "standalone": o.standalone},
                "reasons": {"hook": o.hook_reason, "cliffhanger": o.cliffhanger_reason, "standalone": o.standalone_reason},
                "passed": o.passed, "fix_instruction": o.fix_instruction.strip() or None, "llm": meta_dict(r.meta)}
