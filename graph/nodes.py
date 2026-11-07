"""Graph nodes. Each node does state plumbing (cursors, queues, statuses, trace) and
delegates the actual judgement to `runtime.context.agents` (agents/llm_agents.py, or
agents/stubs.py in tests). An AgentError is recorded in state["errors"] and the run carries on.

Nodes return only the keys they change; reducers in graph/state.py merge them.
"""
from __future__ import annotations

import time
from pathlib import Path

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from agents.base import AgentError
from agents.render import render_promo_text
from graph.context import RunContext
from graph.state import PromoState, Revision, trace_id
from parser.screenplay_parser import parse_show, parse_ts
from memory.store import scope_for
from ranking.rank import rank_promos
from parser.validate import summarize

DECISIONS = {"approve", "reject", "fix", "keep_before"}


def _event(node: str, event: str, promo_id: str | None = None, moment_id: str | None = None, **data) -> dict:
    return {"ts": round(time.time(), 3), "node": node, "event": event,
            "promo_id": promo_id, "moment_id": moment_id, "data": data}


def final_score(scores: dict, ctx: RunContext) -> float:
    w = ctx.settings.scoring.weights
    return round(scores["hook"] * w.hook + scores["cliffhanger"] * w.cliffhanger + scores["standalone"] * w.standalone, 2)


# ------------------------------------------------------------------------------ parse
def parse(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    show = parse_show(Path(state["input_dir"]), ctx.parser_cfg, fallback=ctx.parser_fallback, name=state.get("show"))
    if only := state.get("episode_filter"):
        show.episodes = [ep for ep in show.episodes if ep.number in set(only)]
        if not show.episodes:
            raise ValueError(f"episode filter {only} matches no episode in {state['input_dir']}")
    rows = [summarize(ep) for ep in show.episodes]
    report = {
        "episodes": len(rows),
        "coverage_pct": round(sum(r["coverage_pct"] for r in rows) / len(rows), 2),
        "unparsed_lines": sum(r["unparsed"] for r in rows),
        "warnings": sum(r["warnings"] for r in rows),
        "per_episode": rows,
    }
    parsed_dir = ctx.run_dir(state["run_id"]) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for ep in show.episodes:
        path = parsed_dir / f"episode_{ep.number:02d}.json"
        path.write_text(ep.model_dump_json(), encoding="utf-8")
        index.append({"number": ep.number, "source_file": ep.source_file, "logline": ep.logline,
                      "duration": ep.duration, "items": len(ep.all_items()), "parsed_path": str(path)})
    return {
        "episodes": index, "parse_report": report,
        "reader_cursor": 0, "cutter_cursor": 0, "story_summary": "", "joiner_done": False,
        "pending_eval": [], "revision_queue": [], "review_round": 0, "lessons": [],
        "trace": [_event("parse", "parsed", episodes=len(rows), coverage_pct=report["coverage_pct"])],
    }


# ------------------------------------------------------------------------ memory_read
def memory_read(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """Editor memory for this run: active rules for global / this genre / this show (top N by
    confidence) + approved and rejected examples (same genre first). Injected into the Reader,
    Cutter, Joiner and Evaluator prompts. Records the guidelines version the run used."""
    ctx = runtime.context
    mem, mc = ctx.memory, ctx.settings.memory
    switched_on = bool(state.get("switches", {}).get("memory"))
    version = mem.version_label() if mem else None
    if mem is None or not switched_on:
        used = {"enabled": False, "guidelines_version": version, "rules": [], "examples": [],
                "reason": "memory switched off" if mem else "no memory store (e.g. a dry run)"}
    else:
        used = {"enabled": True, **mem.for_run(state.get("show"), state.get("genre"), mc.top_rules,
                                               mc.approved_examples, mc.rejected_examples)}
    return {"memory_used": used, "trace": [_event("memory_read", "loaded", enabled=used["enabled"],
                                                  guidelines_version=used["guidelines_version"],
                                                  rules=[r["id"] for r in used["rules"]],
                                                  examples=[e["id"] for e in used["examples"]])]}


# ----------------------------------------------------------------------------- reader
def reader(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """One episode per step: the story summary carries over, and every episode read is
    checkpointed, so a rate-limit stop resumes at the next unread episode."""
    ctx = runtime.context
    i = state.get("reader_cursor", 0)
    ep = list(ctx.episodes(state).values())[i]
    try:
        out = ctx.agents.read_episode(ep, state.get("story_summary", ""), state.get("memory_used", {}))
    except AgentError as e:
        # No valid reading after all retries: skip this episode (no moments, summary unchanged).
        return {"reader_cursor": i + 1, "errors": [f"reader: episode {ep.number}: {e}"],
                "episode_notes": {str(ep.number): {"failed": True, "error": str(e)}},
                "trace": [_event("reader", "failed", episode=ep.number, error=str(e))]}
    return {
        "moments": out["moments"],
        "story_summary": out["story_summary"],
        "episode_notes": {str(ep.number): {"new_viewer_context": out["new_viewer_context"],
                                           "story_summary_after": out["story_summary"],
                                           "failed": False, "llm": out.get("llm")}},
        "reader_cursor": i + 1,
        "trace": [_event("reader", "moment", moment_id=m["moment_id"],
                         **{k: v for k, v in m.items() if k != "moment_id"}) for m in out["moments"]],
    }


# ---------------------------------------------------------------------- cutter/joiner
def _apply_revisions(maker: str, state: PromoState, ctx: RunContext) -> dict:
    """Shared by cutter and joiner: apply every queued fix addressed to this maker."""
    mine = [r for r in state.get("revision_queue", []) if r["maker"] == maker]
    rest = [r for r in state.get("revision_queue", []) if r["maker"] != maker]
    candidates, versions, counts, trace, status, errors, done, notes = {}, {}, {}, [], {}, [], [], {}
    for r in mine:
        pid = r["promo_id"]
        old = state["candidates"][pid]
        try:
            new = ctx.agents.revise(maker, old, r["instruction"], ctx.episodes(state), state.get("memory_used", {}),
                                    source=r["source"])
        except AgentError as e:
            errors.append(f"{maker}: revising {pid} v{old['version']}: {e}")
            trace.append(_event(maker, "revise_failed", promo_id=pid, source=r["source"], error=str(e)))
            if r["source"] == "evaluator":
                # The automatic loop can't continue: keep the best version already scored.
                best, ev = _best_version(state, pid, old)
                candidates[pid], status[pid] = best, "kept_best"
                trace.append(_event(maker, "kept_best", promo_id=pid, version=best["version"], final=ev["final"]))
            else:
                # A failed editor fix leaves the promo as it was; it returns to review with a note.
                notes[pid] = f"fix failed, promo unchanged: {e}"
            continue
        new["revised_by"] = r["source"]
        if r["source"] == "editor":
            new["editor_comment"] = r.get("editor_comment")
            new["fix_interpretation"] = r.get("interpretation")
        done.append(pid)
        candidates[pid] = new
        versions[pid] = (state.get("versions", {}).get(pid) or [old]) + [new]
        if r["source"] == "evaluator":
            counts[pid] = state.get("revision_counts", {}).get(pid, 0) + 1
        status[pid] = "revised"
        trace.append(_event(maker, "revised", promo_id=pid, version=new["version"],
                            source=r["source"], instruction=r["instruction"]))
    return {
        "candidates": candidates, "versions": versions, "revision_counts": counts, "status": status,
        "revision_queue": rest, "errors": errors, "review_notes": notes,
        "pending_eval": list(dict.fromkeys(state.get("pending_eval", []) + done)),
        "trace": trace,
    }


def _best_version(state: PromoState, pid: str, current: dict, history: list[dict] | None = None) -> tuple[dict, dict]:
    """Highest-scoring evaluated version of a promo, with its evaluation."""
    history = history if history is not None else state.get("evaluations", {}).get(pid) or []
    best = max(history, key=lambda h: h["final"])
    versions = state.get("versions", {}).get(pid) or [current]
    return next(v for v in versions if v["version"] == best["version"]), best


def _new_promos(maker: str, promos: list[dict], state: PromoState) -> dict:
    return {
        "candidates": {p["promo_id"]: p for p in promos},
        "versions": {p["promo_id"]: [p] for p in promos},
        "status": {p["promo_id"]: "new" for p in promos},
        "pending_eval": state.get("pending_eval", []) + [p["promo_id"] for p in promos],
        "trace": [_event(maker, "created", promo_id=p["promo_id"], version=1, segments=p["segments"],
                         hook_moment_id=p.get("hook_moment_id"),
                         cliffhanger_moment_id=p.get("cliffhanger_moment_id")) for p in promos],
    }


def cutter(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    if any(r["maker"] == "cutter" for r in state.get("revision_queue", [])):
        return _apply_revisions("cutter", state, ctx)
    i = state.get("cutter_cursor", 0)
    ep = list(ctx.episodes(state).values())[i]
    moments = [m for m in state.get("moments", []) if m["episode"] == ep.number]
    try:
        promos = ctx.agents.cut_episode(ep, moments, state.get("memory_used", {}))
    except AgentError as e:
        return {"cutter_cursor": i + 1, "errors": [f"cutter: episode {ep.number}: {e}"],
                "trace": [_event("cutter", "failed", episode=ep.number, error=str(e))]}
    return {**_new_promos("cutter", promos, state), "cutter_cursor": i + 1}


def joiner(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    if any(r["maker"] == "joiner" for r in state.get("revision_queue", [])):
        return _apply_revisions("joiner", state, ctx)
    singles = [p for p in state.get("candidates", {}).values() if p["maker"] == "cutter"]
    try:
        promos = ctx.agents.join(ctx.episodes(state), state.get("moments", []), singles, state.get("memory_used", {}))
    except AgentError as e:
        return {"joiner_done": True, "errors": [f"joiner: {e}"], "trace": [_event("joiner", "failed", error=str(e))]}
    return {**_new_promos("joiner", promos, state), "joiner_done": True}


# -------------------------------------------------------------------------- evaluator
def evaluator(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    ev_cfg = ctx.settings.evaluator
    episodes = ctx.episodes(state)
    queue: list[Revision] = list(state.get("revision_queue", []))
    evaluations, status, candidates, trace, errors = {}, {}, {}, [], []

    # A batch per step: each step is checkpointed, so a crash loses at most one batch.
    pending = list(state.get("pending_eval", []))
    batch, rest = pending[: ev_cfg.batch_size], pending[ev_cfg.batch_size:]
    for pid in batch:
        promo = state["candidates"][pid]
        # The Evaluator sees ONLY the promo's own text (a viewer who never saw the show).
        text = render_promo_text(promo, episodes, ctx.parser_cfg.compiled)
        history = state.get("evaluations", {}).get(pid) or []
        try:
            ev = ctx.agents.evaluate(text, promo, state.get("memory_used", {}))
        except AgentError as e:
            errors.append(f"evaluator: {pid} v{promo['version']}: {e}")
            trace.append(_event("evaluator", "failed", promo_id=pid, version=promo["version"], error=str(e)))
            if history:   # an earlier version was scored: fall back to the best one
                candidates[pid], best = _best_version(state, pid, promo, history)
                status[pid] = "kept_best"
                trace.append(_event("evaluator", "kept_best", promo_id=pid, version=best["version"], final=best["final"]))
            else:
                status[pid] = "eval_failed"     # never scored: left out of ranking
            continue
        final = final_score(ev["scores"], ctx)
        # Pass/fail thresholds are enforced in code from config, not left to the model.
        passed = bool(ev["passed"]) and final >= ev_cfg.pass_threshold and \
            min(ev["scores"].values()) >= ev_cfg.min_each_score
        entry = {"version": promo["version"], "scores": ev["scores"], "final": final, "passed": passed,
                 "fix_instruction": None if passed else ev.get("fix_instruction"),
                 "reasons": ev.get("reasons"), "llm": ev.get("llm"), "promo_text": text}
        history = history + [entry]
        evaluations[pid] = history
        trace.append(_event("evaluator", "evaluated", promo_id=pid, **{k: v for k, v in entry.items() if k != "promo_text"}))

        rounds = state.get("revision_counts", {}).get(pid, 0)
        if promo.get("revised_by") == "editor":
            status[pid] = "editor_revised"      # editor's fix: shown before/after, never auto-reverted
        elif passed:
            status[pid] = "passed"
        elif rounds < ev_cfg.max_revision_rounds:
            status[pid] = "revising"
            queue.append({"promo_id": pid, "maker": promo["maker"], "source": "evaluator",
                          "instruction": ev.get("fix_instruction") or "Improve hook and cliffhanger."})
        else:
            # Out of rounds: keep the best-scoring version and move on.
            candidates[pid], best = _best_version(state, pid, promo, history)
            status[pid] = "kept_best"
            trace.append(_event("evaluator", "kept_best", promo_id=pid, version=best["version"], final=best["final"]))

    return {"evaluations": evaluations, "status": status, "candidates": candidates, "errors": errors,
            "revision_queue": queue, "pending_eval": rest, "trace": trace}


# ------------------------------------------------------------------------------- rank
def rank(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """Final list via ranking/rank.py: score floor, near-duplicate removal, per-episode cap,
    early-episode preference, top N. Every dropped promo is reported with its reason."""
    ctx = runtime.context
    pats = ctx.parser_cfg.compiled
    decisions = state.get("editor_decisions", {})
    rows = []
    for pid, promo in state.get("candidates", {}).items():
        ev = next((e for e in reversed(state.get("evaluations", {}).get(pid) or [])
                   if e["version"] == promo["version"]), None)
        scores = {**ev["scores"], "final": ev["final"]} if ev else {"final": 0.0}
        rows.append({**promo, "scores": scores, "status": state.get("status", {}).get(pid),
                     "editor_decision": decisions.get(pid, {}).get("decision"),
                     "evaluator_reasons": (ev or {}).get("reasons"), "trace_id": trace_id(state["run_id"], pid)})
    if state.get("review_round", 0) and not ctx.settings.feedback.backfill_rejected:
        listed = {p["promo_id"] for p in state.get("promos", [])}
        rows = [r for r in rows if r["promo_id"] in listed]
    ranked, dropped = rank_promos(rows, ctx.settings, [e["number"] for e in state.get("episodes", [])],
                                  lambda ts: parse_ts(ts, pats))
    report = {"considered": len(rows), "ranked": len(ranked), "dropped": dropped,
              "settings": ctx.settings.ranking.model_dump()}
    trace = [_event("rank", "ranked", promo_id=r["promo_id"], rank=r["rank"], final=r["scores"]["final"],
                    rank_score=r["rank_score"], landing_episode=r["landing_episode"]) for r in ranked]
    trace += [_event("rank", "dropped", promo_id=d["promo_id"], **{k: v for k, v in d.items() if k != "promo_id"})
              for d in dropped]
    return {"promos": ranked, "ranking_report": report, "trace": trace}


# ---------------------------------------------------------------------- editor_review
def _review_view(state: PromoState, ctx: RunContext) -> dict:
    """What the editor needs this round. Pure: editor_review re-runs it on resume."""
    decided = state.get("editor_decisions", {})
    promos = state.get("promos", [])
    # Needs a decision: never decided (incl. promos back-filled into the list), or the last
    # decision was "fix" (now shown before/after).
    pending = [p["promo_id"] for p in promos if decided.get(p["promo_id"], {}).get("decision") in (None, "fix")]
    before_after = {}
    for p in promos:
        if p.get("revised_by") == "editor":
            versions = state.get("versions", {}).get(p["promo_id"], [])
            evals = {e["version"]: e for e in state.get("evaluations", {}).get(p["promo_id"], [])}
            prev = next((v for v in reversed(versions) if v["version"] < p["version"]), None)
            before_after[p["promo_id"]] = {
                "before": prev, "before_eval": evals.get(prev["version"]) if prev else None,
                "after": p, "after_eval": evals.get(p["version"]),
                "editor_comment": p.get("editor_comment"), "fix_interpretation": p.get("fix_interpretation"),
            }
    attempts = state.get("fix_attempts", {})
    mx = ctx.settings.feedback.max_fix_attempts
    return {"promos": promos, "pending": pending, "before_after": before_after,
            "fix_attempts": {pid: attempts.get(pid, 0) for pid in pending},
            "can_fix": {pid: attempts.get(pid, 0) < mx for pid in pending}, "max_fix_attempts": mx,
            "notes": {pid: n for pid, n in (state.get("review_notes") or {}).items() if pid in pending}}


def editor_review(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """Pause for the editor. Resumed with Command(resume={promo_id: {decision, comment}}),
    decision = approve | reject | fix | keep_before. Everything before interrupt() re-runs on
    resume, so it must stay side-effect free. Ends review (no pause) when nothing is pending
    or the configured max number of rounds has been held."""
    fb = runtime.context.settings.feedback
    done_rounds = state.get("review_round", 0)
    view = _review_view(state, runtime.context)
    if not view["pending"] or done_rounds >= fb.max_review_rounds:
        summary = {"rounds": done_rounds, "stopped_by_max_rounds": bool(view["pending"]),
                   "unreviewed": view["pending"]}
        return {"review_complete": True, "review_summary": summary,
                "trace": [_event("editor_review", "complete", **summary)]}
    rnd = done_rounds + 1
    decisions = interrupt({"round": rnd, "max_rounds": fb.max_review_rounds, **view})

    clean, errors = {}, []
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for pid, d in (decisions or {}).items():
        d = d or {}
        dec = str(d.get("decision", "")).lower().strip()
        comment = (d.get("comment") or "").strip()
        why = None
        if pid not in view["pending"]:
            why = "is not under review this round"
        elif dec not in DECISIONS:
            why = f"unknown decision {dec!r} (use approve, reject, fix or keep_before)"
        elif dec == "fix" and not comment:
            why = "fix needs a comment saying what to change"
        elif dec == "fix" and not view["can_fix"][pid]:
            why = f"no fix attempts left (max {view['max_fix_attempts']}); approve, reject or keep_before"
        elif dec == "keep_before" and pid not in view["before_after"]:
            why = "keep_before only applies to a promo shown before/after an editor fix"
        if why:
            errors.append(f"editor_review: {pid}: {why}; still pending")
            continue
        clean[pid] = {"decision": dec, "comment": comment, "round": rnd, "decided_at": now}
    return {"editor_decisions": clean, "review_round": rnd, "errors": errors,
            "trace": [_event("editor_review", "decision", promo_id=pid, **d) for pid, d in clean.items()]}


# --------------------------------------------------------------------------- feedback
FIX_TEMPLATE = "Editor's note (authoritative, follow it): {comment}\nHow to apply it (FeedbackAgent): {instruction}"


def feedback(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """FeedbackAgent. Job 1, fix now: each "fix" goes through the LLM FeedbackAgent, which turns
    the comment into a precise instruction; the maker gets the editor's words verbatim (they
    win) plus that reading. "keep_before" undoes the last editor fix. Job 2, learn: every
    decision becomes a lesson for memory_write (Phase 3). After this node the list is ranked
    again, so rejected slots are back-filled (feedback.backfill_rejected) and reviewed."""
    ctx = runtime.context
    rnd = state.get("review_round", 0)
    queue, attempts, trace, errors = list(state.get("revision_queue", [])), {}, [], []
    lessons = list(state.get("lessons", []))
    candidates, status, decisions = {}, {}, {}
    rows = {p["promo_id"]: p for p in state.get("promos", [])}
    for pid, d in state.get("editor_decisions", {}).items():
        cand = state.get("candidates", {}).get(pid)
        if d.get("round") != rnd or cand is None or d.get("applied"):
            continue
        lesson = {"promo_id": pid, "decision": d["decision"], "comment": d.get("comment", ""), "round": rnd,
                  "decided_at": d.get("decided_at"), "version": cand["version"], "maker": cand["maker"],
                  "moment_type": cand.get("moment_type"), "segments": cand["segments"],
                  "final": (rows.get(pid) or {}).get("scores", {}).get("final")}
        if d["decision"] == "keep_before":
            prev = next(v for v in reversed(state["versions"][pid]) if v["version"] < cand["version"])
            ev = next((e for e in reversed(state.get("evaluations", {}).get(pid) or []) if e["version"] == prev["version"]), None)
            candidates[pid] = prev
            status[pid] = "passed" if ev and ev["passed"] else "kept_best"
            decisions[pid] = {**d, "decision": "approve", "kept_before": True, "applied": True}
            trace.append(_event("feedback", "kept_before", promo_id=pid, version=prev["version"], undone=cand["version"]))
        elif d["decision"] == "fix":
            n = state.get("fix_attempts", {}).get(pid, 0)
            if n >= ctx.settings.feedback.max_fix_attempts:      # editor_review already refuses these
                trace.append(_event("feedback", "fix_limit_reached", promo_id=pid, attempts=n))
            else:
                attempts[pid] = n + 1
                interpretation, instruction = None, d["comment"]
                try:
                    plan = ctx.agents.interpret_fix(cand, d["comment"], ctx.episodes(state), state.get("memory_used", {}))
                    interpretation = plan["interpretation"]
                    instruction = FIX_TEMPLATE.format(comment=d["comment"], instruction=plan["instruction"])
                    trace.append(_event("feedback", "fix_interpreted", promo_id=pid, comment=d["comment"],
                                        interpretation=interpretation, instruction=plan["instruction"]))
                except AgentError as e:
                    # FeedbackAgent failed: the editor's words alone still go to the maker.
                    errors.append(f"feedback: interpreting fix for {pid}: {e}")
                    trace.append(_event("feedback", "fix_interpretation_failed", promo_id=pid, error=str(e)))
                lesson["fix_interpretation"] = interpretation
                queue.append({"promo_id": pid, "maker": cand["maker"], "instruction": instruction, "source": "editor",
                              "editor_comment": d["comment"], "interpretation": interpretation})
                trace.append(_event("feedback", "fix_requested", promo_id=pid, attempt=n + 1, comment=d["comment"]))
        lessons.append(lesson)
    return {"revision_queue": queue, "fix_attempts": attempts, "lessons": lessons, "trace": trace, "errors": errors,
            "candidates": candidates, "status": status, "editor_decisions": decisions}


# ----------------------------------------------------------------------- memory_write
def memory_write(state: PromoState, runtime: Runtime[RunContext]) -> dict:
    """FeedbackAgent job 2 (learn). Every editor decision goes to the append-only feedback_log,
    approved / rejected promos become examples; then the learner says which feedback supports or
    contradicts existing rules and proposes new rules backed by >= min_feedback_for_rule items.
    Code decides scope (memory.store.scope_for), status (needs_review unless auto_activate_rules)
    and counts. Idempotent across resumes (memory_writes stage per run)."""
    ctx = runtime.context
    mem, mc = ctx.memory, ctx.settings.memory
    lessons = state.get("lessons") or []
    enabled = bool(state.get("switches", {}).get("memory"))
    base = {"enabled": enabled, "feedback_items": len(lessons)}

    def skip(why: str) -> dict:
        return {"memory_written": {**base, "skipped": why}, "trace": [_event("memory_write", "skipped", reason=why)]}

    if mem is None:
        return skip("no memory store (e.g. a dry run)")
    if not lessons:
        return skip("no editor decisions in this run")
    if not enabled and not mc.write_when_off:
        return skip("memory switched off")
    run_id, show, genre = state["run_id"], state.get("show"), state.get("genre")
    if mem.write_stage(run_id) == "learned":
        return skip("already written")
    version_used = (state.get("memory_used") or {}).get("guidelines_version") or mem.version_label()
    evals = state.get("evaluations") or {}

    def snapshot(pid: str, version: int) -> tuple[dict, dict | None]:
        promo = next((v for v in (state.get("versions") or {}).get(pid, []) if v["version"] == version),
                     (state.get("candidates") or {}).get(pid, {}))
        ev = next((e for e in reversed(evals.get(pid) or []) if e["version"] == version), None)
        return {**promo, "promo_text": (ev or {}).get("promo_text")}, ev

    # 1. log every decision (append-only) ...
    fb_ids = []
    for l in lessons:
        promo, _ = snapshot(l["promo_id"], l.get("version") or 1)
        fb_ids.append(mem.log_feedback(show=show, genre=genre, promo_id=l["promo_id"], promo=promo,
                                       moment_type=l.get("moment_type"), decision=l["decision"],
                                       comment=l.get("comment") or "", guidelines_version=version_used, run_id=run_id,
                                       round=l.get("round") or 0, fix_interpretation=l.get("fix_interpretation")))
    # 2. ... and each promo's final verdict as an example
    final: dict[str, dict] = {}
    comments: dict[str, list[str]] = {}
    for l in sorted(lessons, key=lambda l: l.get("round") or 0):
        final[l["promo_id"]] = l
        if l.get("comment"):
            comments.setdefault(l["promo_id"], []).append(l["comment"])
    n_examples = 0
    for pid, l in final.items():
        label = {"approve": "approved", "keep_before": "approved", "reject": "rejected"}.get(l["decision"])
        if not label:
            continue                                              # last word was "fix": no verdict
        version = state["candidates"][pid]["version"] if label == "approved" else l.get("version") or 1
        promo, ev = snapshot(pid, version)
        mem.add_example(promo=promo, label=label, reason="; ".join(comments.get(pid, [])), show=show, genre=genre,
                        moment_type=promo.get("moment_type"), run_id=run_id,
                        scores={**(ev or {}).get("scores", {}), "final": (ev or {}).get("final")})
        n_examples += 1
    mem.mark_written(run_id, "logged")

    # 3. learn
    new = [f for f in mem.feedback(fb_ids) if f["comment"] or f["decision"] in ("reject", "fix")]
    older = [f for f in mem.unused_feedback(mc.learn_window) if f["id"] not in set(fb_ids)]
    rules_before = mem.rules({"active", "needs_review"})
    errors, trace, new_rule_ids, n_evidence, n_conflicts = [], [], [], 0, 0
    if new or older:
        try:
            out = ctx.agents.learn(new, older, rules_before, {"show": show, "genre": genre,
                                                              "min_support": mc.min_feedback_for_rule})
        except AgentError as e:
            out = None
            errors.append(f"memory_write: learning failed (feedback is kept; a later run can learn from it): {e}")
        if out:
            known = {f["id"]: f for f in new + older}
            new_ids, rule_ids = {f["id"] for f in new}, {r["id"] for r in rules_before}
            for rel in out.get("relations", []):
                if rel["feedback_id"] in new_ids and rel["rule_id"] in rule_ids:
                    n_evidence += mem.add_evidence(rel["rule_id"], rel["feedback_id"], rel["relation"])
            for nr in out.get("new_rules", []):
                sources = sorted({i for i in nr["source_feedback_ids"] if i in known})
                if len(sources) < mc.min_feedback_for_rule:        # the learner's check enforces this too
                    trace.append(_event("memory_write", "rule_rejected", text=nr["text"], reason="too little support"))
                    continue
                scope, value = scope_for([known[i] for i in sources])
                rid = mem.add_rule(nr["text"], scope=scope, scope_value=value, source_feedback_ids=sources,
                                   status="active" if mc.auto_activate_rules else "needs_review", created_by="ai")
                new_rule_ids.append(rid)
                for other in nr.get("conflicts_with") or []:
                    if other in rule_ids:
                        mem.mark_conflict(rid, other, why="new rule conflicts with an existing one")
                        n_conflicts += 1
            for c in out.get("conflicts", []):
                if {c["rule_a"], c["rule_b"]} <= rule_ids and c["rule_a"] != c["rule_b"]:
                    mem.mark_conflict(c["rule_a"], c["rule_b"], why=c.get("why", ""))
                    n_conflicts += 1
    mem.mark_written(run_id, "learned")
    written = {**base, "feedback_logged": len(fb_ids), "examples": n_examples, "evidence": n_evidence,
               "new_rule_ids": new_rule_ids, "conflicts": n_conflicts, "learner_input": len(new) + len(older),
               "version_before": version_used, "version_after": mem.version_label()}
    return {"memory_written": written, "errors": errors,
            "trace": trace + [_event("memory_write", "written", **{k: v for k, v in written.items()})]}
