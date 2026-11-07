"""Write every output of a run to output/runs/<run_id>/ (spec section 9).

    promos.json / promos.csv     the ranked list, spec fields first
    report.md                    readable report: ranked promos, drops, errors, coverage, LLM usage
    raw/candidates.json|csv      raw, unedited results: EVERY promo made, all versions + evaluations,
                                 and what ranking did with it
    raw/moments.json|csv         every Reader moment
    raw/episodes.json            per-episode parse coverage, new-viewer context, story summary
    raw/ranking_report.json      considered / ranked / dropped with reasons
    traces/<promo_id>.json       full trace per promo: moments -> cut -> join -> evaluations ->
                                 revisions -> editor decisions (trace_id points here)
    run_summary.json             state, switches, counts, LLM calls/tokens per agent, model switches
    review_template.json         only while paused for review: decisions to edit and resume with
Raw LLM responses are already in logs/llm/<run_id>/ (one file per attempt).
Safe to call repeatedly: every pause/stop/finish rewrites the files from the checkpoint.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from config.loader import Settings
from graph.state import promo_trace, trace_id
from llm.usage import usage_from_log

SPEC_ORDER = ["rank", "promo_id", "segments", "total_duration_seconds", "hook", "cliffhanger", "why_it_works",
              "moment_type", "scores", "trace_id"]
EXTRA_ORDER = ["trace_file", "rank_score", "landing_episode", "maker", "version", "status", "editor_decision",
               "hook_timestamp", "cliffhanger_timestamp", "join_rationale", "revision_note", "evaluator_reasons",
               "source_moment_ids"]


def _dump(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def segs_text(segments: list[dict]) -> str:
    return " | ".join(f"ep{s['episode']} {s['start_timestamp']}-{s['end_timestamp']}" for s in segments)


def _trace_file(pid: str) -> str:
    return f"traces/{pid}.json"


def _promo_out(p: dict) -> dict:
    """Ranked promo with the spec's fields first, then the useful extras."""
    p = {**p, "trace_file": _trace_file(p["promo_id"])}
    out = {k: p.get(k) for k in SPEC_ORDER}
    out.update({k: p[k] for k in EXTRA_ORDER if p.get(k) is not None})
    return out


def _promo_row(p: dict) -> dict:
    s = p.get("scores") or {}
    return {**{k: p.get(k) for k in SPEC_ORDER + EXTRA_ORDER if k not in ("segments", "scores")},
            "segments": segs_text(p.get("segments") or []),
            "episodes": " ".join(str(e) for e in sorted({x["episode"] for x in p.get("segments") or []})),
            "score_hook": s.get("hook"), "score_cliffhanger": s.get("cliffhanger"),
            "score_standalone": s.get("standalone"), "score_final": s.get("final")}


PROMO_CSV = ["rank", "promo_id", "segments", "episodes", "total_duration_seconds", "hook", "cliffhanger",
             "why_it_works", "moment_type", "score_hook", "score_cliffhanger", "score_standalone", "score_final",
             "rank_score", "landing_episode", "maker", "version", "status", "editor_decision", "trace_id", "trace_file"]


def _candidates(v: dict) -> list[dict]:
    ranked = {p["promo_id"]: p for p in v.get("promos") or []}
    dropped = {d["promo_id"]: d for d in (v.get("ranking_report") or {}).get("dropped", [])}
    out = []
    for pid, c in (v.get("candidates") or {}).items():
        evals = [{k: x for k, x in e.items() if k != "promo_text"} for e in (v.get("evaluations") or {}).get(pid, [])]
        final_eval = next((e for e in reversed(evals) if e["version"] == c["version"]), None)
        if pid in ranked:
            ranking = {"outcome": "ranked", "rank": ranked[pid]["rank"], "rank_score": ranked[pid]["rank_score"]}
        elif pid in dropped:
            ranking = {"outcome": "dropped", **{k: x for k, x in dropped[pid].items() if k != "promo_id"}}
        else:
            ranking = {"outcome": "not_ranked"}
        out.append({"promo_id": pid, "maker": c["maker"], "status": (v.get("status") or {}).get(pid),
                    "final_version": c["version"], "final_scores": final_eval and {**final_eval["scores"], "final": final_eval["final"]},
                    "ranking": ranking, "editor_decision": (v.get("editor_decisions") or {}).get(pid),
                    "versions": (v.get("versions") or {}).get(pid) or [c], "evaluations": evals,
                    "trace_id": trace_id(v["run_id"], pid), "trace_file": _trace_file(pid)})
    return out


def _candidate_row(c: dict) -> dict:
    final = next((x for x in c["versions"] if x["version"] == c["final_version"]), c["versions"][-1])
    s = c.get("final_scores") or {}
    r = c["ranking"]
    return {"promo_id": c["promo_id"], "maker": c["maker"], "status": c["status"], "versions": len(c["versions"]),
            "final_version": c["final_version"], "segments": segs_text(final["segments"]),
            "total_duration_seconds": final.get("total_duration_seconds"), "moment_type": final.get("moment_type"),
            "hook": final.get("hook"), "cliffhanger": final.get("cliffhanger"),
            "score_hook": s.get("hook"), "score_cliffhanger": s.get("cliffhanger"),
            "score_standalone": s.get("standalone"), "score_final": s.get("final"),
            "ranking": r["outcome"], "rank": r.get("rank"), "drop_reason": r.get("reason"),
            "editor_decision": (c.get("editor_decision") or {}).get("decision"), "trace_id": c["trace_id"]}


CANDIDATE_CSV = ["promo_id", "maker", "status", "versions", "final_version", "segments", "total_duration_seconds",
                 "moment_type", "hook", "cliffhanger", "score_hook", "score_cliffhanger", "score_standalone",
                 "score_final", "ranking", "rank", "drop_reason", "editor_decision", "trace_id"]
MOMENT_CSV = ["moment_id", "episode", "kind", "timestamp", "strength", "moment_type", "quote", "description"]


def _trace(v: dict, pid: str) -> dict:
    return {"trace_id": trace_id(v["run_id"], pid), "promo_id": pid,
            "final": (v.get("candidates") or {}).get(pid),
            "moments": [m for m in v.get("moments") or [] if m["moment_id"] in
                        set((v["candidates"][pid].get("source_moment_ids") or []))],
            "versions": (v.get("versions") or {}).get(pid),
            "evaluations": (v.get("evaluations") or {}).get(pid),
            "editor_decisions": [d for d in (v.get("lessons") or []) if d.get("promo_id") == pid]
            or ([v["editor_decisions"][pid]] if pid in (v.get("editor_decisions") or {}) else []),
            "events": promo_trace(v, pid)}


def evaluation_metrics(v: dict, usage: dict) -> dict:
    """Per-run numbers the evaluation compares (spec section 10): memory on/off, guidelines
    version, rules/examples injected, editor approval, retry rate. First-pass approval uses
    round-1 decisions only (before any fix), the fairest with-vs-without-memory signal."""
    mem = v.get("memory_used") or {}
    lessons = v.get("lessons") or []
    first = {d["promo_id"]: d["decision"] for d in lessons if d.get("round") == 1}
    latest: dict[str, str] = {}
    for d in sorted(lessons, key=lambda d: d.get("round", 0)):
        latest[d["promo_id"]] = d["decision"]
    decided = len(first)
    approved = sum(1 for x in first.values() if x == "approve")
    return {
        "memory_enabled": bool(mem.get("enabled", (v.get("switches") or {}).get("memory"))),
        "guidelines_version": mem.get("guidelines_version"),
        "rules_injected": len(mem.get("rules") or []), "examples_injected": len(mem.get("examples") or []),
        "editor": {"decided": decided, "approved": approved,
                   "rejected": sum(1 for x in first.values() if x == "reject"),
                   "fix_requested": sum(1 for x in first.values() if x == "fix"),
                   "first_pass_approval_rate": round(approved / decided, 3) if decided else None,
                   "final_approval_rate": round(sum(1 for x in latest.values() if x in ("approve", "keep_before")) / len(latest), 3)
                   if latest else None},
        "retry_rate": usage.get("retry_rate"),
    }


def review_template(interrupt: dict | None) -> dict:
    """Decisions file for `run.py --resume <id> --decisions <file>`: approve by default."""
    if not interrupt:
        return {}
    promos = {p["promo_id"]: p for p in interrupt.get("promos", [])}
    out = {}
    for pid in interrupt.get("pending", []):
        p = promos.get(pid, {})
        allowed = ["approve", "reject"] + (["fix"] if (interrupt.get("can_fix") or {}).get(pid) else []) \
            + (["keep_before"] if pid in (interrupt.get("before_after") or {}) else [])
        out[pid] = {"decision": "approve", "comment": "", "_allowed": allowed,
                    "_info": f"rank {p.get('rank')} | final {p.get('scores', {}).get('final')} | "
                             f"{segs_text(p.get('segments') or [])} | hook: {p.get('hook')} | cliffhanger: {p.get('cliffhanger')}"}
    return out


def export_run(values: dict, settings: Settings, run_state: str, interrupt: dict | None = None,
               stop_reason: str | None = None, started: float | None = None) -> dict[str, str]:
    v = values
    run_id = v["run_id"]
    d = settings.resolve(settings.paths.output_dir) / "runs" / run_id
    files: dict[str, Path] = {}

    promos = [_promo_out(p) for p in v.get("promos") or []]
    files["promos_json"] = _dump(d / "promos.json", promos)
    files["promos_csv"] = _csv(d / "promos.csv", [_promo_row(p) for p in promos], PROMO_CSV)

    cands = _candidates(v)
    files["candidates_json"] = _dump(d / "raw" / "candidates.json", cands)
    files["candidates_csv"] = _csv(d / "raw" / "candidates.csv", [_candidate_row(c) for c in cands], CANDIDATE_CSV)
    files["moments_json"] = _dump(d / "raw" / "moments.json", v.get("moments") or [])
    files["moments_csv"] = _csv(d / "raw" / "moments.csv", v.get("moments") or [], MOMENT_CSV)
    per_ep = {r.get("episode"): r for r in (v.get("parse_report") or {}).get("per_episode", [])}
    notes = v.get("episode_notes") or {}
    files["episodes_json"] = _dump(d / "raw" / "episodes.json", [
        {**{k: e.get(k) for k in ("number", "source_file", "logline", "duration", "items")},
         "parse": per_ep.get(e["number"]), "reader": notes.get(str(e["number"]))} for e in v.get("episodes") or []])
    files["ranking_report"] = _dump(d / "raw" / "ranking_report.json", v.get("ranking_report") or {})
    if v.get("lessons"):
        files["review_log"] = _dump(d / "raw" / "review_log.json", {"summary": _review_summary(v),
                                                                    "decisions": v.get("lessons") or []})
    for pid in v.get("candidates") or {}:
        _dump(d / _trace_file(pid), _trace(v, pid))
    files["traces_dir"] = d / "traces"

    usage = usage_from_log(settings.resolve(settings.paths.logs_dir) / "llm" / run_id / "calls.jsonl")
    statuses = list((v.get("status") or {}).values())
    summary = {
        "run_id": run_id, "show": v.get("show"), "genre": v.get("genre"), "input_dir": v.get("input_dir"),
        "state": run_state, "stop_reason": stop_reason, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s_this_invocation": round(time.time() - started, 1) if started else None,
        "switches": v.get("switches"), "episode_filter": v.get("episode_filter"),
        "agents_kind": v.get("agents_kind"), "model_overrides": v.get("model_overrides") or {},
        "memory_used": {**{k: x for k, x in (v.get("memory_used") or {}).items() if k not in ("examples", "rules")},
                        "rule_ids": [r["id"] for r in (v.get("memory_used") or {}).get("rules") or []],
                        "rules": [r.get("text") for r in (v.get("memory_used") or {}).get("rules") or []],
                        "example_ids": [e["id"] for e in (v.get("memory_used") or {}).get("examples") or []]},
        "memory_written": v.get("memory_written"),
        "counts": {"episodes": len(v.get("episodes") or []), "moments": len(v.get("moments") or []),
                   "candidates": len(cands), "ranked": len(promos),
                   "status": {s: statuses.count(s) for s in sorted(set(filter(None, statuses)))},
                   "errors": len(v.get("errors") or []), "review_round": v.get("review_round", 0)},
        "parse": {k: x for k, x in (v.get("parse_report") or {}).items() if k != "per_episode"},
        "llm_usage": usage, "evaluation_metrics": evaluation_metrics(v, usage),
        "review": _review_summary(v), "errors": v.get("errors") or [],
        "logs": {"llm_calls": str(settings.resolve(settings.paths.logs_dir) / "llm" / run_id)},
    }
    files["run_summary"] = _dump(d / "run_summary.json", summary)
    if run_state == "awaiting_review":
        files["review_template"] = _dump(d / "review_template.json", review_template(interrupt))
    files["report"] = d / "report.md"
    files["report"].write_text(render_report(v, summary, promos, cands), encoding="utf-8")
    return {k: str(p) for k, p in files.items()}


# ------------------------------------------------------------------------------ report
def _md(x: Any) -> str:
    return str("" if x is None else x).replace("|", "\\|").replace("\n", " ")


def _review_summary(v: dict) -> dict:
    rs = v.get("review_summary") or {}
    return {"rounds": rs.get("rounds", v.get("review_round", 0)), "complete": bool(v.get("review_complete")),
            "stopped_by_max_rounds": rs.get("stopped_by_max_rounds", False), "unreviewed": rs.get("unreviewed", []),
            "decisions": len(v.get("lessons") or [])}


def render_report(v: dict, summary: dict, promos: list[dict], cands: list[dict]) -> str:
    c = summary["counts"]
    u = summary["llm_usage"]
    t = u["totals"]
    L = [f"# Promo report: {v.get('show')}", "",
         f"- Run `{v['run_id']}` | genre: {v.get('genre')} | state: **{summary['state']}** | written {summary['written_at']}",
         f"- Episodes: {c['episodes']} | parse coverage {summary['parse'].get('coverage_pct')}% | "
         f"unparsed lines {summary['parse'].get('unparsed_lines')}",
         f"- Moments: {c['moments']} | promos made: {c['candidates']} {c['status']} | ranked: {c['ranked']}",
         f"- Switches: {summary['switches']}"]
    if summary.get("stop_reason"):
        L.append(f"- **Stopped:** {_md(summary['stop_reason'])[:400]}  \n  Resume: `python run.py --resume {v['run_id']}`")
    if summary["state"] == "awaiting_review":
        L.append(f"- **Waiting for editor review.** Edit `review_template.json`, then "
                 f"`python run.py --resume {v['run_id']} --decisions output/runs/{v['run_id']}/review_template.json`")
    L += ["", "## Ranked promos", "",
          "| # | Promo | Segments | Sec | Type | Hook | Cliff | Stand | Final | Rank score |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in promos:
        s = p["scores"]
        L.append(f"| {p['rank']} | [{p['promo_id']}]({p['trace_file']}) | {_md(segs_text(p['segments']))} | "
                 f"{p['total_duration_seconds']} | {p['moment_type']} | {s.get('hook')} | {s.get('cliffhanger')} | "
                 f"{s.get('standalone')} | **{s.get('final')}** | {p.get('rank_score')} |")
    for p in promos:
        r = p.get("evaluator_reasons") or {}
        L += ["", f"### {p['rank']}. {p['promo_id']} ({p['moment_type']}, {p['total_duration_seconds']}s, final {p['scores'].get('final')})",
              f"- **Segments:** {segs_text(p['segments'])}",
              f"- **Hook:** {_md(p.get('hook'))}", f"- **Cliffhanger:** {_md(p.get('cliffhanger'))}",
              f"- **Why it works:** {_md(p.get('why_it_works'))}"]
        if p.get("join_rationale"):
            L.append(f"- **Join:** {_md(p['join_rationale'])}")
        if r:
            L.append(f"- **Evaluator:** hook: {_md(r.get('hook'))} / cliffhanger: {_md(r.get('cliffhanger'))} / "
                     f"standalone: {_md(r.get('standalone'))}")
        if p.get("version", 1) > 1:
            L.append(f"- **Revised** to v{p['version']}: {_md(p.get('revision_note'))}")
        L.append(f"- Trace: `{p['trace_id']}` -> [{p['trace_file']}]({p['trace_file']})")

    dropped = [x for x in cands if x["ranking"]["outcome"] == "dropped"]
    L += ["", f"## Dropped by ranking ({len(dropped)})", "", "| Promo | Final | Reason | Detail |", "|---|---|---|---|"]
    for x in sorted(dropped, key=lambda x: x["promo_id"]):
        r = x["ranking"]
        detail = {k: y for k, y in r.items() if k not in ("outcome", "reason", "final")}
        L.append(f"| {x['promo_id']} | {r.get('final')} | {r.get('reason')} | {_md(detail)} |")

    L += ["", "## LLM usage", "",
          f"{t['calls']} calls, {t['attempts']} attempts, retry rate {u['retry_rate']}, "
          f"{t['input_tokens']} input + {t['output_tokens']} output tokens, {t['rate_limit_waits']} rate-limit waits, "
          f"{t['model_switches']} model switches.", "",
          "| Agent | Models (calls) | Calls | Retries | Native / JSON mode | Tokens in / out |", "|---|---|---|---|---|---|"]
    for a, x in u["by_agent"].items():
        models = ", ".join(f"{m} ({n})" for m, n in x["models"].items())
        L.append(f"| {a} | {_md(models)} | {x['calls']} | {x['retries']} | {x['native_calls']} / {x['fallback_calls']} | "
                 f"{x['input_tokens']} / {x['output_tokens']} |")
    for s in u["model_switches"]:
        L.append(f"\n- Switch: {s['agent']} {s['from']} -> {s['to']} ({s['reason']})")

    if v.get("lessons"):
        r, m = summary["review"], summary["evaluation_metrics"]["editor"]
        pct = lambda x: "-" if x is None else f"{x:.0%}"
        L += ["", "## Editor review", "",
              f"{r['rounds']} round(s), {r['decisions']} decision(s). First-pass approval "
              f"{pct(m['first_pass_approval_rate'])}, final {pct(m['final_approval_rate'])}."]
        if r["unreviewed"]:
            L.append(f"\n**Stopped at the max number of rounds; not reviewed in final form:** {', '.join(r['unreviewed'])}")
        L += ["", "| Round | Promo | Decision | Comment | FeedbackAgent reading | Version | Final |", "|---|---|---|---|---|---|---|"]
        for l in v["lessons"]:
            L.append(f"| {l.get('round')} | {l['promo_id']} | {l['decision']} | {_md(l.get('comment'))} | "
                     f"{_md(l.get('fix_interpretation'))} | {l.get('version')} | {l.get('final')} |")
    if summary["errors"]:
        L += ["", f"## Errors ({len(summary['errors'])})", ""] + [f"- {_md(e)[:300]}" for e in summary["errors"]]
    L += ["", "## Episodes", "", "| Ep | Coverage % | Items | Warnings | Reader |", "|---|---|---|---|---|"]
    notes = v.get("episode_notes") or {}
    for r in (v.get("parse_report") or {}).get("per_episode", []):
        n = notes.get(str(r.get("episode")), {})
        L.append(f"| {r.get('episode')} | {r.get('coverage_pct')} | {r.get('items')} | {r.get('warnings')} | "
                 f"{'FAILED' if n.get('failed') else 'ok' if n else '-'} |")
    return "\n".join(L) + "\n"
