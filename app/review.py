"""Review logic shared by the Streamlit Review tab and the terminal review (run.py --review).

Input: the editor_review interrupt payload + the run's checkpointed state. Output: one item
per listed promo (pending ones first) with everything the editor needs to decide: the
transcript as a viewer sees it, scores and the evaluator's reasons, the before/after pair
after an editor fix, and the decisions allowed for it right now. No Streamlit here.
"""
from __future__ import annotations

import json
from pathlib import Path

OPTIONS = {
    "approve": "Approve",
    "reject": "Reject",
    "fix": "Fix (say what to change)",
    "keep_before": "Keep before (undo the fix)",
}


def _eval_for(values: dict, pid: str, version: int) -> dict | None:
    return next((e for e in reversed((values.get("evaluations") or {}).get(pid) or []) if e["version"] == version), None)


def review_items(payload: dict, values: dict) -> list[dict]:
    pending = set(payload.get("pending") or [])
    decided = values.get("editor_decisions") or {}
    items = []
    for p in payload.get("promos") or []:
        pid = p["promo_id"]
        ev = _eval_for(values, pid, p["version"]) or {}
        ba = (payload.get("before_after") or {}).get(pid)
        if ba:
            ba = {**ba, "before_text": (ba.get("before_eval") or {}).get("promo_text"),
                  "after_text": (ba.get("after_eval") or {}).get("promo_text")}
        opts = []
        if pid in pending:
            opts = ["approve", "reject"] + (["fix"] if (payload.get("can_fix") or {}).get(pid) else []) \
                + (["keep_before"] if ba else [])
        items.append({"promo_id": pid, "rank": p.get("rank"), "round": payload.get("round"), "promo": p,
                      "pending": pid in pending, "options": opts, "scores": p.get("scores") or {},
                      "reasons": ev.get("reasons") or p.get("evaluator_reasons") or {},
                      "transcript": ev.get("promo_text"), "before_after": ba,
                      "fix_attempts": (payload.get("fix_attempts") or {}).get(pid, 0),
                      "max_fix_attempts": payload.get("max_fix_attempts"),
                      "note": (payload.get("notes") or {}).get(pid), "decision": decided.get(pid)})
    items.sort(key=lambda i: (not i["pending"], i["rank"] or 999))
    return items


def validate(items: list[dict], decisions: dict) -> list[str]:
    """Same rules editor_review enforces, checked before resuming so the editor sees them now."""
    by_id = {i["promo_id"]: i for i in items}
    errors = []
    for pid, d in decisions.items():
        it = by_id.get(pid)
        dec = (d or {}).get("decision")
        if not it or not it["pending"]:
            errors.append(f"{pid}: not under review this round")
        elif dec not in it["options"]:
            errors.append(f"{pid}: '{dec}' is not an option here (allowed: {', '.join(it['options'])})")
        elif dec == "fix" and not ((d or {}).get("comment") or "").strip():
            errors.append(f"{pid}: Fix needs a comment saying what to change")
    return errors


def write_decisions(run_dir: str | Path, rnd: int, decisions: dict) -> Path:
    path = Path(run_dir) / f"review_round{rnd}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
