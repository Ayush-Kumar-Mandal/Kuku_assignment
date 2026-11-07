"""Per-run log across every run, and editor approval with vs without memory.

    python -m evaluate.runs

One row per run (from output/runs/*/run_summary.json): memory on/off, guidelines version,
rules/examples injected, editor decisions and approval rate (first pass = round-1 decisions,
before any fixes), retry rate, calls/tokens, ranked promos, and your post rate if judged.
Writes <evaluation.dir>/runs.csv and runs.md.
"""
from __future__ import annotations

import argparse
import csv
import sys

from config.loader import Settings, load_settings
from evaluate.common import eval_dir, read_json
from evaluate.judge import judgements_path

FIELDS = ["run_id", "show", "written_at", "state", "memory", "guidelines_version", "rules_injected",
          "examples_injected", "editor_decided", "editor_approved", "approval_rate", "final_approval_rate",
          "retry_rate", "calls", "input_tokens", "output_tokens", "ranked", "post_rate"]


def row(summary: dict, post_rate: float | None = None) -> dict:
    m = summary.get("evaluation_metrics") or {}
    ed = m.get("editor") or {}
    u = summary.get("llm_usage") or {}
    t = u.get("totals") or {}
    mem = m.get("memory_enabled")
    if mem is None:
        mem = (summary.get("switches") or {}).get("memory")
    return {"run_id": summary.get("run_id"), "show": summary.get("show"), "written_at": summary.get("written_at"),
            "state": summary.get("state"), "memory": bool(mem), "guidelines_version": m.get("guidelines_version"),
            "rules_injected": m.get("rules_injected", 0), "examples_injected": m.get("examples_injected", 0),
            "editor_decided": ed.get("decided", 0), "editor_approved": ed.get("approved", 0),
            "approval_rate": ed.get("first_pass_approval_rate"), "final_approval_rate": ed.get("final_approval_rate"),
            "retry_rate": u.get("retry_rate"), "calls": t.get("calls"), "input_tokens": t.get("input_tokens"),
            "output_tokens": t.get("output_tokens"), "ranked": (summary.get("counts") or {}).get("ranked"),
            "post_rate": post_rate}


def memory_comparison(rows: list[dict]) -> dict:
    """Pooled first-pass approval rate (approved / decided over all reviewed runs) per group."""
    out = {}
    for key, flag in (("with_memory", True), ("without_memory", False)):
        group = [r for r in rows if r["memory"] is flag]
        reviewed = [r for r in group if r["editor_decided"]]
        decided = sum(r["editor_decided"] for r in reviewed)
        approved = sum(r["editor_approved"] for r in reviewed)
        out[key] = {"runs": len(group), "runs_with_review": len(reviewed), "decided": decided,
                    "approval_rate": round(approved / decided, 3) if decided else None}
    a, b = out["with_memory"]["approval_rate"], out["without_memory"]["approval_rate"]
    out["difference"] = round(a - b, 3) if a is not None and b is not None else None
    return out


def collect(settings: Settings) -> list[dict]:
    root = settings.resolve(settings.paths.output_dir) / "runs"
    rows = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        s = read_json(d / "run_summary.json")
        if not s:
            continue
        js = read_json(judgements_path(settings, d.name), {})
        judged = [j for j in js.values() if j.get("post") is not None]
        rows.append(row(s, round(sum(j["post"] for j in judged) / len(judged), 3) if judged else None))
    return sorted(rows, key=lambda r: r["written_at"] or "")


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    rows = collect(settings)
    cmp_ = memory_comparison(rows)
    d = eval_dir(settings)
    with (d / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda x: "-" if x is None else f"{x:.0%}" if isinstance(x, float) and x <= 1 else str(x)
    L = ["# Runs", "", "| Run | Memory | Guidelines | Rules | Examples | Decided | Approval (1st pass) | "
         "Retry rate | Calls | Ranked | Post rate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run_id']} | {'on' if r['memory'] else 'off'} | {r['guidelines_version'] or '-'} | "
                 f"{r['rules_injected']} | {r['examples_injected']} | {r['editor_decided']} | {fmt(r['approval_rate'])} | "
                 f"{fmt(r['retry_rate'])} | {r['calls'] or '-'} | {r['ranked'] or '-'} | {fmt(r['post_rate'])} |")
    L += ["", "## Approval with vs without memory", "", "| | Runs | Reviewed runs | Decisions | Approval rate |",
          "|---|---|---|---|---|"]
    for k in ("with_memory", "without_memory"):
        c = cmp_[k]
        L.append(f"| {k.replace('_', ' ')} | {c['runs']} | {c['runs_with_review']} | {c['decided']} | {fmt(c['approval_rate'])} |")
    diff = cmp_["difference"]
    L.append("\nDifference (with - without): " + ("- (needs reviewed runs in both groups)" if diff is None
                                                  else f"{diff:+.0%}"))
    (d / "runs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {d / 'runs.csv'}, {d / 'runs.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
