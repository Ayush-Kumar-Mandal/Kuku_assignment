"""Parse a folder of screenplays and report coverage per episode.

    python -m parser.validate --input <folder> [--out output/parsed] [--episodes 1 6 18]
                              [--parser-config config/parser.yaml]

Writes one JSON per episode, logs unparsed lines, prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config.loader import load_parser_config, load_settings
from parser.models import Episode, format_ts
from parser.screenplay_parser import parse_episode


def summarize(ep: Episode) -> dict:
    kinds = Counter(it.kind for it in ep.all_items())
    return {
        "episode": ep.number,
        "file": ep.source_file,
        "scenes": len(ep.scenes),
        "items": sum(kinds.values()),
        **{f"n_{k}": v for k, v in sorted(kinds.items())},
        "characters": len({it.character for it in ep.all_items() if it.character}),
        "duration_s": ep.duration,
        "coverage_pct": ep.coverage.percent,
        "unparsed": ep.coverage.unparsed_lines,
        "llm_recovered": ep.coverage.recovered_by_fallback,
        "ends_inferred": sum(it.end_inferred for it in ep.all_items()),
        "warnings": len(ep.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of screenplay files")
    ap.add_argument("--out", default=None, help="where to write parsed JSON (default: <output_dir>/parsed)")
    ap.add_argument("--parser-config", default=None)
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--episodes", type=int, nargs="*", help="only these episode numbers")
    ap.add_argument("--llm-fallback", action="store_true",
                    help="send lines the patterns can't parse to the LLM fallback parser")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = load_settings(args.settings)
    cfg = load_parser_config(args.parser_config, settings)
    out_dir = Path(args.out) if args.out else settings.resolve(settings.paths.output_dir) / "parsed"
    log_dir = settings.resolve(settings.paths.logs_dir) / "parser"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.input).glob(cfg.file_glob))
    if not files:
        print(f"No files matching {cfg.file_glob!r} in {args.input}")
        return 1

    fallback = None
    if args.llm_fallback:
        from llm.client import LLMClient
        from llm.parser_fallback import make_parser_fallback
        fallback = make_parser_fallback(LLMClient(settings, run_id="parser"), cfg)

    rows, episodes = [], []
    for f in files:
        ep = parse_episode(f, cfg, fallback)
        if args.episodes and ep.number not in args.episodes:
            continue
        episodes.append(ep)
        rows.append(summarize(ep))
        (out_dir / f"episode_{ep.number:02d}.json").write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        if ep.unparsed or ep.warnings:
            with (log_dir / f"episode_{ep.number:02d}.jsonl").open("w", encoding="utf-8") as fh:
                for u in ep.unparsed:
                    fh.write(json.dumps({"type": "unparsed", **u.model_dump()}, ensure_ascii=False) + "\n")
                for w in ep.warnings:
                    fh.write(json.dumps({"type": "warning", "message": w}, ensure_ascii=False) + "\n")

    rows.sort(key=lambda r: r["episode"])
    kinds = sorted({k for r in rows for k in r if k.startswith("n_")})
    cols = ["episode", "scenes", "items", *kinds, "characters", "duration_s", "coverage_pct", "unparsed", "ends_inferred", "warnings"]
    head = {"episode": "ep", "coverage_pct": "cover%", "duration_s": "dur", "ends_inferred": "end_inf", "characters": "chars"}
    labels = [head.get(c, c.removeprefix("n_")) for c in cols]
    widths = [max(len(l), 6) for l in labels]
    print("  ".join(l.rjust(w) for l, w in zip(labels, widths)))
    for r in rows:
        vals = [format_ts(r[c]) if c == "duration_s" else r.get(c, 0) for c in cols]
        print("  ".join(str(v).rjust(w) for v, w in zip(vals, widths)))

    total_content = sum(e.coverage.content_lines for e in episodes)
    total_parsed = sum(e.coverage.parsed_lines for e in episodes)
    totals = {
        "episodes": len(rows),
        "scenes": sum(r["scenes"] for r in rows),
        "items": sum(r["items"] for r in rows),
        **{k.removeprefix("n_"): sum(r.get(k, 0) for r in rows) for k in kinds},
        "total_runtime": format_ts(sum(r["duration_s"] for r in rows)),
        "coverage_pct": round(100 * total_parsed / total_content, 2) if total_content else 0,
        "unparsed_lines": sum(r["unparsed"] for r in rows),
        "warnings": sum(r["warnings"] for r in rows),
    }
    print("\nTOTAL:", "  ".join(f"{k}={v}" for k, v in totals.items()))
    for ep in episodes:
        for u in ep.unparsed:
            print(f"  ep{ep.number:02d} line {u.line_no} UNPARSED ({u.reason}): {u.text[:80]}")
        for w in ep.warnings:
            print(f"  ep{ep.number:02d} WARNING: {w}")
    (out_dir / "_summary.json").write_text(json.dumps({"totals": totals, "episodes": rows}, indent=2), encoding="utf-8")
    print(f"\nParsed JSON -> {out_dir}   logs -> {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
