"""Judge the tool's picks: would you actually post each one? If not, why?

    python -m evaluate.judge --run <run_id> [--labels <file>] [--scope ranked|all]

Run this AFTER labelling: it locks the labels file first, so labels stay blind. Each pick is
shown as the viewer would get it (its own lines, times from 0), with its segments:
    y          would post it
    n          would not -> pick a failure category (number) and add a note
    s          skip for now        q   save and stop (re-run to continue; judged ones are kept)
Judgements are saved to <evaluation.dir>/<run_id>/judgements.json after every answer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

from agents.render import render_promo_text
from config.loader import ParserConfig, Settings, load_parser_config, load_settings
from evaluate import labels as labels_mod
from evaluate.common import eval_dir, read_json, run_episodes, run_path
from graph.export import segs_text
from parser.models import Episode


def judgements_path(settings: Settings, run_id: str) -> Path:
    return eval_dir(settings) / run_id / "judgements.json"


def judge_session(promos: list[dict], episodes: dict[int, Episode], judgements: dict, settings: Settings,
                  pcfg: ParserConfig, ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                  save: Callable[[dict], None] = lambda j: None) -> dict:
    cats = list(settings.evaluation.failure_categories.items())
    menu = "  ".join(f"{i}={label}" for i, (_, label) in enumerate(cats, start=1))
    todo = [p for p in promos if p["promo_id"] not in judgements]
    for k, p in enumerate(todo, start=1):
        rank = f"#{p['rank']} " if p.get("rank") else ""
        say(f"\n{'=' * 70}\n{rank}{p['promo_id']}  ({k}/{len(todo)})  {segs_text(p['segments'])}  "
            f"{p.get('total_duration_seconds')}s\n{'=' * 70}")
        say(render_promo_text(p, episodes, pcfg.compiled))
        say(f"\ntool says - hook: {p.get('hook')} | cliffhanger: {p.get('cliffhanger')}")
        while True:
            a = ask("post it? [y/n/s/q] ").strip().lower()
            if a in ("y", "n", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue
        entry = {"post": a == "y", "category": None, "note": "", "judged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "rank": p.get("rank"), "segments": p["segments"]}
        if a == "n":
            say(menu)
            while True:
                c = ask("category number> ").strip()
                if c.isdigit() and 1 <= int(c) <= len(cats):
                    entry["category"] = cats[int(c) - 1][0]
                    break
            entry["note"] = ask("note (optional)> ").strip()
        judgements[p["promo_id"]] = entry
        save(judgements)
    return judgements


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", help="labels file to lock (default: <evaluation.dir>/labels/<show>.yaml)")
    ap.add_argument("--scope", choices=["ranked", "all"], default="ranked",
                    help="ranked = the tool's final picks (default); all = every promo it made")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    d = run_path(settings, args.run)
    summary = read_json(d / "run_summary.json") or {}
    lab_path = Path(args.labels) if args.labels else labels_mod.default_path(settings, summary.get("show") or args.run)
    if lab_path.exists():
        lab = labels_mod.lock(lab_path)
        print(f"Labels locked ({lab.locked_at}): {lab_path}")
    else:
        print(f"Note: no labels at {lab_path}. Label first (python -m evaluate.labels ...) if you want recall.")
    if args.scope == "ranked":
        promos = read_json(d / "promos.json", [])
    else:
        promos = [{**next(v for v in c["versions"] if v["version"] == c["final_version"]),
                   "rank": c["ranking"].get("rank")} for c in read_json(d / "raw" / "candidates.json", [])]
    path = judgements_path(settings, args.run)
    js = read_json(path, {})
    path.parent.mkdir(parents=True, exist_ok=True)
    save = lambda j: path.write_text(json.dumps(j, indent=1, ensure_ascii=False), encoding="utf-8")
    judge_session(promos, run_episodes(settings, args.run), js, settings, load_parser_config(settings=settings), save=save)
    save(js)
    posted = sum(1 for j in js.values() if j["post"])
    print(f"\n{len(js)} judged, {posted} you'd post. Saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
