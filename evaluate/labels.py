"""Hand-label good promo moments BEFORE looking at the tool's picks.

    python -m evaluate.labels --input <folder> --show <name> --episodes 1-8 [--labels <file>]

Shows each episode's screenplay (never the tool's output) and asks for moments:
    START END [type] [note...]   e.g.  01:21.80 02:20.30 romance stranger proposes
    done   episode finished (counted for recall, even with no moments)
    skip   leave the episode unlabeled for now
    undo   remove the last moment of this episode
    quit   save and stop (re-run to continue)
START must be a line's start time and END a line's end time, as printed. Saved after every
moment. When `evaluate.judge` starts, the labels are LOCKED (locked_at), so they stay blind.
The file is plain YAML and can also be edited by hand before it is locked.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Literal

import yaml
from pydantic import BaseModel

from agents.llm_agents import render_lines
from config.loader import MomentTypes, ParserConfig, Settings, load_moment_types, load_parser_config, load_settings
from evaluate.common import eval_dir, slug
from llm.checks import ScreenplayIndex
from parser.models import Episode
from parser.screenplay_parser import parse_show

COMMANDS = {"done", "skip", "undo", "quit"}


class LabelMoment(BaseModel):
    id: str
    episode: int
    start: str
    end: str
    moment_type: str = "other"
    note: str = ""
    why: str = ""                           # AI labeller's reason (evaluate.ai_labels); empty for hand labels


class Labels(BaseModel):
    show: str
    labeled_by: str = "human"               # anything else (e.g. an AI) is flagged in the evaluation report
    label_source: Literal["human", "ai", "planted"] = "human"   # planted = synthetic show's answer key
    labeler_models: list[str] = []          # AI labels: the models that actually answered (provider:model)
    prompt_version: int | None = None       # AI labels: prompts/labeler.yaml version
    input_dir: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    locked_at: str | None = None            # set by evaluate.judge: labels were made blind
    episodes: list[int] = []                # episodes fully labeled (only these count for recall)
    moments: list[LabelMoment] = []


def default_path(settings: Settings, show: str) -> Path:
    return eval_dir(settings) / "labels" / f"{slug(show)}.yaml"


def load_labels(path: Path) -> Labels:
    return Labels.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def save_labels(lab: Labels, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lab.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    lab.created_at = lab.created_at or lab.updated_at
    path.write_text(yaml.safe_dump(lab.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def lock(path: Path) -> Labels:
    lab = load_labels(path)
    if not lab.locked_at:
        lab.locked_at = time.strftime("%Y-%m-%d %H:%M:%S")
        save_labels(lab, path)
    return lab


def _check(index: ScreenplayIndex, ep: Episode, start: str, end: str, kinds: list[str]) -> list[str]:
    items = [it for it in ep.all_items() if it.kind in kinds]
    starts, ends = sorted({it.start for it in items}), sorted({it.end for it in items})
    s, e = index.seconds(start), index.seconds(end)
    probs = []
    if s is None or not any(abs(x - s) <= index.tol for x in starts):
        probs.append(f"{start} is not a line start; nearest: {index.nearest(starts, s or 0)}")
    if e is None or not any(abs(x - e) <= index.tol for x in ends):
        probs.append(f"{end} is not a line end; nearest: {index.nearest(ends, e or 0)}")
    if not probs and e <= s:
        probs.append("end must be after start")
    return probs


def label_session(lab: Labels, episodes: dict[int, Episode], todo: list[int], settings: Settings,
                  pcfg: ParserConfig, mtypes: MomentTypes, ask: Callable[[str], str] = input,
                  say: Callable[[str], None] = print, save: Callable[[Labels], None] = lambda l: None) -> Labels:
    kinds = settings.agents.screenplay_kinds
    pc = settings.promo
    for n in todo:
        ep = episodes.get(n)
        if ep is None:
            say(f"(episode {n} not found, skipped)")
            continue
        index = ScreenplayIndex([ep], pcfg, pc.boundary_tolerance_s)
        say(f"\n{'=' * 70}\nEpisode {n}  ({ep.duration:.0f}s)\n{'=' * 70}")
        say(render_lines([it for it in ep.all_items() if it.kind in kinds]))
        say(f"\nMoment types: {', '.join(mtypes.moment_types)}.  Commands: {', '.join(sorted(COMMANDS))}.")
        while True:
            raw = ask(f"ep{n} moment> ").strip()
            if not raw:
                continue
            cmd = raw.lower()
            if cmd == "quit":
                save(lab)
                return lab
            if cmd == "skip":
                break
            if cmd == "done":
                if n not in lab.episodes:
                    lab.episodes = sorted(lab.episodes + [n])
                save(lab)
                break
            if cmd == "undo":
                mine = [m for m in lab.moments if m.episode == n]
                if mine:
                    lab.moments.remove(mine[-1])
                    say(f"removed {mine[-1].id}")
                    save(lab)
                continue
            parts = raw.split()
            if len(parts) < 2:
                say("format: START END [type] [note...]")
                continue
            probs = _check(index, ep, parts[0], parts[1], kinds)
            if probs:
                say("rejected: " + "; ".join(probs))
                continue
            rest = parts[2:]
            if rest and rest[0].lower() in mtypes.moment_types:
                mtype, note = rest[0].lower(), " ".join(rest[1:])
            else:                                   # no type given: every word is the note
                mtype, note = mtypes.default_type, " ".join(rest)
            m = LabelMoment(id=f"L{len(lab.moments) + 1:02d}", episode=n, start=parts[0], end=parts[1],
                            moment_type=mtype, note=note)
            lab.moments.append(m)
            dur = index.seconds(parts[1]) - index.seconds(parts[0])
            warn = "" if pc.min_duration_s <= dur <= pc.max_duration_s else \
                f"  (note: {dur:.1f}s is outside {pc.min_duration_s:g}-{pc.max_duration_s:g}s; kept)"
            say(f"saved {m.id}: ep{n} {m.start}-{m.end} {m.moment_type} {m.note}{warn}")
            save(lab)
    return lab


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    from run import parse_episodes
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--show", required=True)
    ap.add_argument("--episodes", required=True, help="e.g. 1-8 (5-10 episodes is enough)")
    ap.add_argument("--labels", help="labels file (default: <evaluation.dir>/labels/<show>.yaml)")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    path = Path(args.labels) if args.labels else default_path(settings, args.show)
    lab = load_labels(path) if path.exists() else Labels(show=args.show, input_dir=str(Path(args.input).resolve()))
    if lab.locked_at:
        raise SystemExit(f"{path} was locked at {lab.locked_at} (judging started): labels must stay blind. "
                         f"Use a new --labels file to label again.")
    pcfg = load_parser_config(settings=settings)
    show = parse_show(Path(args.input), pcfg)
    todo = [n for n in parse_episodes(args.episodes) if n not in lab.episodes]
    print(f"Labelling {len(todo)} episode(s) of {args.show} -> {path}")
    label_session(lab, {e.number: e for e in show.episodes}, todo, settings, pcfg,
                  load_moment_types(settings=settings), save=lambda l: save_labels(l, path))
    save_labels(lab, path)
    print(f"\n{len(lab.moments)} moments over {len(lab.episodes)} finished episode(s) saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
