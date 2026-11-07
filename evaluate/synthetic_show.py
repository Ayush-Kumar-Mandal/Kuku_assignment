"""Synthetic show with planted moments: a second dataset whose answer key is known by construction.

    python -m evaluate.synthetic_show --show "<title>" --genre "<genre>" [--episodes 8] [--seed 7]

1. The writer LLM (models.synthetic_writer) invents an original show bible.
2. Code picks the moments to plant (seeded): per episode 1-2, each with a moment type and a
   position. The writer never chooses what the answers are.
3. The writer writes each episode as plain lines (action / dialogue / sound), marking the
   lines of each planted moment. Everything else is meant to stay ordinary.
4. Code gives every line its time from its word count (config/synthetic.yaml) and writes the
   files in the real screenplay format. Before anything is saved, each episode is parsed with
   the real parser (100% coverage, no warnings) and each plant's spoken length is checked
   against promo.min/max_duration_s; problems go back to the writer to fix.
The answer key is a normal labels file (label_source: planted, locked) at
<evaluation.dir>/labels/<show>.yaml, so `evaluate.compare --run <run>` finds it by show name.
Recall on the plants is exact; precision is approximate (the writer may add strong moments it
wasn't asked for). Drafts are saved as they come, so a stopped run continues where it was.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import DraftLine, DraftScene, EpisodeDraft, EpisodeScript, ShowBible
from config.loader import MomentTypes, ParserConfig, Settings, load_moment_types, load_parser_config, load_settings
from evaluate import labels as labels_mod
from evaluate.common import slug
from llm.client import LLMClient
from llm.prompts import load_chat_prompt
from parser.models import Episode, format_ts
from parser.screenplay_parser import parse_text

AGENT = "synthetic_writer"


# ------------------------------------------------------------------------------ config
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SynthTiming(_Strict):
    words_per_second: float = Field(gt=0)
    min_line_s: float = Field(gt=0)
    max_line_s: float = Field(gt=0)
    gap_s: float = Field(ge=0)
    scene_gap_s: float = Field(ge=0)


class SynthRender(_Strict):
    file_name: str
    header: str
    script_start: str
    script_end: str
    scene: str
    block: str
    dialogue_text: str
    speaker_with_emotion: str
    speaker_plain: str
    kind_labels: dict[str, str]


class SynthConfig(_Strict):
    output_dir: str
    episodes: int = Field(ge=1)
    scenes_per_episode: tuple[int, int]
    lines_per_episode: tuple[int, int]
    plants_per_episode: tuple[int, int]
    plant_lines: tuple[int, int]
    positions: list[str] = Field(min_length=1)
    story_summaries: int = Field(ge=0)
    line_labels: dict[Literal["action", "sound"], list[str]]
    timing: SynthTiming
    render: SynthRender


def load_config(settings: Settings) -> SynthConfig:
    path = settings.resolve(settings.paths.synthetic_config)
    cfg = SynthConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if cfg.plants_per_episode[1] > len(cfg.positions):
        raise ValueError(f"{path}: plants_per_episode max {cfg.plants_per_episode[1]} > {len(cfg.positions)} positions")
    return cfg


# ------------------------------------------------------------------------- the plan
def plan_plants(cfg: SynthConfig, mtypes: MomentTypes, n_episodes: int, seed: int) -> dict[int, list[dict]]:
    """Which moments to plant where. Chosen by code (seeded), never by the writer."""
    rng = random.Random(seed)
    types = [k for k in mtypes.moment_types if k != mtypes.default_type]
    plan = {}
    for ep in range(1, n_episodes + 1):
        k = rng.randint(*cfg.plants_per_episode)
        pos = sorted(rng.sample(cfg.positions, k), key=cfg.positions.index)
        plan[ep] = [{"id": f"P{i + 1}", "moment_type": rng.choice(types), "position": p} for i, p in enumerate(pos)]
    return plan


# ----------------------------------------------------------------------- timing + format
def ts(cs: int) -> str:
    return format_ts(cs / 100)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("*", "")).strip()


def _line_text(ln) -> str:
    t = _clean(ln.text)
    return t.strip("\"“”") if ln.kind == "dialogue" else t


def build_timeline(draft: EpisodeDraft, t: SynthTiming) -> list[list[tuple[int, int]]]:
    """(start, end) in centiseconds for every line, scene by scene. Integer maths, so the
    written times are exactly the times the answer key uses."""
    lo, hi = round(t.min_line_s * 100), round(t.max_line_s * 100)
    gap, scene_gap = round(t.gap_s * 100), round(t.scene_gap_s * 100)
    cur, out = 0, []
    for sc in draft.scenes:
        rows = []
        for ln in sc.lines:
            d = min(max(round(len(_line_text(ln).split()) / t.words_per_second * 100), lo), hi)
            rows.append((cur, cur + d))
            cur += d + gap
        out.append(rows)
        cur = (rows[-1][1] if rows else cur) + scene_gap
    return out


def _label(ln, r: SynthRender) -> str:
    if ln.kind != "dialogue":
        return r.kind_labels[ln.kind]
    name = re.sub(r"[():*\[\]]", "", _clean(ln.speaker)).upper().strip()
    emotion = re.sub(r"[():*\[\]]", "", _clean(ln.emotion)).strip()
    return r.speaker_with_emotion.format(name=name, emotion=emotion) if emotion else r.speaker_plain.format(name=name)


def render_episode(number: int, draft: EpisodeDraft, cfg: SynthConfig) -> str:
    r = cfg.render
    tl = build_timeline(draft, cfg.timing)
    out = [r.header.format(number=number, logline=_clean(draft.logline), continuity=_clean(draft.continuity_notes)),
           "", r.script_start, ""]
    for i, (sc, rows) in enumerate(zip(draft.scenes, tl), start=1):
        out += [r.scene.format(num=i, heading=_clean(sc.heading) or "INT. UNKNOWN - DAY", start=ts(rows[0][0]),
                               end=ts(rows[-1][1])), ""]
        for ln, (a, b) in zip(sc.lines, rows):
            text = _line_text(ln)
            out += [r.block.format(label=_label(ln, r), start=ts(a), end=ts(b),
                                   text=r.dialogue_text.format(text=text) if ln.kind == "dialogue" else text), ""]
    out.append(r.script_end)
    return "\n".join(out) + "\n"


def plant_spans(draft: EpisodeDraft, tl: list[list[tuple[int, int]]]) -> dict[str, tuple[int, int, int, int, int]]:
    """plant id -> (scene index, first line, last line, start cs, end cs), for plants that
    are one consecutive run of lines inside one scene."""
    where: dict[str, list[tuple[int, int]]] = {}
    for si, sc in enumerate(draft.scenes):
        for li, ln in enumerate(sc.lines):
            if ln.plant:
                where.setdefault(ln.plant.strip(), []).append((si, li))
    out = {}
    for pid, locs in where.items():
        scenes = {s for s, _ in locs}
        idx = [i for _, i in locs]
        if len(scenes) == 1 and idx == list(range(idx[0], idx[-1] + 1)):
            si = scenes.pop()
            out[pid] = (si, idx[0], idx[-1], tl[si][idx[0]][0], tl[si][idx[-1]][1])
    return out


def plant_word_range(cfg: SynthConfig, settings: Settings) -> tuple[int, int]:
    """Words a planted moment should hold to land inside promo.min/max_duration_s, with a
    margin each side (a model can't time speech, but it can count words)."""
    wps, pc = cfg.timing.words_per_second, settings.promo
    lo = math.ceil(pc.min_duration_s * 1.15 * wps)
    hi = math.floor(pc.max_duration_s * 0.85 * wps)
    return lo, max(hi, lo + 1)


def story_line(n: int, d: EpisodeDraft) -> str:
    return f"Episode {n}: {_clean(d.summary) or _clean(d.logline)}"


_LINE = re.compile(r"^\s*(?:\[\s*(?P<plant>[A-Za-z]\d+)\s*\]\s*)?(?P<label>[^:\[\]]{1,60}?)\s*:\s*(?P<text>.+?)\s*$")
_SPEAKER = re.compile(r"^(?P<name>[^(]+?)\s*(?:\((?P<emotion>[^)]*)\))?\s*$")


def to_draft(script: EpisodeScript, cfg: SynthConfig) -> tuple[EpisodeDraft, list[str]]:
    """The writer's plain lines -> structured lines. Lines that don't follow the format are
    reported (and left out of the draft)."""
    kinds = {lab.strip().upper(): kind for kind, labs in cfg.line_labels.items() for lab in labs}
    probs, scenes = [], []
    for si, sc in enumerate(script.scenes, start=1):
        lines = []
        for li, raw in enumerate(sc.lines, start=1):
            m = _LINE.match(raw or "")
            sp = _SPEAKER.match(m.group("label")) if m else None
            if not m or not sp:
                probs.append(f"scene {si} line {li} doesn't follow the line format (LABEL: text): '{raw}'")
                continue
            label, plant = m.group("label").strip(), (m.group("plant") or "").upper() or None
            kind = kinds.get(re.sub(r"\s+", " ", label).upper())
            if kind:
                lines.append(DraftLine(kind=kind, text=m.group("text"), plant=plant))
            else:
                lines.append(DraftLine(kind="dialogue", speaker=sp.group("name").strip(),
                                       emotion=(sp.group("emotion") or "").strip(), text=m.group("text"), plant=plant))
        scenes.append(DraftScene.model_construct(heading=sc.heading, lines=lines))
    return EpisodeDraft.model_construct(logline=script.logline, continuity_notes=script.continuity_notes,
                                        summary=script.summary, scenes=scenes), probs


# ----------------------------------------------------------------------------- checks
def episode_check(number: int, plants: list[dict], cfg: SynthConfig, settings: Settings, pcfg: ParserConfig):
    pc = settings.promo
    lo, hi = plant_word_range(cfg, settings)

    def check(d: EpisodeDraft | EpisodeScript) -> list[str]:
        probs = []
        if isinstance(d, EpisodeScript):
            d, probs = to_draft(d, cfg)
        if any(not sc.lines for sc in d.scenes):
            return probs + ["every scene needs at least one line in the format LABEL: text"]
        n_lines = sum(len(s.lines) for s in d.scenes)
        if not cfg.lines_per_episode[0] <= n_lines <= cfg.lines_per_episode[1]:
            probs.append(f"the episode has {n_lines} lines; it needs {cfg.lines_per_episode[0]}-{cfg.lines_per_episode[1]}")
        if not cfg.scenes_per_episode[0] <= len(d.scenes) <= cfg.scenes_per_episode[1]:
            probs.append(f"the episode has {len(d.scenes)} scenes; it needs "
                         f"{cfg.scenes_per_episode[0]}-{cfg.scenes_per_episode[1]}")
        for si, sc in enumerate(d.scenes, start=1):
            for li, ln in enumerate(sc.lines, start=1):
                if not _line_text(ln):
                    probs.append(f"scene {si} line {li}: empty text")
                if ln.kind == "dialogue" and not re.sub(r"[^\w]", "", ln.speaker or ""):
                    probs.append(f"scene {si} line {li}: dialogue needs a speaker")
        wanted = {p["id"] for p in plants}
        marked = {ln.plant.strip() for s in d.scenes for ln in s.lines if ln.plant}
        for pid in sorted(marked - wanted):
            probs.append(f"{pid} is not a planted moment of this episode (plants: {', '.join(sorted(wanted))}); "
                         f"remove the [{pid}] tag from those lines")
        tl = build_timeline(d, cfg.timing)
        spans = plant_spans(d, tl)
        for pid in sorted(wanted):
            if pid not in marked:
                probs.append(f"{pid} is missing: write it, and start every line of it with \"[{pid}] \"")
                continue
            if pid not in spans:
                probs.append(f"{pid}: its lines must be consecutive and inside one scene (no line without [{pid}] "
                             f"between them)")
                continue
            si, first, last, a, b = spans[pid]
            n_words = sum(len(_line_text(ln).split()) for ln in d.scenes[si].lines[first:last + 1])
            if d.scenes[si].lines[first].kind not in ("dialogue", "action"):
                probs.append(f"{pid}: its first line must be a dialogue or action line (the hook), not a sound cue")
            dur = (b - a) / 100
            if dur < pc.min_duration_s:
                probs.append(f"{pid} is {dur:.1f}s when spoken, {n_words} words (needs {pc.min_duration_s:g}-"
                             f"{pc.max_duration_s:g}s, about {lo}-{hi} words): add lines or longer lines to it")
            elif dur > pc.max_duration_s:
                probs.append(f"{pid} is {dur:.1f}s when spoken, {n_words} words (needs {pc.min_duration_s:g}-"
                             f"{pc.max_duration_s:g}s, about {lo}-{hi} words): cut lines from it")
        if not probs:                       # the file it becomes must parse cleanly
            ep = parse_text(render_episode(number, d, cfg), pcfg, cfg.render.file_name.format(number=number))
            if ep.unparsed or ep.warnings or ep.coverage.percent < 100:
                bad = [u.text for u in ep.unparsed][:3] + ep.warnings[:3]
                probs.append(f"some lines don't fit the screenplay format, rewrite them plainly: {bad}")
        return probs
    return check


# ------------------------------------------------------------------------------ prompt bits
def _plants_text(plants: list[dict], mtypes: MomentTypes) -> str:
    return "\n".join(f"  - {p['id']}: {p['moment_type']} ({mtypes.moment_types[p['moment_type']]}), "
                     f"{p['position']} in the episode" for p in plants)


def _characters(bible: ShowBible) -> str:
    return "\n".join(f"  - {c.name} ({c.role}): {c.description}" for c in bible.characters)


# ------------------------------------------------------------------------------ generate
def show_folder(settings: Settings, cfg: SynthConfig, show: str) -> Path:
    return settings.resolve(cfg.output_dir) / slug(show)


def generate(client: LLMClient, settings: Settings, cfg: SynthConfig, pcfg: ParserConfig, mtypes: MomentTypes,
             show: str, genre: str, n_episodes: int, seed: int, say: Callable[..., None] = print) -> dict:
    folder = show_folder(settings, cfg, show)
    drafts = folder / ".drafts"
    key_path = labels_mod.default_path(settings, show)
    if key_path.exists():
        key = labels_mod.load_labels(key_path)
        if key.label_source != "planted":
            raise SystemExit(f"{key_path} holds {key.label_source} labels for a show called '{show}', not a planted "
                             f"key; pick another show title so they aren't overwritten.")
        if key.locked_at:
            raise SystemExit(f"'{show}' was already generated ({folder}); answer key {key_path}. "
                             f"Use --overwrite to make it again.")
    drafts.mkdir(parents=True, exist_ok=True)
    pc = settings.promo

    # the plan (saved first, so a resumed run plants exactly the same moments)
    plan_file = drafts / "plan.json"
    if plan_file.exists():
        saved = json.loads(plan_file.read_text(encoding="utf-8"))
        plan = {int(k): v for k, v in saved["plan"].items()}
        if len(plan) != n_episodes or saved.get("seed") != seed:
            raise SystemExit(f"{folder} holds an unfinished show with {len(plan)} episodes / seed {saved.get('seed')}; "
                             f"re-run with those, or use --overwrite to start again.")
    else:
        plan = plan_plants(cfg, mtypes, n_episodes, seed)
        plan_file.write_text(json.dumps({"seed": seed, "plan": plan}, indent=1), encoding="utf-8")

    bible_file = drafts / "bible.json"
    writers: list[str] = json.loads((drafts / "writers.json").read_text(encoding="utf-8")) \
        if (drafts / "writers.json").exists() else []

    def note_writer(meta):
        m = f"{meta.provider}:{meta.model}"
        if m not in writers:
            writers.append(m)
            (drafts / "writers.json").write_text(json.dumps(writers), encoding="utf-8")

    if bible_file.exists():
        bible = ShowBible.model_validate_json(bible_file.read_text(encoding="utf-8"))
    else:
        def bible_check(b: ShowBible) -> list[str]:
            probs = [] if len(b.episode_beats) == n_episodes else \
                [f"episode_beats has {len(b.episode_beats)} lines; it needs exactly {n_episodes}"]
            names = [c.name.strip().upper() for c in b.characters]
            if len(set(names)) != len(names):
                probs.append("character names must be distinct")
            return probs
        r = client.structured(AGENT, load_chat_prompt(client.prompts_dir, "synthetic_bible"),
                              {"show": show, "genre": genre, "episodes": n_episodes}, ShowBible, [bible_check],
                              trace_id=f"{client.run_id}:bible", label="bible")
        bible = r.output
        note_writer(r.meta)
        bible_file.write_text(bible.model_dump_json(indent=1), encoding="utf-8")
        say(f"bible: {len(bible.characters)} characters, {n_episodes} episode beats")

    ep_prompt = load_chat_prompt(client.prompts_dir, "synthetic_episode")
    fixed = {"show": show, "genre": genre, "premise": bible.premise, "characters": _characters(bible),
             "episodes": n_episodes, "scenes_min": cfg.scenes_per_episode[0], "scenes_max": cfg.scenes_per_episode[1],
             "lines_min": cfg.lines_per_episode[0], "lines_max": cfg.lines_per_episode[1],
             "plant_min": cfg.plant_lines[0], "plant_max": cfg.plant_lines[1], "wps": cfg.timing.words_per_second,
             "plant_words_min": plant_word_range(cfg, settings)[0], "plant_words_max": plant_word_range(cfg, settings)[1],
             "min_s": f"{pc.min_duration_s:g}", "max_s": f"{pc.max_duration_s:g}",
             "hook_window_s": f"{pc.hook_window_s:g}",
             "moment_types": "\n".join(f"  - {k}: {v}" for k, v in mtypes.moment_types.items())}
    eps: dict[int, EpisodeDraft] = {}
    for n in range(1, n_episodes + 1):
        f = drafts / f"ep-{n:03d}.json"
        if f.exists():
            eps[n] = EpisodeDraft.model_validate_json(f.read_text(encoding="utf-8"))
            continue
        story = [story_line(k, eps[k]) for k in sorted(eps)][-cfg.story_summaries:] if cfg.story_summaries else []
        r = client.structured(AGENT, ep_prompt, {**fixed, "episode": n, "beat": bible.episode_beats[n - 1],
                                                 "plants": _plants_text(plan[n], mtypes),
                                                 "story": "\n".join(story) or "(this is the first episode)"},
                              EpisodeScript, [episode_check(n, plan[n], cfg, settings, pcfg)],
                              trace_id=f"{client.run_id}:ep{n}", label=f"episode {n}")
        eps[n] = EpisodeDraft.model_validate(to_draft(r.output, cfg)[0].model_dump())
        note_writer(r.meta)
        f.write_text(eps[n].model_dump_json(indent=1), encoding="utf-8")
        say(f"episode {n}: {sum(len(s.lines) for s in eps[n].scenes)} lines, plants "
            f"{', '.join(p['id'] + ' ' + p['moment_type'] for p in plan[n])} [{r.meta.attempts} attempt(s)]")

    # write the screenplay files and the answer key (only now: everything passed its checks)
    for old in folder.glob("*.txt"):
        old.unlink()
    key = labels_mod.Labels(show=show, label_source="planted", input_dir=str(folder), labeler_models=list(writers),
                            labeled_by=f"planted by evaluate.synthetic_show (writer {', '.join(writers) or '?'}; "
                                       f"seed {seed}): exact by construction")
    for n, d in eps.items():
        name = cfg.render.file_name.format(number=n)
        text = render_episode(n, d, cfg)
        ep: Episode = parse_text(text, pcfg, name)
        assert ep.coverage.percent == 100 and not ep.warnings, f"episode {n} does not round-trip: {ep.warnings}"
        (folder / name).write_text(text, encoding=pcfg.encoding)
        spans = plant_spans(d, build_timeline(d, cfg.timing))
        for p in plan[n]:
            si, first, last, a, b = spans[p["id"]]
            lines = d.scenes[si].lines
            key.moments.append(labels_mod.LabelMoment(
                id=f"E{n:02d}{p['id']}", episode=n, start=ts(a), end=ts(b), moment_type=p["moment_type"],
                note=f"{_line_text(lines[first])[:70]} ... {_line_text(lines[last])[:70]}",
                why=f"planted: {p['moment_type']}, {p['position']}"))
    key.episodes = sorted(eps)
    labels_mod.save_labels(key, key_path)
    labels_mod.lock(key_path)
    return {"folder": folder, "labels_path": key_path, "plan": plan, "writers": writers,
            "moments": len(key.moments)}


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show", required=True, help="title of the made-up show (also its answer key's name)")
    ap.add_argument("--genre", required=True)
    ap.add_argument("--episodes", type=int, help="default: config/synthetic.yaml episodes")
    ap.add_argument("--seed", type=int, default=7, help="decides which moments are planted where")
    ap.add_argument("--overwrite", action="store_true", help="delete this show's generated files and start again")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    cfg = load_config(settings)
    n = args.episodes or cfg.episodes
    folder = show_folder(settings, cfg, args.show)
    key_path = labels_mod.default_path(settings, args.show)
    if args.overwrite:
        if key_path.exists() and labels_mod.load_labels(key_path).label_source != "planted":
            raise SystemExit(f"{key_path} is not a planted key; refusing to delete it.")
        shutil.rmtree(folder / ".drafts", ignore_errors=True)
        for f in folder.glob("*.txt"):
            f.unlink()
        key_path.unlink(missing_ok=True)
    client = LLMClient(settings, run_id=f"synthetic-{slug(args.show)}-{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"Writing '{args.show}' ({n} episodes, seed {args.seed}) with {client.spec(AGENT).provider}:"
          f"{client.spec(AGENT).model} -> {folder}")
    out = generate(client, settings, cfg, load_parser_config(settings=settings), load_moment_types(settings=settings),
                   args.show, args.genre, n, args.seed)
    print(f"\n{n} episodes written to {out['folder']}\n{out['moments']} planted moments: answer key {out['labels_path']}")
    print(f"Next: run the tool on it, then compare:\n"
          f"  python run.py --input \"{out['folder']}\" --show \"{args.show}\" --genre \"{args.genre}\" --no-review\n"
          f"  python -m evaluate.compare --run <that run id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
