"""AI labeller: an independent answer key made by a separate LLM (models.labeler).

    python -m evaluate.ai_labels --input <folder> --show <name> --episodes 2-9 [--genre <g>] [--labels <file>]

Blind by construction: for each episode it gets the screenplay (the same view a human
labeller gets in evaluate.labels) plus earlier episodes' loglines, never a run's moments,
promos or scores. Its own prompt (prompts/labeler.yaml) is written from the brief, not from
the tool's prompts. Every answer is checked like a hand label (start = a real line start,
end = a real line end) plus the promo length and no near-duplicates; problems go back to the
model to fix. Writes a normal labels file, by default <evaluation.dir>/labels/<show>-ai.yaml
so it never mixes with hand labels, records the models that answered, and locks it once
every requested episode is done. Re-run to continue after a quota stop.
Check it against your own labels with evaluate.agreement before trusting it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from agents.llm_agents import render_lines
from agents.schemas import LabelerOutput
from config.loader import MomentTypes, ParserConfig, Settings, load_moment_types, load_parser_config, load_settings
from evaluate import labels as labels_mod
from evaluate.common import eval_dir, overlap, slug
from llm.checks import ScreenplayIndex
from llm.client import LLMClient, RateLimitExhausted, StructuredOutputError
from llm.prompts import load_chat_prompt, load_prompt_file
from parser.models import Episode
from parser.screenplay_parser import parse_show

PROMPT = "labeler"
AGENT = "labeler"
EXIT_STOPPED = 3                      # same as run.py: stopped early (quota), re-run the same command to continue


def default_path(settings: Settings, show: str) -> Path:
    return eval_dir(settings) / "labels" / f"{slug(show)}-ai.yaml"


def make_client(settings: Settings, run_id: str) -> LLMClient:
    return LLMClient(settings, run_id=run_id)


def _story(episodes: dict[int, Episode], n: int, keep: int) -> str:
    earlier = [f"Episode {k}: {episodes[k].logline}" for k in sorted(episodes) if k < n and episodes[k].logline]
    return "\n".join(earlier[-keep:] if keep else []) or "(this is the first episode given)"


def _checks(index: ScreenplayIndex, ep: Episode, settings: Settings, kinds: list[str]):
    pc, cfg, thr = settings.promo, settings.evaluation.ai_labeler, settings.evaluation.match_min_overlap

    def check(out: LabelerOutput) -> list[str]:
        probs = []
        if len(out.moments) > cfg.max_per_episode:
            probs.append(f"list at most {cfg.max_per_episode} stretches (got {len(out.moments)}): keep the best")
        good = []
        for i, m in enumerate(out.moments, start=1):
            bad = labels_mod._check(index, ep, m.start, m.end, kinds)
            if bad:
                probs += [f"stretch {i}: {b}" for b in bad]
                continue
            dur = index.seconds(m.end) - index.seconds(m.start)
            if not pc.min_duration_s <= dur <= pc.max_duration_s:
                probs.append(f"stretch {i} ({m.start}-{m.end}): {dur:.1f}s is outside "
                             f"{pc.min_duration_s:g}-{pc.max_duration_s:g}s")
                continue
            good.append((i, m, [(ep.number, index.seconds(m.start), index.seconds(m.end))]))
        for a in range(len(good)):
            for b in range(a + 1, len(good)):
                (i, mi, si), (j, mj, sj) = good[a], good[b]
                if overlap(si, sj) >= thr:
                    probs.append(f"stretches {i} and {j} overlap too much ({mi.start}-{mi.end} vs {mj.start}-{mj.end}): "
                                 f"keep only the better one")
        return probs
    return check


def label_episodes(client: LLMClient, settings: Settings, pcfg: ParserConfig, mtypes: MomentTypes,
                   episodes: dict[int, Episode], todo: list[int], lab: labels_mod.Labels, show: str,
                   genre: str = "", say: Callable[..., None] = print,
                   save: Callable[[labels_mod.Labels], None] = lambda l: None) -> labels_mod.Labels:
    """Label each episode in `todo` (one LLM call each); an episode is marked finished only
    when the model's answer passed every check. Stops early (saving) on a quota stop."""
    kinds, pc = settings.agents.screenplay_kinds, settings.promo
    prompt = load_chat_prompt(client.prompts_dir, PROMPT)
    version = load_prompt_file(str(client.prompts_dir), PROMPT).version
    fixed = {"show": show, "genre": genre or "(not given)", "min_s": f"{pc.min_duration_s:g}",
             "max_s": f"{pc.max_duration_s:g}", "hook_window_s": f"{pc.hook_window_s:g}",
             "cliffhanger_window_s": f"{pc.cliffhanger_window_s:g}",
             "max_moments": settings.evaluation.ai_labeler.max_per_episode,
             "moment_types": "\n".join(f"    - {k}: {v}" for k, v in mtypes.moment_types.items())}
    for n in todo:
        ep = episodes.get(n)
        if ep is None:
            say(f"(episode {n} not found, skipped)")
            continue
        index = ScreenplayIndex([ep], pcfg, pc.boundary_tolerance_s)
        variables = {**fixed, "episode": n, "logline": ep.logline or "(none)",
                     "story": _story(episodes, n, settings.evaluation.ai_labeler.story_loglines),
                     "lines": render_lines([it for it in ep.all_items() if it.kind in kinds])}
        try:
            r = client.structured(AGENT, prompt, variables, LabelerOutput, [_checks(index, ep, settings, kinds)],
                                  trace_id=f"{client.run_id}:label:ep{n}", label=f"label ep{n}")
        except StructuredOutputError as e:
            say(f"episode {n}: no valid answer ({e}); left unfinished, re-run to retry")
            continue
        except RateLimitExhausted as e:
            say(f"stopped at episode {n}: {e}")
            save(lab)
            return lab
        model = f"{r.meta.provider}:{r.meta.model}"
        if model not in lab.labeler_models:
            lab.labeler_models.append(model)
        lab.moments = [m for m in lab.moments if m.episode != n]           # an episode is labelled once
        for m in r.output.moments:
            mtype = m.moment_type.lower() if m.moment_type.lower() in mtypes.moment_types else mtypes.default_type
            lab.moments.append(labels_mod.LabelMoment(id=f"A{len(lab.moments) + 1:02d}", episode=n, start=m.start,
                                                      end=m.end, moment_type=mtype, note=m.note, why=m.why))
        lab.episodes = sorted(set(lab.episodes) | {n})
        lab.label_source, lab.prompt_version = "ai", version
        lab.labeled_by = (f"AI labeller ({', '.join(lab.labeler_models)}; prompts/{PROMPT}.yaml v{version}). "
                          f"Blind to the tool's picks, but NOT a human editor's judgement")
        say(f"episode {n}: {len(r.output.moments)} stretch(es) [{model}, {r.meta.attempts} attempt(s)]")
        save(lab)
    return lab


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    from run import parse_episodes
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--show", required=True)
    ap.add_argument("--genre", default="")
    ap.add_argument("--episodes", required=True, help="e.g. 2-9")
    ap.add_argument("--labels", help="labels file (default: <evaluation.dir>/labels/<show>-ai.yaml)")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    path = Path(args.labels) if args.labels else default_path(settings, args.show)
    lab = labels_mod.load_labels(path) if path.exists() else \
        labels_mod.Labels(show=args.show, label_source="ai", input_dir=str(Path(args.input).resolve()))
    if lab.label_source != "ai":
        raise SystemExit(f"{path} is not an AI labels file (label_source: {lab.label_source}); "
                         f"AI labels never go into hand-made or planted labels. Use another --labels file.")
    if lab.locked_at:
        raise SystemExit(f"{path} was locked at {lab.locked_at}: labels stay fixed once finished. "
                         f"Use a new --labels file to label again.")
    pcfg = load_parser_config(settings=settings)
    show = parse_show(Path(args.input), pcfg)
    episodes = {e.number: e for e in show.episodes}
    wanted = parse_episodes(args.episodes)
    todo = [n for n in wanted if n not in lab.episodes]
    client = make_client(settings, run_id=f"labels-{slug(args.show)}")
    print(f"AI-labelling {len(todo)} episode(s) of {args.show} with {client.spec(AGENT).provider}:"
          f"{client.spec(AGENT).model} -> {path}")
    label_episodes(client, settings, pcfg, load_moment_types(settings=settings), episodes, todo, lab,
                   show=args.show, genre=args.genre, save=lambda l: labels_mod.save_labels(l, path))
    labels_mod.save_labels(lab, path)
    missing = [n for n in wanted if n in episodes and n not in lab.episodes]
    if missing:
        print(f"Not finished: episodes {missing}. Re-run the same command to continue.")
    else:
        labels_mod.lock(path)
        print("All requested episodes done: labels locked.")
    print(f"{len(lab.moments)} stretch(es) over {len(lab.episodes)} episode(s) saved to {path}")
    return EXIT_STOPPED if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
