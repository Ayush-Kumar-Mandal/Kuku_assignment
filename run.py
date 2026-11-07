"""Command-line entry point: runs the whole pipeline without the UI and writes all outputs.

    python run.py --input <folder> --show <name> --genre <genre> --config config/settings.yaml
                  [--no-memory] [--no-review] [--no-joiner] [--episodes 1-5,8] [--run-id ID]
                  [--agents llm|stub] [--llm-parser-fallback]
    python run.py --resume <run_id> [--decisions <file.json>]

A run is checkpointed after every step (SQLite). If it stops (daily quota with no backup
model left, network, Ctrl+C) or pauses for editor review, `--resume <run_id>` continues
exactly where it left off. Review from the command line: edit the review_template.json
written at the pause (approve / reject / fix + comment) and resume with --decisions.

Only one process may run a given run id at a time: a second start/resume of a run that is
already going is refused (exit 4) instead of doubling the API load.

Exit codes: 0 done, 2 waiting for review, 3 stopped (resumable), 4 already running elsewhere, 1 error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

from langgraph.types import Command

from config.loader import Settings, apply_model_overrides, load_settings
from graph.build import build_graph
from graph.export import export_run
from graph.runner import (RunLocked, build_context, initial_state, new_run_id, open_checkpointer, run, run_lock,
                          status)
from llm.client import LLMConfigError, RateLimitExhausted

EXIT_DONE, EXIT_ERROR, EXIT_REVIEW, EXIT_STOPPED, EXIT_LOCKED = 0, 1, 2, 3, 4
log = logging.getLogger("run")


def parse_episodes(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def parse_models(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model {item}: use AGENT=PROVIDER:MODEL")
        agent, value = item.split("=", 1)
        out[agent.strip()] = value.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Find the best 20-60s promo clips in a folder of screenplays.")
    ap.add_argument("--input", help="folder with one screenplay file per episode")
    ap.add_argument("--show", help="show name (used in prompts and reports)")
    ap.add_argument("--genre", help="genre, e.g. 'romance drama'")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--no-memory", action="store_true", help="don't read/write editor memory")
    ap.add_argument("--no-review", action="store_true", help="skip the editor review pause")
    ap.add_argument("--no-joiner", action="store_true", help="single-episode promos only")
    ap.add_argument("--episodes", help="only these episodes, e.g. 1-5,8 (cheap trial runs)")
    ap.add_argument("--run-id", help="id for a new run (default: <show>-<timestamp>)")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue a stopped or paused run")
    ap.add_argument("--decisions", help="JSON {promo_id: {decision: approve|reject|fix|keep_before, comment}} for a paused run")
    ap.add_argument("--review", action="store_true",
                    help="review in this terminal when the run pauses (approve / reject / fix + comment, before/after)")
    ap.add_argument("--agents", choices=["llm", "stub"], default=None,
                    help="llm (default) or stub = deterministic offline agents (dry run, no API calls); "
                         "a resumed run keeps the agents it started with")
    ap.add_argument("--model", action="append", default=[], metavar="AGENT=PROVIDER:MODEL",
                    help="main model for one agent, e.g. reader=groq:openai/gpt-oss-120b (repeatable); "
                         "its configured backups stay behind it")
    ap.add_argument("--llm-parser-fallback", action="store_true", help="send unparsable lines to the LLM parser")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _setup_logging(verbose: bool):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("run").setLevel(logging.INFO)
    logging.getLogger("llm").setLevel(logging.INFO if verbose else logging.WARNING)
    for noisy in ("httpx", "google_genai", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # langchain-google-genai warns on every call that Gemini 3.x ignores temperature.
    warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _setup_logging(args.verbose)
    settings = settings or load_settings(args.config)
    try:
        cli_models = parse_models(args.model)
        apply_model_overrides(settings, cli_models)          # validate early
    except ValueError as e:
        ap.error(str(e))
    if not args.resume:
        missing = [f"--{k}" for k in ("input", "show", "genre") if not getattr(args, k)]
        if missing:
            ap.error(f"missing {', '.join(missing)} (or use --resume RUN_ID)")
        if not Path(args.input).is_dir():
            ap.error(f"--input {args.input} is not a folder")
    run_id = args.resume or args.run_id or new_run_id(args.show)
    started = time.time()

    try:
        with run_lock(settings, run_id):        # one process per run, held until outputs are written
            return _main_locked(ap, args, settings, run_id, cli_models, started)
    except RunLocked as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_LOCKED


def _main_locked(ap, args, settings: Settings, run_id: str, cli_models: dict[str, str], started: float) -> int:
    with open_checkpointer(settings) as cp:
        graph = build_graph(cp)
        if args.resume:
            st = status(graph, run_id, settings)
            if st["state"] == "not_started":
                ap.error(f"no run '{run_id}' in {settings.paths.checkpoint_db}")
            show, genre = st["values"]["show"], st["values"]["genre"]
            started_with = st["values"].get("agents_kind", "llm")
            if args.agents and args.agents != started_with:
                ap.error(f"run '{run_id}' was started with --agents {started_with}; it can't resume with {args.agents}")
            agents = started_with
            models = {**(st["values"].get("model_overrides") or {}), **cli_models}
            if st["state"] == "awaiting_review" and args.review and not args.decisions:
                inputs = "interactive"                  # review in the terminal, below
            elif st["state"] == "awaiting_review":
                if not args.decisions:
                    print(f"Run {run_id} is waiting for review: add --review to review here, or pass "
                          f"--decisions <file> (template: output/runs/{run_id}/review_template.json).")
                    return EXIT_REVIEW
                decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
                inputs = Command(resume={pid: {k: d.get(k) for k in ("decision", "comment")}
                                         for pid, d in decisions.items()})
            elif st["state"] == "done":
                inputs = "done"
            else:
                inputs = None                           # stopped mid-way: continue from the checkpoint
        else:
            show, genre = args.show, args.genre
            agents = args.agents or "llm"
            models = cli_models
            if status(graph, run_id, settings)["state"] != "not_started":
                ap.error(f"run '{run_id}' already exists; use --resume {run_id}")
            inputs = initial_state(run_id, show, genre, Path(args.input).resolve(), settings,
                                   memory=False if args.no_memory else None,
                                   joiner=False if args.no_joiner else None,
                                   editor_review=False if args.no_review else None,
                                   episodes=parse_episodes(args.episodes), agents=agents, model_overrides=models)

        log.info("run %s: %s (%s)%s", run_id, show, genre, " [resume]" if args.resume else "")
        stop_reason = None
        if inputs != "done":
            ctx = build_context(apply_model_overrides(settings, models), run_id, show, genre, agents=agents,
                                llm_parser_fallback=args.llm_parser_fallback)
            try:
                if inputs != "interactive":
                    run(graph, ctx, run_id, inputs)
                if args.review:
                    review_in_terminal(graph, ctx, run_id, settings)
            except (RateLimitExhausted, LLMConfigError) as e:
                stop_reason = f"{type(e).__name__}: {e}"
            except KeyboardInterrupt:
                stop_reason = "interrupted (Ctrl+C)"
        st = status(graph, run_id, settings)
        files = export_run(st["values"], settings, st["state"], interrupt=st["interrupt"],
                           stop_reason=stop_reason, started=started)

    _print_summary(run_id, st, files, stop_reason, time.time() - started)
    return {"done": EXIT_DONE, "awaiting_review": EXIT_REVIEW}.get(st["state"], EXIT_STOPPED)


# ------------------------------------------------------------------ terminal review
KEYS = {"a": "approve", "r": "reject", "f": "fix", "k": "keep_before"}


def _card(it: dict, n: int, total: int) -> str:
    from graph.export import segs_text
    p, s = it["promo"], it["scores"]
    lines = [f"\n{'=' * 78}",
             f"[{n}/{total}] #{it['rank']} {it['promo_id']} | {segs_text(p['segments'])} | {p.get('total_duration_seconds')}s"
             f" | {p.get('moment_type')} | final {s.get('final')} (hook {s.get('hook')} / cliffhanger "
             f"{s.get('cliffhanger')} / standalone {s.get('standalone')})",
             f"{'=' * 78}",
             f"hook: {p.get('hook')}\ncliffhanger: {p.get('cliffhanger')}\nwhy it works: {p.get('why_it_works')}"]
    for k, v in (it.get("reasons") or {}).items():
        lines.append(f"  evaluator {k}: {v}")
    ba = it.get("before_after")
    if ba:
        be, ae = ba.get("before_eval") or {}, ba.get("after_eval") or {}
        lines += [f"\nYour fix: {ba.get('editor_comment')}",
                  f"FeedbackAgent read it as: {ba.get('fix_interpretation') or '(not available: your words were sent as is)'}",
                  f"Maker's change: {(ba.get('after') or {}).get('revision_note')}",
                  f"\n--- BEFORE (v{(ba.get('before') or {}).get('version')}, final {be.get('final')}) ---",
                  ba.get("before_text") or "(no transcript)",
                  f"\n--- AFTER (v{(ba.get('after') or {}).get('version')}, final {ae.get('final')}) ---",
                  ba.get("after_text") or "(no transcript)"]
    else:
        lines += ["", it.get("transcript") or "(no transcript)"]
    if it.get("note"):
        lines.append(f"\nNOTE: {it['note']}")
    lines.append(f"\nfix attempts used: {it['fix_attempts']}/{it['max_fix_attempts']}")
    return "\n".join(lines)


def terminal_review(st: dict, ask=None, say=print) -> dict | None:
    """One review round in the terminal. Returns decisions, or None if the editor quits (the
    run stays paused and can be reviewed later, here or in the Streamlit Review tab)."""
    from app.review import review_items, validate
    ask = ask or input                       # looked up at call time (tests patch builtins.input)
    payload = st["interrupt"]
    items = review_items(payload, st["values"])
    todo = [i for i in items if i["pending"]]
    while True:
        say(f"\n##### Round {payload['round']} of max {payload.get('max_rounds')}: {len(todo)} promo(s) to review "
            f"({len(items) - len(todo)} already decided) #####")
        decisions = {}
        for n, it in enumerate(todo, start=1):
            say(_card(it, n, len(todo)))
            allowed = [k for k, v in KEYS.items() if v in it["options"]]
            labels = ", ".join(f"{k}={KEYS[k]}" for k in allowed)
            while True:
                a = ask(f"decision [{labels}, s=skip, q=quit] > ").strip().lower()
                if a in ("q", "s") or a in allowed:
                    break
                say(f"'{a}' is not an option here ({labels})")
            if a == "q":
                return None
            if a == "s":
                continue
            dec, comment = KEYS[a], ""
            if dec == "fix":
                while not comment:
                    comment = ask("what should change? > ").strip()
            elif dec == "reject":
                comment = ask("why? (optional, helps the memory learn) > ").strip()
            decisions[it["promo_id"]] = {"decision": dec, "comment": comment}
        errors = validate(items, decisions)
        for e in errors:
            say(f"problem: {e}")
        say("\nDecisions: " + (", ".join(f"{k}={v['decision']}" for k, v in decisions.items()) or "(none)"))
        if not errors and ask("send these decisions? [y/n] > ").strip().lower() == "y":
            return decisions
        say("Starting this round again.")


def review_in_terminal(graph, ctx, run_id: str, settings: Settings, ask=None, say=print) -> None:
    """Keep reviewing while the run pauses for the editor (fixes come back as before/after)."""
    while True:
        st = status(graph, run_id, settings)
        if st["state"] != "awaiting_review":
            return
        decisions = terminal_review(st, ask, say)
        if decisions is None:
            say("Review paused. Continue later with: python run.py --resume " + run_id + " --review")
            return
        say("Applying decisions (fixes are re-made and re-evaluated)...")
        run(graph, ctx, run_id, Command(resume=decisions))


def _print_summary(run_id: str, st: dict, files: dict, stop_reason: str | None, elapsed: float):
    v = st["values"]
    promos = v.get("promos") or []
    print(f"\nRun {run_id}: {st['state']} ({elapsed:.0f}s this invocation)")
    print(f"  episodes {len(v.get('episodes') or [])} | moments {len(v.get('moments') or [])} | "
          f"promos made {len(v.get('candidates') or {})} | ranked {len(promos)} | errors {len(v.get('errors') or [])}")
    for p in promos[:10]:
        segs = " + ".join(f"ep{s['episode']} {s['start_timestamp']}-{s['end_timestamp']}" for s in p["segments"])
        print(f"  #{p['rank']:<2} {p['promo_id']:7} {p['scores']['final']:<5} {p['total_duration_seconds']:>5}s  {segs}")
    if len(promos) > 10:
        print(f"  ... {len(promos) - 10} more")
    print(f"  report: {files['report']}")
    if stop_reason:
        print(f"\nSTOPPED: {stop_reason[:500]}\nResume with: python run.py --resume {run_id}")
    elif st["state"] == "awaiting_review":
        print(f"\nWaiting for editor review. Review here:  python run.py --resume {run_id} --review\n"
              f"or in the app's Review tab, or edit {files['review_template']} and run:\n"
              f"  python run.py --resume {run_id} --decisions {files['review_template']}")


if __name__ == "__main__":
    raise SystemExit(main())
