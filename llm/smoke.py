"""Live end-to-end check of the LLM client against real providers.

    python -m llm.smoke --input <folder> --episode 1 --agents reader evaluator default

For each agent it asks its configured model to pick a hook line from the first part of an
episode, with real custom checks (timestamps must exist and be a line's own start/end),
then prints mode (native/fallback), retries and tokens.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from config.loader import load_parser_config, load_settings
from llm.checks import ScreenplayIndex, check_timestamp
from llm.client import LLMClient, LLMConfigError, RateLimitExhausted, StructuredOutputError
from llm.prompts import load_chat_prompt
from parser.screenplay_parser import parse_show


class HookPick(BaseModel):
    start_timestamp: str = Field(description="start of the chosen line, exactly as written")
    end_timestamp: str = Field(description="end of the chosen line, exactly as written")
    quote: str = Field(description="the line's text")
    why: str = Field(description="one sentence: why it stops a scroller")
    strength: int = Field(ge=1, le=10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=45, help="excerpt length")
    ap.add_argument("--agents", nargs="+", default=["reader", "evaluator", "default"])
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--mode", choices=["auto", "native", "fallback"], default=None,
                    help="force a structured-output path for every agent tested")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings(args.settings)
    if args.mode:
        for a in args.agents:
            settings.models[a] = settings.model_for(a).model_copy(update={"structured_output": args.mode})
    pcfg = load_parser_config(settings=settings)
    show = parse_show(Path(args.input), pcfg)
    ep = show.episode(args.episode)
    index = ScreenplayIndex([ep], pcfg, settings.promo.boundary_tolerance_s)
    items = [it for it in ep.items_between(0, args.seconds) if it.kind != "camera"]
    excerpt = "\n".join(f"[{it.start_ts}-{it.end_ts}] {it.character or it.kind.upper()}: {it.text}" for it in items)
    starts = {it.start_ts: it for it in items}

    def same_line(out: HookPick) -> list[str]:
        probs = check_timestamp(index, ep.number, out.start_timestamp, "start") + \
            check_timestamp(index, ep.number, out.end_timestamp, "end")
        it = starts.get(out.start_timestamp)
        if not probs and (it is None or it.end_ts != out.end_timestamp):
            probs.append("start and end must be the start and end of ONE line in the excerpt")
        return probs

    client = LLMClient(settings, run_id="smoke")
    prompt = load_chat_prompt(client.prompts_dir, "smoke_test")
    ok = True
    for agent in args.agents:
        try:
            r = client.structured(agent, prompt, {"excerpt": excerpt}, HookPick, checks=[same_line], label="smoke")
            m = r.meta
            print(f"\n[{agent}] {m.provider}:{m.model}  mode={m.mode}{' (switched)' if m.switched_to_fallback else ''}"
                  f"  retries={m.retries}  tokens={m.input_tokens}/{m.output_tokens}{'~' if m.tokens_estimated else ''}"
                  f"  {m.latency_s}s")
            print(f"   {r.output.start_timestamp}-{r.output.end_timestamp} \"{r.output.quote}\" "
                  f"(strength {r.output.strength}) - {r.output.why}")
            if m.errors:
                print("   rejected attempts:", m.errors)
        except (LLMConfigError, RateLimitExhausted, StructuredOutputError) as e:
            ok = False
            print(f"\n[{agent}] FAILED: {e}")
    print("\nusage:", client.usage.to_dict()["totals"])
    print("logs ->", client.log_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
