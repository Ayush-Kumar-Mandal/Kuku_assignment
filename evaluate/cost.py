"""Cost & scale estimate from a real run's token counts (spec section 11).

    python -m evaluate.cost --run <run_id> [--shows-per-week 5] [--episodes-per-show 50]

Uses the run's measured calls and tokens per agent (retries included: they're real cost),
scaled per episode to the show size, then:
  free  : $0, but limited by daily quotas -> how many days one show needs on the free tier,
          and the model that is the bottleneck (limits from settings.cost.free_limits)
  paid  : "as run" (each call priced at the model it actually used) and "single model"
          (everything on one paid model), from settings.cost.paid_prices_per_m
Writes <evaluation.dir>/cost_<run_id>.md|json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from config.loader import Settings, load_settings
from evaluate.common import eval_dir, read_json, run_path


def _per_model(agent_usage: dict) -> dict[str, dict]:
    """Calls/tokens per model for one agent: exact when the log has them, else split by calls."""
    if agent_usage.get("model_tokens"):
        return agent_usage["model_tokens"]
    models = agent_usage.get("models") or {}
    total = sum(models.values()) or 1
    return {m: {"calls": n, "input_tokens": agent_usage["input_tokens"] * n / total,
                "output_tokens": agent_usage["output_tokens"] * n / total} for m, n in models.items()}


def estimate(usage: dict, episodes_in_run: int, settings: Settings, episodes_per_show: int | None = None,
             shows_per_week: int | None = None) -> dict:
    c = settings.cost
    eps = episodes_per_show or c.episodes_per_show or episodes_in_run
    weekly = shows_per_week or c.shows_per_week
    scale = eps / episodes_in_run
    by_model: dict[str, dict] = {}
    by_agent = {}
    for agent, u in usage["by_agent"].items():
        by_agent[agent] = {k: round(u[k] * scale) for k in ("calls", "input_tokens", "output_tokens")}
        for m, x in _per_model(u).items():
            t = by_model.setdefault(m, {"calls": 0, "input_tokens": 0.0, "output_tokens": 0.0})
            for k in t:
                t[k] += x[k] * scale
    show = {k: round(sum(a[k] for a in by_agent.values())) for k in ("calls", "input_tokens", "output_tokens")}

    def price(m: str, inp: float, out: float) -> float | None:
        p = c.paid_prices_per_m.get(m)
        return None if p is None else (inp * p.input + out * p.output) / 1e6

    as_run = [price(m, x["input_tokens"], x["output_tokens"]) for m, x in by_model.items()]
    unpriced = [m for m, v in zip(by_model, as_run) if v is None]
    per_show_as_run = sum(v for v in as_run if v is not None)
    single = {m: {"per_show_usd": round(price(m, show["input_tokens"], show["output_tokens"]), 4),
                  "per_week_usd": round(price(m, show["input_tokens"], show["output_tokens"]) * weekly, 2)}
              for m in c.paid_prices_per_m}

    # Free tier: each agent can use its whole model chain (main + backups, switched on quota),
    # so its daily capacity is the sum of the chain's known daily limits. The as-run model mix
    # is NOT used here: it is itself the result of quota switches. A chain containing a model
    # with unpublished limits can't be bounded, so it is flagged and not counted as a bottleneck.
    free_agents = {}
    for agent, x in by_agent.items():
        chain = [f"{m.provider}:{m.model}" for m in settings.model_for(agent).chain()]
        lims = [c.free_limits.get(m) for m in chain]
        unknown_in_chain = any(l is None or (l.rpd is None and l.tpd is None) for l in lims)
        rpd = sum(l.rpd for l in lims if l and l.rpd)
        tpd = sum(l.tpd for l in lims if l and l.tpd)
        tokens = x["input_tokens"] + x["output_tokens"]
        caps = ([rpd / x["calls"]] if rpd and x["calls"] else []) + ([tpd / tokens] if tpd and tokens else [])
        free_agents[agent] = {"chain": chain, "requests_per_day": rpd or None, "tokens_per_day": tpd or None,
                              "demand_calls": x["calls"], "demand_tokens": tokens,
                              "unknown_limit_in_chain": unknown_in_chain,
                              "shows_per_day": None if unknown_in_chain or not caps else round(min(caps), 2)}
    bounded = {a: f for a, f in free_agents.items() if f["shows_per_day"] is not None}
    bottleneck = min(bounded, key=lambda a: bounded[a]["shows_per_day"]) if bounded else None
    spd = bounded[bottleneck]["shows_per_day"] if bottleneck else None
    days_per_show = max(1, math.ceil(1 / spd)) if spd else 1
    latency = sum(u.get("latency_s", 0) for u in usage["by_agent"].values()) * scale
    return {
        "basis": {"episodes_in_run": episodes_in_run, "episodes_per_show": eps, "shows_per_week": weekly,
                  "prices_checked": c.prices_checked},
        "per_show": show, "per_week": {k: v * weekly for k, v in show.items()}, "by_agent_per_show": by_agent,
        "by_model_per_show": {m: {k: round(v) for k, v in x.items()} for m, x in by_model.items()},
        "paid": {"as_run": {"per_show_usd": round(per_show_as_run, 4), "per_week_usd": round(per_show_as_run * weekly, 2),
                            "unpriced_models": unpriced},
                 "single_model": single},
        "free": {"usd": 0, "by_agent": free_agents, "bottleneck_agent": bottleneck, "shows_per_day": spd,
                 "shows_per_week": round(7 * spd, 1) if spd else None, "days_per_show": days_per_show,
                 "fits_weekly_load": (7 * spd >= weekly) if spd else None},
        "time": {"llm_minutes_per_show": round(latency / 60, 1),
                 "note": "sum of call latencies incl. rate-limit waits; agents run one after another"},
    }


def render(e: dict, run_id: str) -> str:
    b, s, w, f = e["basis"], e["per_show"], e["per_week"], e["free"]
    L = [f"# Cost & scale estimate (from run `{run_id}`)", "",
         f"Measured on {b['episodes_in_run']} episodes, scaled to **{b['episodes_per_show']} episodes per show** and "
         f"**{b['shows_per_week']} shows per week**. Retries are included (they are real calls). "
         f"Prices checked {b['prices_checked']}.", "",
         "| | LLM calls | Input tokens | Output tokens |", "|---|---|---|---|",
         f"| Per show | {s['calls']:,} | {s['input_tokens']:,} | {s['output_tokens']:,} |",
         f"| Per week | {w['calls']:,} | {w['input_tokens']:,} | {w['output_tokens']:,} |", "",
         "## Per agent (per show)", "", "| Agent | Calls | Input | Output |", "|---|---|---|---|"]
    for a, x in e["by_agent_per_show"].items():
        L.append(f"| {a} | {x['calls']:,} | {x['input_tokens']:,} | {x['output_tokens']:,} |")
    fits = {True: "fits", False: "does NOT fit", None: "can't be bounded"}[f["fits_weekly_load"]]
    mins = e["time"]["llm_minutes_per_show"]
    by_time = round(24 * 60 / mins, 1) if mins else None
    binding = "the daily quota" if f["shows_per_day"] and by_time and f["shows_per_day"] < by_time else "run time"
    L += ["", "## Free tier", "",
          f"- Cost: **$0**, limited by daily quotas. Each agent can use its whole model chain (main + backups).",
          f"- Bottleneck: **{f['bottleneck_agent']}**, about **{f['shows_per_day']} shows/day = "
          f"{f['shows_per_week']} shows/week**; the weekly load of {b['shows_per_week']} shows {fits}.",
          f"- Measured LLM time: about **{e['time']['llm_minutes_per_show']} min per show** (rate-limit waits "
          f"included), i.e. up to {by_time} shows/day by time alone, so **{binding}** is the binding limit.", "",
          "| Agent | Model chain | Demand per show (calls / tokens) | Free capacity per day (requests / tokens) | Shows/day |",
          "|---|---|---|---|---|"]
    for a, x in f["by_agent"].items():
        cap = f"{x['requests_per_day'] or '?'} / {x['tokens_per_day'] or '?'}"
        spd = "not bounded (a model's limit is unpublished)" if x["unknown_limit_in_chain"] else x["shows_per_day"]
        L.append(f"| {a} | {' -> '.join(x['chain'])} | {x['demand_calls']} / {x['demand_tokens']:,} | {cap} | {spd} |")
    L += ["", "Models shared by several chains are counted for each of them, so capacity is optimistic "
          "when those agents run on the same day.", "",
          "## Paid tier", "", "| Scenario | Per show | Per week |", "|---|---|---|",
          f"| As run (each call at the model it used) | ${e['paid']['as_run']['per_show_usd']:.2f} | "
          f"${e['paid']['as_run']['per_week_usd']:.2f} |"]
    for m, x in sorted(e["paid"]["single_model"].items(), key=lambda kv: kv[1]["per_show_usd"]):
        L.append(f"| Everything on `{m}` | ${x['per_show_usd']:.2f} | ${x['per_week_usd']:.2f} |")
    if e["paid"]["as_run"]["unpriced_models"]:
        L.append(f"\nNo price configured for: {', '.join(e['paid']['as_run']['unpriced_models'])} (left out).")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a finished run to measure (ideally a full show)")
    ap.add_argument("--shows-per-week", type=int)
    ap.add_argument("--episodes-per-show", type=int)
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    summary = read_json(run_path(settings, args.run) / "run_summary.json")
    if not summary or not (summary.get("llm_usage") or {}).get("by_agent"):
        raise SystemExit(f"run '{args.run}' has no LLM usage (not found, or a stub run)")
    e = estimate(summary["llm_usage"], summary["counts"]["episodes"], settings,
                 episodes_per_show=args.episodes_per_show, shows_per_week=args.shows_per_week)
    d = eval_dir(settings)
    (d / f"cost_{args.run}.json").write_text(json.dumps(e, indent=2), encoding="utf-8")
    md = render(e, args.run)
    (d / f"cost_{args.run}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"written: {d / f'cost_{args.run}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
