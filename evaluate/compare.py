"""Compare the tool with your labels and judgements.

    python -m evaluate.compare --run <run_id> [--labels <file>]

Recall as a funnel, per labelled moment (labelled episodes only):
    reader     a Reader moment (hook or cliffhanger) falls inside your span
    candidates some promo the tool made overlaps your span (>= evaluation.match_min_overlap)
    ranked     a promo in the final ranked list overlaps it
"stage_lost" says where a missed moment fell out: reader -> cutter -> ranking.
Post rate and the failure log come from `evaluate.judge`. Writes
<evaluation.dir>/<run_id>/evaluation.md|json and failures.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from config.loader import Settings, load_parser_config, load_settings
from evaluate import labels as labels_mod
from evaluate.common import eval_dir, overlap, read_json, run_path, spans
from evaluate.judge import judgements_path
from graph.export import segs_text
from parser.screenplay_parser import parse_ts


def run_models(summary: dict) -> set[str]:
    """provider:model ids every agent of a run actually used (run_summary.json llm_usage)."""
    by_agent = ((summary or {}).get("llm_usage") or {}).get("by_agent") or {}
    return {m for a in by_agent.values() for m in (a.get("models") or {})}


def label_source(lab: labels_mod.Labels) -> str:
    """human | ai | planted. Older files only set labeled_by (anything but "human" = not a hand label)."""
    return "ai" if lab.label_source == "human" and lab.labeled_by != "human" else lab.label_source


def evaluate_run(lab: labels_mod.Labels, moments: list[dict], candidates: list[dict], promos: list[dict],
                 judgements: dict, settings: Settings, to_s, run_models: set[str] | None = None) -> dict:
    thr = settings.evaluation.match_min_overlap
    labelled = set(lab.episodes)
    per_label: dict[str, dict] = {}

    def final_segments(c: dict) -> list[dict]:
        return next((v for v in c["versions"] if v["version"] == c["final_version"]), c["versions"][-1])["segments"]

    def best(label_span, pool: list[tuple[str, list]]) -> tuple[str | None, float]:
        scored = [(pid, overlap(label_span, spans(segs, to_s))) for pid, segs in pool]
        scored = [x for x in scored if x[1] >= thr]
        return max(scored, key=lambda x: x[1]) if scored else (None, 0.0)

    cand_pool = [(c["promo_id"], segs) for c in candidates for segs in
                 [v["segments"] for v in c["versions"]]]          # any version counts as "made"
    rank_pool = [(p["promo_id"], p["segments"]) for p in promos]
    for m in (m for m in lab.moments if m.episode in labelled):
        s, e = to_s(m.start), to_s(m.end)
        span = [(m.episode, s, e)]
        reader = [x["moment_id"] for x in moments if x["episode"] == m.episode and s - 1e-6 <= to_s(x["timestamp"]) <= e + 1e-6]
        cand, cand_ov = best(span, cand_pool)
        ranked, rank_ov = best(span, rank_pool)
        lost = None if ranked else "ranking" if cand else "cutter" if reader else "reader"
        per_label[m.id] = {"episode": m.episode, "span": f"{m.start}-{m.end}", "moment_type": m.moment_type,
                           "note": m.note, "reader_moments": reader, "made_by": cand,
                           "made_overlap": round(cand_ov, 2), "ranked_by": ranked,
                           "ranked_overlap": round(rank_ov, 2), "stage_lost": lost}
    n = len(per_label)
    recall = {"labels": n, "episodes_labelled": len(labelled),
              "reader": sum(1 for x in per_label.values() if x["reader_moments"]),
              "candidates": sum(1 for x in per_label.values() if x["made_by"]),
              "ranked": sum(1 for x in per_label.values() if x["ranked_by"])}
    recall.update({f"{k}_rate": round(recall[k] / n, 3) if n else None for k in ("reader", "candidates", "ranked")})
    # How many of the tool's picks in labelled episodes match one of your moments (a precision proxy).
    in_labelled = [p for p in promos if any(s["episode"] in labelled for s in p["segments"])]
    matched = [p for p in in_labelled if any(
        overlap([(m.episode, to_s(m.start), to_s(m.end))], spans(p["segments"], to_s)) >= thr
        for m in lab.moments if m.episode in labelled)]

    judged = {k: v for k, v in judgements.items() if v.get("post") is not None}
    posts = sum(1 for v in judged.values() if v["post"])
    fails = [{"promo_id": k, **v} for k, v in judged.items() if not v["post"]]
    by_cat = Counter(f["category"] for f in fails)
    return {
        "recall": recall, "per_label": per_label,
        "picks_in_labelled_episodes": {"picks": len(in_labelled), "match_a_label": len(matched),
                                       "rate": round(len(matched) / len(in_labelled), 3) if in_labelled else None},
        "post_rate": {"judged": len(judged), "would_post": posts,
                      "rate": round(posts / len(judged), 3) if judged else None},
        "failures": {"by_category": dict(by_cat), "items": fails},
        "labels_locked_at": lab.locked_at, "labeled_by": lab.labeled_by, "match_min_overlap": thr,
        "label_source": label_source(lab),
        # An AI labeller that is one of the run's own models is grading its own work.
        "independence": {"labeler_models": lab.labeler_models,
                         "shared_with_run": sorted(set(lab.labeler_models) & set(run_models or ()))},
    }


def source_notes(r: dict) -> list[tuple[str, str]]:
    """What a reader must know about who made the labels, as (level, markdown) with level
    info | warning. Used by the evaluation.md report and the app's Evaluation page."""
    out: list[tuple[str, str]] = []
    source = r.get("label_source") or ("human" if r.get("labeled_by", "human") == "human" else "ai")
    if source == "planted":
        out.append(("info", f"**Answer key planted by construction** ({r['labeled_by']}): the synthetic show was "
                            f"written around these moments, so recall is exact. Precision is approximate: the writer "
                            f"may also have produced strong moments it wasn't asked for."))
    elif source == "ai":
        out.append(("warning", f"**Labels are NOT hand labels: labeled by {r['labeled_by']}.** Recall below measures "
                               f"agreement with that labeler, not with a human editor."))
        out.append(("info", r["agreement"] if r.get("agreement") else
                    "Not yet checked against a human: run `python -m evaluate.agreement --reference <your labels> "
                    "--candidate <these labels>`."))
    shared = (r.get("independence") or {}).get("shared_with_run") or []
    if shared and source == "planted":
        out.append(("warning", f"**Not independent: a model that wrote the answer key also ran in this run "
                               f"({', '.join(shared)}).** It may find moments in its own writing more easily; read "
                               f"recall as an upper bound."))
    elif shared:
        out.append(("warning", f"**Not independent: the labeller used the same model(s) as this run "
                               f"({', '.join(shared)}).** It is partly grading its own work; point models.labeler at "
                               f"another family and label again."))
    return out


def write_report(r: dict, settings: Settings, run_id: str) -> dict[str, Path]:
    d = eval_dir(settings) / run_id
    d.mkdir(parents=True, exist_ok=True)
    paths = {"json": d / "evaluation.json", "md": d / "evaluation.md", "failures_csv": d / "failures.csv"}
    paths["json"].write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    with paths["failures_csv"].open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["promo_id", "rank", "category", "note", "segments", "judged_at"],
                           extrasaction="ignore")
        w.writeheader()
        for x in r["failures"]["items"]:
            w.writerow({**x, "segments": segs_text(x.get("segments") or [])})
    rc, pr, pk = r["recall"], r["post_rate"], r["picks_in_labelled_episodes"]
    cats = settings.evaluation.failure_categories
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    L = [f"# Evaluation: {run_id}", "",
         f"Labels locked at: {r['labels_locked_at'] or 'NOT LOCKED (run evaluate.judge to lock them)'} · "
         f"match = shared screen time >= {r['match_min_overlap']:.0%} of the shorter clip", ""]
    for _, note in source_notes(r):
        L += [f"> {note}", ""]
    L += ["## Recall (your moments the tool found)", "",
         f"{rc['labels']} labelled moments in {rc['episodes_labelled']} episodes.", "",
         "| Stage | Found | Rate |", "|---|---|---|",
         f"| Reader marked a moment inside it | {rc['reader']} | {pct(rc['reader_rate'])} |",
         f"| A promo was made for it | {rc['candidates']} | {pct(rc['candidates_rate'])} |",
         f"| It made the final ranked list | {rc['ranked']} | {pct(rc['ranked_rate'])} |", "",
         "| Label | Ep | Span | Type | Reader moments | Made by | Ranked by | Lost at | Note |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, x in r["per_label"].items():
        L.append(f"| {k} | {x['episode']} | {x['span']} | {x['moment_type']} | {', '.join(x['reader_moments']) or '-'} | "
                 f"{x['made_by'] or '-'} | {x['ranked_by'] or '-'} | {x['stage_lost'] or 'found'} | {x['note']} |")
    L += ["", "## Picks you'd post", "",
          f"- Judged: {pr['judged']} · would post: {pr['would_post']} · **post rate {pct(pr['rate'])}**",
          f"- Ranked picks in your labelled episodes: {pk['picks']}, of which {pk['match_a_label']} match one of "
          f"your moments ({pct(pk['rate'])})", "", "## Failure log", "", "| Category | Count |", "|---|---|"]
    for key, label in cats.items():
        L.append(f"| {label} | {r['failures']['by_category'].get(key, 0)} |")
    if r["failures"]["items"]:
        L += ["", "| Promo | Rank | Category | Note |", "|---|---|---|---|"]
        for x in r["failures"]["items"]:
            L.append(f"| {x['promo_id']} | {x.get('rank') or '-'} | {cats.get(x['category'], x['category'])} | {x.get('note', '')} |")
    paths["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    d = run_path(settings, args.run)
    summary = read_json(d / "run_summary.json") or {}
    lab_path = Path(args.labels) if args.labels else labels_mod.default_path(settings, summary.get("show") or args.run)
    lab = labels_mod.load_labels(lab_path) if lab_path.exists() else labels_mod.Labels(show=summary.get("show") or "")
    if not lab_path.exists():
        print(f"Note: no labels at {lab_path}; recall will be empty.")
    pcfg = load_parser_config(settings=settings)
    r = evaluate_run(lab, read_json(d / "raw" / "moments.json", []), read_json(d / "raw" / "candidates.json", []),
                     read_json(d / "promos.json", []), read_json(judgements_path(settings, args.run), {}),
                     settings, lambda ts: parse_ts(ts, pcfg.compiled), run_models=run_models(summary))
    if r["label_source"] == "ai":
        from evaluate import agreement
        found = agreement.latest_for(settings, lab_path)
        r["agreement"] = agreement.summary_line(found) if found else None
    paths = write_report(r, settings, args.run)
    rc, pr = r["recall"], r["post_rate"]
    print(f"recall: {rc['ranked']}/{rc['labels']} ranked, {rc['candidates']}/{rc['labels']} made, "
          f"{rc['reader']}/{rc['labels']} marked by Reader | post rate: {pr['would_post']}/{pr['judged']}")
    print(f"report: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
