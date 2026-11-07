"""Agreement between two label sets: can the AI labeller be trusted?

    python -m evaluate.agreement --reference <your labels.yaml> --candidate <ai labels.yaml>

The reference is the answer you trust: your own blind labels (evaluate.labels) or a
synthetic show's planted key (evaluate.synthetic_show). The candidate is usually the AI
labeller's file (evaluate.ai_labels). Only episodes finished in BOTH files count.
Moments are matched one-to-one, best overlap first, with the same rule as everywhere else
(shared screen time >= evaluation.match_min_overlap of the shorter one). Reports:
  recall     share of the reference moments the candidate also marked
  precision  share of the candidate's moments that are in the reference
  F1         both in one number
and a verdict from evaluation.agreement: trust it on other episodes / don't rely on it /
too few reference moments to say. Writes <evaluation.dir>/agreement/<candidate>__vs__<reference>.md|json;
evaluate.compare quotes it whenever it uses those AI labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config.loader import Settings, load_parser_config, load_settings
from evaluate import labels as labels_mod
from evaluate.common import eval_dir, overlap, read_json
from parser.screenplay_parser import parse_ts


def _source(lab: labels_mod.Labels) -> str:
    return "ai" if lab.label_source == "human" and lab.labeled_by != "human" else lab.label_source


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def compare_labels(reference: labels_mod.Labels, candidate: labels_mod.Labels, settings: Settings, to_s) -> dict:
    cfg, thr = settings.evaluation.agreement, settings.evaluation.match_min_overlap
    episodes = sorted(set(reference.episodes) & set(candidate.episodes))
    ref = [m for m in reference.moments if m.episode in episodes]
    cand = [m for m in candidate.moments if m.episode in episodes]
    span = lambda m: [(m.episode, to_s(m.start), to_s(m.end))]

    scored = sorted(((overlap(span(r), span(c)), i, j) for i, r in enumerate(ref) for j, c in enumerate(cand)
                     if r.episode == c.episode), reverse=True)
    used_r, used_c, pairs = set(), set(), []
    for ov, i, j in scored:                      # greedy, best overlap first: each moment matched once
        if ov < thr or i in used_r or j in used_c:
            continue
        used_r.add(i), used_c.add(j)
        pairs.append({"reference": ref[i].id, "candidate": cand[j].id, "episode": ref[i].episode,
                      "reference_span": f"{ref[i].start}-{ref[i].end}", "candidate_span": f"{cand[j].start}-{cand[j].end}",
                      "overlap": round(ov, 2)})
    pairs.sort(key=lambda p: (p["episode"], p["reference"]))
    describe = lambda m: {"id": m.id, "episode": m.episode, "span": f"{m.start}-{m.end}", "moment_type": m.moment_type,
                          "note": m.note}
    n = len(pairs)
    recall, precision = _rate(n, len(ref)), _rate(n, len(cand))
    f1 = round(2 * recall * precision / (recall + precision), 3) if recall and precision else (0.0 if ref and cand else None)
    per_ep = {e: {"reference": sum(m.episode == e for m in ref), "candidate": sum(m.episode == e for m in cand),
                  "matched": sum(p["episode"] == e for p in pairs)} for e in episodes}

    if not episodes:
        verdict = ("no_overlap", "No episode is finished in both files: nothing to compare yet.")
    elif len(ref) < cfg.min_reference_moments:
        verdict = ("too_few", f"Too few reference moments to say ({len(ref)}; need at least "
                              f"{cfg.min_reference_moments}). Label more episodes and run this again.")
    elif (recall or 0) >= cfg.trust_min_recall and (precision or 0) >= cfg.trust_min_precision:
        verdict = ("trust", f"Trust it on the other episodes: it found {recall:.0%} of the reference moments "
                            f"(needs {cfg.trust_min_recall:.0%}) and {precision:.0%} of its own picks are in the "
                            f"reference (needs {cfg.trust_min_precision:.0%}).")
    else:
        verdict = ("dont_rely", f"Don't rely on it: recall {recall or 0:.0%} (needs {cfg.trust_min_recall:.0%}), "
                                f"precision {precision or 0:.0%} (needs {cfg.trust_min_precision:.0%}).")
    warnings = []
    if _source(reference) == "ai":
        warnings.append("The reference is itself AI-made: this is AI-vs-AI agreement, not a check against a human.")
    if _source(candidate) == "human":
        warnings.append("The candidate is a hand-made file: this measures agreement between two people.")
    return {"reference": reference.labeled_by, "reference_source": _source(reference),
            "candidate": candidate.labeled_by, "candidate_source": _source(candidate),
            "candidate_models": candidate.labeler_models, "episodes": episodes,
            "reference_moments": len(ref), "candidate_moments": len(cand), "matched": n,
            "recall": recall, "precision": precision, "f1": f1, "match_min_overlap": thr,
            "per_episode": per_ep, "pairs": pairs,
            "missed": [describe(ref[i]) for i in range(len(ref)) if i not in used_r],
            "extra": [describe(cand[j]) for j in range(len(cand)) if j not in used_c],
            "verdict": {"key": verdict[0], "text": verdict[1]}, "warnings": warnings}


def report_path(settings: Settings, candidate: Path, reference: Path) -> Path:
    """Base path (no suffix) of the report for this pair of files."""
    return eval_dir(settings) / "agreement" / f"{Path(candidate).stem}__vs__{Path(reference).stem}"


def latest_for(settings: Settings, candidate: Path) -> dict | None:
    """Newest agreement report for this candidate labels file, if any."""
    found = sorted((eval_dir(settings) / "agreement").glob(f"{Path(candidate).stem}__vs__*.json"),
                   key=lambda p: p.stat().st_mtime)
    return read_json(found[-1]) if found else None


def summary_line(r: dict) -> str:
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    return (f"Checked against {r['reference']} ({r['reference_source']}) on {len(r['episodes'])} episode(s), "
            f"{r['reference_moments']} reference moments: recall {pct(r['recall'])}, precision {pct(r['precision'])}. "
            f"{r['verdict']['text']}")


def write_report(r: dict, base: Path, reference_path: Path, candidate_path: Path) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    r = {**r, "reference_file": str(reference_path), "candidate_file": str(candidate_path),
         "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    paths = {"json": base.with_suffix(".json"), "md": base.with_suffix(".md")}
    paths["json"].write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    L = [f"# Label agreement: {Path(candidate_path).name} vs {Path(reference_path).name}", "",
         f"- Reference (trusted): {r['reference']} [{r['reference_source']}]",
         f"- Candidate (checked): {r['candidate']} [{r['candidate_source']}]",
         f"- Episodes finished in both: {', '.join(map(str, r['episodes'])) or 'none'} · "
         f"match = shared screen time >= {r['match_min_overlap']:.0%} of the shorter moment", ""]
    L += [f"> **{w}**" for w in r["warnings"]] + ([""] if r["warnings"] else [])
    L += [f"## Verdict", "", f"**{r['verdict']['text']}**", "",
          "| Measure | Value | Meaning |", "|---|---|---|",
          f"| Recall | {pct(r['recall'])} ({r['matched']}/{r['reference_moments']}) | reference moments the candidate also marked |",
          f"| Precision | {pct(r['precision'])} ({r['matched']}/{r['candidate_moments']}) | candidate moments that are in the reference |",
          f"| F1 | {'-' if r['f1'] is None else r['f1']} | both together |", "",
          "## Per episode", "", "| Episode | Reference | Candidate | Matched |", "|---|---|---|---|"]
    for e, x in r["per_episode"].items():
        L.append(f"| {e} | {x['reference']} | {x['candidate']} | {x['matched']} |")
    L += ["", "## Matched", "", "| Ep | Reference | Candidate | Overlap |", "|---|---|---|---|"]
    L += [f"| {p['episode']} | {p['reference']} {p['reference_span']} | {p['candidate']} {p['candidate_span']} | "
          f"{p['overlap']:.0%} |" for p in r["pairs"]] or ["| - | - | - | - |"]
    for title, key in (("Missed by the candidate (in the reference only)", "missed"),
                       ("Extra (marked by the candidate only)", "extra")):
        L += ["", f"## {title}", "", "| Id | Ep | Span | Type | Note |", "|---|---|---|---|---|"]
        L += [f"| {m['id']} | {m['episode']} | {m['span']} | {m['moment_type']} | {m['note']} |" for m in r[key]] \
            or ["| - | - | - | - | - |"]
    paths["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, help="the labels you trust (yours, or a planted key)")
    ap.add_argument("--candidate", required=True, help="the labels to check (usually the AI labeller's)")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = settings or load_settings(args.config)
    ref_p, cand_p = Path(args.reference), Path(args.candidate)
    pcfg = load_parser_config(settings=settings)
    r = compare_labels(labels_mod.load_labels(ref_p), labels_mod.load_labels(cand_p), settings,
                       lambda ts: parse_ts(ts, pcfg.compiled))
    paths = write_report(r, report_path(settings, cand_p, ref_p), ref_p, cand_p)
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    for w in r["warnings"]:
        print(f"WARNING: {w}")
    print(f"episodes {r['episodes']} | recall {pct(r['recall'])} ({r['matched']}/{r['reference_moments']}) | "
          f"precision {pct(r['precision'])} ({r['matched']}/{r['candidate_moments']}) | F1 {r['f1']}")
    print(r["verdict"]["text"])
    print(f"report: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
