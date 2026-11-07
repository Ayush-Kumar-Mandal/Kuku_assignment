"""Turn a promo into the text a viewer would actually get: the Evaluator's ONLY input.

No show name, logline, story summary or episode numbers: the Evaluator simulates someone
who has never seen the show. Times are relative to the promo's own start, so the hook
window ("first 3 seconds") is visible to it.
"""
from __future__ import annotations

from config.loader import CompiledPatterns
from parser.models import Episode, format_ts
from parser.screenplay_parser import parse_ts


def render_promo_text(promo: dict, episodes: dict[int, Episode], pats: CompiledPatterns) -> str:
    out: list[str] = []
    offset = 0.0
    segs = promo["segments"]
    for i, seg in enumerate(segs, start=1):
        ep = episodes[seg["episode"]]
        s, e = parse_ts(seg["start_timestamp"], pats), parse_ts(seg["end_timestamp"], pats)
        if len(segs) > 1:
            out.append(f"--- cut {i} ---")
        for it in ep.all_items():
            if s - 1e-6 <= it.start < e:
                who = it.character or it.kind.upper()
                tone = f" ({', '.join(it.emotions)})" if it.emotions else ""
                out.append(f"[{format_ts(offset + it.start - s)}] {who}{tone}: {it.text}")
        offset += e - s
    out.append(f"[{format_ts(offset)}] END OF PROMO")
    return "\n".join(out)
