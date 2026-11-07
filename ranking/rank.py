"""Rank evaluated promos into the final list. Pure functions: no LLM, no graph state.

Steps (all limits from settings.ranking):
  1. Filter: editor-rejected, unusable status (eval_failed, revising...), final < min_final_score.
  2. Score: rank_score = final * (1 + early_episode_preference * earliness), where earliness is
     1.0 for the run's first episode and 0.0 for its last. A promo's landing episode is the
     earliest episode it shows (where a viewer who taps it would start watching). The
     preference only settles close calls: with 0.15, ep 1 at 7.0 (8.05) still loses to
     ep 50 at 8.5.
  3. Pick greedily, best first:
       - near-duplicate: shares >= overlap_threshold of the SHORTER promo's screen time with an
         already-picked promo (same episode, overlapping spans) -> dropped
       - episode cap: an episode already used by max_per_episode picked promos -> dropped
         (a joined promo counts against every episode it uses)
       - top_n reached -> dropped
     Promos the editor approved, or sent back with "fix" (shown again with before/after), are
     pinned: picked first and exempt from the score floor and all three rules above, so they
     never disappear from a later review round.
Every dropped promo is reported with its reason, for the UI, the trace and evaluation.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from config.loader import Settings

ToSeconds = Callable[[str], float]
USABLE = {"passed", "kept_best", "editor_revised"}
PINNED = {"approve", "fix"}          # editor decisions that keep a promo listed


def _spans(promo: dict, to_s: ToSeconds) -> list[tuple[int, float, float]]:
    return [(s["episode"], to_s(s["start_timestamp"]), to_s(s["end_timestamp"])) for s in promo["segments"]]


def duration(promo: dict, to_s: ToSeconds) -> float:
    return sum(e - s for _, s, e in _spans(promo, to_s))


def overlap_ratio(a: dict, b: dict, to_s: ToSeconds) -> float:
    """Shared screen time / the shorter promo's duration (1.0 = one sits inside the other)."""
    shared = sum(max(0.0, min(e1, e2) - max(s1, s2))
                 for ep1, s1, e1 in _spans(a, to_s) for ep2, s2, e2 in _spans(b, to_s) if ep1 == ep2)
    shorter = min(duration(a, to_s), duration(b, to_s))
    return round(shared / shorter, 4) if shorter > 0 else 0.0


def rank_promos(rows: list[dict], settings: Settings, episodes: list[int], to_s: ToSeconds
                ) -> tuple[list[dict], list[dict]]:
    """rows: promos with `scores.final`, `status`, `editor_decision` (approve/reject/fix/None).
    Returns (ranked promos with rank / rank_score / landing_episode, dropped [{promo_id, reason, ...}])."""
    rc = settings.ranking
    first, last = (min(episodes), max(episodes)) if episodes else (1, 1)
    dropped: list[dict] = []
    pool: list[dict] = []

    def drop(r: dict, reason: str, **detail):
        dropped.append({"promo_id": r["promo_id"], "reason": reason, "final": r["scores"]["final"], **detail})

    for r in rows:
        decision = r.get("editor_decision")
        pinned = decision in PINNED
        if decision == "reject":
            drop(r, "rejected_by_editor")
            continue
        if not pinned and r.get("status") not in USABLE:
            drop(r, "status", status=r.get("status"))
            continue
        if not pinned and r["scores"]["final"] < rc.min_final_score:
            drop(r, "below_min_score", min_final_score=rc.min_final_score)
            continue
        landing = min(s["episode"] for s in r["segments"])
        earliness = (last - landing) / (last - first) if last > first else 0.0
        rank_score = round(r["scores"]["final"] * (1 + rc.early_episode_preference * earliness), 3)
        pool.append({**r, "landing_episode": landing, "rank_score": rank_score, "pinned": pinned})

    order = lambda r: (-r["rank_score"], -r["scores"]["final"], r["landing_episode"], r["promo_id"])
    picked: list[dict] = []
    per_episode: Counter = Counter()
    for r in sorted(pool, key=lambda r: (not r["pinned"],) + order(r)):
        eps = {s["episode"] for s in r["segments"]}
        if not r["pinned"]:
            dup = next(((p, o) for p in picked if (o := overlap_ratio(r, p, to_s)) >= rc.overlap_threshold), None)
            if dup:
                drop(r, "duplicate", of=dup[0]["promo_id"], overlap=dup[1])
                continue
            full = sorted(e for e in eps if per_episode[e] >= rc.max_per_episode)
            if full:
                drop(r, "episode_cap", episodes=full, max_per_episode=rc.max_per_episode)
                continue
            if len(picked) >= rc.top_n:
                drop(r, "below_top_n", top_n=rc.top_n)
                continue
        picked.append(r)
        per_episode.update(eps)

    picked.sort(key=order)
    for i, r in enumerate(picked, start=1):
        r["rank"] = i
    return picked, dropped
