"""Ranking: pure functions over evaluated promos (no LLM, no graph)."""
import pytest

from config.loader import load_settings
from parser.models import format_ts
from ranking.rank import overlap_ratio, rank_promos


def secs(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def row(pid, segs, final, status="passed", decision=None, maker="cutter"):
    segments = [{"episode": e, "start_timestamp": format_ts(a), "end_timestamp": format_ts(b)} for e, a, b in segs]
    return {"promo_id": pid, "maker": maker, "segments": segments,
            "total_duration_seconds": round(sum(b - a for _, a, b in segs), 2),
            "scores": {"hook": 8, "cliffhanger": 8, "standalone": 8, "final": final},
            "status": status, "editor_decision": decision}


@pytest.fixture
def settings():
    s = load_settings()
    s.ranking.top_n, s.ranking.max_per_episode = 20, 2
    s.ranking.overlap_threshold, s.ranking.early_episode_preference, s.ranking.min_final_score = 0.5, 0.15, 6.0
    return s


EPS = list(range(1, 51))


def rank(rows, settings, eps=EPS):
    return rank_promos(rows, settings, eps, secs)


def reasons(dropped):
    return {d["promo_id"]: d["reason"] for d in dropped}


def test_overlap_ratio_uses_shorter_promo():
    a = row("a", [(2, 10, 40)], 8)                     # 30s
    b = row("b", [(2, 30, 50)], 8)                     # 20s, 10s shared
    assert overlap_ratio(a, b, secs) == pytest.approx(0.5)
    assert overlap_ratio(a, row("c", [(3, 10, 40)], 8), secs) == 0  # other episode


def test_sorted_ranked_and_cut_to_top_n(settings):
    settings.ranking.top_n = 3
    rows = [row(f"c{i:02d}_1", [(i, 0, 30)], 6 + i * 0.5) for i in range(1, 6)]
    ranked, dropped = rank(rows, settings, eps=[1, 2, 3, 4, 5])
    assert [r["promo_id"] for r in ranked] == ["c05_1", "c04_1", "c03_1"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert reasons(dropped) == {"c02_1": "below_top_n", "c01_1": "below_top_n"}


def test_near_duplicate_dropped_partial_overlap_kept(settings):
    rows = [row("a", [(2, 10, 40)], 8.0),
            row("b", [(2, 15, 42)], 7.0),               # 25s of its 27s inside a
            row("c", [(2, 35, 60)], 6.9)]               # 5s of 25s overlap: different promo
    ranked, dropped = rank(rows, settings)
    assert [r["promo_id"] for r in ranked] == ["a", "c"]
    [d] = dropped
    assert d["promo_id"] == "b" and d["reason"] == "duplicate" and d["of"] == "a"


def test_joined_promo_duplicating_a_single_keeps_the_better(settings):
    rows = [row("j01", [(1, 0, 10), (3, 0, 20)], 7.0, maker="joiner"),
            row("c03_1", [(3, 0, 25)], 8.0)]            # 20s shared / shorter 25s = 0.8
    ranked, dropped = rank(rows, settings)
    assert [r["promo_id"] for r in ranked] == ["c03_1"] and reasons(dropped) == {"j01": "duplicate"}


def test_per_episode_cap_counts_joined_promos(settings):
    rows = [row("c05_1", [(5, 0, 25)], 9.0), row("c05_2", [(5, 30, 55)], 8.5),
            row("j01", [(4, 0, 10), (5, 60, 80)], 8.0, maker="joiner"),   # third promo touching ep5
            row("c04_1", [(4, 20, 50)], 7.0)]
    ranked, dropped = rank(rows, settings)
    assert [r["promo_id"] for r in ranked] == ["c05_1", "c05_2", "c04_1"]
    assert reasons(dropped) == {"j01": "episode_cap"}


def test_early_episode_preference_breaks_close_calls_only(settings):
    ranked, _ = rank([row("late", [(50, 0, 30)], 7.0), row("early", [(1, 0, 30)], 7.0)], settings)
    assert [r["promo_id"] for r in ranked] == ["early", "late"]
    assert ranked[0]["rank_score"] == pytest.approx(7.0 * 1.15) and ranked[1]["rank_score"] == 7.0
    ranked, _ = rank([row("late", [(50, 0, 30)], 8.5), row("early", [(1, 0, 30)], 7.0)], settings)
    assert ranked[0]["promo_id"] == "late"                       # 8.5 beats 7.0 * 1.15 = 8.05
    assert ranked[0]["landing_episode"] == 50


def test_filters_status_editor_reject_and_min_score(settings):
    rows = [row("ok", [(1, 0, 30)], 7.0), row("bad", [(2, 0, 30)], 8.0, status="eval_failed"),
            row("rej", [(3, 0, 30)], 9.0, decision="reject"), row("weak", [(4, 0, 30)], 5.9, status="kept_best")]
    ranked, dropped = rank(rows, settings)
    assert [r["promo_id"] for r in ranked] == ["ok"]
    assert reasons(dropped) == {"bad": "status", "rej": "rejected_by_editor", "weak": "below_min_score"}


def test_editor_approved_promos_are_never_dropped(settings):
    settings.ranking.top_n = 1
    rows = [row("star", [(2, 10, 40)], 9.0),
            row("liked", [(2, 15, 42)], 5.5, decision="approve")]   # duplicate, low score, past top_n
    ranked, dropped = rank(rows, settings)
    assert [r["promo_id"] for r in ranked] == ["liked"]              # approved claims its spot first
    assert reasons(dropped) == {"star": "duplicate"}


def test_promo_sent_back_for_a_fix_stays_listed_for_before_after_review(settings):
    rows = [row("star", [(2, 10, 40)], 9.0),
            row("fixed", [(2, 15, 42)], 5.0, status="editor_revised", decision="fix")]
    ranked, dropped = rank(rows, settings)
    assert "fixed" in [r["promo_id"] for r in ranked] and reasons(dropped) == {"star": "duplicate"}
