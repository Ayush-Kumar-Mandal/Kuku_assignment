"""Phase 3: MemoryStore (SQLite): feedback_log, guidelines, examples, versions, retrieval."""
import sqlite3

import pytest

from memory.store import MemoryStore, scope_for


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def fb(mem, show="Show A", genre="romance", comment="", decision="reject", promo_id="c01_1", run_id="r1", rnd=1):
    return mem.log_feedback(show=show, genre=genre, promo_id=promo_id, promo={"promo_id": promo_id, "hook": "h"},
                            moment_type="reveal", decision=decision, comment=comment, guidelines_version="v0",
                            run_id=run_id, round=rnd)


# ------------------------------------------------------------------------ feedback_log
def test_feedback_log_is_append_only(mem):
    fid = fb(mem, comment="too slow")
    assert mem.feedback([fid])[0]["comment"] == "too slow"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        mem.conn.execute("UPDATE feedback_log SET comment = 'x' WHERE id = ?", (fid,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        mem.conn.execute("DELETE FROM feedback_log WHERE id = ?", (fid,))


def test_same_decision_logged_twice_is_not_duplicated(mem):
    assert fb(mem, comment="a") == fb(mem, comment="a")          # re-run of memory_write after a crash
    assert len(mem.feedback()) == 1


# -------------------------------------------------------------------------- guidelines
def test_rules_versions_and_editor_actions(mem):
    assert mem.version() == 0 and mem.version_label() == "v0"
    rid = mem.add_rule("Start on the line that names the conflict.", scope="global", source_feedback_ids=[],
                       status="needs_review", created_by="ai")
    assert mem.version() == 1 and mem.rule(rid)["status"] == "needs_review"
    mem.set_status(rid, "active")
    mem.edit_rule(rid, "Start on the line that names the conflict, never on scenery.")
    mem.set_status(rid, "retired")
    mem.set_status(rid, "active")                                  # re-activate
    assert mem.version() == 5
    hist = mem.history()
    assert [h["version"] for h in hist] == [5, 4, 3, 2, 1]
    assert hist[0]["active_rules"][0]["text"].endswith("never on scenery.")
    mem.delete_rule(rid)
    assert mem.rule(rid) is None and mem.version() == 6
    with pytest.raises(ValueError):
        mem.set_status(999, "active")


def test_evidence_updates_confidence_and_flags_contradicted_rules(mem):
    rid = mem.add_rule("End right on the unanswered question.", scope="global", source_feedback_ids=[], status="active")
    c0 = mem.rule(rid)["confidence"]
    f1, f2, f3, f4 = (fb(mem, comment=f"c{i}", promo_id=f"p{i}") for i in range(4))
    assert mem.add_evidence(rid, f1, "supports") is True
    assert mem.add_evidence(rid, f1, "supports") is False         # same feedback counted once
    r = mem.rule(rid)
    assert r["supporting_count"] == 1 and r["confidence"] > c0 and r["status"] == "active"
    v = mem.version()
    for f in (f2, f3):
        mem.add_evidence(rid, f, "contradicts")
    r = mem.rule(rid)
    assert (r["supporting_count"], r["contradicting_count"]) == (1, 2)
    assert r["status"] == "needs_review" and mem.version() == v + 1  # never auto-deleted, flagged once
    mem.add_evidence(rid, f4, "contradicts")
    assert mem.rule(rid)["status"] == "needs_review"


def test_conflicting_rules_both_need_review(mem):
    a = mem.add_rule("Keep promos under 30 seconds.", scope="global", source_feedback_ids=[], status="active")
    b = mem.add_rule("Let promos breathe: 45 seconds or more.", scope="global", source_feedback_ids=[], status="active")
    mem.mark_conflict(a, b, why="opposite length advice")
    assert mem.rule(a)["status"] == mem.rule(b)["status"] == "needs_review"
    assert b in mem.rule(a)["conflicts_with"]


@pytest.mark.parametrize("rows, expect", [
    ([("A", "romance")], ("show", "A")),
    ([("A", "romance"), ("A", "romance")], ("show", "A")),
    ([("A", "romance"), ("B", "romance")], ("genre", "romance")),
    ([("A", "romance"), ("B", "thriller")], ("global", None)),
])
def test_scope_from_source_feedback(rows, expect):
    assert scope_for([{"show": s, "genre": g} for s, g in rows]) == expect


# ---------------------------------------------------------------------------- reading
def test_for_run_filters_by_scope_and_orders_by_confidence(mem):
    f = [fb(mem, comment=f"c{i}", promo_id=f"p{i}") for i in range(3)]
    g = mem.add_rule("global rule", scope="global", source_feedback_ids=[], status="active")
    mem.add_rule("romance rule", scope="genre", scope_value="romance", source_feedback_ids=[], status="active")
    mem.add_rule("show A rule", scope="show", scope_value="Show A", source_feedback_ids=[], status="active")
    mem.add_rule("show B rule", scope="show", scope_value="Show B", source_feedback_ids=[], status="active")
    mem.add_rule("thriller rule", scope="genre", scope_value="thriller", source_feedback_ids=[], status="active")
    mem.add_rule("proposed rule", scope="global", source_feedback_ids=[], status="needs_review")
    for x in f:
        mem.add_evidence(g, x, "supports")                         # global rule now most confident
    used = mem.for_run(show="Show A", genre="Romance", top_rules=3, approved=0, rejected=0)
    assert [r["text"] for r in used["rules"]] == ["global rule", "romance rule", "show A rule"]
    assert used["guidelines_version"] == mem.version_label()
    assert len(mem.for_run(show="Show A", genre="romance", top_rules=1, approved=0, rejected=0)["rules"]) == 1


def test_examples_same_genre_first_then_any_highest_scores_first(mem):
    def ex(label, genre, final, pid):
        mem.add_example(promo={"promo_id": pid, "hook": f"hook {pid}", "cliffhanger": "c", "moment_type": "reveal",
                               "total_duration_seconds": 30}, label=label, reason="r", show="S", genre=genre,
                        moment_type="reveal", scores={"final": final}, run_id="r1")
    ex("approved", "romance", 7.0, "a1"); ex("approved", "romance", 9.0, "a2"); ex("approved", "thriller", 9.5, "a3")
    ex("rejected", "thriller", 4.0, "x1")                          # no romance rejects -> fall back to any genre
    used = mem.for_run(show="S", genre="romance", top_rules=0, approved=2, rejected=1)
    assert [e["promo_id"] for e in used["examples"] if e["label"] == "approved"] == ["a2", "a1"]
    assert [e["promo_id"] for e in used["examples"] if e["label"] == "rejected"] == ["x1"]
    assert "hook a2" in used["examples"][0]["summary"]


def test_unused_feedback_for_learning(mem):
    used_one = fb(mem, comment="too slow", promo_id="p1")
    fb(mem, comment="", decision="approve", promo_id="p2")          # nothing to learn from
    free = fb(mem, comment="ends too late", promo_id="p3")
    mem.add_rule("x", scope="show", scope_value="Show A", source_feedback_ids=[used_one], status="needs_review")
    assert [f["id"] for f in mem.unused_feedback(limit=10)] == [free]


def test_run_write_markers(mem):
    assert mem.write_stage("r1") is None
    mem.mark_written("r1", "logged")
    assert mem.write_stage("r1") == "logged"
    mem.mark_written("r1", "learned")
    assert mem.write_stage("r1") == "learned"
