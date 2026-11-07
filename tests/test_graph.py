"""Graph skeleton tests: real StateGraph + SQLite checkpointer + stub agents."""
from pathlib import Path

import pytest
from langgraph.types import Command

from agents.stubs import StubAgents, StubCrash
from config.loader import load_moment_types, load_parser_config, load_settings
from graph.build import build_graph
from graph.context import RunContext
from graph.runner import initial_state, open_checkpointer, run, status
from graph.state import promo_trace
from synthetic import write_show


@pytest.fixture
def show_dir(tmp_path):
    return write_show(tmp_path / "show")


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.logs_dir = str(tmp_path / "logs")
    s.paths.output_dir = str(tmp_path / "output")
    return s


def make(settings, **stub_kw):
    return RunContext(settings=settings, parser_cfg=load_parser_config(settings=settings),
                      moment_types=load_moment_types(settings=settings), agents=StubAgents(settings, **stub_kw))


@pytest.fixture
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def start(settings, db, show_dir, ctx, run_id="r1", **switches):
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        out = run(g, ctx, run_id, initial_state(run_id, "Test Show", "drama", show_dir, settings, **switches))
        return out, status(g, run_id, settings)


def test_full_run_without_review(settings, db, show_dir):
    ctx = make(settings)
    out, st = start(settings, db, show_dir, ctx, editor_review=False)
    assert st["state"] == "done"
    assert out["reader_cursor"] == 3 and len(out["moments"]) == 6
    assert out["story_summary"].count("Ep") == 3                     # summary carried over
    # every promo failed v1, was revised once by its maker, passed v2
    assert set(out["candidates"]) == {"c01_1", "c02_1", "c03_1", "j01"}
    assert all(out["status"][p] == "passed" for p in out["candidates"])
    assert all(len(out["evaluations"][p]) == 2 for p in out["candidates"])
    assert out["revision_counts"] == {p: 1 for p in out["candidates"]}
    assert ctx.agents.calls["revise_cutter"] == 3 and ctx.agents.calls["revise_joiner"] == 1
    # ranked output carries the fields the spec asks for
    p = out["promos"][0]
    assert p["rank"] == 1 and p["trace_id"] == f"r1:{p['promo_id']}" and {"hook", "cliffhanger", "standalone", "final"} <= set(p["scores"])
    # per-promo trace: reader moment -> created -> evaluated -> revised -> evaluated -> ranked
    events = [e["event"] for e in promo_trace(out, "c01_1")]
    assert events[0] == "moment" and events[-1] == "ranked"
    assert ["created", "evaluated", "revised", "evaluated"] == [e for e in events if e in {"created", "evaluated", "revised"}]


def test_evaluator_sees_only_promo_text(settings, db, show_dir):
    out, _ = start(settings, db, show_dir, make(settings), editor_review=False)
    text = out["evaluations"]["j01"][0]["promo_text"]
    assert "--- cut 2 ---" in text and "END OF PROMO" in text
    assert "Test Show" not in text and "Synthetic episode" not in text   # no show name / logline


def test_keep_best_after_max_rounds(settings, db, show_dir):
    ctx = make(settings, always_fail={"c01_1"})
    out, _ = start(settings, db, show_dir, ctx, editor_review=False, joiner=False)
    rounds = settings.evaluator.max_revision_rounds
    assert out["status"]["c01_1"] == "kept_best"
    assert out["revision_counts"]["c01_1"] == rounds
    assert len(out["evaluations"]["c01_1"]) == rounds + 1
    best = max(out["evaluations"]["c01_1"], key=lambda e: e["final"])
    assert out["candidates"]["c01_1"]["version"] == best["version"]
    assert "j01" not in out["candidates"]                                # joiner switched off


def test_editor_review_interrupt_fix_and_resume(settings, db, show_dir):
    ctx = make(settings)
    out, st = start(settings, db, show_dir, ctx, joiner=False)
    assert st["state"] == "awaiting_review"
    payload = st["interrupt"]
    assert payload["round"] == 1 and set(payload["pending"]) == {"c01_1", "c02_1", "c03_1"}

    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        run(g, ctx, "r1", Command(resume={
            "c01_1": {"decision": "approve"},
            "c02_1": {"decision": "fix", "comment": "End on the slap, not after it"},
            "c03_1": {"decision": "reject", "comment": "needs earlier context"},
        }))
        st = status(g, "r1", settings)
        # fix went cutter -> evaluator -> rank -> review again, with before/after
        assert st["state"] == "awaiting_review"
        p2 = st["interrupt"]
        assert p2["round"] == 2 and p2["pending"] == ["c02_1"]
        ba = p2["before_after"]["c02_1"]
        assert ba["before"]["version"] == 2 and ba["after"]["version"] == 3
        # the maker got the editor's words verbatim (authoritative) plus the FeedbackAgent's reading
        assert "End on the slap, not after it" in ba["after"]["revision_note"]
        assert "FeedbackAgent" in ba["after"]["revision_note"]
        assert ba["editor_comment"] == "End on the slap, not after it" and ba["fix_interpretation"]
        assert "c03_1" not in [p["promo_id"] for p in p2["promos"]]       # rejected promo dropped

        out = run(g, ctx, "r1", Command(resume={"c02_1": {"decision": "approve"}}))
        assert status(g, "r1", settings)["state"] == "done"
    assert out["fix_attempts"] == {"c02_1": 1}
    assert out["status"]["c02_1"] == "editor_revised"
    assert {(l["promo_id"], l["decision"]) for l in out["lessons"]} == {
        ("c01_1", "approve"), ("c02_1", "fix"), ("c03_1", "reject"), ("c02_1", "approve")}
    assert out["memory_written"]["feedback_items"] == 4


def test_crash_then_resume_from_checkpoint(settings, db, show_dir):
    crashing = make(settings, crash_on={"read_episode": 2})
    with pytest.raises(StubCrash):
        start(settings, db, show_dir, crashing, editor_review=False)

    fresh = make(settings)                                   # e.g. new process after a rate-limit stop
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        st = status(g, "r1", settings)
        assert st["state"] == "stopped" and st["next"] == ["reader"]
        assert st["values"]["reader_cursor"] == 1                # episode 1 was saved
        out = run(g, fresh, "r1")                                # input None = resume
        assert status(g, "r1", settings)["state"] == "done"
    assert fresh.agents.calls["read_episode"] == 2               # only episodes 2 and 3
    assert out["reader_cursor"] == 3 and len(out["moments"]) == 6


def test_memory_switch_recorded(settings, db, show_dir):
    out, _ = start(settings, db, show_dir, make(settings), memory=False, editor_review=False)
    assert out["memory_used"]["enabled"] is False and out["switches"]["memory"] is False


# ---------------------------------------------------------------- Step 4: agent failures
def test_reader_failure_skips_episode_and_run_continues(settings, db, show_dir):
    ctx = make(settings, error_on={"read_episode": {2}})        # episode 2's read fails
    out, st = start(settings, db, show_dir, ctx, editor_review=False, joiner=False)
    assert st["state"] == "done"
    assert out["reader_cursor"] == 3
    assert {m["episode"] for m in out["moments"]} == {1, 3}
    assert any("reader" in e and "episode 2" in e for e in out["errors"])
    assert out["episode_notes"]["2"]["failed"] is True
    assert "c02_1" not in out["candidates"]                        # nothing to cut


def test_evaluator_failure_marks_promo_and_skips_ranking(settings, db, show_dir):
    ctx = make(settings, error_on={"evaluate": {1}})              # first evaluation fails
    out, _ = start(settings, db, show_dir, ctx, editor_review=False, joiner=False)
    assert out["status"]["c01_1"] == "eval_failed"
    assert "c01_1" not in [p["promo_id"] for p in out["promos"]]
    assert any("evaluator" in e and "c01_1" in e for e in out["errors"])


def test_failed_revision_keeps_best_evaluated_version(settings, db, show_dir):
    ctx = make(settings, error_on={"revise_cutter": {1}})         # c01_1's revision fails
    out, _ = start(settings, db, show_dir, ctx, editor_review=False, joiner=False)
    assert out["status"]["c01_1"] == "kept_best"
    assert out["candidates"]["c01_1"]["version"] == 1
    assert out["status"]["c02_1"] == "passed"


def test_episode_filter_limits_the_run(settings, db, show_dir):
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        out = run(g, make(settings), "r1", initial_state("r1", "Test Show", "drama", show_dir, settings,
                                                         editor_review=False, episodes=[1, 3]))
    assert [e["number"] for e in out["episodes"]] == [1, 3]
    assert {m["episode"] for m in out["moments"]} == {1, 3}


# ---------------------------------------------------------------- Step 5: ranking node
def test_rank_node_uses_ranking_and_reports_drops(settings, db, show_dir):
    settings.ranking.max_per_episode = 1        # each stub episode also feeds the joined promo j01
    out, _ = start(settings, db, show_dir, make(settings), editor_review=False)
    rep = out["ranking_report"]
    ranked = [p["promo_id"] for p in out["promos"]]
    assert rep["considered"] == 4 and len(ranked) + len(rep["dropped"]) == 4
    assert {d["reason"] for d in rep["dropped"]} <= {"duplicate", "episode_cap"}
    p = out["promos"][0]
    assert {"rank", "rank_score", "landing_episode", "trace_id", "hook", "cliffhanger", "why_it_works",
            "moment_type", "segments", "total_duration_seconds"} <= set(p)
    events = [e for e in out["trace"] if e["node"] == "rank"]
    assert {e["event"] for e in events} == {"ranked", "dropped"}


def test_final_list_drops_rejected_promos_when_review_ends(settings, db, show_dir):
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        out = run(g, ctx, "r1", Command(resume={"c01_1": {"decision": "approve"},
                                               "c02_1": {"decision": "reject", "comment": "weak"},
                                               "c03_1": {"decision": "approve"}}))
        assert status(g, "r1", settings)["state"] == "done"
    assert [p["promo_id"] for p in out["promos"]] == ["c01_1", "c03_1"]
    assert [p["rank"] for p in out["promos"]] == [1, 2]
    assert {"promo_id": "c02_1", "reason": "rejected_by_editor"}.items() <= next(
        d for d in out["ranking_report"]["dropped"] if d["promo_id"] == "c02_1").items()


# ------------------------------------------------- evaluator works in checkpointed batches
def test_evaluator_batches_survive_a_crash(settings, db, show_dir):
    settings.evaluator.batch_size = 2
    crashing = make(settings, crash_on={"evaluate": 3})          # dies inside the 2nd batch
    with pytest.raises(StubCrash):
        start(settings, db, show_dir, crashing, editor_review=False, joiner=False)
    fresh = make(settings)
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        assert len(status(g, "r1", settings)["values"]["evaluations"]) == 2   # 1st batch was saved
        out = run(g, fresh, "r1")
    # 3 promos x (v1 + v2) = 6 evaluations in all; the 2 saved ones are not redone
    assert fresh.agents.calls["evaluate"] == 4
    assert all(out["status"][p] == "passed" for p in ("c01_1", "c02_1", "c03_1"))


def test_evaluator_loops_until_pending_is_empty(settings, db, show_dir):
    settings.evaluator.batch_size = 1
    ctx = make(settings)
    out, _ = start(settings, db, show_dir, ctx, editor_review=False, joiner=False)
    assert ctx.agents.calls["evaluate"] == 6 and not out["pending_eval"]
    assert ctx.agents.calls["revise_cutter"] == 3                 # revisions still go round by round



# ------------------------------------------------------------- Phase 2: review loop
def decide(settings, db, ctx, decisions, run_id="r1"):
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        out = run(g, ctx, run_id, Command(resume=decisions))
        return out, status(g, run_id, settings)


def test_rejected_slot_is_backfilled_and_newcomer_reviewed(settings, db, show_dir):
    settings.ranking.top_n = 2                                   # c03_1 starts below the cut
    ctx = make(settings)
    _, st = start(settings, db, show_dir, ctx, joiner=False)
    assert st["interrupt"]["pending"] == ["c01_1", "c02_1"]
    _, st = decide(settings, db, ctx, {"c01_1": {"decision": "approve"}, "c02_1": {"decision": "reject"}})
    assert st["state"] == "awaiting_review" and st["interrupt"]["round"] == 2
    assert st["interrupt"]["pending"] == ["c03_1"]               # pulled up into the free slot
    out, st = decide(settings, db, ctx, {"c03_1": {"decision": "approve"}})
    assert st["state"] == "done" and out["review_complete"] is True
    assert [p["promo_id"] for p in out["promos"]] == ["c01_1", "c03_1"]


def test_backfill_off_just_removes_rejected(settings, db, show_dir):
    settings.ranking.top_n, settings.feedback.backfill_rejected = 2, False
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    out, st = decide(settings, db, ctx, {"c01_1": {"decision": "approve"}, "c02_1": {"decision": "reject"}})
    assert st["state"] == "done" and [p["promo_id"] for p in out["promos"]] == ["c01_1"]


def test_all_approved_ends_without_another_pause(settings, db, show_dir):
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    out, st = decide(settings, db, ctx, {p: {"decision": "approve"} for p in ("c01_1", "c02_1", "c03_1")})
    assert st["state"] == "done" and out["review_round"] == 1


def test_keep_before_reverts_the_editor_fix(settings, db, show_dir):
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    approve_rest = {"c02_1": {"decision": "approve"}, "c03_1": {"decision": "approve"}}
    _, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix", "comment": "Start later"}, **approve_rest})
    assert st["interrupt"]["pending"] == ["c01_1"] and "c01_1" in st["interrupt"]["before_after"]
    out, st = decide(settings, db, ctx, {"c01_1": {"decision": "keep_before"}})
    assert st["state"] == "done"
    p = next(p for p in out["promos"] if p["promo_id"] == "c01_1")
    assert p["version"] == 2 and out["candidates"]["c01_1"]["version"] == 2     # v3 (the fix) undone
    assert out["editor_decisions"]["c01_1"]["decision"] == "approve"
    assert out["editor_decisions"]["c01_1"]["kept_before"] is True


def test_invalid_decisions_keep_the_promo_pending(settings, db, show_dir):
    settings.feedback.max_fix_attempts = 1
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    approve_rest = {"c02_1": {"decision": "approve"}, "c03_1": {"decision": "approve"}}
    _, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix"}, **approve_rest})     # fix without a comment
    assert st["interrupt"]["pending"] == ["c01_1"] and st["interrupt"]["round"] == 2
    _, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix", "comment": "tighter"}})
    assert st["interrupt"]["can_fix"]["c01_1"] is False           # the one allowed fix is used
    out, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix", "comment": "again"},  # over the limit
                                         "c09_9": {"decision": "approve"}})                 # not under review
    assert st["state"] == "awaiting_review" and st["interrupt"]["pending"] == ["c01_1"]
    errs = " ".join(st["values"]["errors"])
    assert "comment" in errs and "no fix attempts left" in errs and "c09_9" in errs


def test_max_review_rounds_stops_and_flags_unreviewed(settings, db, show_dir):
    settings.feedback.max_review_rounds = 1
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    out, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix", "comment": "tighter"},
                                         "c02_1": {"decision": "approve"}, "c03_1": {"decision": "approve"}})
    assert st["state"] == "done"
    assert out["review_summary"] == {"rounds": 1, "stopped_by_max_rounds": True, "unreviewed": ["c01_1"]}


def test_lessons_carry_what_memory_will_need(settings, db, show_dir):
    ctx = make(settings)
    start(settings, db, show_dir, ctx, joiner=False)
    out, _ = decide(settings, db, ctx, {p: {"decision": "approve", "comment": "good"} for p in ("c01_1", "c02_1", "c03_1")})
    l = out["lessons"][0]
    assert {"promo_id", "decision", "comment", "round", "decided_at", "version", "maker", "moment_type", "final"} <= set(l)


def test_feedback_agent_failure_falls_back_to_the_editors_words(settings, db, show_dir):
    ctx = make(settings, error_on={"interpret_fix": {1}})
    start(settings, db, show_dir, ctx, joiner=False)
    _, st = decide(settings, db, ctx, {"c01_1": {"decision": "fix", "comment": "End on the slap"},
                                       "c02_1": {"decision": "approve"}, "c03_1": {"decision": "approve"}})
    ba = st["interrupt"]["before_after"]["c01_1"]
    assert ba["after"]["revision_note"] == "applied: End on the slap" and ba["fix_interpretation"] is None
    assert any("feedback" in e and "c01_1" in e for e in st["values"]["errors"])
