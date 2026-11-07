"""Phase 3: memory_read / memory_write in the real graph (stub agents, temp memory DB)."""
import pytest
from langgraph.types import Command

from agents.stubs import StubAgents
from config.loader import load_moment_types, load_parser_config, load_settings
from graph import nodes
from graph.build import build_graph
from graph.context import RunContext
from graph.runner import initial_state, open_checkpointer, run, status
from memory.store import MemoryStore
from synthetic import write_show


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.logs_dir = str(tmp_path / "logs")
    s.paths.output_dir = str(tmp_path / "output")
    return s


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


@pytest.fixture
def show_dir(tmp_path):
    return write_show(tmp_path / "show")


def ctx_for(settings, mem, **kw):
    return RunContext(settings=settings, parser_cfg=load_parser_config(settings=settings),
                      moment_types=load_moment_types(settings=settings), agents=StubAgents(settings, **kw), memory=mem)


def reviewed_run(settings, tmp_path, show_dir, ctx, run_id, decisions, show="Show A", genre="romance", memory=True):
    db = tmp_path / f"{run_id}.sqlite"
    with open_checkpointer(settings, db) as cp:
        g = build_graph(cp)
        run(g, ctx, run_id, initial_state(run_id, show, genre, show_dir, settings, joiner=False, memory=memory))
        out = run(g, ctx, run_id, Command(resume=decisions))
        assert status(g, run_id, settings)["state"] == "done"
        return out


SLOW = {"c01_1": {"decision": "reject", "comment": "starts too slow"},
        "c02_1": {"decision": "reject", "comment": "starts too slow"},
        "c03_1": {"decision": "approve", "comment": "great hook"}}
ALL_OK = {p: {"decision": "approve"} for p in ("c01_1", "c02_1", "c03_1")}


def test_review_is_written_to_memory_and_a_rule_is_proposed(settings, mem, tmp_path, show_dir):
    out = reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r1", SLOW)
    w = out["memory_written"]
    assert (w["feedback_logged"], w["examples"], len(w["new_rule_ids"])) == (3, 3, 1)
    assert w["version_before"] == "v0" and w["version_after"] == "v1"
    [rule] = mem.rules()
    assert rule["status"] == "needs_review"                       # AI proposals wait for the editor
    assert (rule["scope"], rule["scope_value"]) == ("show", "Show A") and len(rule["source_feedback_ids"]) == 2
    fbs = mem.feedback()
    assert {f["decision"] for f in fbs} == {"reject", "approve"} and all(f["show"] == "Show A" for f in fbs)
    assert fbs[0]["promo"]["promo_id"] and "END OF PROMO" in fbs[0]["promo"]["promo_text"]
    assert {e["label"] for e in mem.examples()} == {"approved", "rejected"}


def test_next_run_reads_memory_only_after_the_editor_activates_the_rule(settings, mem, tmp_path, show_dir):
    reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r1", SLOW)
    ctx2 = ctx_for(settings, mem)
    out2 = reviewed_run(settings, tmp_path, show_dir, ctx2, "r2", ALL_OK)
    used = out2["memory_used"]
    assert used["enabled"] and used["rules"] == [] and used["guidelines_version"] == "v1"
    assert {e["label"] for e in used["examples"]} == {"approved", "rejected"}       # examples need no approval
    [rule] = mem.rules()
    mem.set_status(rule["id"], "active")                          # editor, in the Memory tab
    ctx3 = ctx_for(settings, mem)
    out3 = reviewed_run(settings, tmp_path, show_dir, ctx3, "r3", ALL_OK)
    assert [r["id"] for r in out3["memory_used"]["rules"]] == [rule["id"]]
    assert out3["memory_used"]["guidelines_version"] == "v2"         # v1 proposed, v2 activated by the editor
    assert any(r["id"] == rule["id"] for r in ctx3.agents.memory_seen["read_episode"]["rules"])  # injected into agents


def test_feedback_supporting_an_active_rule_raises_its_confidence(settings, mem, tmp_path, show_dir):
    reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r1", SLOW)
    [rule] = mem.rules()
    mem.set_status(rule["id"], "active")
    c0 = mem.rule(rule["id"])["confidence"]
    reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r2",
                 {**ALL_OK, "c01_1": {"decision": "reject", "comment": "starts too slow"}})
    r = mem.rule(rule["id"])
    assert r["supporting_count"] == 1 and r["confidence"] > c0


def test_memory_off_injects_nothing_but_still_records_feedback(settings, mem, tmp_path, show_dir):
    reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r1", SLOW)
    mem.set_status(mem.rules()[0]["id"], "active")
    ctx = ctx_for(settings, mem)
    out = reviewed_run(settings, tmp_path, show_dir, ctx, "r2", ALL_OK, memory=False)
    assert out["memory_used"]["enabled"] is False and out["memory_used"]["rules"] == []
    assert ctx.agents.memory_seen["read_episode"]["rules"] == []
    assert out["memory_written"]["feedback_logged"] == 3              # settings.memory.write_when_off


def test_memory_write_is_idempotent(settings, mem, tmp_path, show_dir):
    ctx = ctx_for(settings, mem)
    out = reviewed_run(settings, tmp_path, show_dir, ctx, "r1", SLOW)

    class RT:                                                          # re-run the node, as after a crash
        context = ctx
    again = nodes.memory_write(out, RT())
    assert again["memory_written"]["skipped"] == "already written"
    assert len(mem.feedback()) == 3 and len(mem.rules()) == 1 and len(mem.examples()) == 3


def test_learning_failure_keeps_the_feedback(settings, mem, tmp_path, show_dir):
    out = reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem, error_on={"learn": {1}}), "r1", SLOW)
    assert out["memory_written"]["feedback_logged"] == 3 and mem.rules() == []
    assert any("memory_write" in e for e in out["errors"])
    # the next run's learner still sees that feedback (unused_feedback) and can build the rule from it
    reviewed_run(settings, tmp_path, show_dir, ctx_for(settings, mem), "r2", ALL_OK)
    assert len(mem.rules()) == 1
