"""Phase 2: shared review logic (app/review.py) and the interactive terminal review (run.py --review)."""
import builtins
import json

import pytest

import run as cli
from app import review
from config.loader import load_settings
from graph.build import build_graph
from graph.runner import open_checkpointer, status
from synthetic import write_show


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.paths.logs_dir = str(tmp_path / "logs")
    s.paths.checkpoint_db = str(tmp_path / "ckpt.sqlite")
    return s


@pytest.fixture
def paused(settings, tmp_path):
    """A stub run (3 cutter promos, no joiner) paused at review round 1."""
    show = write_show(tmp_path / "show")
    code = cli.main(["--input", str(show), "--show", "Test Show", "--genre", "drama", "--agents", "stub",
                     "--run-id", "r1", "--no-joiner"], settings=settings)
    assert code == cli.EXIT_REVIEW
    return settings


def state(settings):
    with open_checkpointer(settings) as cp:
        return status(build_graph(cp), "r1", settings)


def resume_with(settings, decisions, tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps(decisions), encoding="utf-8")
    return cli.main(["--resume", "r1", "--decisions", str(f)], settings=settings)


# --------------------------------------------------------------------------- items
def test_items_for_a_first_round(paused):
    st = state(paused)
    items = review.review_items(st["interrupt"], st["values"])
    assert [i["promo_id"] for i in items] == ["c01_1", "c02_1", "c03_1"] and all(i["pending"] for i in items)
    it = items[0]
    assert it["options"] == ["approve", "reject", "fix"]                 # no keep_before without a fix
    assert "END OF PROMO" in it["transcript"] and it["scores"]["final"] == it["promo"]["scores"]["final"]
    assert it["before_after"] is None and it["round"] == 1


def test_before_after_and_validation_after_a_fix(paused, tmp_path):
    resume_with(paused, {"c01_1": {"decision": "fix", "comment": "Start later"},
                         "c02_1": {"decision": "approve"}, "c03_1": {"decision": "approve"}}, tmp_path)
    st = state(paused)
    items = review.review_items(st["interrupt"], st["values"])
    fixed = items[0]
    assert fixed["promo_id"] == "c01_1" and fixed["pending"] and fixed["round"] == 2
    assert fixed["options"] == ["approve", "reject", "fix", "keep_before"]
    ba = fixed["before_after"]
    assert ba["before"]["version"] == 2 and ba["after"]["version"] == 3
    assert "END OF PROMO" in ba["before_text"] and "END OF PROMO" in ba["after_text"]
    assert ba["editor_comment"] == "Start later" and ba["fix_interpretation"]
    assert [i["pending"] for i in items[1:]] == [False, False]          # already approved: read-only
    errs = review.validate(items, {"c01_1": {"decision": "fix"}, "c02_1": {"decision": "reject"},
                                   "zz": {"decision": "approve"}})
    assert len(errs) == 3 and any("comment" in e for e in errs)
    assert review.validate(items, {"c01_1": {"decision": "keep_before"}}) == []


def test_write_decisions(tmp_path):
    p = review.write_decisions(tmp_path, 2, {"a": {"decision": "approve", "comment": ""}})
    assert p.name == "review_round2.json" and json.loads(p.read_text(encoding="utf-8"))["a"]["decision"] == "approve"


# ------------------------------------------------------------- terminal review (run.py)
def scripted(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def test_terminal_review_fix_then_keep_before(paused, monkeypatch, capsys):
    # round 1: fix c01_1 (with comment), approve c02_1, reject c03_1 (with a reason); confirm
    # round 2: c01_1 shown before/after -> keep_before; confirm
    scripted(monkeypatch, "f", "End on the slap", "a", "r", "too slow to start", "y", "k", "y")
    assert cli.main(["--resume", "r1", "--review"], settings=paused) == cli.EXIT_DONE
    out = capsys.readouterr().out
    assert "Round 1" in out and "Round 2" in out and "BEFORE" in out and "AFTER" in out
    v = state(paused)["values"]
    assert v["editor_decisions"]["c01_1"]["kept_before"] is True
    assert [p["promo_id"] for p in v["promos"]] == ["c01_1", "c02_1"]   # c03_1 rejected


def test_terminal_review_quit_leaves_run_paused(paused, monkeypatch):
    scripted(monkeypatch, "q")
    assert cli.main(["--resume", "r1", "--review"], settings=paused) == cli.EXIT_REVIEW
    assert state(paused)["state"] == "awaiting_review"


def test_terminal_review_only_offers_allowed_options(paused, monkeypatch, capsys):
    # "k" (keep_before) isn't allowed in round 1 -> asked again; then approve all three
    scripted(monkeypatch, "k", "a", "a", "a", "y")
    assert cli.main(["--resume", "r1", "--review"], settings=paused) == cli.EXIT_DONE
    assert "not an option" in capsys.readouterr().out


def test_review_outputs_are_exported(paused, monkeypatch, tmp_path):
    d = paused.resolve(paused.paths.output_dir) / "runs" / "r1"
    tmpl = json.loads((d / "review_template.json").read_text(encoding="utf-8"))
    assert tmpl["c01_1"]["_allowed"] == ["approve", "reject", "fix"]
    scripted(monkeypatch, "f", "End on the slap", "a", "r", "too slow", "y", "a", "y")
    assert cli.main(["--resume", "r1", "--review"], settings=paused) == cli.EXIT_DONE
    log = json.loads((d / "raw" / "review_log.json").read_text(encoding="utf-8"))
    assert [(l["promo_id"], l["decision"], l["round"]) for l in log["decisions"]] == [
        ("c01_1", "fix", 1), ("c02_1", "approve", 1), ("c03_1", "reject", 1), ("c01_1", "approve", 2)]
    assert log["summary"]["rounds"] == 2
    report = (d / "report.md").read_text(encoding="utf-8")
    assert "## Editor review" in report and "too slow" in report and "End on the slap" in report
    summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["review"]["rounds"] == 2 and summary["evaluation_metrics"]["editor"]["first_pass_approval_rate"] == 0.333
