"""run.py end to end with stub agents on a synthetic show: outputs, review pause, resume."""
import csv
import json
import os
import signal

import pytest

import run as cli
from config.loader import load_settings
from llm.client import RateLimitExhausted
from llm.usage import usage_from_log
from synthetic import write_show

SPEC_FIELDS = {"promo_id", "segments", "total_duration_seconds", "hook", "cliffhanger", "why_it_works",
               "moment_type", "scores", "rank", "trace_id"}


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.paths.logs_dir = str(tmp_path / "logs")
    s.paths.checkpoint_db = str(tmp_path / "ckpt.sqlite")
    return s


@pytest.fixture
def show_dir(tmp_path):
    return write_show(tmp_path / "show")


def go(settings, *argv):
    return cli.main(list(argv), settings=settings)


def base(show_dir, *extra):
    return ["--input", str(show_dir), "--show", "Test Show", "--genre", "drama", "--agents", "stub",
            "--run-id", "r1", *extra]


def test_full_run_writes_every_output(settings, show_dir, tmp_path):
    assert go(settings, *base(show_dir, "--no-review")) == cli.EXIT_DONE
    d = tmp_path / "output" / "runs" / "r1"
    promos = json.loads((d / "promos.json").read_text(encoding="utf-8"))
    assert promos and SPEC_FIELDS <= set(promos[0]) and promos[0]["rank"] == 1
    assert {"hook", "cliffhanger", "standalone", "final"} <= set(promos[0]["scores"])
    assert (d / promos[0]["trace_file"]).exists()

    rows = list(csv.DictReader((d / "promos.csv").open(encoding="utf-8")))
    assert len(rows) == len(promos)
    assert {"rank", "promo_id", "segments", "score_hook", "score_final", "trace_id"} <= set(rows[0])

    # raw, unedited results: every candidate (ranked or not) with all its versions and evaluations
    raw = json.loads((d / "raw" / "candidates.json").read_text(encoding="utf-8"))
    assert {c["promo_id"] for c in raw} == {"c01_1", "c02_1", "c03_1", "j01"}
    c = next(c for c in raw if c["promo_id"] == "c01_1")
    assert [v["version"] for v in c["versions"]] == [1, 2] and len(c["evaluations"]) == 2
    assert c["ranking"]["outcome"] in {"ranked", "dropped"}
    assert len(list(csv.DictReader((d / "raw" / "candidates.csv").open(encoding="utf-8")))) == 4
    assert len(json.loads((d / "raw" / "moments.json").read_text(encoding="utf-8"))) == 6

    trace = json.loads((d / "traces" / "c01_1.json").read_text(encoding="utf-8"))
    kinds = [e["event"] for e in trace["events"]]
    assert kinds[0] == "moment" and "created" in kinds and "evaluated" in kinds and "revised" in kinds

    report = (d / "report.md").read_text(encoding="utf-8")
    assert "Test Show" in report and "c01_1" in report and "Dropped" in report

    summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["switches"]["editor_review"] is False
    assert summary["counts"]["episodes"] == 3 and summary["counts"]["ranked"] == len(promos)


def test_review_pause_then_resume_with_decisions_file(settings, show_dir, tmp_path):
    assert go(settings, *base(show_dir, "--no-joiner")) == cli.EXIT_REVIEW
    d = tmp_path / "output" / "runs" / "r1"
    template = json.loads((d / "review_template.json").read_text(encoding="utf-8"))
    assert set(template) == {"c01_1", "c02_1", "c03_1"} and template["c01_1"]["decision"] == "approve"
    assert (d / "promos.json").exists()                       # provisional outputs written at the pause

    template["c03_1"] = {"decision": "reject", "comment": "needs context"}
    f = tmp_path / "decisions.json"
    f.write_text(json.dumps(template), encoding="utf-8")
    assert go(settings, "--resume", "r1", "--decisions", str(f)) == cli.EXIT_DONE
    promos = json.loads((d / "promos.json").read_text(encoding="utf-8"))
    assert "c03_1" not in [p["promo_id"] for p in promos]


def test_quota_stop_writes_partial_outputs_and_resumes(settings, show_dir, tmp_path, monkeypatch):
    from agents.stubs import StubAgents
    real = StubAgents.cut_episode
    calls = {"n": 0}

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RateLimitExhausted("evaluator daily quota used up", daily=True)
        return real(self, *a, **kw)

    monkeypatch.setattr(StubAgents, "cut_episode", flaky)
    assert go(settings, *base(show_dir, "--no-review")) == cli.EXIT_STOPPED
    summary = json.loads((tmp_path / "output" / "runs" / "r1" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "stopped" and "daily quota" in summary["stop_reason"]
    assert go(settings, "--resume", "r1") == cli.EXIT_DONE      # resumes with the run's own agents (stub)
    assert not (tmp_path / "logs" / "llm" / "r1" / "calls.jsonl").exists()   # no real LLM call was made


def test_existing_run_id_needs_resume(settings, show_dir):
    assert go(settings, *base(show_dir, "--no-review")) == cli.EXIT_DONE
    with pytest.raises(SystemExit):
        go(settings, *base(show_dir, "--no-review"))


def test_run_already_running_is_refused(settings, show_dir, capsys):
    from graph.runner import holder, run_lock
    with run_lock(settings, "r1") as owner:
        assert holder(settings, "r1")["pid"] == owner["pid"]
        assert go(settings, *base(show_dir, "--no-review")) == cli.EXIT_LOCKED
        assert "already running" in capsys.readouterr().err
    assert holder(settings, "r1") is None                      # released: the run can go now
    assert go(settings, *base(show_dir, "--no-review")) == cli.EXIT_DONE
    assert holder(settings, "r1") is None                      # run.py releases it when it finishes
    with run_lock(settings, "r1"):
        assert go(settings, "--resume", "r1") == cli.EXIT_LOCKED


def test_lock_held_by_other_process_and_freed_when_it_is_killed(settings, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from graph.runner import RunLocked, holder, run_lock
    root = Path(cli.__file__).resolve().parent
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, sys.argv[1]);"
         "from config.loader import load_settings; from graph.runner import run_lock;"
         "s = load_settings(); s.paths.checkpoint_db = sys.argv[2]\n"
         "with run_lock(s, 'r1'):\n print('ready', flush=True); time.sleep(60)",
         str(root), settings.paths.checkpoint_db],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        pid = holder(settings, "r1")["pid"]        # the interpreter itself (a venv python.exe on Windows
        assert pid != os.getpid()                  # is a launcher, so not necessarily child.pid)
        with pytest.raises(RunLocked, match=str(pid)):
            with run_lock(settings, "r1"):
                pass
        os.kill(pid, signal.SIGTERM)               # abrupt (TerminateProcess on Windows): the OS must drop the lock
        child.wait(timeout=30)
    finally:
        child.kill()
    assert holder(settings, "r1") is None
    with run_lock(settings, "r1"):
        pass


@pytest.mark.parametrize("text, expect", [("1-3,5", [1, 2, 3, 5]), ("7", [7]), ("2,1-2", [1, 2])])
def test_episode_ranges(text, expect):
    assert cli.parse_episodes(text) == expect


def test_usage_from_log_sums_across_resumes(tmp_path):
    log = tmp_path / "calls.jsonl"
    rows = [{"agent": "reader", "provider": "p", "model": "a", "attempts": 2, "retries": 1, "success": True,
             "mode": "native", "input_tokens": 100, "output_tokens": 10, "rate_limit_waits": 0, "latency_s": 1.0,
             "model_switches": []},
            {"agent": "reader", "provider": "p", "model": "b", "attempts": 1, "retries": 0, "success": True,
             "mode": "fallback", "input_tokens": 50, "output_tokens": 5, "rate_limit_waits": 1, "latency_s": 2.0,
             "model_switches": [{"from": "p:a", "to": "p:b", "reason": "daily quota"}]}]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    u = usage_from_log(log)
    r = u["by_agent"]["reader"]
    assert r["calls"] == 2 and r["retries"] == 1 and r["input_tokens"] == 150 and r["models"] == {"p:a": 1, "p:b": 1}
    assert u["totals"]["model_switches"] == 1 and u["retry_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert u["model_switches"][0]["to"] == "p:b"
