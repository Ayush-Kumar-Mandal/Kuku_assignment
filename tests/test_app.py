"""Streamlit app: data layer (pure functions + run launcher) and a headless AppTest smoke test."""
import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from app import data
from config.loader import ROOT, load_settings
from synthetic import write_show


@pytest.fixture
def settings_file(tmp_path):
    """A full settings copy whose output/logs/checkpoint paths point into tmp_path."""
    raw = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    raw["paths"].update(output_dir=str(tmp_path / "output"), logs_dir=str(tmp_path / "logs"),
                        checkpoint_db=str(tmp_path / "ckpt.sqlite"))
    raw["memory"]["db_path"] = str(tmp_path / "memory.db")       # never the real editor memory in tests
    f = tmp_path / "settings.yaml"
    f.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return f


def wait(proc, timeout=90):
    t = time.time()
    while proc.poll() is None and time.time() - t < timeout:
        time.sleep(0.3)
    return proc.poll()


# ------------------------------------------------------------------------ pure helpers
def test_model_choices_come_from_config():
    s = load_settings()
    choices = data.model_choices(s)
    for spec in s.models.values():
        for c in spec.chain():
            assert f"{c.provider}:{c.model}" in choices
    assert choices == sorted(set(choices))


def test_model_override_puts_choice_first_and_keeps_configured_backups():
    from config.loader import apply_model_overrides
    base = load_settings()
    s = apply_model_overrides(base, {"evaluator": "groq:qwen/qwen3.8-27b"})
    ev = s.models["evaluator"]
    assert (ev.provider, ev.model) == ("groq", "qwen/qwen3.8-27b")
    assert [f.model for f in ev.fallbacks] == [f.model for f in base.models["evaluator"].fallbacks
                                               if f.model != "qwen/qwen3.8-27b"]   # not its own backup
    assert s.models["reader"] == base.models["reader"] and base.models["evaluator"].model == "openai/gpt-oss-120b"


def test_upload_names_are_sanitised(tmp_path):
    class F:
        def __init__(self, name, body): self.name, self._b = name, body
        def getvalue(self): return self._b
    dest = data.save_uploads([F("../../evil.txt", b"x"), F("sub/dir/ep 1.txt", b"y")], tmp_path / "up")
    assert sorted(p.name for p in dest.iterdir()) == ["ep 1.txt", "evil.txt"]
    assert not (tmp_path / "evil.txt").exists()


def test_progress_from_state_values():
    v = {"episodes": [{"number": n} for n in range(1, 11)], "reader_cursor": 4, "cutter_cursor": 0,
         "candidates": {}, "evaluations": {}}
    p = data.progress(v, ["reader"])
    assert p["stage"] == "reader" and p["read"] == 4 and p["total"] == 10 and 0 < p["fraction"] < 0.5
    v.update(reader_cursor=10, cutter_cursor=10, joiner_done=True, candidates={"a": {}, "b": {}},
             evaluations={"a": [{}]})
    p = data.progress(v, ["evaluator"])
    assert p["stage"] == "evaluator" and p["evaluated"] == 1 and p["fraction"] > 0.7


def test_pid_alive():
    assert data.pid_alive(os.getpid()) is True
    assert data.pid_alive(999999) is False


def _fake_launch(settings_file, run_id, started_at):
    """launch.json naming a live process that is NOT the run (Windows reuses process ids)."""
    s = load_settings(settings_file)
    d = data.run_dir(s, run_id)
    d.mkdir(parents=True)
    (d / "launch.json").write_text(json.dumps({"pid": os.getpid(), "started_at": started_at}), encoding="utf-8")
    return s


def test_reused_pid_of_an_old_launch_is_not_a_running_run(settings_file):
    # Real case: a run launched at 12:13 had exited; at 15:35 Windows gave its pid to the
    # Streamlit app, so the Review tab showed "the run is working" forever.
    s = _fake_launch(settings_file, "old", "2020-01-01 00:00:00")
    assert data.is_running(s, "old") is False
    assert data.stop(s, "old") is False              # and Stop must not kill that unrelated process


def test_a_just_launched_run_counts_as_running_before_it_takes_its_lock(settings_file):
    s = _fake_launch(settings_file, "new", time.strftime("%Y-%m-%d %H:%M:%S"))
    assert data.is_running(s, "new") is True


# --------------------------------------------------------------- launcher (real process)
def test_launch_stub_run_then_list_and_load_it(settings_file, tmp_path):
    show = write_show(tmp_path / "show")
    proc = data.launch(settings_file, run_id="ui1", input_dir=show, show="Test Show", genre="drama",
                       agents="stub", review=False)
    assert wait(proc) == 0, (tmp_path / "logs" / "runs" / "ui1.log").read_text(encoding="utf-8")[-2000:]
    s = load_settings(settings_file)
    runs = data.list_runs(s)
    assert [r["run_id"] for r in runs] == ["ui1"] and runs[0]["state"] == "done"
    run = data.load_run(s, "ui1")
    assert run["promos"] and run["summary"]["counts"]["episodes"] == 3
    tr = data.load_trace(s, "ui1", run["promos"][0]["promo_id"])
    assert tr["events"] and tr["versions"]


def test_model_choice_is_kept_when_the_run_resumes(settings_file, tmp_path):
    show = write_show(tmp_path / "show")
    proc = data.launch(settings_file, run_id="ui2", input_dir=show, show="Test Show", genre="drama",
                       agents="stub", review=True, models={"evaluator": "groq:qwen/qwen3.8-27b"})
    assert "evaluator=groq:qwen/qwen3.8-27b" in proc.args
    assert wait(proc) == 2                                          # paused for review
    d = tmp_path / "output" / "runs" / "ui2"
    decisions = tmp_path / "d.json"
    decisions.write_text((d / "review_template.json").read_text(encoding="utf-8"), encoding="utf-8")
    proc = data.resume(settings_file, "ui2", decisions=decisions)
    assert not any(a.startswith("evaluator=") for a in proc.args)   # not passed again...
    assert wait(proc) == 0
    summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done"
    assert summary["model_overrides"] == {"evaluator": "groq:qwen/qwen3.8-27b"}   # ...the run kept it


# -------------------------------------------------------------------- headless app
def test_app_renders_run_and_promos(settings_file, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    show = write_show(tmp_path / "show")
    assert wait(data.launch(settings_file, run_id="ui3", input_dir=show, show="Test Show", genre="drama",
                            agents="stub", review=False)) == 0
    monkeypatch.setenv("PROMO_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("ui3" in str(o) for o in at.sidebar.selectbox[0].options)
    at.sidebar.selectbox[0].select("ui3").run()
    assert not at.exception, at.exception
    cards = [b for b in at.button if (b.key or "").startswith("det_")]
    assert cards                                                   # ranked promos as cards (default view)
    cards[0].click().run()                                         # Details opens that promo's trace
    assert not at.exception, at.exception
    assert any("timeline" in s.value or cards[0].key[4:] in s.value for s in at.subheader)
    assert any("Download" in b.label for b in at.get("download_button"))
    at.session_state["promo_view"] = "Table"
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1                                  # ...or as a table


# ------------------------------------------------------------- Phase 2: Review tab
def test_review_tab_submits_decisions_and_resumes(settings_file, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    show = write_show(tmp_path / "show")
    assert wait(data.launch(settings_file, run_id="rv1", input_dir=show, show="Test Show", genre="drama",
                            agents="stub", review=True, joiner=False)) == 2          # paused for review
    monkeypatch.setenv("PROMO_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    at.sidebar.selectbox[0].select("rv1").run()
    assert not at.exception, at.exception
    assert any("Round 1" in s.value for s in at.subheader)
    radios = [r for r in at.radio if r.key and r.key.startswith("dec_1_")]
    assert len(radios) == 3 and "keep_before" not in radios[0].options              # no fix yet
    for r, choice in zip(radios, ["approve", "reject", "approve"]):
        r.set_value(choice)
    next(b for b in at.button if b.label == "Submit decisions").click().run()
    assert not at.exception, at.exception
    d = tmp_path / "output" / "runs" / "rv1"
    sent = json.loads((d / "review_round1.json").read_text(encoding="utf-8"))
    assert {k: v["decision"] for k, v in sent.items()} == {"c01_1": "approve", "c02_1": "reject", "c03_1": "approve"}
    t = time.time()                                  # the app resumed the run in the background
    while time.time() - t < 60 and (data._read_json(d / "run_summary.json") or {}).get("state") != "done":
        time.sleep(0.5)
    summary = data._read_json(d / "run_summary.json")
    assert summary["state"] == "done" and summary["evaluation_metrics"]["editor"]["decided"] == 3


def test_review_tab_blocks_fix_without_comment(settings_file, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    show = write_show(tmp_path / "show")
    assert wait(data.launch(settings_file, run_id="rv2", input_dir=show, show="Test Show", genre="drama",
                            agents="stub", review=True, joiner=False)) == 2
    monkeypatch.setenv("PROMO_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    at.sidebar.selectbox[0].select("rv2").run()
    [r for r in at.radio if r.key == "dec_1_c01_1"][0].set_value("fix")
    next(b for b in at.button if b.label == "Submit decisions").click().run()
    assert any("comment" in e.value for e in at.error)
    assert not (tmp_path / "output" / "runs" / "rv2" / "review_round1.json").exists()   # nothing sent



# ------------------------------------------------------------- Phase 3: Memory page
def memory_app(settings_file, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PROMO_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    at.button(key="nav_memory").click().run()
    assert not at.exception, at.exception
    return at


def filled_memory(settings_file):
    from memory.store import MemoryStore
    mem = MemoryStore(load_settings(settings_file).resolve(load_settings(settings_file).memory.db_path))
    ids = [mem.log_feedback(show="S", genre="romance", promo_id=f"c0{i}_1", promo={"promo_id": f"c0{i}_1"},
                            moment_type="reveal", decision="reject", comment="starts on scenery", guidelines_version="v0",
                            run_id="r1", round=1) for i in (1, 2)]
    rid = mem.add_rule("Open on open conflict, never on scenery.", scope="show", scope_value="S",
                       source_feedback_ids=ids, status="needs_review")
    return mem, rid


def test_memory_page_lets_the_editor_activate_a_proposed_rule(settings_file, monkeypatch):
    mem, rid = filled_memory(settings_file)
    at = memory_app(settings_file, monkeypatch)
    assert any("Waiting for you (1)" in t.label for t in at.tabs)
    assert any("Open on open conflict" in str(t.value) for t in at.text_area)
    next(b for b in at.button if b.key == f"activate_{rid}").click().run()
    assert not at.exception, at.exception
    assert mem.rule(rid)["status"] == "active" and mem.version() == 2


def test_memory_page_edit_retire_and_delete_need_the_editor(settings_file, monkeypatch):
    mem, rid = filled_memory(settings_file)
    at = memory_app(settings_file, monkeypatch)
    [t for t in at.text_area if t.key == f"text_{rid}"][0].set_value("Open on the first clash.").run()   # leaving the box reruns
    next(b for b in at.button if b.key == f"save_{rid}").click().run()
    assert mem.rule(rid)["text"] == "Open on the first clash."
    next(b for b in at.button if b.key == f"delete_{rid}").click().run()   # without ticking "confirm"
    assert mem.rule(rid) is not None and any("confirm" in w.value.lower() for w in at.warning)
    [c for c in at.checkbox if c.key == f"confirm_{rid}"][0].check().run()
    next(b for b in at.button if b.key == f"delete_{rid}").click().run()
    assert mem.rule(rid) is None


def test_memory_page_editor_can_add_a_rule(settings_file, monkeypatch):
    from memory.store import MemoryStore
    at = memory_app(settings_file, monkeypatch)
    at.text_input(key="new_rule_text").set_value("Never end on a calm moment.")
    next(b for b in at.button if b.label == "Add rule").click().run()
    mem = MemoryStore(load_settings(settings_file).resolve(load_settings(settings_file).memory.db_path))
    [r] = mem.rules()
    assert (r["text"], r["status"], r["created_by"], r["scope"]) == ("Never end on a calm moment.", "active", "editor", "global")
