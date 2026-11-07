"""Streamlit Evaluation page: shows the evaluate/ reports (label files, tool vs labels,
labeller agreement). It only reads them; the reports are made by the evaluate CLIs, whose
real writers build the files used here."""
import pytest
import yaml

from app import data
from config.loader import ROOT, load_parser_config, load_settings
from evaluate import agreement, compare, labels
from parser.screenplay_parser import parse_ts


@pytest.fixture
def settings_file(tmp_path):
    raw = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    raw["paths"].update(output_dir=str(tmp_path / "output"), logs_dir=str(tmp_path / "logs"),
                        checkpoint_db=str(tmp_path / "ckpt.sqlite"))
    raw["memory"]["db_path"] = str(tmp_path / "memory.db")
    raw["evaluation"]["dir"] = str(tmp_path / "eval")             # never the real reports in tests
    f = tmp_path / "settings.yaml"
    f.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return f


def _m(i, ep, start, end, mtype="reveal"):
    return labels.LabelMoment(id=i, episode=ep, start=start, end=end, moment_type=mtype, note=f"note {i}")


def fill(settings):
    """A planted key, AI labels, their agreement report and one run evaluation."""
    pcfg = load_parser_config(settings=settings)
    to_s = lambda ts: parse_ts(ts, pcfg.compiled)
    key = labels.Labels(show="Glass", label_source="planted", labeled_by="planted by the test", episodes=[1, 2],
                        locked_at="2026-11-07 05:00:00", labeler_models=["google_genai:gemma-4-31b-it"],
                        moments=[_m("E01P1", 1, "00:10.00", "00:40.00"), _m("E02P1", 2, "00:05.00", "00:35.00", "romance")])
    ai = labels.Labels(show="Glass", label_source="ai", labeled_by="AI labeller (gemma)", episodes=[1, 2],
                       labeler_models=["google_genai:gemma-4-31b-it"],
                       moments=[_m("A01", 1, "00:10.00", "00:40.00"), _m("A02", 2, "01:00.00", "01:30.00")])
    d = settings.resolve(settings.evaluation.dir) / "labels"
    kp, ap = d / "glass.yaml", d / "glass-ai.yaml"
    labels.save_labels(key, kp)
    labels.save_labels(ai, ap)
    agreement.write_report(agreement.compare_labels(key, ai, settings, to_s),
                           agreement.report_path(settings, ap, kp), kp, ap)
    compare.write_report(compare.evaluate_run(key, [], [], [], {}, settings, to_s,
                                              run_models={"google_genai:gemma-4-31b-it"}), settings, "glass-run1")
    return kp, ap


# ------------------------------------------------------------------------ data helpers
def test_label_files_list_who_made_them(settings_file):
    s = load_settings(settings_file)
    fill(s)
    rows = {r["file"]: r for r in data.list_label_files(s)}
    assert set(rows) == {"glass.yaml", "glass-ai.yaml"}
    assert (rows["glass.yaml"]["source"], rows["glass.yaml"]["moments"], rows["glass.yaml"]["episodes"]) == ("planted", 2, "1-2")
    assert rows["glass.yaml"]["locked"] == "2026-11-07 05:00:00"
    assert rows["glass-ai.yaml"]["source"] == "ai" and rows["glass-ai.yaml"]["models"] == "google_genai:gemma-4-31b-it"
    assert rows["glass-ai.yaml"]["locked"] == ""


def test_agreements_and_run_evaluations_are_listed_and_broken_files_skipped(settings_file):
    s = load_settings(settings_file)
    fill(s)
    base = s.resolve(s.evaluation.dir)
    (base / "agreement" / "broken__vs__x.json").write_text("{not json", encoding="utf-8")
    (base / "labels" / "broken.yaml").write_text("moments: [", encoding="utf-8")
    [a] = data.list_agreements(s)
    assert a["name"] == "glass-ai__vs__glass" and a["report"]["recall"] == 0.5 and a["report"]["precision"] == 0.5
    [e] = data.list_run_evaluations(s)
    assert e["run_id"] == "glass-run1" and e["report"]["label_source"] == "planted"
    assert e["md"].name == "evaluation.md"
    assert [r["file"] for r in data.list_label_files(s)] == ["glass-ai.yaml", "glass.yaml"]


def test_nothing_to_show_gives_empty_lists(settings_file):
    s = load_settings(settings_file)
    assert data.list_label_files(s) == [] and data.list_agreements(s) == [] and data.list_run_evaluations(s) == []


# -------------------------------------------------------------------- headless page
def eval_app(settings_file, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PROMO_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    at.button(key="nav_evaluation").click().run()
    assert not at.exception, at.exception
    return at


def text(at) -> str:
    return " ".join(str(x.value) for x in [*at.markdown, *at.caption, *at.info, *at.warning, *at.success, *at.error])


def test_evaluation_page_shows_run_results_agreement_and_label_files(settings_file, monkeypatch):
    fill(load_settings(settings_file))
    at = eval_app(settings_file, monkeypatch)
    labels = [t.label for t in at.tabs]
    assert all(any(name in l for l in labels) for name in ("Tool vs labels", "Labeller agreement", "Label files"))
    body = text(at)
    assert "planted" in body.lower()                               # who made the answer key is always shown
    assert "Not independent" in body                               # the run shared a model with the key's writer
    assert "Too few reference moments" in body                     # the agreement verdict
    assert len(at.dataframe) >= 3                                  # per-label, per-episode, label files
    assert any(m.label == "Recall" and m.value == "50%" for m in at.metric)


def test_evaluation_page_explains_how_to_start_when_empty(settings_file, monkeypatch):
    at = eval_app(settings_file, monkeypatch)
    body = text(at)
    assert "evaluate.compare" in body and "evaluate.agreement" in body and "evaluate.labels" in body
