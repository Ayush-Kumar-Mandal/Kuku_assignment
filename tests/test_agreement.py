"""Agreement between two label sets (evaluate.agreement): is the AI labeller trustworthy?"""
import pytest

from config.loader import load_parser_config, load_settings
from evaluate import agreement, compare, labels
from parser.screenplay_parser import parse_ts


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.evaluation.dir = str(tmp_path / "eval")
    s.evaluation.agreement.min_reference_moments = 3
    return s


@pytest.fixture
def to_s(settings):
    pcfg = load_parser_config(settings=settings)
    return lambda ts: parse_ts(ts, pcfg.compiled)


def L(i, ep, a, b, src="human"):
    return labels.LabelMoment(id=i, episode=ep, start=a, end=b)


def labs(src, episodes, moments, by=None):
    return labels.Labels(show="S", label_source=src, labeled_by=by or src, episodes=episodes, moments=moments)


REF = labs("human", [1, 2, 3], [L("L01", 1, "00:00.00", "00:30.00"), L("L02", 1, "00:40.00", "01:00.00"),
                                L("L03", 2, "00:10.00", "00:40.00"), L("L04", 3, "00:00.00", "00:25.00")])


def test_one_to_one_matching_recall_precision_f1(settings, to_s):
    cand = labs("ai", [1, 2, 3], [
        L("A01", 1, "00:05.00", "00:30.00"),    # matches L01 (25s shared / 25s shorter = 1.0)
        L("A02", 1, "00:10.00", "00:40.00"),    # also overlaps L01 (20/30 = 0.67), but A01 fits it better
        L("A03", 2, "00:30.00", "00:55.00"),    # 10s of 25s with L03 = 0.4 < 0.5: no match
        L("A04", 3, "00:00.00", "00:20.00")])   # matches L04
    r = agreement.compare_labels(REF, cand, settings, to_s)
    assert [(p["reference"], p["candidate"]) for p in r["pairs"]] == [("L01", "A01"), ("L04", "A04")]
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["f1"] == 0.5
    assert sorted(m["id"] for m in r["missed"]) == ["L02", "L03"]
    assert sorted(m["id"] for m in r["extra"]) == ["A02", "A03"]
    assert r["per_episode"][1] == {"reference": 2, "candidate": 2, "matched": 1}


def test_only_episodes_finished_in_both_count(settings, to_s):
    cand = labs("ai", [1, 4], [L("A01", 1, "00:00.00", "00:30.00"), L("A02", 1, "00:40.00", "01:00.00"),
                               L("A03", 4, "00:00.00", "00:30.00")])
    r = agreement.compare_labels(REF, cand, settings, to_s)
    assert r["episodes"] == [1]
    assert (r["reference_moments"], r["candidate_moments"], r["recall"], r["precision"]) == (2, 2, 1.0, 1.0)


def test_verdicts(settings, to_s):
    good = labs("ai", [1, 2, 3], [L("A1", 1, "00:00.00", "00:30.00"), L("A2", 1, "00:40.00", "01:00.00"),
                                  L("A3", 2, "00:10.00", "00:40.00")])
    assert agreement.compare_labels(REF, good, settings, to_s)["verdict"]["key"] == "trust"
    poor = labs("ai", [1, 2, 3], [L("A1", 1, "00:00.00", "00:30.00")])
    assert agreement.compare_labels(REF, poor, settings, to_s)["verdict"]["key"] == "dont_rely"
    settings.evaluation.agreement.min_reference_moments = 10
    v = agreement.compare_labels(REF, good, settings, to_s)["verdict"]
    assert v["key"] == "too_few" and "10" in v["text"]
    none = labs("ai", [7], [])
    assert agreement.compare_labels(REF, none, settings, to_s)["verdict"]["key"] == "no_overlap"


def test_reference_that_is_itself_ai_is_flagged(settings, to_s):
    r = agreement.compare_labels(REF.model_copy(update={"label_source": "ai", "labeled_by": "claude (AI)"}),
                                 REF.model_copy(update={"label_source": "ai"}), settings, to_s)
    assert r["reference_source"] == "ai"
    assert any("AI-vs-AI" in w for w in r["warnings"])


def test_cli_writes_report_and_compare_quotes_it(settings, tmp_path, to_s):
    ref_p, cand_p = tmp_path / "s.yaml", tmp_path / "s-ai.yaml"
    labels.save_labels(REF, ref_p)
    labels.save_labels(labs("ai", [1, 2, 3], [L("A1", 1, "00:00.00", "00:30.00")], by="AI labeller (gemma)"), cand_p)
    assert agreement.main(["--reference", str(ref_p), "--candidate", str(cand_p)], settings=settings) == 0
    out = agreement.report_path(settings, cand_p, ref_p)
    assert out.with_suffix(".md").exists() and out.with_suffix(".json").exists()
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "Recall" in md and "L02" in md
    # evaluate.compare finds the agreement for these AI labels and quotes it in its report
    found = agreement.latest_for(settings, cand_p)
    assert found and found["recall"] == 0.25
    r = compare.evaluate_run(labels.load_labels(cand_p), [], [], [], {}, settings, to_s)
    r["agreement"] = agreement.summary_line(found)
    text = compare.write_report(r, settings, "run1")["md"].read_text(encoding="utf-8")
    assert "Checked against" in text and "25%" in text
