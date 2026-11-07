"""Step 8: blind labelling, judging the tool's picks, comparison, per-run log, cost estimate."""
import json

import pytest

from config.loader import load_moment_types, load_parser_config, load_settings
from evaluate import compare, cost, judge, labels, runs
from parser.screenplay_parser import parse_show, parse_ts
from synthetic import write_show


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.paths.logs_dir = str(tmp_path / "logs")
    s.evaluation.dir = str(tmp_path / "eval")
    return s


@pytest.fixture
def pcfg(settings):
    return load_parser_config(settings=settings)


@pytest.fixture
def episodes(tmp_path, pcfg):
    show = parse_show(write_show(tmp_path / "show", (1, 2, 3)), pcfg)
    return {e.number: e for e in show.episodes}


def scripted(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def seg(ep, a, b):
    return {"episode": ep, "start_timestamp": a, "end_timestamp": b}


# ---------------------------------------------------------------------------- labelling
def test_label_session_validates_and_saves(settings, pcfg, episodes, tmp_path):
    lab = labels.Labels(show="Test Show")
    out = []
    saved = []
    labels.label_session(lab, episodes, [1, 2], settings, pcfg, load_moment_types(settings=settings),
                         ask=scripted("00:05.00 00:31.50", "00:04.00 00:31.50 slap the clash", "done", "skip"),
                         say=out.append, save=lambda l: saved.append(l.model_copy(deep=True)))
    text = "\n".join(map(str, out))
    assert "Line 1 of episode 1." in text                          # screenplay is shown...
    assert "00:05.00" in text and "00:04.00" in text               # ...bad start rejected with nearest
    [m] = lab.moments
    assert (m.id, m.episode, m.start, m.end, m.moment_type, m.note) == \
        ("L01", 1, "00:04.00", "00:31.50", "other", "slap the clash")   # no known type: all words are the note
    assert lab.episodes == [1]                                     # ep 2 skipped: not counted for recall
    assert saved and saved[-1].moments


def test_labels_roundtrip_and_lock(settings, tmp_path):
    p = tmp_path / "l.yaml"
    lab = labels.Labels(show="S", episodes=[1], moments=[labels.LabelMoment(id="L01", episode=1, start="00:04.00",
                                                                         end="00:31.50")])
    labels.save_labels(lab, p)
    again = labels.load_labels(p)
    assert again.moments[0].end == "00:31.50" and again.locked_at is None
    labels.lock(p)
    assert labels.load_labels(p).locked_at is not None
    with pytest.raises(SystemExit, match="locked"):
        labels.main(["--input", str(tmp_path), "--show", "S", "--episodes", "1", "--labels", str(p)], settings=settings)


# ------------------------------------------------------------------------------ judging
PROMOS = [{"promo_id": "c01_1", "rank": 1, "segments": [seg(1, "00:04.00", "00:31.50")], "hook": "h", "cliffhanger": "c",
           "total_duration_seconds": 27.5, "scores": {"final": 8}},
          {"promo_id": "c02_1", "rank": 2, "segments": [seg(2, "00:04.00", "00:31.50")], "hook": "h", "cliffhanger": "c",
           "total_duration_seconds": 27.5, "scores": {"final": 7}},
          {"promo_id": "c03_1", "rank": 3, "segments": [seg(3, "00:04.00", "00:31.50")], "hook": "h", "cliffhanger": "c",
           "total_duration_seconds": 27.5, "scores": {"final": 6}}]


def test_judge_session_records_posts_and_failures(settings, pcfg, episodes):
    js, out = {}, []
    cats = list(settings.evaluation.failure_categories)
    judge.judge_session(PROMOS, episodes, js, settings, pcfg, ask=scripted("y", "n", "1", "needs the wedding first", "s"),
                        say=out.append, save=lambda j: None)
    assert js["c01_1"]["post"] is True
    assert js["c02_1"] == {**js["c02_1"], "post": False, "category": cats[0], "note": "needs the wedding first"}
    assert "c03_1" not in js                                        # skipped: judge later
    assert any("END OF PROMO" in str(o) for o in out)               # shown as the viewer sees it
    # resume: already-judged promos are not asked again
    judge.judge_session(PROMOS, episodes, js, settings, pcfg, ask=scripted("y"), say=out.append, save=lambda j: None)
    assert js["c03_1"]["post"] is True


# ----------------------------------------------------------------------------- compare
def test_compare_funnel_post_rate_and_failure_log(settings, pcfg):
    to_s = lambda ts: parse_ts(ts, pcfg.compiled)
    lab = labels.Labels(show="S", episodes=[1, 2], moments=[
        labels.LabelMoment(id="L01", episode=1, start="00:04.00", end="00:31.50"),   # found all the way
        labels.LabelMoment(id="L02", episode=2, start="00:40.00", end="00:55.50"),   # reader only
        labels.LabelMoment(id="L03", episode=2, start="00:00.00", end="00:15.50")])  # missed
    moments = [{"moment_id": "m01_h1", "episode": 1, "timestamp": "00:04.00"},
               {"moment_id": "m02_c1", "episode": 2, "timestamp": "00:44.00"},
               {"moment_id": "m03_h1", "episode": 3, "timestamp": "00:04.00"}]          # unlabeled episode
    cands = [{"promo_id": "c01_1", "versions": [{"version": 1, "segments": [seg(1, "00:08.00", "00:31.50")]}],
              "final_version": 1, "ranking": {"outcome": "ranked", "rank": 1}},
             {"promo_id": "c02_1", "versions": [{"version": 1, "segments": [seg(2, "00:20.00", "00:35.50")]}],
              "final_version": 1, "ranking": {"outcome": "dropped", "reason": "below_min_score"}}]
    promos = [{"promo_id": "c01_1", "rank": 1, "segments": [seg(1, "00:08.00", "00:31.50")]}]
    judgements = {"c01_1": {"post": True}, "c02_1": {"post": False, "category": "resolves_too_much"},
                  "c09_1": {"post": False, "category": "needs_context"}}
    r = compare.evaluate_run(lab, moments, cands, promos, judgements, settings, to_s)
    f = r["recall"]
    assert (f["labels"], f["reader"], f["candidates"], f["ranked"]) == (3, 2, 1, 1)
    assert r["per_label"]["L01"]["ranked_by"] == "c01_1" and r["per_label"]["L03"]["stage_lost"] == "reader"
    assert r["per_label"]["L02"]["stage_lost"] == "cutter"
    assert r["post_rate"] == {"judged": 3, "would_post": 1, "rate": pytest.approx(1 / 3, abs=1e-3)}
    assert r["failures"]["by_category"] == {"resolves_too_much": 1, "needs_context": 1}
    paths = compare.write_report(r, settings, "run1")
    assert "Recall" in (paths["md"]).read_text(encoding="utf-8")
    assert len((paths["failures_csv"]).read_text(encoding="utf-8").strip().splitlines()) == 3


# --------------------------------------------------------------------------- run log
def summary(run_id, memory, decisions, retry_rate, rules=0):
    return {"run_id": run_id, "show": "S", "written_at": "2026-11-06 10:00:00", "switches": {"memory": memory},
            "llm_usage": {"retry_rate": retry_rate, "totals": {"calls": 10, "input_tokens": 100, "output_tokens": 10}},
            "counts": {"ranked": 5},
            "evaluation_metrics": {"memory_enabled": memory, "guidelines_version": "v2" if memory else None,
                                   "rules_injected": rules, "examples_injected": 0,
                                   "editor": {"decided": len(decisions), "approved": decisions.count("approve"),
                                              "first_pass_approval_rate": decisions.count("approve") / len(decisions)
                                              if decisions else None}}}


def test_runs_log_compares_approval_with_and_without_memory(settings, tmp_path):
    rows = [runs.row(summary("a", True, ["approve", "approve", "reject"], 0.1, rules=4)),
            runs.row(summary("b", False, ["approve", "reject", "reject", "fix"], 0.3)),
            runs.row(summary("c", False, [], 0.2))]
    cmp_ = runs.memory_comparison(rows)
    assert cmp_["with_memory"]["runs"] == 1 and cmp_["with_memory"]["approval_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert cmp_["without_memory"]["runs_with_review"] == 1 and cmp_["without_memory"]["approval_rate"] == 0.25
    assert rows[0]["rules_injected"] == 4 and rows[1]["retry_rate"] == 0.3


def test_export_writes_evaluation_metrics(tmp_path):
    from graph.export import evaluation_metrics
    v = {"switches": {"memory": True}, "memory_used": {"enabled": True, "guidelines_version": "v3",
                                                        "rules": [1, 2], "examples": [1]},
         "lessons": [{"promo_id": "a", "decision": "approve", "round": 1},
                     {"promo_id": "b", "decision": "fix", "round": 1},
                     {"promo_id": "c", "decision": "reject", "round": 1},
                     {"promo_id": "b", "decision": "approve", "round": 2}]}
    m = evaluation_metrics(v, {"retry_rate": 0.2})
    assert m["memory_enabled"] and m["guidelines_version"] == "v3" and m["rules_injected"] == 2
    e = m["editor"]
    assert e["decided"] == 3 and e["first_pass_approval_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert e["final_approval_rate"] == pytest.approx(2 / 3, abs=1e-3) and m["retry_rate"] == 0.2


# ------------------------------------------------------------------------------- cost
USAGE = {"by_agent": {
    "reader": {"calls": 50, "input_tokens": 100_000, "output_tokens": 20_000,
               "models": {"google_genai:gemini-3.5-flash-lite": 50}},
    "evaluator": {"calls": 150, "input_tokens": 200_000, "output_tokens": 100_000,
                  "models": {"groq:openai/gpt-oss-120b": 100, "groq:openai/gpt-oss-20b": 50}}},
    "totals": {"calls": 200, "input_tokens": 300_000, "output_tokens": 120_000}}


def test_cost_per_show_and_week(settings):
    settings.cost.shows_per_week = 4
    e = cost.estimate(USAGE, episodes_in_run=50, settings=settings, episodes_per_show=50)
    assert e["per_show"]["calls"] == 200 and e["per_week"]["calls"] == 800
    assert e["per_show"]["input_tokens"] == 300_000
    # paid, priced at each call's own model: reader on flash-lite, evaluator split 2/3 : 1/3
    p = settings.cost.paid_prices_per_m
    fl, big, small = p["google_genai:gemini-3.5-flash-lite"], p["groq:openai/gpt-oss-120b"], p["groq:openai/gpt-oss-20b"]
    want = (100_000 * fl.input + 20_000 * fl.output) / 1e6 + \
        (200_000 * 2 / 3 * big.input + 100_000 * 2 / 3 * big.output) / 1e6 + \
        (200_000 / 3 * small.input + 100_000 / 3 * small.output) / 1e6
    assert e["paid"]["as_run"]["per_show_usd"] == pytest.approx(want, rel=1e-3)
    assert e["paid"]["as_run"]["per_week_usd"] == pytest.approx(4 * want, rel=1e-3)
    one = e["paid"]["single_model"]["google_genai:gemini-3.6-flash"]
    assert one["per_show_usd"] == pytest.approx((300_000 * 0.75 + 120_000 * 3.75) / 1e6, rel=1e-3)
    assert e["free"]["usd"] == 0 and e["free"]["days_per_show"] >= 1


def test_cost_scales_with_episodes(settings):
    e = cost.estimate(USAGE, episodes_in_run=10, settings=settings, episodes_per_show=50)
    assert e["per_show"]["calls"] == 1000                           # 10-episode run scaled to 50 episodes


def test_free_tier_capacity_uses_each_agents_whole_model_chain(settings):
    # evaluator: 300K tokens/show; its chain (gpt-oss-120b -> gpt-oss-20b -> qwen) has 3 x 200K tokens/day
    e = cost.estimate(USAGE, episodes_in_run=50, settings=settings, episodes_per_show=50)
    ev = e["free"]["by_agent"]["evaluator"]
    assert ev["chain"] == [f"{c.provider}:{c.model}" for c in settings.models["evaluator"].chain()]
    assert ev["tokens_per_day"] == 600_000 and ev["shows_per_day"] == pytest.approx(2.0)
    assert e["free"]["by_agent"]["reader"]["unknown_limit_in_chain"] is True     # flash-lite: unpublished
    assert e["free"]["bottleneck_agent"] == "evaluator" and e["free"]["shows_per_week"] == pytest.approx(14.0)
