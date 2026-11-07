"""AI labeller (evaluate.ai_labels): blind, checked, recorded. Fake chat model, no network."""
import pytest

from config.loader import load_moment_types, load_parser_config, load_settings
from evaluate import ai_labels, compare, labels
from llm.client import LLMClient
from parser.screenplay_parser import parse_show, parse_ts
from synthetic import write_show
from test_agents import RouterChat


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.paths.logs_dir = str(tmp_path / "logs")
    s.evaluation.dir = str(tmp_path / "eval")
    return s


@pytest.fixture
def episodes(tmp_path, settings):
    show = parse_show(write_show(tmp_path / "show", (1, 2, 3)), load_parser_config(settings=settings))
    return {e.number: e for e in show.episodes}


def mom(start, end, mtype="confrontation"):
    return {"start": start, "end": end, "moment_type": mtype, "note": "Mara accuses Tomas", "why": "opens on a clash"}


def label(settings, episodes, script, todo, lab=None):
    fake = RouterChat({"LabelerOutput": script})
    client = LLMClient(settings, run_id="lab", model_factory=lambda spec, st: fake, sleep=lambda s: None)
    lab = lab or labels.Labels(show="Test Show", label_source="ai")
    out = ai_labels.label_episodes(client, settings, load_parser_config(settings=settings),
                                   load_moment_types(settings=settings), episodes, todo, lab,
                                   show="Test Show", genre="romance", say=lambda *_: None)
    return out, fake


def test_labeller_sees_the_screenplay_and_story_and_records_its_model(settings, episodes):
    lab, fake = label(settings, episodes, [{"moments": [mom("00:04.00", "00:31.50")]}, {"moments": []}], [2, 3])
    prompt = fake.text("LabelerOutput", 0)
    assert "Line 1 of episode 2." in prompt                       # the screenplay, as a human labeller sees it
    assert "Synthetic episode 1." in prompt                       # earlier loglines = story so far
    assert "Line 1 of episode 3." not in prompt                   # never a later episode
    assert lab.episodes == [2, 3]                                 # an empty answer still finishes the episode
    [m] = lab.moments
    assert (m.episode, m.start, m.end, m.moment_type) == (2, "00:04.00", "00:31.50", "confrontation")
    assert m.why == "opens on a clash"
    assert lab.label_source == "ai" and lab.labeler_models == ["google_genai:gemma-4-31b-it"]
    assert lab.prompt_version == 1 and "NOT a human" in lab.labeled_by


def test_invented_timestamp_and_bad_length_trigger_a_retry(settings, episodes):
    lab, fake = label(settings, episodes, [{"moments": [mom("00:05.00", "00:31.50"), mom("00:40.00", "00:47.50")]},
                                           {"moments": [mom("00:04.00", "00:31.50")]}], [1])
    retry = fake.text("LabelerOutput", 1)
    assert "00:05.00 is not a line start" in retry                # the client's retry message carries our checks
    assert "7.5s" in retry and "20-60s" in retry
    assert [(m.start, m.end) for m in lab.moments] == [("00:04.00", "00:31.50")]


def test_overlapping_moments_and_too_many_are_rejected(settings, episodes):
    settings.evaluation.ai_labeler.max_per_episode = 1
    lab, fake = label(settings, episodes, [{"moments": [mom("00:04.00", "00:31.50"), mom("00:08.00", "00:31.50")]},
                                           {"moments": [mom("00:04.00", "00:31.50")]}], [1])
    retry = fake.text("LabelerOutput", 1)
    assert "at most 1" in retry and "overlap" in retry
    assert len(lab.moments) == 1


def test_unknown_type_becomes_the_default(settings, episodes):
    lab, _ = label(settings, episodes, [{"moments": [mom("00:04.00", "00:31.50", "slapfest")]}], [1])
    assert lab.moments[0].moment_type == load_moment_types(settings=settings).default_type


def test_cli_writes_a_separate_locked_ai_file_and_refuses_human_files(settings, tmp_path, monkeypatch):
    show_dir = write_show(tmp_path / "show2", (1,))
    fake = RouterChat({"LabelerOutput": [{"moments": [mom("00:04.00", "00:31.50")]}]})
    monkeypatch.setattr(ai_labels, "make_client",
                        lambda s, run_id: LLMClient(s, run_id=run_id, model_factory=lambda spec, st: fake,
                                                    sleep=lambda x: None))
    assert ai_labels.main(["--input", str(show_dir), "--show", "Test Show", "--episodes", "1"], settings=settings) == 0
    path = ai_labels.default_path(settings, "Test Show")
    assert path.name == "test-show-ai.yaml" and path != labels.default_path(settings, "Test Show")
    saved = labels.load_labels(path)
    assert saved.locked_at and saved.label_source == "ai" and saved.episodes == [1]
    with pytest.raises(SystemExit, match="locked"):
        ai_labels.main(["--input", str(show_dir), "--show", "Test Show", "--episodes", "1"], settings=settings)
    human = tmp_path / "human.yaml"
    labels.save_labels(labels.Labels(show="Test Show"), human)
    with pytest.raises(SystemExit, match="not an AI labels file"):
        ai_labels.main(["--input", str(show_dir), "--show", "Test Show", "--episodes", "1", "--labels", str(human)],
                       settings=settings)


def test_cli_exits_3_and_leaves_the_file_unlocked_when_it_stops_early(settings, tmp_path, monkeypatch):
    """Same code as run.py's 'stopped (resumable)', so a script can tell it must re-run."""
    show_dir = write_show(tmp_path / "show3", (1, 2))
    monkeypatch.setattr(ai_labels, "make_client", lambda s, run_id: LLMClient(
        s, run_id=run_id, model_factory=lambda spec, st: RouterChat({}), sleep=lambda x: None))
    monkeypatch.setattr(ai_labels, "label_episodes", lambda *a, **k: None)     # quota ran out before any episode
    assert ai_labels.main(["--input", str(show_dir), "--show", "Test Show", "--episodes", "1-2"], settings=settings) == 3
    assert not labels.load_labels(ai_labels.default_path(settings, "Test Show")).locked_at


# ------------------------------------------------------------- compare: who made the labels
def _eval(settings, lab, run_models):
    pcfg = load_parser_config(settings=settings)
    return compare.evaluate_run(lab, [], [], [], {}, settings, lambda ts: parse_ts(ts, pcfg.compiled),
                                run_models=run_models)


def test_compare_warns_when_the_labeller_shares_a_model_with_the_run(settings):
    lab = labels.Labels(show="S", label_source="ai", labeler_models=["groq:openai/gpt-oss-120b"], labeled_by="AI")
    r = _eval(settings, lab, {"groq:openai/gpt-oss-120b", "google_genai:gemini-3.6-flash"})
    assert r["independence"]["shared_with_run"] == ["groq:openai/gpt-oss-120b"]
    md = compare.write_report(r, settings, "run1")["md"].read_text(encoding="utf-8")
    assert "NOT hand labels" in md and "same model" in md
    clean = _eval(settings, lab.model_copy(update={"labeler_models": ["google_genai:gemma-4-31b-it"]}),
                  {"google_genai:gemini-3.6-flash"})
    assert clean["independence"]["shared_with_run"] == []
    assert "same model" not in compare.write_report(clean, settings, "run2")["md"].read_text(encoding="utf-8")


def test_compare_flags_a_planted_key_written_by_one_of_the_runs_models(settings):
    lab = labels.Labels(show="S", label_source="planted", labeled_by="planted", labeler_models=["google_genai:x"])
    md = compare.write_report(_eval(settings, lab, {"google_genai:x"}), settings, "run4")["md"].read_text(encoding="utf-8")
    assert "Not independent" in md and "wrote the answer key" in md


def test_compare_describes_planted_answer_keys(settings):
    r = _eval(settings, labels.Labels(show="S", label_source="planted", labeled_by="synthetic generator"), set())
    md = compare.write_report(r, settings, "run3")["md"].read_text(encoding="utf-8")
    assert "planted" in md and "NOT hand labels" not in md
