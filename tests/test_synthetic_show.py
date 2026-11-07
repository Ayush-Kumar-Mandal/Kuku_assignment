"""Synthetic show with planted moments (evaluate.synthetic_show). Fake writer, no network."""
import pytest

from agents.schemas import DraftLine, DraftScene, EpisodeDraft, EpisodeScript
from config.loader import load_moment_types, load_parser_config, load_settings
from evaluate import labels, synthetic_show as syn
from llm.checks import ScreenplayIndex
from llm.client import LLMClient
from parser.screenplay_parser import parse_show
from test_agents import RouterChat


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths.output_dir = str(tmp_path / "output")
    s.paths.logs_dir = str(tmp_path / "logs")
    s.evaluation.dir = str(tmp_path / "eval")
    return s


@pytest.fixture
def cfg(settings, tmp_path):
    c = syn.load_config(settings)
    c.output_dir = str(tmp_path / "synthetic")
    c.lines_per_episode = (4, 30)
    c.scenes_per_episode = (1, 3)
    c.plants_per_episode = (1, 1)
    c.timing = c.timing.model_copy(update={"words_per_second": 1.0, "min_line_s": 1.0, "max_line_s": 10.0,
                                           "gap_s": 0.5, "scene_gap_s": 1.0})
    return c


@pytest.fixture
def pcfg(settings):
    return load_parser_config(settings=settings)


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def draft(plant_words=5, plant_lines=5, plant="P1", split=False):
    """Scene 1: three ordinary lines. Scene 2: the plant (5 lines x 5 words = 5 s each), then one line."""
    ordinary = [DraftLine(kind="action", text=words(3, "calm")),
                DraftLine(kind="dialogue", speaker="Mara", emotion="Tired", text=words(3, "hello")),
                DraftLine(kind="sound", text="A clock ticks.")]
    planted = [DraftLine(kind="dialogue", speaker="Tomas" if i % 2 else "Mara", emotion="Angry, Cold" if i == 0 else "",
                         text=words(plant_words, f"p{i}x"), plant=plant) for i in range(plant_lines)]
    if split:                                   # an unplanted line inside the plant
        planted.insert(2, DraftLine(kind="action", text="She looks away."))
    tail = [DraftLine(kind="dialogue", speaker="Mara", text=words(2, "bye"))]
    return EpisodeDraft(logline="Mara confronts Tomas.", continuity_notes="Follows the wedding.",
                        summary="Mara confronts Tomas.",
                        scenes=[DraftScene(heading="INT. HALL - DAY", lines=ordinary),
                                DraftScene(heading="EXT. GARDEN - NIGHT", lines=planted + tail)])


# ------------------------------------------------------------------------ timing + format
def test_timeline_follows_word_counts_clamps_and_gaps(cfg):
    s1, s2 = syn.build_timeline(draft(), cfg.timing)          # centiseconds
    assert s1 == [(0, 300), (350, 650), (700, 1000)]          # 3 words at 1 word/s = 3 s each, 0.5 s gaps
    assert s2[0][0] == 1100                                   # scene gap 1.0 s after the last line
    long = EpisodeDraft(logline="l", continuity_notes="c", summary="s",
                        scenes=[DraftScene(heading="H", lines=[DraftLine(kind="action", text=words(40)),
                                                               DraftLine(kind="action", text="Hi")])])
    [(a1, b1), (a2, b2)] = syn.build_timeline(long, cfg.timing)[0]
    assert (b1 - a1, b2 - a2) == (1000, 100)                  # clamped to max_line_s / min_line_s


def test_rendered_episode_round_trips_through_the_real_parser(cfg, pcfg):
    d = draft()
    text = syn.render_episode(3, d, cfg)
    ep = syn.parse_text(text, pcfg, "episode-003.txt")
    assert ep.number == 3 and ep.logline == "Mara confronts Tomas."
    assert ep.coverage.percent == 100.0 and not ep.unparsed and not ep.warnings
    items = ep.all_items()
    assert len(items) == sum(len(s.lines) for s in d.scenes)
    first_plant = items[3]
    assert (first_plant.character, first_plant.emotions, first_plant.kind) == ("MARA", ["Angry", "Cold"], "dialogue")
    assert first_plant.text == words(5, "p0x")
    assert items[2].kind == "sound" and items[0].kind == "action"
    assert [it.start_ts for it in items[:2]] == ["00:00.00", "00:03.50"]


def test_plant_spans_map_to_exact_line_boundaries(cfg, pcfg, settings):
    d = draft()
    spans = syn.plant_spans(d, syn.build_timeline(d, cfg.timing))
    sc, first, last, a, b = spans["P1"]
    assert (sc, first, last) == (1, 0, 4) and (b - a) / 100 == pytest.approx(27.0)
    ep = syn.parse_text(syn.render_episode(1, d, cfg), pcfg, "episode-001.txt")
    idx = ScreenplayIndex([ep], pcfg, settings.promo.boundary_tolerance_s)
    assert labels._check(idx, ep, syn.ts(a), syn.ts(b), settings.agents.screenplay_kinds) == []


# ------------------------------------------------------------------------------ checks
def _problems(cfg, settings, pcfg, d, plants=({"id": "P1", "moment_type": "betrayal", "position": "middle"},)):
    return syn.episode_check(1, list(plants), cfg, settings, pcfg)(d)


def test_good_draft_passes_checks(cfg, settings, pcfg):
    assert _problems(cfg, settings, pcfg, draft()) == []


def test_too_short_plant_is_sent_back_with_its_length(cfg, settings, pcfg):
    probs = _problems(cfg, settings, pcfg, draft(plant_words=2, plant_lines=4))
    assert any("P1 is 9.5s" in p and "8 words" in p and "add lines" in p for p in probs)


def test_writer_word_target_matches_the_promo_length(cfg, settings):
    lo, hi = syn.plant_word_range(cfg, settings)                   # 1 word/s in the test timing
    assert lo >= settings.promo.min_duration_s and hi <= settings.promo.max_duration_s and lo < hi


def test_missing_summary_is_accepted_and_the_logline_stands_in():
    d = EpisodeDraft.model_validate({"logline": "L", "continuity_notes": "c",
                                     "scenes": [{"heading": "H", "lines": [{"kind": "action", "text": "x"}]}]})
    assert d.summary == "" and syn.story_line(1, d) == "Episode 1: L"


def test_broken_missing_and_unknown_plants_are_sent_back(cfg, settings, pcfg):
    assert any("P1" in p and "consecutive" in p for p in _problems(cfg, settings, pcfg, draft(split=True)))
    assert any("P1 is missing" in p and "[P1]" in p for p in _problems(cfg, settings, pcfg, draft(plant="P9")))
    assert any("P9" in p and "not a planted" in p and "[P9] tag" in p for p in _problems(cfg, settings, pcfg, draft(plant="P9")))


def test_dialogue_needs_a_speaker_and_line_count_is_enforced(cfg, settings, pcfg):
    d = draft()
    d.scenes[0].lines[1].speaker = ""
    assert any("speaker" in p for p in _problems(cfg, settings, pcfg, d))
    cfg.lines_per_episode = (20, 30)
    assert any("lines" in p and "20-30" in p for p in _problems(cfg, settings, pcfg, draft()))


def test_plan_is_seeded_and_uses_real_moment_types(cfg, settings):
    mt = load_moment_types(settings=settings)
    cfg.plants_per_episode = (1, 2)
    a, b = syn.plan_plants(cfg, mt, 6, seed=7), syn.plan_plants(cfg, mt, 6, seed=7)
    assert a == b and sorted(a) == [1, 2, 3, 4, 5, 6]
    for plants in a.values():
        assert 1 <= len(plants) <= 2 and [p["id"] for p in plants] == [f"P{i + 1}" for i in range(len(plants))]
        assert all(p["moment_type"] in mt.moment_types and p["moment_type"] != mt.default_type for p in plants)
        assert len({p["position"] for p in plants}) == len(plants)


# ----------------------------------------------------------------------------- end to end
BIBLE = {"premise": "A bride is abandoned at the altar.", "episode_beats": ["Ep one.", "Ep two."],
         "characters": [{"name": "Mara", "role": "heroine", "description": "d"},
                        {"name": "Tomas", "role": "rival", "description": "d"}]}


def as_script(d: EpisodeDraft) -> dict:
    """What the writer LLM answers: plain screenplay-style lines, planted ones prefixed [P1]."""
    def line(ln):
        tag = f"[{ln.plant}] " if ln.plant else ""
        if ln.kind == "dialogue":
            return f"{tag}{ln.speaker}{f' ({ln.emotion})' if ln.emotion else ''}: {ln.text}"
        return f"{tag}{'ACTION' if ln.kind == 'action' else 'SOUND'}: {ln.text}"
    return {"logline": d.logline, "continuity_notes": d.continuity_notes, "summary": d.summary,
            "scenes": [{"heading": sc.heading, "lines": [line(ln) for ln in sc.lines]} for sc in d.scenes]}


def test_script_lines_become_structured_lines(cfg):
    script = EpisodeScript.model_validate({"logline": "L", "scenes": [{"heading": "H", "lines": [
        "ACTION: The doors open.", "[P1] MARA (Angry, Cold): You sold it?", "[P1] sound fx: Glass breaks.",
        "Tomas: Yes.", "this line has no label"]}]})
    d, probs = syn.to_draft(script, cfg)
    kinds = [(ln.kind, ln.speaker, ln.emotion, ln.plant) for ln in d.scenes[0].lines]
    assert kinds == [("action", "", "", None), ("dialogue", "MARA", "Angry, Cold", "P1"), ("sound", "", "", "P1"),
                     ("dialogue", "Tomas", "", None)]
    assert d.scenes[0].lines[1].text == "You sold it?"
    assert probs == ["scene 1 line 5 doesn't follow the line format (LABEL: text): 'this line has no label'"]


def _client(settings, script):
    fake = RouterChat(script)
    return LLMClient(settings, run_id="syn", model_factory=lambda spec, st: fake, sleep=lambda s: None), fake


def test_generate_writes_a_parsable_show_and_a_locked_planted_key(settings, cfg, pcfg, tmp_path):
    good = as_script(draft())
    client, fake = _client(settings, {"ShowBible": [BIBLE], "EpisodeScript": [good, as_script(draft(plant_words=2)), good]})
    out = syn.generate(client, settings, cfg, pcfg, load_moment_types(settings=settings), show="Test Show",
                       genre="romance", n_episodes=2, seed=1, say=lambda *_: None)
    show = parse_show(out["folder"], pcfg)
    assert [e.number for e in show.episodes] == [1, 2] and all(e.coverage.percent == 100 for e in show.episodes)
    key = labels.load_labels(out["labels_path"])
    assert key.label_source == "planted" and key.locked_at and key.episodes == [1, 2]
    assert [(m.episode, m.moment_type) for m in key.moments] == [(1, out["plan"][1][0]["moment_type"]),
                                                                 (2, out["plan"][2][0]["moment_type"])]
    assert all(m.start == "00:11.00" and m.end == "00:38.00" for m in key.moments)   # scene 2, 5 x 5 s + gaps
    assert "planted" in key.labeled_by and "groq:openai/gpt-oss-120b" in key.labeled_by
    assert key.labeler_models == ["groq:openai/gpt-oss-120b"]          # so compare can flag a shared model
    assert "Ep two." in fake.text("EpisodeScript", 2) and "Mara confronts Tomas." in fake.text("EpisodeScript", 2)
    assert "P1" in fake.text("EpisodeScript", 0)                  # the plants are in the prompt
    assert labels.default_path(settings, "Test Show") == out["labels_path"]     # compare finds it by show name


def test_generate_resumes_from_saved_drafts_and_refuses_to_clobber(settings, cfg, pcfg):
    mt = load_moment_types(settings=settings)
    client, _ = _client(settings, {"ShowBible": [BIBLE], "EpisodeScript": [as_script(draft())]})
    with pytest.raises(Exception):                               # 2nd episode: the fake has no answer left
        syn.generate(client, settings, cfg, pcfg, mt, show="Test Show", genre="g", n_episodes=2, seed=1,
                     say=lambda *_: None)
    client2, fake2 = _client(settings, {"EpisodeScript": [as_script(draft())]})       # no bible call needed
    out = syn.generate(client2, settings, cfg, pcfg, mt, show="Test Show", genre="g", n_episodes=2, seed=1,
                       say=lambda *_: None)
    assert len(fake2.seen["EpisodeScript"]) == 1 and labels.load_labels(out["labels_path"]).episodes == [1, 2]
    with pytest.raises(SystemExit, match="already"):
        syn.generate(client2, settings, cfg, pcfg, mt, show="Test Show", genre="g", n_episodes=2, seed=1,
                     say=lambda *_: None)


def test_never_overwrites_hand_labels_of_the_same_show_name(settings, cfg, pcfg):
    labels.save_labels(labels.Labels(show="Test Show"), labels.default_path(settings, "Test Show"))
    client, _ = _client(settings, {"ShowBible": [BIBLE]})
    with pytest.raises(SystemExit, match="not a planted"):
        syn.generate(client, settings, cfg, pcfg, load_moment_types(settings=settings), show="Test Show",
                     genre="g", n_episodes=2, seed=1, say=lambda *_: None)
