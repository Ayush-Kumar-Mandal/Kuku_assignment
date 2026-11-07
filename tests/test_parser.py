"""Parser tests on a synthetic fixture (real screenplays are confidential, never in the repo)."""
from pathlib import Path

import pytest
import yaml

from config.loader import ParserConfig, load_parser_config
from parser.models import Item, format_ts
from parser.screenplay_parser import parse_episode, parse_ts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ep.txt"


@pytest.fixture(scope="module")
def cfg():
    return load_parser_config()


@pytest.fixture(scope="module")
def ep(cfg):
    return parse_episode(FIXTURE, cfg)


def item(ep, text_part):
    return next(it for it in ep.all_items() if text_part in it.text)


def test_timestamps(cfg):
    p = cfg.compiled
    assert parse_ts("01:05.60", p) == 65.6
    assert parse_ts("00:05.21", p) == 5.21
    assert parse_ts("00:04.8", p) == 4.8
    assert parse_ts("1:02:03.5", p) == 3723.5
    assert format_ts(65.6) == "01:05.60"


def test_header(ep):
    assert ep.number == 7
    # wrapped logline joined onto one line
    assert ep.logline.endswith("just as the town fair begins.")
    assert ep.continuity_notes.startswith("Synthetic")


def test_scenes_including_uppercase_metadata(ep):
    assert [s.number for s in ep.scenes] == [1, 2]
    s2 = ep.scenes[1]
    assert s2.heading == "EXT. TOWN SQUARE – DAY"
    assert (s2.start, s2.end) == (20.0, 65.6)


def test_block_kinds(ep):
    kinds = [it.kind for it in ep.all_items()]
    assert kinds == ["camera", "action", "dialogue", "dialogue", "sound", "action", "dialogue", "camera"]


def test_speaker_split(ep):
    it = item(ep, "recipe")
    assert it.character == "MARA"
    assert it.voice_tags == ["O.S."]
    assert it.emotions == ["Furious", "Shaking"]
    assert it.text == "You stole my recipe!"          # quotes stripped
    assert (it.start, it.end, it.start_ts, it.end_ts) == (3.6, 5.21, "00:03.60", "00:05.21")


def test_multiline_body(ep):
    assert item(ep, "rolling pin").text == "Mara slams a rolling pin onto the counter. Flour bursts into the air."


def test_end_backfill(ep):
    tomas = item(ep, "stole nothing")
    assert tomas.end_inferred and tomas.end == 8.0 and tomas.end_ts == "00:08.00"   # next item's start
    last = item(ep, "smiles and turns away")
    assert last.end_inferred and last.end == 65.6                                    # scene end


def test_unparsed_line_logged(ep):
    assert [u.text for u in ep.unparsed] == ["this line matches nothing at all"]
    assert ep.unparsed[0].reason == "unrecognised line"
    assert ep.coverage.unparsed_lines == 1
    assert 0 < ep.coverage.percent < 100


def test_ids_and_boundaries(ep):
    assert ep.all_items()[0].item_id == "e07_s01_i001"
    assert ep.scenes[1].items[0].item_id == "e07_s02_i001"
    b = ep.boundaries()
    assert b == sorted(set(b)) and 3.6 in b and 65.6 in b
    assert "00:05.21" in ep.timestamp_index()
    assert ep.duration == 65.6


def test_fallback_hook(cfg):
    calls = []

    def fake_llm(unparsed, episode):
        calls.append([u.line_no for u in unparsed])
        u = unparsed[0]
        return [Item(item_id="x", episode=0, scene=0, index=99, kind="action", label="ACTION",
                     text=u.text, start=10.0, end=11.0, start_ts="00:10.00", end_ts="00:11.00", line_no=u.line_no)]

    ep = parse_episode(FIXTURE, cfg, fallback=fake_llm)
    rec = [it for it in ep.all_items() if it.source == "llm_fallback"]
    assert len(calls) == 1 and len(rec) == 1
    assert rec[0].scene == 1 and rec[0].item_id.startswith("e07_s01_f")
    assert ep.coverage.recovered_by_fallback == 1


def test_new_format_needs_config_change_only(tmp_path):
    """Same content, different spelling: timestamps as [0:03,6 -> 0:05,2] and
    speaker blocks as 'MARA [Furious]:'. Only parser.yaml changes."""
    src = (
        "EPISODE NUMBER: Episode 3\n"
        "SCENE 1\nScene heading: INT. ROOM\nScene start: 0:00,0\nScene end: 0:10,0\n"
        ">> ACTION\n[0:00,0 -> 0:03,6]\nA door opens.\n"
        ">> MARA {Furious}\n[0:03,6 -> 0:05,2]\n\"Get out.\"\n"
    )
    f = tmp_path / "ep3.txt"
    f.write_text(src, encoding="utf-8")
    raw = yaml.safe_load(Path("config/parser.yaml").read_text(encoding="utf-8"))
    raw["block_header"] = r"^>>\s*(?P<label>.+?)\s*(?P<body>)$"
    raw["timing_line"] = r"^\[(?P<start>{ts})\s*->\s*(?P<end>{ts})\]$"
    raw["speaker"]["name"] = r"^(?P<name>[^{]+?)\s*(?:\{|$)"
    raw["speaker"]["parenthetical"] = r"\{([^}]*)\}"
    ep = parse_episode(f, ParserConfig.model_validate(raw))
    assert ep.coverage.percent == 100.0
    mara = ep.all_items()[1]
    assert (mara.character, mara.emotions, mara.text, mara.start, mara.end) == ("MARA", ["Furious"], "Get out.", 3.6, 5.2)
