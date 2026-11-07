import pytest
import yaml

from config.loader import Settings, load_moment_types, load_parser_config, load_settings


def test_all_configs_load():
    s = load_settings()
    assert s.promo.min_duration_s == 20 and s.promo.max_duration_s == 60
    assert s.promo.hook_window_s == 3
    assert s.model_for("reader").provider
    assert s.model_for("unknown_agent") == s.models["default"]
    mt = load_moment_types(settings=s)
    assert {"confrontation", "reveal", "humiliation", "revenge", "romance", "betrayal", "danger", "other"} <= set(mt.names)
    assert load_parser_config(settings=s).compiled.block_kinds["CAMERA"] == "camera"


def test_bad_settings_fail_fast():
    raw = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    raw["scoring"]["weights"]["hook"] = 0.9           # weights no longer sum to 1
    with pytest.raises(ValueError, match="sum to 1.0"):
        Settings.model_validate(raw)
    raw = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    raw["promo"]["typo_key"] = 1                       # unknown key
    with pytest.raises(ValueError):
        Settings.model_validate(raw)
