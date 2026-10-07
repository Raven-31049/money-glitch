"""Tests for config.py: YAML variant loading and range validation.

The examples in configs/ are exercised too, so a drifting example file breaks
the suite instead of silently rotting.
"""

from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingModuleSource]

from bet_engine.config import VariantConfig, load_all, load_variant, variant_config_paths  # pyright: ignore[reportMissingImports]

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _merge(base: dict, overrides: dict) -> dict:
    """Overlay `overrides` onto `base`, recursing into nested dicts."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def valid_config(**overrides) -> dict:
    base = {
        "name": "test_variant",
        "market": "match_winner",
        "league": "E0",
        "seasons": ["1819", "1920"],
        "model": {"type": "naive_frequency", "params": {"k": 2}},
        "staking": {
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "max_stake_frac": 0.05,
        },
        "bankroll": {
            "starting": 1000.0,
            "max_drawdown": 0.3,
            "stop_on_breach": True,
        },
        "backtest": {"min_train_days": 60, "refit_every_days": 1},
    }
    return _merge(base, overrides)


def _write(tmp_path: Path, filename: str, data: dict) -> Path:
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_variant_parses_all_fields(tmp_path):
    path = _write(tmp_path, "v.yaml", valid_config())

    config = load_variant(path)

    assert isinstance(config, VariantConfig)
    assert config.name == "test_variant"
    assert config.market == "match_winner"
    assert config.league == "E0"
    assert config.seasons == ["1819", "1920"]
    assert config.odds_policy == "pinnacle_only"
    assert config.model.type == "naive_frequency"
    assert config.model.params == {"k": 2}
    assert config.staking.ev_threshold == 0.05
    assert config.staking.kelly_fraction == 0.25
    assert config.staking.max_stake_frac == 0.05
    assert config.bankroll.starting == 1000.0
    assert config.bankroll.max_drawdown == 0.3
    assert config.bankroll.stop_on_breach is True
    assert config.uncertainty_percentile is None
    assert config.backtest.min_train_days == 60
    assert config.backtest.refit_every_days == 1


def test_defaults_when_omitted(tmp_path):
    config = valid_config()
    del config["backtest"]["refit_every_days"]
    config["bankroll"].pop("stop_on_breach", None)
    path = _write(tmp_path, "v.yaml", config)

    loaded = load_variant(path)

    assert loaded.uncertainty_percentile is None
    assert loaded.odds_policy == "pinnacle_only"
    assert loaded.backtest.refit_every_days == 1
    assert loaded.bankroll.stop_on_breach is False


def test_fallback_odds_policy_is_accepted(tmp_path):
    path = _write(tmp_path, "v.yaml", valid_config(odds_policy="fallback"))

    assert load_variant(path).odds_policy == "fallback"


def test_unquoted_season_numbers_are_coerced(tmp_path):
    config = valid_config(seasons=[1819, 1920])
    path = _write(tmp_path, "v.yaml", config)

    assert load_variant(path).seasons == ["1819", "1920"]


def test_boundary_values_accepted(tmp_path):
    config = valid_config()
    config["staking"]["kelly_fraction"] = 1.0  # upper inclusive bound
    config["staking"]["ev_threshold"] = 0.0    # >= 0
    config["bankroll"]["max_drawdown"] = 0.5
    path = _write(tmp_path, "v.yaml", config)

    assert load_variant(path).staking.kelly_fraction == 1.0


def test_load_all_returns_variants_keyed_by_name(tmp_path):
    _write(tmp_path, "a.yaml", valid_config(name="alpha"))
    _write(tmp_path, "b.yml", valid_config(name="beta"))

    loaded = load_all(tmp_path)

    assert set(loaded) == {"alpha", "beta"}


def test_load_all_rejects_duplicate_names(tmp_path):
    _write(tmp_path, "a.yaml", valid_config(name="dup"))
    _write(tmp_path, "b.yaml", valid_config(name="dup"))

    with pytest.raises(ValueError, match="duplicate variant name"):
        load_all(tmp_path)


def test_load_all_rejects_non_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        load_all(tmp_path / "missing")


def test_load_variant_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "v.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_variant(path)


def test_missing_section_reports_locations(tmp_path):
    config = valid_config()
    del config["staking"]
    path = _write(tmp_path, "v.yaml", config)

    with pytest.raises(ValueError, match="staking"):
        load_variant(path)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"staking": {"kelly_fraction": 0.0}}, "kelly_fraction"),
        ({"staking": {"kelly_fraction": 1.5}}, "kelly_fraction"),
        ({"staking": {"ev_threshold": -0.1}}, "ev_threshold"),
        ({"staking": {"max_stake_frac": 0.0}}, "max_stake_frac"),
        ({"staking": {"max_stake_frac": 1.2}}, "max_stake_frac"),
        ({"bankroll": {"starting": 0.0}}, "starting"),
        ({"bankroll": {"max_drawdown": 0.0}}, "max_drawdown"),
        ({"bankroll": {"max_drawdown": 1.0}}, "max_drawdown"),
        ({"backtest": {"min_train_days": 0}}, "min_train_days"),
        ({"backtest": {"refit_every_days": 0}}, "refit_every_days"),
        ({"seasons": []}, "seasons must not be empty"),
        ({"seasons": ["1819", "x"]}, "4-digit"),
        ({"uncertainty_percentile": -1.0}, "uncertainty_percentile"),
        ({"uncertainty_percentile": 101.0}, "uncertainty_percentile"),
        ({"market": ""}, "must not be blank"),
        ({"name": "   "}, "must not be blank"),
        ({"model": {"type": "   "}}, "model.type"),
        ({"odds_policy": "pinnacle"}, "odds_policy must be one of"),
        ({"odds_policy": ""}, "odds_policy must be one of"),
    ],
    ids=[
        "kelly_fraction_zero",
        "kelly_fraction_above_one",
        "ev_threshold_negative",
        "max_stake_frac_zero",
        "max_stake_frac_above_one",
        "starting_zero",
        "max_drawdown_zero",
        "max_drawdown_one",
        "min_train_days_zero",
        "refit_every_days_zero",
        "seasons_empty",
        "season_bad_code",
        "percentile_negative",
        "percentile_above_100",
        "market_blank",
        "name_blank",
        "model_type_blank",
        "odds_policy_unknown",
        "odds_policy_blank",
    ],
)
def test_validation_failures(tmp_path, override, match):
    path = _write(tmp_path, "v.yaml", valid_config(**override))

    with pytest.raises(ValueError, match=match):
        load_variant(path)


def test_example_configs_load():
    loaded = load_all(CONFIGS_DIR)

    for name in ("epl_mw_control", "epl_mw_naive"):
        assert name in loaded
        assert loaded[name].league == "E0"
        assert loaded[name].market == "match_winner"
        assert loaded[name].bankroll.starting > 0


def test_example_configs_state_their_odds_policy_explicitly():
    """Checked in the YAML, not just via the default: a report reader must be
    able to see the policy in the config file the run quotes."""
    paths = variant_config_paths(CONFIGS_DIR)
    assert paths, "no example configs found"

    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw.get("odds_policy") == "pinnacle_only", (
            f"{path} must state odds_policy: pinnacle_only"
        )