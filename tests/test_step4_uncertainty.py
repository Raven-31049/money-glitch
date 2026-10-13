"""Guards for the step-4 uncertainty diagnostic.

The script is loaded by path (``scripts/`` is not an importable package) and —
because it defines dataclasses — must be registered in ``sys.modules`` before
``exec_module``, or the dataclass decorator cannot resolve the module's globals.

What is pinned here:

* the holdout guard refuses a holdout season by name;
* :func:`load_adopted_model` reads the adopted model from
  ``configs/phase1_cards.yaml`` and splits the feature-build half-life out of the
  constructor params (the config plumbing the whole step rests on);
* the per-match table is built from marginals that are never renormalised, and
  it refuses a draw that does not sum to 1 across outcomes;
* the per-line point helper is ordered (P(over 3.5) > P(over 4.5) > P(over 5.5)).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase1_step4_uncertainty.py"
)


def _load_diagnostic():
    spec = importlib.util.spec_from_file_location("phase1_step4_uncertainty", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diag = _load_diagnostic()

LINES = diag.SAMPLE_LINES


def _recorder(draws: int = 64, matches: int = 6) -> diag.SamplingRecorder:
    rec = diag.SamplingRecorder(
        draws=draws, outcomes=("over", "under"), lines=LINES
    )
    rng = np.random.default_rng(0)
    ids = [f"m{i}" for i in range(matches)]
    # (n, matches, lines, outcomes): over varies a little, under is its complement.
    over = np.clip(
        0.5 + rng.normal(0.0, 0.05, size=(draws, matches, len(LINES))), 0.01, 0.99
    )
    samples = np.stack([over, 1.0 - over], axis=-1)
    rec.sample_blocks.append(samples)
    rec.ids = ids
    rec.sides = {match_id: (2.2, 2.0) for match_id in ids}
    rec.totals = {match_id: 4.2 for match_id in ids}
    return rec


def _features(matches: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(matches)],
            "date": pd.date_range("2021-08-07", periods=matches, freq="3D"),
            "season": ["2122"] * matches,
            "total_cards": [4.0, 3.0, 5.0, 2.0, 6.0, 4.0][:matches],
            "ref_past_count": [5.0, 20.0, 3.0, 40.0, 8.0, 25.0][:matches],
            "home_past_matches": [15.0, 30.0, 2.0, 50.0, 10.0, 22.0][:matches],
            "away_past_matches": [12.0, 40.0, 8.0, 60.0, 5.0, 20.0][:matches],
        }
    )


def test_holdout_guard_refuses_a_holdout_season():
    with pytest.raises(ValueError, match="refusing to load holdout"):
        diag._guard_holdout(["1819", "2425"])

    # Development seasons pass untouched.
    diag._guard_holdout(["1819", "1920", "2021", "2122", "2223", "2324"])


def test_load_adopted_model_splits_the_feature_parameter_out():
    adopted = diag.load_adopted_model()

    assert adopted.model_type == "cards_team_ref"
    assert adopted.line == 4.5
    assert adopted.half_life_days == 365.0
    # The feature-build half-life must NOT reach the constructor.
    assert "half_life_days" not in adopted.params
    assert adopted.params.get("k") == 6.0
    assert "2425" not in adopted.config.seasons


def test_point_over_per_line_is_ordered_and_per_line():
    rec = _recorder()
    points = diag.point_over_per_line(rec, rec.ids)

    assert points.shape == (len(rec.ids), len(LINES))
    # A higher line is harder to clear, so P(over) falls with the line.
    assert np.all(points[:, 0] > points[:, 1])
    assert np.all(points[:, 1] > points[:, 2])


def test_build_match_table_reports_ordered_widths_and_marginals():
    rec = _recorder()
    table = diag.build_match_table(_features(), rec)

    required = {
        "point_3_5", "point_4_5", "point_5_5",
        "width_3_5", "width_4_5", "width_5_5",
        "low_4_5", "high_4_5",
    }
    assert required <= set(table.columns)
    # width == high - low, and every width is in [0, 1].
    width = table["width_4_5"].to_numpy(dtype=float)
    assert np.allclose(width, table["high_4_5"] - table["low_4_5"])
    assert np.all((width >= 0.0) & (width <= 1.0))


def test_build_match_table_keeps_marginals_unrenormalised():
    """The 10th percentiles of over and under are read independently, so their
    sum is not forced to 1 (invariant 5)."""
    rec = _recorder()
    table = diag.build_match_table(_features(), rec)

    lower_sum = (table["low_4_5"] + (1.0 - table["high_4_5"])).to_numpy(dtype=float)
    # The lower marginal of under is 1 - the *upper* marginal of over; if the
    # percentiles had been renormalised across outcomes this identity's sum
    # would be pinned to 1 and could not vary.
    assert lower_sum.std() > 0.0


def test_build_match_table_refuses_a_draw_that_does_not_sum_to_one():
    rec = _recorder()
    rec.sample_blocks[0] = rec.sample_blocks[0] * 2.0  # no longer a distribution

    with pytest.raises(AssertionError, match="does not sum to 1"):
        diag.build_match_table(_features(), rec)


def test_width_and_reliability_helpers_run_on_the_table():
    table = diag.build_match_table(_features(), _recorder())

    widths = diag.width_summary(table, LINES)
    assert [row["line"] for row in widths] == list(LINES)
    assert all(row["p05"] <= row["p50"] <= row["p95"] for row in widths)

    bands = diag.width_band_calibration(table, 4.5, diag.WIDTH_BANDS)
    assert all(0.0 <= row["mae"] <= 1.0 for row in bands)

    reliability = diag.reliability_tables(table, LINES)
    assert set(reliability) == set(LINES)

    losses = diag.line_log_losses(table, LINES)
    assert set(losses) == set(LINES)
    assert all(value > 0.0 for value in losses.values())


def test_line_tag_is_filesystem_and_column_safe():
    assert diag._line_tag(4.5) == "4_5"
    assert diag._line_tag(3.5) == "3_5"
