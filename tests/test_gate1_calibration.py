"""Guards for the Phase 1 gate-1 calibration check.

The script is loaded by path (``scripts/`` is not an importable package) and —
because it defines a dataclass — must be registered in ``sys.modules`` before
``exec_module``, or the dataclass decorator cannot resolve the module's globals.

What is pinned here:

* the holdout guard refuses a holdout season by name;
* :func:`load_adopted_model` reads the quoted line set from the config and never
  lets it (or the feature half-life) reach the model constructor;
* the frozen pass-criterion constants are exactly the ones recorded in
  IMPLEMENTATION_NOTES §5 — a drifting threshold breaks the suite;
* :func:`evaluate_line` passes a hand-built perfectly-calibrated line and fails a
  deliberately biased one, on the right condition;
* :func:`propose_fixes` proposes nothing when the gate passes, and at most two
  fixes (the shelved Fix B first) when it does not;
* :func:`propose_fixes` adds the monotone Fix C only for a purely local gap.
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
    / "phase1_gate1_calibration.py"
)


def _load_gate():
    spec = importlib.util.spec_from_file_location("phase1_gate1_calibration", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _calibrated_table() -> pd.DataFrame:
    """A line 3.5 table that is perfectly calibrated by construction.

    Two P(over) groups (0.35 and 0.45); in each the over-rate equals the
    prediction, so the overall gap is 0, the two-point calibration slope is 1,
    and every n >= 100 bin has gap 0.
    """
    p = np.concatenate([np.full(300, 0.35), np.full(700, 0.45)])
    over = np.concatenate(
        [np.array([1] * 105 + [0] * 195), np.array([1] * 315 + [0] * 385)]
    )
    total = np.where(over == 1, 4.0, 3.0)  # 4 > 3.5 > 3
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(p.size)],
            "season": ["2122"] * p.size,
            "actual_total": total,
            "point_3_5": p,
        }
    )


def _biased_table() -> pd.DataFrame:
    """Same construction but 15 points too many over — a level (a) failure."""
    p = np.concatenate([np.full(300, 0.35), np.full(700, 0.45)])
    over = np.concatenate(
        [np.array([1] * 150 + [0] * 150), np.array([1] * 450 + [0] * 250)]
    )
    total = np.where(over == 1, 4.0, 3.0)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(p.size)],
            "season": ["2122"] * p.size,
            "actual_total": total,
            "point_3_5": p,
        }
    )


def test_holdout_guard_refuses_a_holdout_season():
    with pytest.raises(ValueError, match="refusing to load holdout"):
        gate._guard_holdout(["1819", "2425"])

    gate._guard_holdout(["1819", "1920", "2021", "2122", "2223", "2324"])


def test_frozen_criterion_constants_match_the_recorded_rule():
    assert gate.CAL_IN_LARGE_TOL == 0.02
    assert (gate.SLOPE_LOW, gate.SLOPE_HIGH) == (0.8, 1.2)
    assert (gate.BIN_MIN_N, gate.BIN_TOL) == (100, 0.05)
    assert gate.N_BINS == 10


def test_load_adopted_model_reads_the_quoted_line_set():
    adopted = gate.load_adopted_model()

    assert adopted.model_type == "cards_team_ref"
    assert adopted.lines == (3.5, 4.5, 5.5)
    assert adopted.line == 4.5
    assert adopted.half_life_days == 365.0
    # Neither the report line set nor the feature half-life may reach the
    # constructor; the scalar line is derived for it instead.
    assert "lines" not in adopted.params
    assert "half_life_days" not in adopted.params
    assert adopted.params["line"] == 4.5
    assert adopted.params["k"] == 6.0


def test_evaluate_line_passes_a_perfectly_calibrated_line():
    result = gate.evaluate_line(_calibrated_table(), 3.5)

    assert result["pass_large"] is True
    assert result["pass_slope"] is True
    assert result["pass_bins"] is True
    assert result["passed"] is True
    assert result["slope"] == pytest.approx(1.0, abs=0.05)


def test_evaluate_line_fails_a_biased_line_on_the_level_condition():
    result = gate.evaluate_line(_biased_table(), 3.5)

    assert result["pass_large"] is False
    assert result["passed"] is False
    assert result["gap"] < 0.0  # under-predicts the over rate


def test_diagnose_names_the_broken_condition():
    result = gate.evaluate_line(_biased_table(), 3.5)

    reasons = gate.diagnose(result)

    assert any("level (a)" in reason for reason in reasons)


def test_over_reliability_bins_a_hand_checked_example():
    rows = gate.over_reliability([0.15, 0.25, 0.45, 0.45], [0, 1, 1, 0])

    by_low = {row["low"]: row for row in rows}
    assert by_low[0.1]["n"] == 1 and by_low[0.1]["actual"] == 0.0
    assert by_low[0.2]["n"] == 1 and by_low[0.2]["actual"] == 1.0
    assert by_low[0.4]["n"] == 2
    assert by_low[0.4]["predicted"] == pytest.approx(0.45)
    assert by_low[0.4]["actual"] == pytest.approx(0.5)


def test_count_frequency_shares_sum_to_one_and_bucket_ten_plus():
    sides = {f"m{i}": (2.0, 2.0) for i in range(4)}
    ids = np.array([f"m{i}" for i in range(4)])
    actual = np.array([0.0, 2.0, 8.0, 14.0])  # 14 falls in the 10+ bucket

    rows = gate.count_frequency(sides, ids, actual)

    predicted = np.array([row["predicted"] for row in rows])
    observed = np.array([row["actual"] for row in rows])
    assert predicted.sum() == pytest.approx(1.0)
    assert observed.sum() == pytest.approx(1.0)
    by_count = {row["count"]: row for row in rows}
    assert by_count["0"]["actual"] == pytest.approx(0.25)
    assert by_count["9+"]["actual"] == pytest.approx(0.25)
    # 14 cards lands in the pooled top bucket, not its own.
    assert observed[-1] == pytest.approx(0.25)


def test_point_over_per_line_is_ordered():
    sides = {"m": (2.2, 2.0)}

    points = gate.point_over_per_line(sides, np.array(["m"]), (3.5, 4.5, 5.5))

    assert points.shape == (1, 3)
    assert points[0, 0] > points[0, 1] > points[0, 2]


def test_propose_fixes_says_nothing_when_the_gate_passes():
    assert gate.propose_fixes([]) == []


def test_propose_fixes_offers_only_the_shelved_fix_b_for_a_level_failure():
    level_failure = {"pass_large": False, "pass_slope": True, "pass_bins": True}

    proposals = gate.propose_fixes([level_failure])

    assert len(proposals) == 1
    assert proposals[0]["name"] == gate.FIX_B_NAME


def test_propose_fixes_adds_the_monotone_fix_c_for_a_local_gap():
    local_failure = {"pass_large": True, "pass_slope": True, "pass_bins": False}

    proposals = gate.propose_fixes([local_failure])

    assert len(proposals) == gate.MAX_FIXES == 2
    assert proposals[0]["name"] == gate.FIX_B_NAME
    assert proposals[1]["name"] == gate.FIX_C_NAME


def test_propose_fixes_never_exceeds_two():
    failures = [
        {"pass_large": False, "pass_slope": False, "pass_bins": False},
        {"pass_large": True, "pass_slope": True, "pass_bins": False},
    ]

    assert len(gate.propose_fixes(failures)) <= gate.MAX_FIXES
