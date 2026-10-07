"""Tests for eval/calibration.py: Brier, log loss, reliability bins.

The score cases are hand-calculated so the formula — squared error over the
whole probability vector, natural log of the settled outcome's probability —
is pinned to numbers rather than to the implementation, for both 2-way and
3-way markets (invariant 8: the scorer never sees a market, only columns).

The calibration case is synthetic and exactly known: probabilities drawn from
a distribution, outcomes drawn *from those same probabilities*, so the data is
perfectly calibrated by construction and the table must come out
near-diagonal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.eval import (
    ReliabilityBin,
    brier_score,
    log_loss,
    reliability_table,
)


def _two_way(homes, labels, aways=None):
    """A 2-way frame from home probabilities, with labels naming the winner.

    ``aways`` is explicit for the hand-checked cases: ``1.0 - 0.9`` is a hair
    below 0.9 in binary, which would split a bin edge the example means to
    land on.
    """
    if aways is None:
        aways = [1.0 - p for p in homes]
    return pd.DataFrame({"H": homes, "A": aways}), labels


# --- Brier score -------------------------------------------------------------


def test_brier_scores_a_two_way_pair_against_the_hand_calculation():
    predicted, actual = _two_way([0.7, 0.4], ["H", "H"])

    # Row 1: (0.7-1)^2 + (0.3-0)^2 = 0.18. Row 2: 0.36 + 0.36 = 0.72.
    assert brier_score(predicted, actual) == pytest.approx(0.45)


def test_brier_scores_a_three_way_row_against_the_hand_calculation():
    predicted = pd.DataFrame({"H": [0.5], "D": [0.3], "A": [0.2]})

    # The settled outcome was A: (0.5-0)^2 + (0.3-0)^2 + (0.2-1)^2 = 0.98.
    assert brier_score(predicted, ["A"]) == pytest.approx(0.98)


def test_a_perfect_prediction_scores_zero():
    predicted, actual = _two_way([1.0, 0.0], ["H", "A"])

    assert brier_score(predicted, actual) == pytest.approx(0.0)


def test_integer_positions_and_labels_agree():
    predicted, actual = _two_way([0.7, 0.4], ["H", "H"])

    by_labels = brier_score(predicted, actual)
    by_positions = brier_score(predicted.to_numpy(), [0, 0])

    assert by_labels == by_positions


# --- log loss ----------------------------------------------------------------


def test_log_loss_scores_a_two_way_pair_against_the_hand_calculation():
    predicted, actual = _two_way([0.7, 0.4], ["H", "H"])

    # -(ln 0.7 + ln 0.4) / 2: the settled outcome's own probability each row.
    assert log_loss(predicted, actual) == pytest.approx(0.6364828379064439)


def test_log_loss_scores_a_three_way_row_against_the_hand_calculation():
    predicted = pd.DataFrame({"H": [0.5], "D": [0.3], "A": [0.2]})

    assert log_loss(predicted, ["A"]) == pytest.approx(-np.log(0.2))


def test_log_loss_refuses_a_zero_probability_on_the_settled_outcome():
    # Clipping would score this model finite; it declared the result impossible.
    predicted, actual = _two_way([0.0], ["H"])

    with pytest.raises(ValueError, match="log loss is undefined"):
        log_loss(predicted, actual)


# --- reliability table -------------------------------------------------------


def test_reliability_bins_split_a_hand_checked_example():
    predicted, actual = _two_way(
        [0.1, 0.1, 0.9, 0.9], ["A", "A", "H", "H"], aways=[0.9, 0.9, 0.1, 0.1]
    )

    table = reliability_table(predicted, actual, n_bins=2)

    # Each bin holds four pairs: 0.1 appeared with the outcome absent four
    # times (twice as H, twice as A), 0.9 with it present four times.
    assert len(table) == 2
    lower, upper = table
    assert isinstance(lower, ReliabilityBin)
    assert (lower.low, lower.high) == (0.0, 0.5)
    assert lower.predicted_mean == pytest.approx(0.1)
    assert lower.actual_frequency == 0.0
    assert lower.count == 4
    assert (upper.low, upper.high) == (0.5, 1.0)
    assert upper.predicted_mean == pytest.approx(0.9)
    assert upper.actual_frequency == 1.0
    assert upper.count == 4


def test_empty_bins_keep_their_shape_with_nan_statistics():
    predicted, actual = _two_way(
        [0.1, 0.1, 0.9, 0.9], ["A", "A", "H", "H"], aways=[0.9, 0.9, 0.1, 0.1]
    )

    table = reliability_table(predicted, actual)  # default 10 bins

    assert len(table) == 10
    assert sum(bin_.count for bin_ in table) == 8
    populated = [bin_ for bin_ in table if bin_.count]
    assert [bin_.predicted_mean for bin_ in populated] == pytest.approx([0.1, 0.9])
    assert [bin_.actual_frequency for bin_ in populated] == pytest.approx([0.0, 1.0])
    for bin_ in table:
        if bin_.count == 0:
            assert np.isnan(bin_.predicted_mean)
            assert np.isnan(bin_.actual_frequency)


def test_probabilities_on_a_bin_edge_stay_inside_the_table():
    predicted, actual = _two_way([1.0], ["H"])

    table = reliability_table(predicted, actual, n_bins=5)

    assert sum(bin_.count for bin_ in table) == 2
    assert table[-1].count == 1  # p == 1.0 belongs to the last bin
    assert table[0].count == 1  # p == 0.0 belongs to the first


def test_a_three_way_table_counts_one_pair_per_outcome():
    predicted = pd.DataFrame({"H": [0.5, 0.2], "D": [0.3, 0.5], "A": [0.2, 0.3]})

    table = reliability_table(predicted, ["H", "A"], n_bins=5)

    assert sum(bin_.count for bin_ in table) == 6  # 2 bets x 3 outcomes


def test_perfectly_calibrated_probabilities_produce_a_near_diagonal_table():
    # Outcomes are drawn from the predicted probabilities themselves, so
    # calibration is exact in expectation; only binomial noise separates the
    # two columns of the table.
    rng = np.random.default_rng(1234)
    n_bets = 6000
    homes = rng.uniform(0.02, 0.98, n_bets)
    won = rng.random(n_bets) < homes
    predicted, actual = _two_way(homes, np.where(won, "H", "A"))

    table = reliability_table(predicted, actual)

    assert len(table) == 10
    assert sum(bin_.count for bin_ in table) == 2 * n_bets
    populated = [bin_ for bin_ in table if bin_.count]
    assert len(populated) >= 8
    for bin_ in populated:
        assert bin_.low <= bin_.predicted_mean <= bin_.high
        # Each bin's observed frequency tracks its own mean probability.
        assert abs(bin_.actual_frequency - bin_.predicted_mean) < 0.06


# --- input validation --------------------------------------------------------


def test_a_scalar_prediction_is_refused():
    with pytest.raises(ValueError, match="must be 2-D"):
        brier_score(0.5, [0])


def test_a_one_dimensional_prediction_is_refused():
    with pytest.raises(ValueError, match="must be 2-D"):
        brier_score(np.array([0.5, 0.5]), [0, 0])


def test_a_ragged_prediction_is_a_type_error():
    with pytest.raises(TypeError, match="DataFrame or a 2-D array"):
        brier_score([[0.5, 0.5], [0.9]], [0, 0])


def test_a_single_outcome_column_is_refused():
    with pytest.raises(ValueError, match="at least two outcome"):
        brier_score(pd.DataFrame({"H": [0.5]}), [0])


def test_an_empty_prediction_frame_is_refused():
    empty = pd.DataFrame({"H": [], "A": []})

    with pytest.raises(ValueError, match="at least one row"):
        brier_score(empty, [])


def test_a_non_numeric_prediction_is_refused():
    predicted = pd.DataFrame({"H": ["a"], "A": ["b"]})

    with pytest.raises(ValueError, match="numeric probabilities"):
        brier_score(predicted, ["H"])


def test_a_non_finite_prediction_is_refused():
    predicted, actual = _two_way([0.5, np.nan], ["H", "A"])

    with pytest.raises(ValueError, match="finite on every row"):
        brier_score(predicted, actual)


def test_a_probability_outside_the_unit_interval_is_refused():
    predicted, actual = _two_way([1.2], ["H"])

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        brier_score(predicted, actual)


def test_a_row_that_does_not_sum_to_one_is_refused_not_renormalised():
    predicted, actual = _two_way([0.5], ["H"])  # 0.5 + 0.4 != 1
    predicted.loc[0, "A"] = 0.4

    with pytest.raises(ValueError, match="sum to 1"):
        brier_score(predicted, actual)


def test_duplicate_outcome_columns_are_refused():
    predicted = pd.DataFrame([[0.5, 0.5]], columns=["H", "H"])

    with pytest.raises(ValueError, match="duplicate outcome"):
        brier_score(predicted, [0])


def test_labels_with_an_array_prediction_are_refused():
    with pytest.raises(ValueError, match="DataFrame whose columns"):
        brier_score(np.array([[0.5, 0.5]]), np.array(["H"]))


def test_an_unknown_label_is_refused():
    predicted, _ = _two_way([0.5], ["H"])

    with pytest.raises(ValueError, match="not among"):
        brier_score(predicted, ["X"])


def test_an_outcome_count_that_disagrees_with_the_frame_is_refused():
    predicted, _ = _two_way([0.5, 0.5], ["H", "H"])

    with pytest.raises(ValueError, match="predicted has 2 row"):
        brier_score(predicted, ["H"])


def test_float_positions_are_refused_as_ambiguous():
    predicted, _ = _two_way([0.5, 0.5], ["H", "H"])

    with pytest.raises(ValueError, match="outcome labels"):
        brier_score(predicted, np.array([0.0, 1.0]))


def test_boolean_outcomes_are_refused_as_ambiguous():
    predicted, _ = _two_way([0.5, 0.5], ["H", "H"])

    with pytest.raises(ValueError, match="outcome labels"):
        brier_score(predicted, np.array([True, False]))


def test_an_integer_position_outside_the_market_is_refused():
    predicted, _ = _two_way([0.5], ["H"])

    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        brier_score(predicted, [2])


def test_a_two_dimensional_actual_is_refused():
    predicted, _ = _two_way([0.5, 0.5], ["H", "H"])

    with pytest.raises(ValueError, match=r"one outcome per bet \(1-D\)"):
        brier_score(predicted, [["H"], ["A"]])


@pytest.mark.parametrize("n_bins", [2.5, "5", True])
def test_a_non_integer_bin_count_is_refused(n_bins):
    predicted, actual = _two_way([0.5], ["H"])

    with pytest.raises(TypeError, match="n_bins must be an integer"):
        reliability_table(predicted, actual, n_bins=n_bins)


def test_a_non_positive_bin_count_is_refused():
    predicted, actual = _two_way([0.5], ["H"])

    with pytest.raises(ValueError, match="n_bins must be"):
        reliability_table(predicted, actual, n_bins=0)
