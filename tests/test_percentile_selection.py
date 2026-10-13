"""Tests for percentile-EV selection (Part B): marginals, strict threshold.

The uncertainty-aware path bets on the *conservative* edge: for each
(match, outcome) it takes the q-th percentile of that outcome's own sampled
probability — a marginal, never renormalised (invariant 5) — and compares
``p_q * raw_odds - 1`` to the threshold, exactly as the point path does. So the
spine here is: (a) percentiles are per-outcome and unnormalised, and (b) an
exactly-fair price still never clears a zero threshold, which is what keeps the
pure-market control at zero bets on this path too (invariant 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import select_bets_at_percentile
from bet_engine.eval import percentile_probabilities

COLUMNS = ["match_id", "market", "outcome", "odds", "model_prob"]


def _predictions(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


# --- percentile_probabilities -----------------------------------------------


def test_percentiles_are_marginal_per_outcome_and_never_renormalised():
    # Two outcomes whose draws are deliberately *not* complements; a correct
    # implementation reports each outcome's own 10th percentile (0.26, 0.34),
    # whose sum is far below 1. Renormalising would push them to 0.43/0.57.
    samples = np.array(
        [
            [[0.2, 0.3]],
            [[0.8, 0.7]],
        ]
    )
    out = percentile_probabilities(samples, 10.0, ["m1"], ["over", "under"])

    assert list(out.columns) == ["match_id", "outcome", "prob"]
    assert out["prob"].sum() == pytest.approx(0.60)
    assert out.loc[out["outcome"] == "over", "prob"].item() == pytest.approx(0.26)
    assert out.loc[out["outcome"] == "under", "prob"].item() == pytest.approx(0.34)


def test_percentile_zero_is_the_minimum_and_hundred_the_maximum():
    samples = np.array([[[0.10, 0.90]], [[0.30, 0.70]], [[0.20, 0.80]]])

    low = percentile_probabilities(samples, 0.0, ["m1"], ["over", "under"])
    high = percentile_probabilities(samples, 100.0, ["m1"], ["over", "under"])

    assert low["prob"].tolist() == pytest.approx([0.10, 0.70])
    assert high["prob"].tolist() == pytest.approx([0.30, 0.90])


def test_the_table_covers_every_match_and_outcome_in_order():
    samples = np.full((4, 2, 2), 0.5)
    out = percentile_probabilities(samples, 50.0, ["m1", "m2"], ["over", "under"])

    assert out["match_id"].tolist() == ["m1", "m1", "m2", "m2"]
    assert out["outcome"].tolist() == ["over", "under", "over", "under"]


def test_a_bad_percentile_is_refused():
    samples = np.full((4, 1, 2), 0.5)
    for bad in (-1.0, 101.0, float("nan"), True):
        with pytest.raises(ValueError, match="percentile"):
            percentile_probabilities(samples, bad, ["m1"], ["over", "under"])


def test_a_shape_mismatch_is_refused():
    samples = np.full((4, 2, 2), 0.5)
    with pytest.raises(ValueError, match="match_ids"):
        percentile_probabilities(samples, 50.0, ["m1"], ["over", "under"])
    with pytest.raises(ValueError, match="outcomes"):
        percentile_probabilities(samples, 50.0, ["m1", "m2"], ["over"])


def test_two_dimensional_samples_are_refused_with_the_expected_shape():
    with pytest.raises(ValueError, match="draws, n_matches, n_outcomes"):
        percentile_probabilities(np.full((4, 2), 0.5), 50.0, ["m1"], ["over"])


# --- select_bets_at_percentile ----------------------------------------------


def _marginals(over, under, draws=50):
    """``(draws, n_matches, 2)`` from constant per-match marginal probabilities.

    Constant across draws keeps the chosen percentile equal to the given value,
    so the expected edge is hand-checkable. The marginals need not sum to 1:
    that is the whole point of taking them per outcome (invariant 5).
    """
    o = np.repeat(np.asarray(over, dtype=float)[None, :], draws, axis=0)
    u = np.repeat(np.asarray(under, dtype=float)[None, :], draws, axis=0)
    return np.stack([o, u], axis=-1)


def _both_outcomes(*matches):
    rows = []
    for match_id, over_odds, under_odds in matches:
        rows.append((match_id, "cards_total", "over", over_odds, 0.5))
        rows.append((match_id, "cards_total", "under", under_odds, 0.5))
    return _predictions(rows)


def test_only_matches_whose_percentile_edge_clears_the_threshold_bet():
    predictions = _both_outcomes(("m1", 2.0, 2.0), ("m2", 2.0, 2.0))
    # m1 over: 0.62 * 2 - 1 = +0.24 (bets); every other cell is below +0.05.
    samples = _marginals(over=[0.62, 0.45], under=[0.45, 0.45])

    bets = select_bets_at_percentile(predictions, samples, 0.05, 25.0)

    assert bets["match_id"].tolist() == ["m1"]
    assert bets["outcome"].tolist() == ["over"]


def test_an_exactly_fair_price_clears_a_zero_threshold_zero_times():
    # percentile prob 0.50 at odds 2.0 is EV exactly 0; strict '>' must drop it,
    # so the pure-market control places no bets on this path either.
    predictions = _both_outcomes(("m1", 2.0, 2.0))
    samples = _marginals(over=[0.50], under=[0.50])

    assert select_bets_at_percentile(predictions, samples, 0.0, 50.0).empty


def test_one_bet_per_match_and_market_is_kept():
    predictions = _both_outcomes(("m1", 2.0, 2.0))
    # Both outcomes clear +0.05, so the higher-EV one is kept (and only one).
    samples = _marginals(over=[0.62], under=[0.55])

    bets = select_bets_at_percentile(predictions, samples, 0.05, 25.0)

    assert len(bets) == 1
    assert bets.iloc[0]["outcome"] == "over"


def test_a_lower_percentile_is_a_stricter_edge():
    predictions = _both_outcomes(("m1", 2.5, 2.5))
    # Over prob varies across draws 0.2..0.8, so its 10th percentile is well
    # below its 90th: the conservative percentile refuses the bet the optimistic
    # one accepts. Under is the complement, so it moves the other way.
    over = np.linspace(0.2, 0.8, 100)
    samples = np.stack(
        [over[:, None], (1.0 - over)[:, None]], axis=-1
    )  # (100, 1, 2)

    assert select_bets_at_percentile(predictions, samples, 0.0, 10.0).empty
    assert not select_bets_at_percentile(predictions, samples, 0.0, 90.0).empty


def test_a_percentile_missing_for_a_match_is_an_error():
    predictions = _both_outcomes(("m1", 2.0, 2.0), ("m2", 2.0, 2.0))
    samples = _marginals(over=[0.6], under=[0.6])  # only one match's worth

    with pytest.raises(ValueError):
        select_bets_at_percentile(predictions, samples, 0.05, 25.0)

