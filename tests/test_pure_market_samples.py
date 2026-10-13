"""Tests for PureMarket.predict_samples: the control's percentile path.

Invariant 4 has to hold on *both* selection paths. The point path is covered in
test_pure_market.py; this file pins the sampled one. The control has no
parameters to be unsure about, so its sampled distribution is a point mass on
the de-vigged probability — the q-th percentile of a constant is that constant
for every q, so the percentile path must place exactly zero bets too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import select_bets_at_percentile
from bet_engine.models.pure_market import PureMarket


def _priced_frame():
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "over_odds": [1.9, 2.2],
            "under_odds": [1.95, 1.7],
        }
    )


def test_samples_broadcast_the_devigged_probability():
    frame = _priced_frame()
    model = PureMarket(("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"})
    point = model.predict(frame).set_index(["match_id", "outcome"])["prob"]

    samples = model.predict_samples(frame, n=64)

    assert samples.shape == (64, len(frame), 2)
    # Every draw is identical to the point estimate (a point mass).
    assert np.allclose(samples, samples[0])
    for match in frame["match_id"]:
        assert samples[0, frame.index[frame["match_id"] == match][0]] == pytest.approx(
            [point.loc[(match, "over")], point.loc[(match, "under")]]
        )


def test_samples_sum_to_one_per_draw():
    model = PureMarket(("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"})
    samples = model.predict_samples(_priced_frame(), n=10)

    assert np.allclose(samples.sum(axis=-1), 1.0)


def test_the_outcome_axis_follows_the_declared_order():
    frame = _priced_frame()
    over_first = PureMarket(
        ("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"}
    )
    under_first = PureMarket(
        ("under", "over"), odds_columns={"over": "over_odds", "under": "under_odds"}
    )

    a = over_first.predict_samples(frame, n=4)
    b = under_first.predict_samples(frame, n=4)

    assert np.allclose(a[..., 0], b[..., 1])
    assert np.allclose(a[..., 1], b[..., 0])


def test_a_bad_draw_count_is_refused():
    model = PureMarket(("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"})
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="n must be"):
            model.predict_samples(_priced_frame(), n=bad)


def test_an_unpriced_row_raises_rather_than_broadcasting_a_nan():
    frame = _priced_frame()
    frame.loc[1, "under_odds"] = np.nan
    model = PureMarket(
        ("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"}
    )

    with pytest.raises(ValueError, match="cannot price match"):
        model.predict_samples(frame, n=8)


def test_the_default_column_naming_is_odds_outcome():
    frame = pd.DataFrame(
        {"match_id": ["m1"], "odds_H": [1.8], "odds_A": [2.2]}
    )
    model = PureMarket(("H", "A"))

    samples = model.predict_samples(frame, n=5)

    assert samples.shape == (5, 1, 2)
    assert np.allclose(samples.sum(axis=-1), 1.0)


def test_the_control_places_no_bets_on_the_percentile_path_either():
    """The whole point of the control: zero bets on both selection paths."""
    frame = _priced_frame()
    model = PureMarket(
        ("over", "under"), odds_columns={"over": "over_odds", "under": "under_odds"}
    )
    point = model.predict(frame)
    predictions = pd.DataFrame(
        {
            "match_id": point["match_id"],
            "market": "cards_total",
            "outcome": point["outcome"],
            "model_prob": point["prob"],  # overwritten per percentile, kept valid
            "odds": [
                frame.loc[frame["match_id"] == match_id, f"{outcome}_odds"].item()
                for match_id, outcome in zip(point["match_id"], point["outcome"])
            ],
        }
    )

    samples = model.predict_samples(frame, n=32)

    for q in (0.0, 25.0, 50.0, 90.0, 100.0):
        assert select_bets_at_percentile(predictions, samples, 0.0, q).empty

