"""Tests for the model-side Part B API: per-row-line over/under/**push**.

A quoted card total carries its own line, and a whole-number line can be hit
exactly — a *push*, which is void rather than an under. These tests pin that the
model reads the line **per row** (not the constructor's fixed line), reports the
three states in the market's order, and (for a half-line) never invents a push.
Both card targets are covered: the convolved two-side model and the single-total
negative-binomial model. The sampled paths return one distribution per draw that
must sum to 1; the percentiles a caller later takes are marginals (invariant 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.models import (
    CardsTeamRefPoisson,
    CardsTotalNegativeBinomial,
    create_model,
)
from bet_engine.models.cards_poisson import OVER_UNDER_PUSH

OUTCOMES = ("over", "under")


def _point_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Counts plus the columns both point estimators read (home/away/ref rate)."""
    rng = np.random.default_rng(seed)
    home_rate = rng.uniform(0.8, 3.0, n)
    away_rate = rng.uniform(0.8, 3.0, n)
    ref_rate = rng.uniform(2.5, 5.0, n)
    total = rng.poisson(home_rate + away_rate + ref_rate).astype(float)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "league": ["E0"] * n,
            "date": pd.date_range("2021-08-01", periods=n, freq="D"),
            "home": [f"h{i}" for i in range(n)],
            "away": [f"a{i}" for i in range(n)],
            "home_cards": np.floor(total / 2.0),
            "away_cards": total - np.floor(total / 2.0),
            "total_cards": total,
            "home_rate": home_rate,
            "away_rate": away_rate,
            "ref_rate": ref_rate,
            "league_mean": [2.0] * n,
        }
    )


def _sample_frame(n: int = 300, seed: int = 4, k: float = 6.0) -> pd.DataFrame:
    """``_point_frame`` plus the Gamma sufficient statistics the draws need."""
    frame = _point_frame(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    home_matches = rng.uniform(20.0, 60.0, n)
    away_matches = rng.uniform(20.0, 60.0, n)
    ref_matches = rng.uniform(20.0, 60.0, n)
    league_mean = frame["league_mean"].to_numpy()
    home_rate = (k * league_mean + rng.uniform(1.0, 4.0, n) * home_matches) / (
        k + home_matches
    )
    away_rate = (k * league_mean + rng.uniform(1.0, 4.0, n) * away_matches) / (
        k + away_matches
    )
    total_prior = np.full(n, 5.0)
    ref_rate = (k * total_prior + rng.uniform(2.0, 8.0, n) * ref_matches) / (
        k + ref_matches
    )
    return frame.assign(
        home_rate=home_rate,
        away_rate=away_rate,
        ref_rate=ref_rate,
        home_past_cards=rng.uniform(1.0, 4.0, n) * home_matches,
        home_past_matches=home_matches,
        away_past_cards=rng.uniform(1.0, 4.0, n) * away_matches,
        away_past_matches=away_matches,
        ref_past_cards=rng.uniform(2.0, 8.0, n) * ref_matches,
        ref_past_count=ref_matches,
        league_mean_total=total_prior,
    )


# --- point estimate: long form, three states per match -----------------------


def test_the_point_estimate_is_long_form_over_under_push_per_match():
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    out = model.over_under(frame.assign(line=4.5))

    assert list(out.columns) == ["match_id", "line", "outcome", "prob"]
    assert len(out) == 3 * len(frame)
    assert list(out["outcome"].drop_duplicates()) == list(OVER_UNDER_PUSH)
    # One over/under/push triple per match, each summing to exactly 1.
    assert np.allclose(out.groupby("match_id")["prob"].sum(), 1.0)


def test_a_half_line_reports_zero_push():
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    out = model.over_under(frame.assign(line=4.5))
    push = out.loc[out["outcome"] == "push", "prob"]

    assert (push == 0.0).all()


def test_a_whole_line_reports_a_nonzero_push():
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    out = model.over_under(frame.assign(line=4))
    push = out.loc[out["outcome"] == "push", "prob"]
    over = out.loc[out["outcome"] == "over", "prob"].to_numpy()
    under = out.loc[out["outcome"] == "under", "prob"].to_numpy()

    assert (push > 0.0).all()
    assert push.to_numpy() == pytest.approx(1.0 - over - under, abs=1e-12)


def test_the_line_is_read_per_row_not_from_the_constructor():
    """The constructor line is the *fixed* step-1 quote; over_under must honour
    each row's own line, so moving one row's line moves only that match."""
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    baseline = model.over_under(frame.assign(line=3.5)).set_index(
        ["match_id", "outcome"]
    )["prob"]
    # Re-run with a single match quoted at a higher line.
    mixed = frame.assign(line=3.5)
    mixed.loc[5, "line"] = 6.5
    out = model.over_under(mixed).set_index(["match_id", "outcome"])["prob"]

    touched = frame.loc[5, "match_id"]
    untouched = frame.loc[6, "match_id"]
    # A higher line raises under and lowers over for exactly that match.
    assert out.loc[(touched, "under")] > baseline.loc[(touched, "under")]
    assert out.loc[(touched, "over")] < baseline.loc[(touched, "over")]
    # Every other match is unchanged.
    assert out.loc[(untouched, "over")] == baseline.loc[(untouched, "over")]


def test_the_total_model_reports_the_same_long_form():
    frame = _point_frame()
    model = CardsTotalNegativeBinomial(OUTCOMES, line=4.5, k=6.0).fit(frame)

    out = model.over_under(frame.assign(line=4))

    assert list(out.columns) == ["match_id", "line", "outcome", "prob"]
    assert np.allclose(out.groupby("match_id")["prob"].sum(), 1.0)
    assert (out.loc[out["outcome"] == "push", "prob"] > 0.0).all()


def test_a_missing_line_column_names_the_join():
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    with pytest.raises(ValueError, match="join_odds"):
        model.over_under(frame)  # no `line` column


def test_a_negative_line_is_refused():
    frame = _point_frame()
    model = CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(frame)

    with pytest.raises(ValueError, match="negative line"):
        model.over_under(frame.assign(line=-1.0))


# --- sampled: one distribution per draw, no renormalisation ------------------


@pytest.mark.parametrize(
    "model_type",
    [
        CardsTeamRefPoisson,
        CardsTotalNegativeBinomial,
    ],
)
def test_sampled_over_under_returns_one_distribution_per_draw(model_type):
    frame = _sample_frame()
    model = model_type(OUTCOMES, line=4.5, k=6.0).fit(frame)

    samples = model.over_under(frame.assign(line=4), draws=120, seed=7)

    assert samples.shape == (120, len(frame), len(OVER_UNDER_PUSH))
    assert np.allclose(samples.sum(axis=-1), 1.0)
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_sampled_push_is_zero_on_a_half_line_and_positive_on_a_whole_line():
    frame = _sample_frame()
    model = CardsTotalNegativeBinomial(OUTCOMES, line=4.5, k=6.0).fit(frame)

    half = model.over_under(frame.assign(line=4.5), draws=50, seed=1)
    whole = model.over_under(frame.assign(line=4), draws=50, seed=1)

    push_axis = OVER_UNDER_PUSH.index("push")
    assert (half[..., push_axis] == 0.0).all()
    assert (whole[..., push_axis] > 0.0).any()


def test_sampled_over_under_needs_a_fit_first():
    frame = _sample_frame()
    with pytest.raises(RuntimeError, match="before fit"):
        CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).over_under(
            frame.assign(line=4), draws=10
        )


def test_sampled_over_under_is_reproducible_for_a_fixed_seed():
    frame = _sample_frame()
    model = CardsTotalNegativeBinomial(OUTCOMES, line=4.5, k=6.0).fit(frame)

    data = frame.assign(line=4)
    first = model.over_under(data, draws=40, seed=99)
    again = model.over_under(data, draws=40, seed=99)
    other = model.over_under(data, draws=40, seed=100)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_over_under_reports_the_fixed_state_order_regardless_of_declared_order():
    """over_under is settled over/under/push in that fixed order; the model's
    declared outcome order only affects ``predict``, not this API."""
    frame = _sample_frame()
    over_first = CardsTotalNegativeBinomial(("over", "under"), line=4.5, k=6.0).fit(
        frame
    )
    under_first = CardsTotalNegativeBinomial(("under", "over"), line=4.5, k=6.0).fit(
        frame
    )

    data = frame.assign(line=4)
    a = over_first.over_under(data, draws=30, seed=2)
    b = under_first.over_under(data, draws=30, seed=2)

    assert np.allclose(a, b)


def test_the_market_name_and_registry_are_untouched_by_this_api():
    # over_under is a model method, not a market; the registry key is the market.
    assert create_model("cards_total_nbinom", {"line": 4.5}, OUTCOMES) is not None
