"""Tests for the negative-binomial cards models and the count convolution.

Step 3 asks whether estimating a dispersion parameter, and convolving the two
per-side distributions into a match total, fixes the tails the Poisson model
under-covers. Two things therefore need testing beyond the Poisson suite:

* the convolution itself — that it is a real probability distribution (rows sum
  to 1, non-negative) and that it reproduces the Poisson total exactly, which is
  the one case with a closed form to check against (home + away Poisson is
  Poisson);
* that the negative-binomial fit *estimates* dispersion from the data (recovers
  a known alpha on synthetic overdispersed counts, and collapses to ~0 —
  Poisson — on data that is not overdispersed), while sharing the Poisson
  design column-for-column so any score difference is the likelihood alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from bet_engine.backtest import WALK_FORWARD_COLUMNS, walk_forward
from bet_engine.models import (
    CardsNegativeBinomial,
    CardsPoisson,
    CardsTeamRefNegativeBinomial,
    create_model,
)
from bet_engine.models.base import check_predictions
from bet_engine.models.cards_poisson import add_card_features, add_referee_feature
from bet_engine.models.count_distributions import (
    SIDE_MAX_COUNT,
    convolve_rows,
    side_pmf,
    total_pmf,
)

OUTCOMES = ("over", "under")
LINES = (3.5, 4.5, 5.5)
COUNT_GRID = np.arange(SIDE_MAX_COUNT)


def _scored_frame(
    home_cards: list[float],
    away_cards: list[float],
    home_rate: list[float],
    away_rate: list[float],
    match_ids: list[str] | None = None,
) -> pd.DataFrame:
    """A frame carrying the feature columns directly, as add_card_features would."""
    n = len(home_cards)
    ids = match_ids or [f"m{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "match_id": ids,
            "league": ["E0"] * n,
            "date": pd.date_range("2021-08-01", periods=n, freq="7D"),
            "home": [f"home{i}" for i in range(n)],
            "away": [f"away{i}" for i in range(n)],
            "home_cards": home_cards,
            "away_cards": away_cards,
            "total_cards": np.add(home_cards, away_cards),
            "home_rate": home_rate,
            "away_rate": away_rate,
            "league_mean": [2.0] * n,
        }
    )


def _rates_equal_cards(n: int = 12) -> pd.DataFrame:
    home_rate = ([1.0, 2.0, 3.0, 4.0] * ((n + 3) // 4))[:n]
    away_rate = ([4.0, 3.0, 2.0, 1.0] * ((n + 3) // 4))[:n]
    return _scored_frame(home_rate, away_rate, home_rate, away_rate)


# ---------------------------------------------------------------------------
# the convolution
# ---------------------------------------------------------------------------


def test_convolving_two_poisson_sides_reproduces_the_poisson_total():
    """The one closed form available: home + away Poisson is Poisson with the
    summed mean, so the convolution must equal it to float precision."""
    mu_home = np.array([1.5, 2.0, 4.3])
    mu_away = np.array([2.5, 3.0, 0.7])
    counts, pmf = total_pmf(mu_home, mu_away, None)

    expected = poisson.pmf(counts[None, :], (mu_home + mu_away)[:, None])

    assert np.allclose(pmf, expected, atol=1e-12)
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-12)


def test_the_convolved_total_is_a_probability_distribution():
    rng = np.random.default_rng(3)
    mu_home = rng.uniform(0.5, 4.0, 50)
    mu_away = rng.uniform(0.5, 4.0, 50)

    counts, pmf = total_pmf(mu_home, mu_away, 0.1)

    assert pmf.shape == (50, counts.size)
    assert (pmf >= 0.0).all()
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-9)


def test_zero_alpha_is_the_poisson_limit():
    mu = np.array([1.0, 2.0, 3.5])

    as_nbinom = side_pmf(mu, 1e-9, COUNT_GRID)
    as_poisson = side_pmf(mu, None, COUNT_GRID)

    assert np.allclose(as_nbinom, as_poisson, atol=1e-6)


def test_a_per_row_alpha_matches_looping_the_scalar_case():
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = np.array([0.0, 0.05, 0.2, 0.4])

    batched = side_pmf(mu, alpha, COUNT_GRID)

    for row in range(len(mu)):
        scalar = side_pmf(
            mu[row : row + 1], float(alpha[row]) if alpha[row] > 0 else None, COUNT_GRID
        )
        assert np.allclose(batched[row], scalar[0], atol=1e-12)


def test_convolve_rows_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="equal-shaped"):
        convolve_rows(np.ones((2, 5)), np.ones((2, 4)))


def test_known_overdispersion_widens_the_convolved_total():
    """A larger alpha puts more mass in the tails of the total, which is the
    whole reason to carry a dispersion parameter into the convolution."""
    mu_home = np.array([2.0])
    mu_away = np.array([2.0])

    counts, tight = total_pmf(mu_home, mu_away, 1e-9)
    _, wide = total_pmf(mu_home, mu_away, 0.5)

    low = counts <= 1
    high = counts >= 6
    assert wide[:, low].sum() > tight[:, low].sum()
    assert wide[:, high].sum() > tight[:, high].sum()


# ---------------------------------------------------------------------------
# the negative-binomial model
# ---------------------------------------------------------------------------


def test_the_nbinom_model_predicts_the_rate_when_cards_equal_the_rate():
    frame = _rates_equal_cards()
    model = CardsNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)
    model.predict(frame)

    for match_id, expected in zip(
        frame["match_id"], frame["home_rate"] + frame["away_rate"]
    ):
        assert model.predicted_totals[match_id] == pytest.approx(expected, abs=1e-4)


def test_the_nbinom_model_records_both_side_means_and_their_sum():
    frame = _rates_equal_cards()
    model = CardsNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)
    model.predict(frame)

    for match_id in frame["match_id"]:
        home, away = model.predicted_sides[match_id]
        assert model.predicted_totals[match_id] == pytest.approx(home + away, abs=1e-9)


def test_the_nbinom_fit_estimates_a_positive_dispersion_on_overdispersed_data():
    rng = np.random.default_rng(7)
    n = 1500
    alpha = 0.25
    home_rate = rng.uniform(1.0, 3.5, n)
    away_rate = rng.uniform(1.0, 3.5, n)
    home_cards = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * home_rate))
    away_cards = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * away_rate))
    frame = _scored_frame(
        home_cards.tolist(), away_cards.tolist(), home_rate.tolist(), away_rate.tolist()
    )

    model = CardsNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    assert model.alpha_ is not None
    assert 0.1 < model.alpha_ < 0.5  # the true alpha is 0.25


def test_the_nbinom_fit_collapses_towards_poisson_on_poisson_data():
    rng = np.random.default_rng(11)
    n = 1500
    home_rate = rng.uniform(1.0, 3.5, n)
    away_rate = rng.uniform(1.0, 3.5, n)
    home_cards = rng.poisson(home_rate)
    away_cards = rng.poisson(away_rate)
    frame = _scored_frame(
        home_cards.tolist(), away_cards.tolist(), home_rate.tolist(), away_rate.tolist()
    )

    model = CardsNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    assert model.alpha_ is not None
    assert model.alpha_ < 0.05


def test_the_poisson_family_leaves_dispersion_unset():
    model = CardsPoisson(OUTCOMES, line=3.5, k=6.0).fit(_rates_equal_cards())
    assert model.alpha_ is None


def test_the_nbinom_design_is_column_for_column_the_poisson_design():
    frame = _rates_equal_cards()
    poisson_terms = (
        CardsPoisson(OUTCOMES, line=3.5).fit(frame).coefficient_table()["term"].tolist()
    )
    nbinom_terms = (
        CardsNegativeBinomial(OUTCOMES, line=3.5)
        .fit(frame)
        .coefficient_table()["term"]
        .tolist()
    )

    assert nbinom_terms == poisson_terms
    assert "alpha" not in nbinom_terms


def test_the_nbinom_referee_variant_adds_exactly_one_design_column():
    frame = _rates_equal_cards()
    frame["ref_rate"] = 3.0 + np.arange(len(frame)) % 5 * 0.5
    frame["ref_past_count"] = 20.0

    team_terms = set(
        CardsNegativeBinomial(OUTCOMES, line=3.5)
        .fit(frame)
        .coefficient_table()["term"]
        .tolist()
    )
    ref_terms = set(
        CardsTeamRefNegativeBinomial(OUTCOMES, line=3.5)
        .fit(frame)
        .coefficient_table()["term"]
        .tolist()
    )

    assert ref_terms == team_terms | {"log_ref_rate"}


@pytest.mark.parametrize("line", LINES)
def test_nbinom_over_and_under_sum_to_one(line):
    frame = _rates_equal_cards()
    prediction = CardsNegativeBinomial(OUTCOMES, line=line, k=6.0).fit(frame).predict(
        frame
    )

    check_predictions(prediction, list(OUTCOMES))
    assert prediction["prob"].between(0.0, 1.0).all()
    assert (prediction["prob"] > 0.0).all()


def test_the_nbinom_referee_variant_refuses_a_frame_without_ref_rate():
    with pytest.raises(ValueError, match="ref_rate"):
        CardsTeamRefNegativeBinomial(OUTCOMES, line=3.5).fit(_rates_equal_cards())


def test_changing_one_rows_rate_moves_only_that_match_for_nbinom():
    frame = _rates_equal_cards()
    model = CardsNegativeBinomial(OUTCOMES, line=3.5).fit(frame)

    touched = frame.copy()
    touched.loc[5, "away_rate"] *= 4.0
    moved = model.predict(touched).set_index("match_id")["prob"]
    original = model.predict(frame).set_index("match_id")["prob"]

    disturbed = frame.loc[5, "match_id"]
    assert not np.allclose(moved.loc[disturbed], original.loc[disturbed])
    others = original.index != disturbed
    assert np.allclose(moved.loc[others], original.loc[others], atol=1e-6)


# ---------------------------------------------------------------------------
# registry + walk-forward
# ---------------------------------------------------------------------------


def test_the_registry_builds_the_nbinom_models():
    team = create_model("cards_nbinom", {"line": 4.5}, OUTCOMES)
    team_ref = create_model("cards_team_ref_nbinom", {"line": 5.5}, OUTCOMES)

    assert type(team) is CardsNegativeBinomial
    assert type(team_ref) is CardsTeamRefNegativeBinomial
    assert team.line == 4.5
    assert team_ref.line == 5.5
    assert "negative-binomial" in repr(team)
    assert "team+referee" in repr(team_ref)


def test_walk_forward_end_to_end_with_the_nbinom_variant():
    frame = _rates_equal_cards(n=24)
    frame["ref_rate"] = 3.0 + np.arange(len(frame)) % 5 * 0.5
    frame["ref_past_count"] = 20.0
    models: list[CardsTeamRefNegativeBinomial] = []

    def factory() -> CardsTeamRefNegativeBinomial:
        model = CardsTeamRefNegativeBinomial(OUTCOMES, line=3.5, k=6.0)
        models.append(model)
        return model

    result = walk_forward(frame, factory, min_train_days=21, refit_every_days=1)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert result.groupby("match_id").size().eq(2).all()
    assert np.allclose(result.groupby("match_id")["model_prob"].sum(), 1.0)
    assert all(model.alpha_ is not None for model in models)
    captured = {key for model in models for key in model.predicted_totals}
    assert captured == set(result["match_id"])
