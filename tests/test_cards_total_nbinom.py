"""Tests for the match-*total* negative-binomial model (Phase 1 step 3b).

Step 3 found that the per-side counts carry practically no dispersion of their
own, while the *total's* variance/mean sits above 1 because home and away cards
are correlated through the match. Step 3b therefore models the total directly:
one row per match, one negative-binomial likelihood, one dispersion estimated
from the data. These tests pin the pieces that make that model what it is:

* the design is match-level (four columns: const, log home rate, log away rate,
  log referee rate) rather than the two-side design the other models share;
* the likelihood really estimates dispersion (recovers a known alpha on
  overdispersed totals, collapses to ~0 on Poisson totals);
* over/under is read off the single total pmf, and the two outcomes partition
  the pmf (sum to 1) exactly as the convolved model does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import WALK_FORWARD_COLUMNS, walk_forward
from bet_engine.models import (
    CardsTeamRefPoisson,
    CardsTotalNegativeBinomial,
    create_model,
)
from bet_engine.models.base import check_predictions

OUTCOMES = ("over", "under")
LINES = (3.5, 4.5, 5.5)


def _total_frame(
    total_cards: list[float],
    home_rate: list[float],
    away_rate: list[float],
    ref_rate: list[float],
    *,
    freq: str = "7D",
) -> pd.DataFrame:
    """A frame carrying the columns add_card_features/add_referee_feature make."""
    n = len(total_cards)
    total = np.asarray(total_cards, dtype=float)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "league": ["E0"] * n,
            "date": pd.date_range("2021-08-01", periods=n, freq=freq),
            "home": [f"home{i}" for i in range(n)],
            "away": [f"away{i}" for i in range(n)],
            "home_cards": np.floor(total / 2.0),
            "away_cards": total - np.floor(total / 2.0),
            "total_cards": total,
            "home_rate": home_rate,
            "away_rate": away_rate,
            "ref_rate": ref_rate,
            "league_mean": [2.0] * n,
        }
    )


def _synthetic(
    *, alpha: float, n: int = 2000, seed: int, freq: str = "7D"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Overdispersed (alpha > 0) or Poisson (alpha == 0) total cards.

    The true mean is a log-linear function of the three rates, which is what the
    model's design can express, so the fitted coefficients are comparable to the
    ones that generated the data.
    """
    rng = np.random.default_rng(seed)
    home_rate = rng.uniform(0.8, 3.0, n)
    away_rate = rng.uniform(0.8, 3.0, n)
    ref_rate = rng.uniform(2.5, 5.0, n)
    beta = np.array([0.3, 0.4, 0.4, 0.3])
    mu = np.exp(
        beta[0]
        + beta[1] * np.log(home_rate)
        + beta[2] * np.log(away_rate)
        + beta[3] * np.log(ref_rate)
    )
    if alpha > 0:
        total = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu))
    else:
        total = rng.poisson(mu)
    frame = _total_frame(
        total.tolist(),
        home_rate.tolist(),
        away_rate.tolist(),
        ref_rate.tolist(),
        freq=freq,
    )
    return frame, mu


def test_the_registry_builds_the_total_model():
    model = create_model("cards_total_nbinom", {"line": 4.5}, OUTCOMES)

    assert type(model) is CardsTotalNegativeBinomial
    assert model.line == 4.5
    assert "negative-binomial" in repr(model)
    assert "total" in repr(model)


def test_the_design_is_match_level_with_four_named_terms():
    frame, _ = _synthetic(alpha=0.2, n=400, seed=5)
    model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    terms = model.coefficient_table()["term"].tolist()

    assert terms == ["const", "log_home_rate", "log_away_rate", "log_ref_rate"]


def test_the_fit_recovers_the_dispersion_of_overdispersed_totals():
    frame, _ = _synthetic(alpha=0.25, n=2000, seed=7)

    model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    assert model.alpha_ is not None
    assert 0.1 < model.alpha_ < 0.5  # the true alpha is 0.25


def test_the_fit_collapses_towards_poisson_on_poisson_totals():
    frame, _ = _synthetic(alpha=0.0, n=2000, seed=11)

    model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    assert model.alpha_ is not None
    assert model.alpha_ < 0.05


def test_the_fit_recovers_the_generating_coefficients():
    frame, _ = _synthetic(alpha=0.2, n=3000, seed=13)

    table = (
        CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0)
        .fit(frame)
        .coefficient_table()
        .set_index("term")["coef"]
    )

    # beta = [0.3, 0.4, 0.4, 0.3]; generous tolerance for finite-sample noise.
    assert table["const"] == pytest.approx(0.3, abs=0.15)
    assert table["log_home_rate"] == pytest.approx(0.4, abs=0.15)
    assert table["log_away_rate"] == pytest.approx(0.4, abs=0.15)
    assert table["log_ref_rate"] == pytest.approx(0.3, abs=0.15)


def test_the_model_records_the_expected_total_per_match():
    frame, mu = _synthetic(alpha=0.2, n=600, seed=17)
    model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)
    model.predict(frame)

    predicted = np.array([model.predicted_totals[mid] for mid in frame["match_id"]])

    assert set(model.predicted_totals) == set(frame["match_id"])
    assert np.corrcoef(predicted, mu)[0, 1] > 0.99
    assert np.mean(np.abs(predicted - mu) / mu) < 0.05


@pytest.mark.parametrize("line", LINES)
def test_the_total_model_over_and_under_sum_to_one(line):
    frame, _ = _synthetic(alpha=0.2, n=300, seed=19)
    prediction = (
        CardsTotalNegativeBinomial(OUTCOMES, line=line, k=6.0)
        .fit(frame)
        .predict(frame)
    )

    check_predictions(prediction, list(OUTCOMES))
    assert prediction["prob"].between(0.0, 1.0).all()
    assert (prediction["prob"] > 0.0).all()


def test_the_total_model_refuses_a_frame_without_ref_rate():
    frame, _ = _synthetic(alpha=0.2, n=100, seed=23)

    with pytest.raises(ValueError, match="ref_rate"):
        CardsTotalNegativeBinomial(OUTCOMES, line=3.5).fit(frame.drop(columns="ref_rate"))


def test_changing_one_rows_rate_moves_only_that_match():
    frame, _ = _synthetic(alpha=0.2, n=40, seed=29)
    model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0).fit(frame)

    touched = frame.copy()
    touched.loc[5, "away_rate"] *= 4.0
    moved = model.predict(touched).set_index("match_id")["prob"]
    original = model.predict(frame).set_index("match_id")["prob"]

    disturbed = frame.loc[5, "match_id"]
    assert not np.allclose(moved.loc[disturbed], original.loc[disturbed])
    others = original.index != disturbed
    assert np.allclose(moved.loc[others], original.loc[others], atol=1e-6)


def test_walk_forward_end_to_end_with_the_total_model():
    # Daily dates: the folds need enough training rows for a four-column design
    # plus the dispersion parameter, which a 7-day cadence would not provide.
    frame, _ = _synthetic(alpha=0.2, n=40, seed=31, freq="D")
    models: list[CardsTotalNegativeBinomial] = []

    def factory() -> CardsTotalNegativeBinomial:
        model = CardsTotalNegativeBinomial(OUTCOMES, line=3.5, k=6.0)
        models.append(model)
        return model

    result = walk_forward(frame, factory, min_train_days=21, refit_every_days=1)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert result.groupby("match_id").size().eq(2).all()
    assert np.allclose(result.groupby("match_id")["model_prob"].sum(), 1.0)
    assert all(model.alpha_ is not None for model in models)
    captured = {key for model in models for key in model.predicted_totals}
    assert captured == set(result["match_id"])


def test_the_total_design_differs_from_the_side_designs():
    """The whole point of step 3b: a match-level target and design, not a
    variant of the two-side models."""
    frame, _ = _synthetic(alpha=0.2, n=200, seed=37)
    total_terms = set(
        CardsTotalNegativeBinomial(OUTCOMES, line=3.5)
        .fit(frame)
        .coefficient_table()["term"]
    )
    side_terms = set(
        CardsTeamRefPoisson(OUTCOMES, line=3.5).fit(frame).coefficient_table()["term"]
    )

    assert total_terms == {"const", "log_home_rate", "log_away_rate", "log_ref_rate"}
    assert side_terms == {
        "const",
        "log_own_rate",
        "log_opp_rate",
        "log_ref_rate",
        "home_flag",
    }
    assert total_terms != side_terms
