"""Tests for models/pure_market.py: the shared de-vig and the control model.

Invariant 4 rests on one arithmetic identity: the control's ``model_prob``
must be the run's own ``market_prob``, bit for bit. EV is measured on the raw
odds actually paid (PLAN.md section 5): the de-vigged probability times the
book's price is the inverse overround, so the control's edge is
``1 / overround - 1``, strictly negative, and ``select_bets``' strict ``>``
refuses every row. These tests prove that identity rather than trusting it,
and prove the edge itself can never come out positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.markets import get_market
from bet_engine.models import PureMarket, devig_long
from bet_engine.run import market_long

OUTCOMES = ("H", "D", "A")


def _priced_frame(n: int = 4, seed: int = 7) -> pd.DataFrame:
    """``n`` matches with plausible decimal prices (no other columns needed)."""
    rng = np.random.default_rng(seed)
    odds = rng.uniform(1.2, 15.0, size=(n, 3))
    data: dict[str, object] = {"match_id": [f"m{i}" for i in range(n)]}
    for column, values in zip((f"odds_{o}" for o in OUTCOMES), odds.T):
        data[column] = values
    return pd.DataFrame(data)


# --- devig_outcomes ----------------------------------------------------------

def test_devig_matches_a_hand_calculation():
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "odds_H": [2.0],
            "odds_D": [4.0],
            "odds_A": [5.0],
        }
    )

    long = devig_long(frame, OUTCOMES, value_name="market_prob")

    # implied 0.5/0.25/0.2 sum to 0.95; each divided by the row sum.
    assert long["market_prob"].tolist() == pytest.approx(
        [0.5 / 0.95, 0.25 / 0.95, 0.2 / 0.95]
    )
    assert long["outcome"].tolist() == ["H", "D", "A"]
    assert long["match_id"].tolist() == ["m1", "m1", "m1"]


def test_devig_output_is_long_form_with_rows_sums_of_one():
    frame = _priced_frame(5)

    long = devig_long(frame, OUTCOMES, value_name="market_prob")

    assert list(long.columns) == ["match_id", "outcome", "market_prob"]
    assert len(long) == 5 * len(OUTCOMES)
    sums = long.groupby("match_id")["market_prob"].sum()
    assert sums.tolist() == pytest.approx([1.0] * 5)


def test_devig_on_a_batch_equals_devig_on_the_full_frame_bit_for_bit():
    """The invariant-4 identity: batch size must not change a single bit.

    The run computes ``market_prob`` over the whole frame while the control
    predicts day-batches; if either path were batch-dependent, the control's
    edge would be float noise instead of exactly 0.0.
    """
    frame = _priced_frame(6)
    full = devig_long(frame, OUTCOMES, value_name="market_prob")
    full_indexed = full.set_index(["match_id", "outcome"]).sort_index()

    for keep in ([0], [3], [1, 4], [2, 0, 5]):
        batch = devig_long(frame.iloc[keep], OUTCOMES, value_name="market_prob")
        batch_indexed = batch.set_index(["match_id", "outcome"]).sort_index()
        expected = full_indexed.loc[batch_indexed.index, "market_prob"]
        assert (
            batch_indexed["market_prob"].to_numpy()
            == expected.to_numpy()
        ).all(), f"batch {keep} differed from the full frame"


def test_devig_leaves_rows_without_a_price_as_nan():
    frame = _priced_frame(3)
    frame.loc[1, "odds_D"] = np.nan

    long = devig_long(frame, OUTCOMES, value_name="market_prob")

    poisoned = long.loc[long["match_id"] == "m1", "market_prob"]
    assert poisoned.isna().all()
    untouched = long.loc[long["match_id"].isin(["m0", "m2"]), "market_prob"]
    assert untouched.notna().all()


def test_devig_rejects_input_it_cannot_arithmetic_on():
    with pytest.raises(TypeError, match="DataFrame"):
        devig_long([1, 2, 3], OUTCOMES)
    with pytest.raises(ValueError, match="must not be empty"):
        devig_long(_priced_frame(1), ())
    with pytest.raises(ValueError, match="missing price column"):
        devig_long(pd.DataFrame({"match_id": ["m1"]}), OUTCOMES)


# --- the control model -------------------------------------------------------

def test_control_prediction_is_the_runs_market_prob_exactly():
    """Bit-identity between the control and the run's own market column."""
    frame = _priced_frame(8)
    market = get_market("match_winner")

    from_run = market_long(frame, market).rename(columns={"market_prob": "run_prob"})
    from_model = PureMarket(market.outcomes).predict(frame)

    merged = from_model.merge(
        from_run, on=["match_id", "outcome"], validate="one_to_one"
    )
    assert len(merged) == len(from_model)
    assert (
        merged["prob"].to_numpy() == merged["run_prob"].to_numpy()
    ).all(), "control prediction is not bit-identical to market_prob"


def test_control_edge_is_never_positive():
    """``market_prob * raw_odds - 1`` must never exceed 0 for a book's prices.

    This is invariant 4's arithmetic core under the raw-odds EV definition
    (PLAN.md section 5). A real book prices every outcome with a margin, so the
    implied probabilities sum to an overround >= 1; the control's edge is then
    ``1 / overround - 1 <= 0``. A positive edge here would mean the control
    could bet, which is a bug in the EV code, not an edge.
    """
    rng = np.random.default_rng(11)
    n = 200
    fair = rng.dirichlet(np.ones(3), size=n)  # (n, 3), each row sums to 1
    overround = 1.0 + rng.uniform(0.01, 0.15, size=n)  # book margin, always > 1
    prices = 1.0 / (fair * overround[:, None])  # (n, 3) book prices

    frame = pd.DataFrame({"match_id": [f"m{i}" for i in range(n)]})
    for column, values in zip((f"odds_{o}" for o in OUTCOMES), prices.T):
        frame[column] = values

    long = devig_long(frame, OUTCOMES, value_name="p")
    raw = frame.melt(
        id_vars="match_id",
        value_vars=[f"odds_{o}" for o in OUTCOMES],
        var_name="column",
        value_name="raw_odds",
    )
    raw["outcome"] = raw["column"].str.removeprefix("odds_")
    merged = long.merge(
        raw[["match_id", "outcome", "raw_odds"]], on=["match_id", "outcome"]
    )

    edges = merged["p"] * merged["raw_odds"] - 1.0

    assert (edges <= 0.0).all()
    assert edges.max() <= 0.0


def test_control_fit_learns_nothing_and_returns_itself():
    model = PureMarket(OUTCOMES)
    train = _priced_frame(3)

    assert model.fit(train) is model

    # Predictions before and after seeing training rows are the same: the
    # control has no state to estimate.
    frame = _priced_frame(2, seed=99)
    before = model.predict(frame)
    model.fit(_priced_frame(10, seed=5))
    after = model.predict(frame)
    pd.testing.assert_frame_equal(before, after)


def test_control_refuses_a_row_it_cannot_price():
    frame = _priced_frame(2)
    frame.loc[1, "odds_A"] = np.nan
    model = PureMarket(OUTCOMES)

    with pytest.raises(ValueError, match="cannot price.*m1"):
        model.predict(frame)


def test_control_requires_outcomes():
    with pytest.raises(ValueError, match="must not be empty"):
        PureMarket(())
