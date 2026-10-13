"""End-to-end synthetic proof of the Part B cards over/under pipeline.

No real odds exist yet, so this test drives the *real* modules — the push-aware
market, the truth model's probabilities, the odds join, selection, push-aware
settlement and the full-sequence bootstrap — over data whose true probabilities
and true edge are known by construction. That is what lets each assertion be a
statement about correctness rather than about our own output:

* outcomes are drawn from a Poisson whose line probabilities are computed, so the
  settled over/under/push label is ground truth, not something the code invented;
* the book is priced to a fixed overround (so the pure-market control provably
  has no edge), and one known subset is deliberately mispriced in our favour, so
  a correct selector must bet exactly that subset and profit on it;
* the bootstrap then has to tell the known edge apart from the no-edge matches.

The lines are whole numbers (3, 4, 5) so a genuine push really is exercised, and
the report is grouped by line as the task requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from bet_engine.backtest import select_bets, select_bets_at_percentile, settle_bets
from bet_engine.eval import bootstrap_pnl
from bet_engine.markets.cards_totals import CardsTotalLine
from bet_engine.models.pure_market import PureMarket

LINES = (3, 4, 5)
BOOK_MARGIN = 0.05  # the book's overround minus one; > 0 so the control is clean
MISPRICE = 0.08  # how far below the true over probability the edge price sits
EDGE_FRACTION = 0.4
STAKE = 10.0
N = 3000
SEED = 20260101


def _synthetic():
    """Generate matches, true line probabilities, book prices, and outcomes.

    Returns a frame carrying, per match: the line, the true over/under/push
    probabilities, the priced over/under odds (overround exactly ``1+BOOK_MARGIN``
    for every match), whether the match is in the mispriced subset, and the
    *actually drawn* settled label from a Poison total.
    """
    rng = np.random.default_rng(SEED)
    n = N
    line = rng.choice(LINES, size=n).astype(float)
    mu = rng.uniform(3.0, 6.0, n)

    p_over = 1.0 - poisson.cdf(line, mu)
    p_under = poisson.cdf(line - 1.0, mu)
    p_push = poisson.pmf(line, mu)

    is_edge = rng.random(n) < EDGE_FRACTION

    # Every match is priced to the same overround ``1 + BOOK_MARGIN`` so the
    # control's de-vigged edge is the fixed negative ``1/(1+BOOK_MARGIN) - 1``.
    #
    # No-edge matches: both sides are priced above their *push-aware* fair level
    # (fair implied over is ``p_over / (1 - p_push)`` because a push returns the
    # stake), so the push-aware EV of either side is negative — the bet truly has
    # no edge, not merely a negative two-outcome approximation.
    #
    # Edge matches: implied-over is dropped below the true over probability, so
    # the selector's ``p_over * odds - 1`` turns positive on the over side only;
    # implied-under absorbs the difference and stays negative.
    fair_over_implied = p_over * (1.0 + BOOK_MARGIN) / (1.0 - p_push)
    fair_under_implied = p_under * (1.0 + BOOK_MARGIN) / (1.0 - p_push)
    implied_over = np.where(is_edge, p_over * (1.0 - MISPRICE), fair_over_implied)
    implied_under = np.where(
        is_edge, (1.0 + BOOK_MARGIN) - implied_over, fair_under_implied
    )
    over_odds = 1.0 / implied_over
    under_odds = 1.0 / implied_under

    total = poisson.rvs(mu, random_state=np.random.default_rng(SEED + 1))
    market = CardsTotalLine()
    settled = [
        market.settle({"hy": t, "ay": 0, "hr": 0, "ar": 0, "line": ln})
        for t, ln in zip(total, line)
    ]

    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "date": pd.date_range("2021-08-01", periods=n, freq="h"),
            "home": [f"home{i}" for i in range(n)],
            "away": [f"away{i}" for i in range(n)],
            "line": line,
            "p_over": p_over,
            "p_under": p_under,
            "p_push": p_push,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "is_edge": is_edge,
            "settled": settled,
        }
    )
    return frame


def _winners(frame):
    return dict(zip(frame["match_id"], frame["settled"]))


def _truth_predictions(frame):
    """The truth model's over/under probabilities, in the selection long form."""
    over = frame[["match_id", "over_odds", "p_over"]].copy()
    over.columns = ["match_id", "odds", "model_prob"]
    over["market"] = "cards_total"
    over["outcome"] = "over"
    under = frame[["match_id", "under_odds", "p_under"]].copy()
    under.columns = ["match_id", "odds", "model_prob"]
    under["market"] = "cards_total"
    under["outcome"] = "under"
    return pd.concat([over, under], ignore_index=True)


def test_the_truth_model_bets_exactly_the_mispriced_subset():
    frame = _synthetic()
    predictions = _truth_predictions(frame)

    bets = select_bets(predictions, ev_threshold=0.0)

    assert set(bets["outcome"]) == {"over"}
    assert set(bets["match_id"]) == set(frame.loc[frame["is_edge"], "match_id"])
    # One bet per match, and every one carries a usable price.
    assert bets["match_id"].is_unique
    assert (bets["odds"] > 1.0).all()


def test_the_mispriced_subset_profits_and_the_rest_does_not():
    frame = _synthetic()
    winners = _winners(frame)
    bets = select_bets(_truth_predictions(frame), ev_threshold=0.0)
    bets = bets.assign(stake=STAKE)

    settled = settle_bets(bets, winners)
    edge_pnl = settled["pnl"].sum()
    # A real, positive return on a large sample, not a single lucky bet.
    assert len(settled) > 500
    assert edge_pnl > 0.0
    assert edge_pnl / (len(settled) * STAKE) > 0.01

    # The no-edge matches the model declined would have lost, on average.
    no_edge = frame.loc[
        ~frame["is_edge"] & (frame["settled"] != "push"), :
    ].copy()
    no_edge["pnl"] = np.where(
        no_edge["settled"] == "over",
        STAKE * (no_edge["over_odds"] - 1.0),
        -STAKE,
    )
    assert no_edge["pnl"].mean() < 0.0


def test_a_push_returns_the_stake_and_is_excluded_from_scoring():
    frame = _synthetic()
    winners = _winners(frame)
    bets = select_bets(_truth_predictions(frame), ev_threshold=0.0)
    bets = bets.assign(stake=STAKE)

    settled = settle_bets(bets, winners)

    pushes = settled.loc[settled["void"]]
    assert len(pushes) > 0  # whole lines really did land exactly
    assert (pushes["result"] == "push").all()
    assert np.allclose(pushes["pnl"], 0.0)
    assert not pushes["won"].any()


def test_the_bootstrap_tells_the_known_edge_from_the_no_edge():
    frame = _synthetic()
    winners = _winners(frame)

    edge_bets = select_bets(_truth_predictions(frame), ev_threshold=0.0).assign(
        stake=STAKE
    )
    edge_pnl = settle_bets(edge_bets, winners)["pnl"].to_numpy()

    # A no-edge sequence: bet the over on every no-edge match. The selector
    # declined these (its ``p*odds-1`` is negative), and the push-aware EV is
    # negative too, so the bootstrap must read it as no edge.
    no_edge = frame.loc[~frame["is_edge"], ["match_id", "over_odds"]].copy()
    no_edge_bets = no_edge.rename(columns={"over_odds": "odds"}).assign(
        market="cards_total", outcome="over", model_prob=0.5, stake=STAKE
    )
    no_edge_pnl = settle_bets(no_edge_bets, winners)["pnl"].to_numpy()

    edge = bootstrap_pnl(edge_pnl, n=4000, seed=7)
    none = bootstrap_pnl(no_edge_pnl, n=4000, seed=7)

    assert edge.fraction_profitable > 0.95
    assert edge.mean_roi > 0.0
    assert edge.fraction_profitable > none.fraction_profitable


def test_the_pure_market_control_places_zero_bets_on_both_paths():
    frame = _synthetic()
    control = PureMarket(
        ("over", "under"),
        odds_columns={"over": "over_odds", "under": "under_odds"},
    )

    # Point path: the control's de-vigged edge is 1/(1+margin) - 1 < 0 everywhere.
    point = control.predict(frame).rename(columns={"prob": "model_prob"})
    predictions = point.assign(
        market="cards_total",
        odds=[
            frame.loc[frame["match_id"] == match_id, f"{outcome}_odds"].item()
            for match_id, outcome in zip(point["match_id"], point["outcome"])
        ],
    )
    assert select_bets(predictions, ev_threshold=0.0).empty

    # Percentile path: the control has no uncertainty, so every percentile is the
    # same de-vigged number, and it must still place zero bets.
    samples = control.predict_samples(frame, n=32)
    for percentile in (10.0, 50.0, 90.0):
        assert select_bets_at_percentile(
            predictions, samples, 0.0, percentile
        ).empty


def test_results_are_reported_per_line():
    frame = _synthetic()
    winners = _winners(frame)
    bets = select_bets(_truth_predictions(frame), ev_threshold=0.0).assign(
        stake=STAKE
    )
    bets = bets.merge(frame[["match_id", "line"]], on="match_id")

    settled = settle_bets(bets, winners)
    per_line = settled.groupby("line").agg(
        n_bets=("pnl", "size"), total_pnl=("pnl", "sum"), pushes=("void", "sum")
    )

    assert list(per_line.index) == [float(line) for line in LINES]
    assert (per_line["n_bets"] > 0).all()
    assert (per_line["total_pnl"] > 0.0).all()
