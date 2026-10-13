"""Tests for backtest/over_under.py: void-aware settlement and PnL.

A whole-number cards line can push, so the generic bankroll's two-outcome "won
or lost" is not enough. These tests pin the three-way result mapping and the
hand-calculated PnL: a push returns the stake (PnL exactly 0) and is *not* a win,
which is the difference a report must not blur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import (
    bet_pnl,
    bet_result,
    is_push,
    settle_bets,
    total_pnl,
)
from bet_engine.markets.base import PUSH

BETS = ["match_id", "outcome", "odds", "stake"]


def _bets(rows):
    return pd.DataFrame(rows, columns=BETS)


# --- the three-way result ---------------------------------------------------


def test_a_settled_win_loss_and_push_are_told_apart():
    assert bet_result("over", "over") == "win"
    assert bet_result("over", "under") == "loss"
    assert bet_result("over", PUSH) == "push"
    assert bet_result("under", PUSH) == "push"


def test_an_unsettled_match_is_none_not_a_push():
    assert bet_result("over", None) is None
    assert bet_result("over", np.nan) is None
    assert bet_result("over", pd.NA) is None


def test_is_push_only_matches_the_void_sentinel():
    assert is_push(PUSH)
    assert not is_push("over")
    assert not is_push("under")
    assert not is_push(None)


# --- hand-calculated PnL ----------------------------------------------------


def test_bet_pnl_matches_the_hand_calculation():
    assert bet_pnl(2.5, 10.0, "win") == pytest.approx(15.0)  # 10 * (2.5 - 1)
    assert bet_pnl(2.5, 10.0, "loss") == pytest.approx(-10.0)
    assert bet_pnl(2.5, 10.0, "push") == pytest.approx(0.0)


def test_a_push_does_not_move_the_bankroll():
    assert bet_pnl(4.0, 100.0, "push") == 0.0


def test_bet_pnl_refuses_bad_inputs():
    with pytest.raises(ValueError, match="odds"):
        bet_pnl(1.0, 10.0, "win")  # price at even money is not a real price
    with pytest.raises(ValueError, match="odds"):
        bet_pnl(float("inf"), 10.0, "win")
    with pytest.raises(ValueError, match="stake"):
        bet_pnl(2.0, -1.0, "win")
    with pytest.raises(ValueError, match="result"):
        bet_pnl(2.0, 10.0, "draw")


# --- settle_bets ------------------------------------------------------------


def test_settle_bets_adds_result_won_void_and_pnl():
    bets = _bets(
        [
            ("m1", "over", 2.0, 10.0),
            ("m2", "over", 3.0, 10.0),
            ("m3", "under", 1.5, 10.0),
        ]
    )
    winners = {"m1": "over", "m2": "under", "m3": PUSH}

    settled = settle_bets(bets, winners)

    assert list(settled["result"]) == ["win", "loss", "push"]
    assert list(settled["won"]) == [True, False, False]
    assert list(settled["void"]) == [False, False, True]
    assert settled["pnl"].tolist() == pytest.approx([10.0, -10.0, 0.0])


def test_a_push_is_void_but_not_won():
    bets = _bets([("m1", "over", 2.0, 5.0)])
    settled = settle_bets(bets, {"m1": PUSH})

    row = settled.iloc[0]
    assert row["void"] and not row["won"]
    assert row["pnl"] == 0.0


def test_settle_bets_refuses_a_bet_on_an_unsettled_match():
    bets = _bets([("m1", "over", 2.0, 5.0), ("m2", "under", 2.5, 5.0)])

    with pytest.raises(ValueError, match="unsettled match"):
        settle_bets(bets, {"m1": "over"})  # m2 has no result


def test_settle_bets_refuses_a_bad_price():
    bets = _bets([("m1", "over", float("nan"), 5.0)])
    with pytest.raises(ValueError, match="odds"):
        settle_bets(bets, {"m1": "over"})


def test_settle_bets_refuses_a_bad_stake():
    bets = _bets([("m1", "over", 2.0, -1.0)])
    with pytest.raises(ValueError, match="stake"):
        settle_bets(bets, {"m1": "over"})


def test_settle_bets_needs_the_required_columns():
    with pytest.raises(ValueError, match="missing required column"):
        settle_bets(pd.DataFrame({"match_id": ["m1"]}), {"m1": "over"})


def test_settle_bets_honours_a_renamed_stake_column():
    bets = pd.DataFrame(
        [{"match_id": "m1", "outcome": "over", "odds": 2.0, "units": 4.0}]
    )
    settled = settle_bets(bets, {"m1": "over"}, stake_column="units")

    assert settled["pnl"].tolist() == pytest.approx([4.0])


# --- total_pnl --------------------------------------------------------------


def test_total_pnl_sums_the_sequence_with_voids_at_zero():
    bets = _bets(
        [
            ("m1", "over", 2.0, 10.0),   # win  +10
            ("m2", "over", 2.0, 10.0),   # push   0
            ("m3", "over", 2.0, 10.0),   # loss -10
        ]
    )
    winners = {"m1": "over", "m2": PUSH, "m3": "under"}

    assert total_pnl(bets, winners) == pytest.approx(0.0)
