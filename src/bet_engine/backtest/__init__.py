"""Backtest engine: fold iteration, staking, bankroll."""

from .bankroll import Simulation, simulate
from .over_under import bet_pnl, bet_result, is_push, settle_bets, total_pnl
from .staking import devig, ev, kelly_stake, select_bets, select_bets_at_percentile
from .walkforward import (
    WALK_FORWARD_COLUMNS,
    SplitFn,
    assert_train_before_predict,
    split_by_day,
    walk_forward,
)

__all__ = [
    "WALK_FORWARD_COLUMNS",
    "Simulation",
    "SplitFn",
    "assert_train_before_predict",
    "bet_pnl",
    "bet_result",
    "devig",
    "ev",
    "is_push",
    "kelly_stake",
    "select_bets",
    "select_bets_at_percentile",
    "settle_bets",
    "simulate",
    "split_by_day",
    "total_pnl",
    "walk_forward",
]
