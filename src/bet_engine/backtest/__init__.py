"""Backtest engine: fold iteration, staking, bankroll."""

from .bankroll import Simulation, simulate
from .staking import devig, ev, kelly_stake, select_bets
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
    "devig",
    "ev",
    "kelly_stake",
    "select_bets",
    "simulate",
    "split_by_day",
    "walk_forward",
]
