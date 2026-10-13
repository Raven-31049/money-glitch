"""Void-aware settlement and PnL for the over/under cards market (Part B).

The generic bankroll (backtest/bankroll.py) knows two outcomes: a bet won or
lost. A cards over/under at a *whole-number* line can also **push** — the total
lands exactly on the line, the stake comes back, and no one wins
(``markets.base.PUSH``). Modelling that as a loss would understate the run;
modelling it as a win would overstate it. So this module is the small, tested
layer that maps a bet and its match's settled label to one of three results and
the matching PnL, and the generic bankroll is deliberately left alone (a push is
a cards-line concept, and keeping it out of bankroll keeps that engine
market-agnostic — invariant 8).

WHY a push is excluded from calibration, not scored 0/1: a win/loss reliability
or Brier score asks "how often was the event right?"; a push has no event to be
right about, so it is dropped and counted, never scored as an under or a loss.
Callers filter ``void`` rows out of the scoring frame (see the flag this module
returns) instead of letting a fabricated 0/1 leak into a score.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..markets.base import PUSH, valid_price

#: Result labels this module produces from a bet and its settled match.
RESULT_COLUMNS: tuple[str, ...] = ("result", "won", "void", "pnl")


def is_push(settled_label: object) -> bool:
    """True when a settled match label is the void state (exact hit on the line)."""
    return settled_label == PUSH


def bet_result(outcome: str, settled_label: object) -> str | None:
    """``"win"``/``"loss"``/``"push"`` for one bet, or None if unsettled.

    None means the match has no result yet (``settle`` returned None): the bet
    is neither settled nor scored, distinct from a push, which *is* settled and
    simply pays nothing.
    """
    if settled_label is None or settled_label is pd.NA:
        return None
    if _isna(settled_label):
        return None
    if is_push(settled_label):
        return "push"
    return "win" if settled_label == outcome else "loss"


def bet_pnl(odds: float, stake: float, result: str) -> float:
    """PnL of ``stake`` at decimal ``odds`` for a settled ``result``.

    ``win`` pays ``stake * (odds - 1)``; ``loss`` pays ``-stake``; ``push``
    returns the stake, so its PnL is exactly ``0.0`` (the bet is void, the
    bankroll does not move). A zero or negative stake is not a bet this
    function prices — it returns 0.0, matching how fraction/kelly staking
    refuses a non-positive edge — and a non-finite or non-positive price is an
    error, not a silent number.
    """
    price = _number(odds, "odds")
    if not valid_price(price):
        raise ValueError(f"odds must be finite and > 1.0, got {odds!r}")
    amount = _number(stake, "stake")
    if not np.isfinite(amount) or amount < 0.0:
        raise ValueError(f"stake must be a finite amount >= 0, got {stake!r}")
    if result == "win":
        return amount * (price - 1.0)
    if result == "loss":
        return -amount
    if result == "push":
        return 0.0
    raise ValueError(f"result must be 'win', 'loss' or 'push', got {result!r}")


def settle_bets(
    bets: pd.DataFrame,
    winners: Mapping[str, object],
    *,
    stake_column: str = "stake",
) -> pd.DataFrame:
    """``bets`` plus ``result``, ``won``, ``void`` and ``pnl``.

    ``winners`` maps ``match_id`` to what the market settled it as: a winning
    outcome label, ``markets.base.PUSH`` for a void match, or None for an
    unsettled one. A bet on an unsettled match is an error — a bet must be
    settled to have a PnL — and the message names the matches.

    ``won`` is True only for an outright win (a push is *not* a win); ``void``
    is True only for a push, so a caller scores calibration on
    ``~void`` rows and lets a void carry zero PnL. ``pnl`` is in the same
    currency as ``stake_column`` (the per-bet amount, not a bankroll fraction).
    """
    if not isinstance(bets, pd.DataFrame):
        raise TypeError(f"bets must be a DataFrame, got {type(bets).__name__}")
    required = ("match_id", "outcome", "odds", stake_column)
    missing = [column for column in required if column not in bets.columns]
    if missing:
        raise ValueError(
            f"bets is missing required column(s) {missing}; have {list(bets.columns)}"
        )
    frame = bets.reset_index(drop=True)

    labels = frame["match_id"].map(lambda match_id: winners.get(match_id))
    unresolved = labels.map(_isna).to_numpy(dtype=bool)
    if unresolved.any():
        bad = frame.loc[unresolved, "match_id"].unique()[:5].tolist()
        raise ValueError(
            f"bet(s) on unsettled match(es) {bad}: a bet must be settled before "
            f"its PnL is defined"
        )

    odds = pd.to_numeric(frame["odds"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(odds) & (odds > 1.0)
    if not usable.all():
        bad = frame.loc[~usable, "match_id"].unique()[:5].tolist()
        raise ValueError(f"odds must be a finite price > 1.0; bad match(es) {bad}")

    stakes = pd.to_numeric(frame[stake_column], errors="coerce").to_numpy(dtype=float)
    stake_ok = np.isfinite(stakes) & (stakes >= 0.0)
    if not stake_ok.all():
        bad = frame.loc[~stake_ok, "match_id"].unique()[:5].tolist()
        raise ValueError(
            f"stake must be a finite amount >= 0 on every bet; bad match(es) {bad}"
        )

    void = labels.to_numpy() == PUSH
    won = (~void) & (labels.to_numpy() == frame["outcome"].to_numpy())
    pnl = np.where(
        void, 0.0, np.where(won, stakes * (odds - 1.0), -stakes)
    )
    settled = frame.copy()
    settled["result"] = np.where(void, "push", np.where(won, "win", "loss"))
    settled["won"] = won
    settled["void"] = void
    settled["pnl"] = pnl
    return settled


def total_pnl(bets: pd.DataFrame, winners: Mapping[str, object], *, stake_column: str = "stake") -> float:
    """Sum of :func:`settle_bets`' ``pnl`` over ``bets``.

    A convenience for tests and reports: the money a sequence of settled bets
    returned, voids included at exactly 0.
    """
    settled = settle_bets(bets, winners, stake_column=stake_column)
    return float(settled["pnl"].sum())


def _isna(value: object) -> bool:
    """True for None, NaN, NaT and pandas' NA (not for an outcome label)."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


def _number(value: Any, name: str) -> float:
    """Coerce to float, or raise ValueError naming ``name``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


__all__ = [
    "RESULT_COLUMNS",
    "bet_pnl",
    "bet_result",
    "is_push",
    "settle_bets",
    "total_pnl",
]
