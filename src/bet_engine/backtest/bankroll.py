"""Bankroll simulation: bankroll, drawdown, and breach handling.

Every run must be executable with stop_on_breach=True and stop_on_breach=False,
and reports must show both, because "we stopped" and "we kept going" are
different claims (invariant 2).

Two rules shape everything below:

* **A calendar day is the staking atom.** Each bet's stake is a fraction of the
  balance the day *opened* with, not of a bankroll that moves mid-day, so a
  losing morning cannot shrink the afternoon's stakes and a day's stakes can
  never exceed the balance it started with. A frame whose stake_fracs sum above
  1 in one day is refused rather than allowed to drain below zero.
* **A calendar day is also the settlement atom.** Every bet of a day is placed
  and settled before anything is judged: a day whose drawdown crosses the limit
  partway through still settles all of its bets, and the limit is checked only
  against the balance the day *closed* with. A stop therefore lands on a day
  boundary — the next day is never started — and no intraday dip (which the
  day's opening-balance staking has already decided to risk) can cut a day in
  half. This also makes a day's result independent of the order its bets are
  listed in: only the day's closing balance is ever measured.
* **Drawdown is measured from the running peak**, not from the starting bankroll:
  "how far down from the best this run has ever been" is what a drawdown limit
  means to a bettor, and it is the only definition under which a recovered run
  can fall back into breach. Peak and drawdown are both evaluated on closing
  balances — start of run plus end of each day.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..markets.base import valid_price

#: Columns every input frame must carry. The bet size arrives as ``stake_frac``
#: rather than currency so the same frame prices identically under both stop
#: modes — simulate applies the day's opening balance to it.
_REQUIRED_COLUMNS = (
    "match_id",
    "date",
    "market",
    "outcome",
    "odds",
    "won",
    "stake_frac",
)

#: Provenance carried through to the ledger when the input has it, so a stored
#: bet can still say which league and which book priced it.
_PASS_THROUGH = ("league", "odds_source")

#: Ledger columns: the input's provenance plus what the simulation adds.
#: A strict subset of the db ``bets`` table (simulation inputs ``won`` and
#: ``stake_frac`` are not stored facts), so the ledger persists by stamping
#: ``run_id`` on each row — no column mapping, no dropped field.
_LEDGER_TAIL = ("seq_no", "stake", "pnl", "bankroll_after")

#: ISO date prefix: four-digit year first, so day/month order is never guessed.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

#: Float accumulation may put a day's stake_fracs a hair over 1.0; real
#: over-staking is orders of magnitude larger than that.
_DAY_BUDGET_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Simulation:
    """One executed bankroll run: its ledger and the summary of its curve.

    ``bets`` is the ledger in placement order. ``breach_seq_no`` is 1-based and
    names the LAST bet of the day whose closing balance first hit the limit
    (days settle atomically, so the breach is a whole day, not a mid-day bet),
    and it is recorded in *both* stop modes, so "when did it break" is
    answerable even from a run that played on; ``stopped`` then says whether
    any bets were actually left unplaced. ``peak``/``worst_drawdown`` are
    computed from closing balances only.
    """

    bets: pd.DataFrame
    starting: float
    final_bankroll: float
    peak: float
    worst_drawdown: float
    breach_seq_no: int | None
    breach_date: pd.Timestamp | None
    stopped: bool

    @property
    def breached(self) -> bool:
        """Whether the drawdown limit was ever hit, stopped or not."""
        return self.breach_seq_no is not None


@dataclass(frozen=True)
class _Prepared:
    """Validated inputs, flattened into per-bet arrays for the sim loop."""

    records: list[dict[str, Any]]
    days: list[pd.Timestamp]
    odds: list[float]
    won: list[bool]
    stake_fracs: list[float]
    ledger_columns: tuple[str, ...]


def simulate(
    bets: pd.DataFrame,
    starting: float,
    max_drawdown: float,
    stop_on_breach: bool,
) -> Simulation:
    """Walk ``bets`` chronologically and return the bankroll curve's summary.

    WHY the two modes are both first-class (invariant 2): stopping at the
    limit and playing through it make different claims — "the system would
    have been shut down" versus "the system survived the dip" — so the same
    inputs are simulated either way and the breach is recorded either way.
    ``stopped`` is True only when the limit ended the run early: a breach on
    the final bet leaves nothing to stop.

    Stakes come from ``stake_frac`` of the day's opening balance (see module
    docstring), PnL is ``stake * (odds - 1)`` on a win and ``-stake`` on a
    loss. Days settle atomically: every bet of a day is placed and settled,
    then the closing balance is measured against the running peak. In stop
    mode, a day that closes at or below the limit is fully settled, the breach
    is recorded against that day's final seq_no, and no later day is placed.
    """
    if not isinstance(bets, pd.DataFrame):
        raise TypeError(f"bets must be a DataFrame, got {type(bets).__name__}")
    start = _amount(starting, "starting")
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError(f"starting must be a finite amount > 0, got {starting!r}")
    limit = _amount(max_drawdown, "max_drawdown")
    if not (math.isfinite(limit) and 0.0 < limit < 1.0):
        raise ValueError(f"max_drawdown must be in (0, 1), got {max_drawdown!r}")
    if not isinstance(stop_on_breach, (bool, np.bool_)):
        raise TypeError(
            f"stop_on_breach must be a bool, got {type(stop_on_breach).__name__}"
        )
    stop = bool(stop_on_breach)

    prepared = _prepare(bets)
    bankroll = start
    peak = start
    worst = 0.0
    breach_seq: int | None = None
    breach_day: pd.Timestamp | None = None
    current_day: pd.Timestamp | None = None
    day_open = start
    day_end_seq = 0  # seq_no of the last bet placed on current_day
    halted = False
    ledger: list[dict[str, Any]] = []

    def close_day() -> bool:
        """Judge the just-finished day on its closing balance; True = breached."""
        nonlocal peak, worst, breach_seq, breach_day
        if bankroll > peak:
            peak = bankroll
        drawdown = (peak - bankroll) / peak  # peak >= starting > 0, never zero
        if drawdown > worst:
            worst = drawdown
        if breach_seq is None and drawdown >= limit:
            breach_seq = day_end_seq
            breach_day = current_day
            return True
        return False

    for position, record in enumerate(prepared.records):
        day = prepared.days[position]
        if day != current_day:
            if current_day is not None and close_day() and stop:
                halted = True  # judged on a day boundary; that day still settled
            if halted:
                break
            current_day = day
            day_open = bankroll
        stake = prepared.stake_fracs[position] * day_open
        pnl = (
            stake * (prepared.odds[position] - 1.0)
            if prepared.won[position]
            else -stake
        )
        bankroll += pnl
        day_end_seq = position + 1
        placed = dict(record)
        placed["seq_no"] = position + 1
        placed["stake"] = stake
        placed["pnl"] = pnl
        placed["bankroll_after"] = bankroll
        ledger.append(placed)
    else:
        # Ran off the end: the final day still needs its settlement.
        if current_day is not None:
            close_day()

    total = len(prepared.records)
    stopped = stop and breach_seq is not None and breach_seq < total
    ledger_frame = pd.DataFrame(ledger, columns=prepared.ledger_columns)
    # The flag lets eval.bootstrap refuse this ledger on its own (invariant 3)
    # — it travels with the frame, not only with its Simulation, and it is an
    # attr rather than a column because the ledger stays a strict subset of
    # the db bets table.
    ledger_frame.attrs["truncated_by_stop"] = stopped
    return Simulation(
        bets=ledger_frame,
        starting=start,
        final_bankroll=bankroll,
        peak=peak,
        worst_drawdown=worst,
        breach_seq_no=breach_seq,
        breach_date=breach_day,
        stopped=stopped,
    )


def _prepare(bets: pd.DataFrame) -> _Prepared:
    """Validate the input frame once and flatten it into arrays.

    Every rejection lives here so the simulation loop can stay arithmetic: a
    frame that reaches the loop has parseable dates in non-decreasing order,
    usable prices, boolean outcomes, and a per-day stake budget that cannot
    stake itself below zero.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in bets.columns]
    if missing:
        raise ValueError(
            f"bets is missing required column(s) {missing}; have {list(bets.columns)}"
        )
    frame = bets.reset_index(drop=True)

    dates = _parse_dates(frame["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "bets.date must be in chronological (non-decreasing) date order; "
            "the day freeze and seq_no replay both assume it"
        )
    days = dates.dt.normalize()

    odds = pd.to_numeric(frame["odds"], errors="coerce")
    usable = odds.map(valid_price).astype(bool)
    if not usable.all():
        raise ValueError(
            f"odds must be a finite price > 1.0 on every bet; "
            f"bad row(s): {list(frame.index[~usable])}"
        )

    won_ok = frame["won"].isin([True, False])
    if not won_ok.all():
        raise ValueError(
            f"won must be a true/false outcome; "
            f"bad row(s): {list(frame.index[~won_ok])}"
        )

    stakes = pd.to_numeric(frame["stake_frac"], errors="coerce")
    bad_stakes = stakes.isna() | (stakes < 0.0) | (stakes > 1.0)
    if bad_stakes.any():
        raise ValueError(
            f"stake_frac must be numeric and in [0, 1]; "
            f"bad row(s): {list(frame.index[bad_stakes])}"
        )
    # The day's budget IS the balance it opens with, so refusing the overflow
    # here is what guarantees no day can stake itself below zero.
    daily = stakes.groupby(days).sum()
    over = daily[daily > 1.0 + _DAY_BUDGET_TOLERANCE]
    if not over.empty:
        detail = ", ".join(f"{day.date()}: {total:.6g}" for day, total in over.items())
        raise ValueError(
            f"stake_frac sums above 1.0 within a calendar day ({detail}); a day "
            f"cannot stake more than it opens with"
        )

    passthrough = [column for column in _PASS_THROUGH if column in frame.columns]
    work = frame[
        ["match_id", "date", "market", "outcome", "odds", *passthrough]
    ].copy()
    work["date"] = dates
    work["odds"] = odds.astype(float)
    return _Prepared(
        records=work.to_dict("records"),
        days=days.tolist(),
        odds=odds.astype(float).tolist(),
        won=frame["won"].astype(bool).tolist(),
        stake_fracs=stakes.astype(float).tolist(),
        ledger_columns=(*work.columns, *_LEDGER_TAIL),
    )


def _parse_dates(values: pd.Series) -> pd.Series:
    """Calendar dates as datetime64, with day/month order never guessed.

    Text must carry an ISO ``YYYY-MM-DD`` prefix: "01/08/2021" is ambiguous
    between January and August, and a bet booked on the wrong day freezes its
    stake against the wrong opening balance. Missing dates are rejected rather
    than tolerated — the day freeze has no meaning without a day.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        dates = values
    else:
        for value in values.tolist():
            if pd.isna(value):
                continue  # rejected below, as a date with no value
            if isinstance(value, str):
                if not _ISO_DATE.match(value.strip()):
                    raise ValueError(
                        f"date {value!r} is not ISO 'YYYY-MM-DD...'; refusing "
                        f"to guess day/month order"
                    )
            elif not isinstance(value, (date, np.datetime64)):
                raise ValueError(
                    f"date contains {value!r}, which is not a date; expected "
                    f"datetime64 or ISO 'YYYY-MM-DD...' text"
                )
        dates = pd.to_datetime(values, format="mixed", errors="coerce")
    if dates.isna().any():
        raise ValueError(
            f"date must be present and parseable on every bet; "
            f"bad row(s): {list(dates.index[dates.isna()])}"
        )
    return dates


def _amount(value: Any, name: str) -> float:
    """Coerce to float, or raise ValueError naming ``name``.

    One helper so ``"abc"``/``None`` are rejected with the parameter's own
    name instead of Python's TypeError, which does not say which argument was
    wrong.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


__all__ = ["Simulation", "simulate"]
