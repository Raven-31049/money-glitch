"""Staking: edge threshold, fractional Kelly, hard per-bet caps.

The pure market control (de-vigged odds, zero model weight) must place ~0 bets,
so this module's threshold behaviour is the tripwire for EV/Kelly bugs
(invariant 4). Everything here is therefore written to be provably strict: the
edge comparison is ``>`` (never ``>=``) so an exactly-fair price never clears a
zero threshold, and a row with no price is unselectable rather than
selectable-by-accident.

De-vig lives here rather than in probability/ because the edge filter cannot
mean anything without it: an EV measured against an overrounded price mixes
model edge with bookmaker margin, and a control that compares *that* to a
threshold is comparing the wrong quantity. Proportional de-vig is the only
method for now; ``devig`` stays importable from here meanwhile.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from ..markets.base import Outcome, valid_price

#: What a frame must carry before an edge can be computed at all.
_REQUIRED_COLUMNS = ("match_id", "market", "outcome", "odds", "model_prob")


def devig(odds: Mapping[Outcome, float]) -> tuple[dict[Outcome, float], float]:
    """Proportionally remove the overround: ``(probabilities, overround)``.

    Each price's implied probability (``1 / odds``) is rescaled by the sum of
    the whole vector — the standard multiplicative de-vig. It keeps the
    market's ordering (the favourite stays the favourite) and needs nothing but
    the prices themselves, which is why it needs no book-level prior. The
    overround comes back alongside the probabilities so a caller can still
    report how much vig the run paid instead of losing it inside the division.

    Every price must be a usable one (``valid_price``). One corrupt or missing
    price would distort *every* probability in the vector, because the sum of
    the vector is the denominator — so a single bad cell is an error here, not
    a skipped entry.
    """
    if not odds:
        raise ValueError("no prices supplied to devig")
    implied: dict[Outcome, float] = {}
    for outcome, price in odds.items():
        try:
            number = float(price)
        except (TypeError, ValueError):
            raise ValueError(
                f"price for outcome {outcome!r} is not numeric: {price!r}"
            ) from None
        if not valid_price(number):
            raise ValueError(
                f"price for outcome {outcome!r} must be finite and > 1.0, "
                f"got {price!r}"
            )
        implied[outcome] = 1.0 / number
    overround = sum(implied.values())
    probabilities = {
        outcome: value / overround for outcome, value in implied.items()
    }
    return probabilities, overround


def ev(prob: float | pd.Series, odds: float | pd.Series) -> float | pd.Series:
    """Expected value of one unit staked at ``odds``: ``prob * odds - 1``.

    Deliberately bare arithmetic: it runs on scalars in tests and elementwise on
    a Series inside :func:`select_bets`, and it validates nothing. Input checks
    belong at the boundary, because a silent clamp inside the EV itself would
    hide exactly the edge bug the pure-market control exists to expose
    (invariant 4).
    """
    return prob * odds - 1


def kelly_stake(
    prob: float,
    odds: float,
    fraction: float,
    bankroll: float,
    max_stake_frac: float,
) -> float:
    """Fractional-Kelly stake in currency for one bet, capped per bet.

    ``f* = (prob * odds - 1) / (odds - 1)`` is the full-Kelly fraction of the
    bankroll for this edge. ``fraction`` scales it down because the probability
    is an estimate, never a known quantity, and full Kelly over-bets on noise;
    ``max_stake_frac`` is the hard ceiling applied *afterwards*, so a
    mis-tuned probability cannot ask for the whole bankroll even when
    fractional Kelly would permit it. The cap binds last on purpose: a safety
    rail that the thing it guards could override is not a rail.

    A non-positive ``f*`` stakes nothing — a stake on zero or negative edge is
    the bug invariant 4 hunts, not a bet to be priced. A bankroll of zero
    stakes nothing for the same reason (there is nothing to stake); a
    non-finite bankroll is a computation error and raises instead.
    """
    p = _number(prob, "prob")
    price = _number(odds, "odds")
    scale = _number(fraction, "fraction")
    cap = _number(max_stake_frac, "max_stake_frac")
    bank = _number(bankroll, "bankroll")

    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob!r}")
    if not valid_price(price):
        raise ValueError(f"odds must be finite and > 1.0, got {odds!r}")
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"max_stake_frac must be in (0, 1], got {max_stake_frac!r}")
    if not math.isfinite(bank):
        raise ValueError(f"bankroll must be a finite amount, got {bankroll!r}")
    if bank < 0.0:
        raise ValueError(f"bankroll must be >= 0, got {bankroll!r}")
    if bank == 0.0:
        return 0.0

    kelly = (p * price - 1.0) / (price - 1.0)
    if kelly <= 0.0:
        return 0.0
    return float(min(kelly * scale, cap) * bank)


def select_bets(predictions: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    """One bet per (match_id, market): the highest-EV outcome above the filter.

    WHY it is written this way:

    * **Strict ``>``.** The pure-market control runs this with
      ``ev_threshold=0`` on de-vigged prices, where the model's own fair price
      lands on EV exactly 0. ``>=`` would let every fair row through, so the
      control would place bets and invariant 4 would read a staking bug as an
      edge.
    * **EV uses the raw odds actually paid.** This function never chooses the
      price basis — the caller passes the frame's ``odds`` column as given —
      but the basis that decides a real bet is the raw book price, because the
      money is won or lost at the price the book offered. De-vigged (fair) odds
      are used only to produce the market probability, never to size or select a
      bet. Deciding a different basis here would silently change what a
      threshold means.
    * **NaN odds is "no price", never a bet.** EV against NaN is not an edge,
      so such rows fall out of the comparison instead of being treated as 0 or
      skipped by accident. A *present but invalid* price (<= 1 or non-finite)
      is an error instead: it means the odds contract upstream was bypassed.
    * **Ties go to the first row**, so a date-ordered frame selects
      deterministically and reruns are reproducible.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(
            f"predictions must be a DataFrame, got {type(predictions).__name__}"
        )
    threshold = _number(ev_threshold, "ev_threshold")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"ev_threshold must be a finite number >= 0, got {ev_threshold!r}")
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"predictions is missing required column(s) {missing}; "
            f"have {list(predictions.columns)}"
        )

    frame = predictions.reset_index(drop=True)
    for column in ("match_id", "market", "outcome"):
        if frame[column].isna().any():
            raise ValueError(
                f"predictions has {int(frame[column].isna().sum())} null "
                f"{column} cell(s); a bet needs an identity"
            )

    # between() is False for NaN too, so one mask covers both "not a number"
    # (coerced to NaN) and "a number that is not a probability".
    probs = pd.to_numeric(frame["model_prob"], errors="coerce")
    bad_probs = ~probs.between(0.0, 1.0)
    if bad_probs.any():
        raise ValueError(
            f"model_prob must be numeric and in [0, 1]; "
            f"bad row(s): {list(frame.index[bad_probs])}"
        )

    odds = pd.to_numeric(frame["odds"], errors="coerce")
    priced = odds.map(valid_price).astype(bool)
    invalid = odds.notna() & ~priced
    if invalid.any():
        raise ValueError(
            f"odds must be a finite price > 1.0; bad row(s): "
            f"{list(frame.index[invalid])}"
        )

    edges = ev(probs, odds)  # NaN odds leave NaN edges: never selected
    clearing = edges[edges > threshold]
    if clearing.empty:
        # No edge clears the filter: an empty result, not the whole input.
        return frame.iloc[0:0]
    keys = frame.loc[clearing.index, ["match_id", "market"]]
    best = clearing.groupby([keys["match_id"], keys["market"]], sort=False).idxmax()
    positions = sorted(int(position) for position in best.to_numpy())
    return frame.iloc[positions].reset_index(drop=True)


def _number(value: Any, name: str) -> float:
    """Coerce ``value`` to float, or raise ValueError naming ``name``.

    One helper so every numeric argument here rejects ``"abc"``/``None`` with
    the same message shape instead of leaking Python's own TypeError, which
    does not say which parameter was wrong.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


__all__ = ["devig", "ev", "kelly_stake", "select_bets"]
