"""Market contract: outcomes, settlement, and odds resolution.

The validation engine (walk-forward, staking, bankroll, bootstrap, calibration,
control) is written against this contract only and never against a concrete
market — that is what keeps it market-agnostic (invariant 8). Everything a
market knows about itself (which columns hold which bookmaker's prices, how a
result code maps to an outcome label) lives in the concrete subclass and
nowhere else.

Three rules the contract enforces for every market:

* ``settle(row)`` returns the winning outcome label, or ``None`` when the row
  carries no usable result yet. ``None`` rather than an exception, because a
  backtest walks rows chronologically and an unsettled row is a normal state,
  not an error.
* ``odds(row)`` never mixes books inside one price vector. A source either
  supplies a complete, valid price for every outcome or it is skipped: a vector
  stitched together from two books is a price nobody could actually have bet,
  and de-vigging it would produce a fiction.
* which sources are permitted at all is a *policy* (``pinnacle_only`` by
  default), not an accident of column availability. A row only Bet365 can price
  is unpriced under ``pinnacle_only`` — skipped and counted, not quietly priced
  off a different book than the rest of the run.

Tick size and de-vig wiring are Phase 2/3 work (see docs/PLAN.md) and are
deliberately absent here rather than stubbed as placeholders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

#: A row as the engine sees it: a Series from a canonical league frame, or a
#: plain dict record from ``DataFrame.to_dict("records")``.
Row = Mapping[str, Any] | pd.Series

Outcome = str

#: Every unavailable price is this, never ``None``/0/guesswork, so "no price"
#: has one representation across every market and every consumer.
NAN = float("nan")

#: Which bookmakers an odds policy permits. ``None`` means "every source the
#: market declares, in the market's declared priority order".
#:
#: WHY this lives here and not in config: the policy changes *odds resolution*,
#: so the set of policies is a property of the odds contract; config only names
#: one of them. Importing the other way would make markets depend on config and
#: the two lists could drift.
POLICY_BOOKMAKERS: Mapping[str, frozenset[str] | None] = {
    # PSC -> PS only. A sharp reference price on every row the run accepts;
    # Bet365 coverage is bought at the cost of price noise, so it is opt-in.
    "pinnacle_only": frozenset({"pinnacle"}),
    # PSC -> PS -> B365: maximum coverage, mixed sharp/soft provenance
    # recorded per row so reports can still split the two.
    "fallback": None,
}

#: The policies a variant config may name, in the order they are documented.
ODDS_POLICIES: tuple[str, ...] = tuple(POLICY_BOOKMAKERS)

#: What a run uses when its config says nothing: sharp prices only.
DEFAULT_ODDS_POLICY = "pinnacle_only"


def _is_null(value: object) -> bool:
    """True for None, NaN, NaT, and pandas' NA sentinel.

    ``value != value`` is the ordinary NaN test; pandas' NA returns *itself*
    from that comparison and then refuses to be coerced to bool, hence the
    TypeError branch.
    """
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


def field(row: Row, column: str) -> Any:
    """Value of ``column`` in ``row``, or None when it is absent or null.

    Absent and null deliberately collapse to the same answer: a frame that
    never had a bookmaker's columns (a season where the source dropped them)
    must behave exactly like a row where that book's price is simply missing,
    so no caller ever has to branch on which of the two happened.
    """
    try:
        value = row[column]
    except (KeyError, IndexError, TypeError):
        return None
    return None if _is_null(value) else value


def _price(row: Row, column: str) -> float:
    """Read ``column`` as a decimal price; anything unusable becomes NaN."""
    value = field(row, column)
    if value is None:
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def valid_price(price: float) -> bool:
    """A usable decimal price is finite and beats even money.

    Odds at or below 1.0 never occur in a real book, so one appearing means a
    corrupt cell, not a bargain; treating it as missing keeps a single bad
    value out of every EV, Kelly and de-vig computation downstream.
    """
    return math.isfinite(price) and price > 1.0


@dataclass(frozen=True)
class OddsSource:
    """One bookmaker's complete price vector, keyed by outcome label.

    ``label`` is recorded as provenance on predictions and in reports, so it is
    a stable identifier: rename columns freely, never rename the label.

    ``bookmaker`` is the book *policy* reasons about ("pinnacle", "b365"),
    deliberately separate from ``label``, which identifies the exact price
    (closing vs opening). Two sources from the same book share a bookmaker and
    differ in label, which is exactly how PSC -> PS resolves as one book.
    """

    label: str
    bookmaker: str
    columns: Mapping[Outcome, str]

    def read(self, row: Row) -> dict[Outcome, float]:
        """This source's prices for every outcome; unusable entries are NaN."""
        return {
            outcome: _price(row, column)
            for outcome, column in self.columns.items()
        }


def validate_sources(
    outcomes: Sequence[Outcome], sources: Sequence[OddsSource]
) -> None:
    """Reject a source list that cannot price this market's outcomes exactly.

    Called once when a market is constructed, so a mis-declared market fails at
    import time instead of quietly returning NaN rows halfway through a
    backtest that would then be reported as "no market prices found".
    """
    expected = tuple(outcomes)
    seen: set[str] = set()
    for source in sources:
        if source.label in seen:
            raise ValueError(f"duplicate odds source label {source.label!r}")
        seen.add(source.label)
        if not isinstance(source.bookmaker, str) or not source.bookmaker.strip():
            raise ValueError(
                f"odds source {source.label!r} has a blank bookmaker; policy "
                f"filtering needs to know which book a price came from"
            )
        if tuple(source.columns) != expected:
            raise ValueError(
                f"odds source {source.label!r} prices {tuple(source.columns)}, "
                f"expected {expected}"
            )


def sources_for_policy(
    sources: Sequence[OddsSource], policy: str
) -> tuple[OddsSource, ...]:
    """The sources ``policy`` permits, still in the market's priority order.

    An unknown policy raises rather than degrading to "everything": a typo like
    ``pinnacle`` silently widening the run to soft prices would invalidate it
    without anyone noticing.
    """
    try:
        allowed_bookmakers = POLICY_BOOKMAKERS[policy]
    except KeyError:
        raise ValueError(
            f"unknown odds policy {policy!r}; known policies: {list(ODDS_POLICIES)}"
        ) from None
    if allowed_bookmakers is None:
        return tuple(sources)
    return tuple(s for s in sources if s.bookmaker in allowed_bookmakers)


def resolve_odds(
    row: Row,
    outcomes: Sequence[Outcome],
    sources: Sequence[OddsSource],
    *,
    policy: str = DEFAULT_ODDS_POLICY,
) -> tuple[dict[Outcome, float], str | None]:
    """First source the policy permits that has a complete, valid vector.

    The policy narrows *which* sources are considered; the market's declared
    order decides priority within them. Returns ``(prices, source_label)``.
    When no permitted source is complete the prices are all NaN and the label
    is None — an honest "no price on this row" rather than an estimate from
    whichever book happened to have two of three prices, and rather than a
    price from a book the run's policy excluded. The label is what makes a
    downstream EV number traceable back to the price it came from.
    """
    expected = tuple(outcomes)
    for source in sources_for_policy(sources, policy):
        prices = source.read(row)
        if tuple(prices) != expected:
            raise ValueError(
                f"odds source {source.label!r} prices {tuple(prices)}, "
                f"expected {expected}"
            )
        if all(valid_price(price) for price in prices.values()):
            return prices, source.label
    return {outcome: NAN for outcome in outcomes}, None


@runtime_checkable
class Market(Protocol):
    """The market surface the validation engine is allowed to see."""

    name: str
    outcomes: Sequence[Outcome]

    def settle(self, row: Row) -> Outcome | None:
        """Winning outcome for this row, or None if it has no usable result."""
        ...

    def odds(
        self, row: Row, *, policy: str = DEFAULT_ODDS_POLICY
    ) -> dict[Outcome, float]:
        """Decimal odds per outcome; NaN wherever unavailable."""
        ...

    def odds_with_source(
        self, row: Row, *, policy: str = DEFAULT_ODDS_POLICY
    ) -> tuple[dict[Outcome, float], str | None]:
        """``odds`` plus the label of the book that produced it (None if none)."""
        ...


class BaseMarket:
    """Reference implementation of the odds half of the contract.

    Subclasses set ``name``, ``outcomes`` and ``sources`` and implement
    ``settle``. ``settle`` is intentionally *not* provided here, so a subclass
    that forgets it fails the :class:`Market` protocol check instead of
    answering with something that is not a market outcome.
    """

    name: str
    outcomes: tuple[Outcome, ...]
    sources: tuple[OddsSource, ...]

    def __init__(self) -> None:
        validate_sources(self.outcomes, self.sources)

    def odds_with_source(
        self, row: Row, *, policy: str = DEFAULT_ODDS_POLICY
    ) -> tuple[dict[Outcome, float], str | None]:
        return resolve_odds(row, self.outcomes, self.sources, policy=policy)

    def odds(
        self, row: Row, *, policy: str = DEFAULT_ODDS_POLICY
    ) -> dict[Outcome, float]:
        return self.odds_with_source(row, policy=policy)[0]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


def attach_odds(
    df: pd.DataFrame, market: Market, *, policy: str = DEFAULT_ODDS_POLICY
) -> pd.DataFrame:
    """``df`` plus ``odds_<outcome>`` price columns and an ``odds_source`` column.

    WHY the whole frame rather than a filtered one: the rows the policy cannot
    price are exactly the rows a report has to account for. They are kept, with
    NaN prices and a NULL source, and the caller counts them
    (``df["odds_source"].isna().sum()``) instead of losing them where no one
    can see them — every skipped match becomes a row in the ``skipped`` table.

    Raises ``TypeError`` for anything that is not a frame or not a
    :class:`Market`, so a caller passes a real market and not a lookalike.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    if not isinstance(market, Market):
        raise TypeError(
            f"market must satisfy the Market protocol, got {type(market).__name__}"
        )

    labels: list[str | None] = []
    prices: dict[Outcome, list[float]] = {outcome: [] for outcome in market.outcomes}
    for record in df.to_dict("records"):
        odds, source = market.odds_with_source(record, policy=policy)
        labels.append(source)
        for outcome in market.outcomes:
            prices[outcome].append(odds[outcome])

    priced = df.copy()
    priced["odds_source"] = pd.Series(labels, index=df.index, dtype="object")
    for outcome, values in prices.items():
        priced[f"odds_{outcome}"] = pd.Series(
            values, index=df.index, dtype="float64"
        )
    return priced


__all__ = [
    "DEFAULT_ODDS_POLICY",
    "NAN",
    "ODDS_POLICIES",
    "POLICY_BOOKMAKERS",
    "BaseMarket",
    "Market",
    "OddsSource",
    "Outcome",
    "Row",
    "attach_odds",
    "field",
    "resolve_odds",
    "sources_for_policy",
    "valid_price",
    "validate_sources",
]
