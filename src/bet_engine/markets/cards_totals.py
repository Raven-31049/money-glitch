"""Cards over/under settlement — the market half of Phase 1, before odds.

WHY this exists while there are still no prices: settlement (which side of a
line a match landed on) is market knowledge, and invariant 8 puts market
knowledge in ``markets/`` and nowhere else. Keeping the rule here means the
scoring code, the future odds adapter and any backtest all settle a card total
the same one way, instead of each re-deriving ``hy + ay + hr + ar > line``.

WHY it is not in ``REGISTRY`` yet: registry keys are market names, and this
market's identity includes a line that a variant config does not have a field
for. It is registered when the odds source lands and a config can name
``line`` — until then a config could only ask for a market nobody can price,
and ``sources`` is deliberately empty because Phase 1 step 1 has no odds at all.

PROVISIONAL counting rule (see docs/DATA_NOTES.md): a match's total cards is
``hy + ay + hr + ar`` exactly as football-data publishes them. The bookmaker's
own counting rule — whether a second yellow leading to a red counts once or
twice, whether cards to staff count — is unconfirmed, so any comparison of
these probabilities to a real line stays provisional until the odds source is
chosen.
"""

from __future__ import annotations

import math
from typing import Sequence

from .base import PUSH, BaseMarket, OddsSource, Outcome, Row, field

#: The four published card counts that make up a match total.
CARD_COLUMNS: Sequence[str] = ("hy", "ay", "hr", "ar")

#: Where the per-match line and prices live once an odds table has been joined
#: onto a match frame (data.odds). Kept here — not in the odds module — because
#: which columns a market prices off is market knowledge (invariant 8).
LINE_COLUMN = "line"
OVER_ODDS_COLUMN = "over_odds"
UNDER_ODDS_COLUMN = "under_odds"


class CardsTotals(BaseMarket):
    """Over/under a card total: ``hy + ay + hr + ar`` against ``line``."""

    outcomes: tuple[Outcome, ...] = ("over", "under")
    #: No odds source yet — Phase 1 step 1 is a calibration exercise.
    sources: tuple[OddsSource, ...] = ()

    def __init__(self, line: float) -> None:
        # bool is excluded because isinstance(True, int) is True, and a config
        # that says `line: true` must not become a line of 1.
        if isinstance(line, bool) or not isinstance(line, (int, float)):
            raise ValueError(f"line must be a positive number, got {line!r}")
        if not math.isfinite(float(line)) or float(line) <= 0:
            raise ValueError(f"line must be finite and > 0, got {line!r}")
        self.line = float(line)
        self.name = f"cards_over_{self.line:g}"
        super().__init__()

    def settle(self, row: Row) -> Outcome | None:
        """``over`` or ``under``, or None when the row has no card counts yet.

        None rather than an exception: a row missing card columns is an
        unsettled match as far as this market is concerned, and the scoring
        code excludes and counts it instead of scoring it as a loss.
        """
        counts: list[float] = []
        for column in CARD_COLUMNS:
            value = field(row, column)
            if value is None:
                return None
            try:
                counts.append(float(value))
            except (TypeError, ValueError):
                return None
        if not all(math.isfinite(count) for count in counts):
            return None
        total = sum(counts)
        return "over" if total > self.line else "under"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} line={self.line:g}>"


class CardsTotalLine(BaseMarket):
    """Over/under a card total at each match's OWN line, push-aware (Part B).

    WHERE this sits next to :class:`CardsTotals`: the frozen step-1 market sets
    its line at construction and calls an exact hit *under*. Part B's odds data
    carries a line per quoted over/under, and a whole-number line with an exact
    hit is a real-world **push** — the stake returns and neither side wins. So
    this market reads its line from the row and answers three states, while
    :class:`CardsTotals` is left untouched so steps 1-4 stay reproducible.

    It is the one cards market in the registry (``REGISTRY["cards_total"]``):
    its identity is no longer a fixed line a config would have to name, so a
    config can ask for it by name and let the odds table supply the line.

    Prices come from the joined odds columns (``over_odds``/``under_odds``),
    declared as one source so the odds policy still governs whether a row is
    priced at all. The book is whatever policy the run chose; the *line* is the
    row's, not the market's, which is why ``name`` has no number in it.
    """

    name = "cards_total"
    outcomes: tuple[Outcome, ...] = ("over", "under")
    sources: tuple[OddsSource, ...] = (
        OddsSource(
            "cards_total",
            "pinnacle",
            {"over": OVER_ODDS_COLUMN, "under": UNDER_ODDS_COLUMN},
        ),
    )

    def settle(self, row: Row) -> Outcome | None:
        """``over``, ``under``, ``PUSH`` (void), or None when unsettleable.

        None when the row carries no result: missing card counts *or* no line.
        A total exactly on a whole-number line is :data:`~.base.PUSH` — not
        ``under`` — because that is what voids the bet; a half-line never
        produces one.
        """
        counts: list[float] = []
        for column in CARD_COLUMNS:
            value = field(row, column)
            if value is None:
                return None
            try:
                counts.append(float(value))
            except (TypeError, ValueError):
                return None
        line_value = field(row, LINE_COLUMN)
        if line_value is None:
            return None
        try:
            line = float(line_value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(count) for count in counts) or not math.isfinite(
            line
        ):
            return None
        total = sum(counts)
        if total > line:
            return "over"
        if total < line:
            return "under"
        return PUSH


#: The registry instance: one stateless market, line-per-row from the odds.
CARDS_TOTAL = CardsTotalLine()

__all__ = [
    "CARD_COLUMNS",
    "CARDS_TOTAL",
    "LINE_COLUMN",
    "OVER_ODDS_COLUMN",
    "UNDER_ODDS_COLUMN",
    "CardsTotalLine",
    "CardsTotals",
]
