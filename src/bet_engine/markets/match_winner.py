"""Match-winner (1X2): three-way home / draw / away settlement.

The first concrete market, chosen because it forces a draw outcome into the
probability vector and so exercises the contract's hardest case early.

Price priority within the run's odds policy (see ``markets.base``):
pinnacle_close, then pinnacle (opening), then — under the ``fallback`` policy
only — b365. Closing prices are the least noisy estimate of the true line, so
they win whenever they exist; the fallback chain exists because football-data
drops Pinnacle columns for whole seasons and for stretches of a season (see
docs/DATA_NOTES.md). Under the default ``pinnacle_only`` policy a row with no
Pinnacle price at all prices as NaN and is counted as skipped rather than
borrowed from a different book.
"""

from __future__ import annotations

from .base import BaseMarket, OddsSource, Row, field

_SOURCE_PRIORITY: tuple[OddsSource, ...] = (
    OddsSource(
        "pinnacle_close", "pinnacle", {"H": "PSCH", "D": "PSCD", "A": "PSCA"}
    ),
    OddsSource("pinnacle", "pinnacle", {"H": "PSH", "D": "PSD", "A": "PSA"}),
    OddsSource("b365", "b365", {"H": "B365H", "D": "B365D", "A": "B365A"}),
)


class MatchWinner(BaseMarket):
    """Three-way match result, settled from the ``ftr`` column (H/D/A)."""

    name = "match_winner"
    outcomes = ("H", "D", "A")
    sources = _SOURCE_PRIORITY

    def settle(self, row: Row) -> str | None:
        """Outcome label from ``ftr``, or None when the match has no result.

        Case and surrounding whitespace are tolerated because the raw column
        is hand-maintained upstream; an unrecognised code is None, not an
        exception, so a single odd row cannot abort a walk-forward run.
        """
        raw = field(row, "ftr")
        if raw is None:
            return None
        label = str(raw).strip().upper()
        return label if label in self.outcomes else None


#: The registry instance. Markets are stateless singletons, so every consumer
#: shares one object rather than re-constructing equivalent ones per run.
MATCH_WINNER = MatchWinner()

__all__ = ["MATCH_WINNER", "MatchWinner", "_SOURCE_PRIORITY"]
