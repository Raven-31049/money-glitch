"""Market adapters. Market-specific logic lives here and nowhere else.

The registry maps a variant config's ``market:`` string to a singleton market
object, so a config can only ever name a market that actually exists.
"""

from __future__ import annotations

from .base import (
    DEFAULT_ODDS_POLICY,
    ODDS_POLICIES,
    POLICY_BOOKMAKERS,
    BaseMarket,
    Market,
    OddsSource,
    Row,
    attach_odds,
    field,
    resolve_odds,
    sources_for_policy,
    validate_sources,
)
from .cards_totals import CARDS_TOTAL, CardsTotalLine, CardsTotals
from .match_winner import MATCH_WINNER, MatchWinner

#: Markets a config can name. CardsTotals (the frozen fixed-line step-1 market)
#: is deliberately absent: its identity includes a line no config field carries.
#: Its Part B successor CARDS_TOTAL reads the line per row from the odds, so it
#: IS nameable — "cards_total" — and is registered here.
REGISTRY: dict[str, Market] = {
    MATCH_WINNER.name: MATCH_WINNER,
    CARDS_TOTAL.name: CARDS_TOTAL,
}


def get_market(name: str) -> Market:
    """Look a market up by registry name, failing loudly if it is unknown."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown market {name!r}; known markets: {sorted(REGISTRY)}"
        ) from None


__all__ = [
    "DEFAULT_ODDS_POLICY",
    "ODDS_POLICIES",
    "POLICY_BOOKMAKERS",
    "BaseMarket",
    "CARDS_TOTAL",
    "CardsTotalLine",
    "CardsTotals",
    "MATCH_WINNER",
    "Market",
    "MatchWinner",
    "OddsSource",
    "REGISTRY",
    "Row",
    "attach_odds",
    "field",
    "get_market",
    "resolve_odds",
    "sources_for_policy",
    "validate_sources",
]
