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
from .match_winner import MATCH_WINNER, MatchWinner

REGISTRY: dict[str, Market] = {MATCH_WINNER.name: MATCH_WINNER}


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
