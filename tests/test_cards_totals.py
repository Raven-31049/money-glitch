"""Tests for markets/cards_totals.py: how a card total settles.

Settlement lives in markets/ because it is market knowledge (invariant 8);
these tests pin the counting rule (`hy+ay+hr+ar`), the integer-line boundary
(a total sitting exactly on the line is *under*, never a push), and the
contract that an unpriced/unsettled row answers None instead of raising.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bet_engine.markets import REGISTRY
from bet_engine.markets.cards_totals import CardsTotals


def _row(hy=1, ay=1, hr=1, ar=1, **extra):
    row = {"hy": hy, "ay": ay, "hr": hr, "ar": ar}
    row.update(extra)
    return row


def test_a_total_above_the_line_is_over_and_below_is_under():
    market = CardsTotals(3.5)

    assert market.settle(_row(2, 2, 1, 0)) == "over"  # 5 > 3.5
    assert market.settle(_row(1, 1, 0, 1)) == "under"  # 3 < 3.5


def test_a_total_sitting_exactly_on_the_line_is_under_never_a_push():
    market = CardsTotals(4)

    assert market.settle(_row(2, 1, 1, 0)) == "under"  # 4 is not > 4
    assert market.settle(_row(2, 1, 1, 1)) == "over"  # 5 > 4


def test_the_total_is_the_four_published_card_counts():
    market = CardsTotals(5.5)

    # 3 + 2 + 1 + 0 = 6, and 6 > 5.5 — referee cards (hr/ar) count like any other.
    assert market.settle(_row(3, 2, 1, 0)) == "over"
    assert market.settle(_row(3, 2, 0, 0)) == "under"  # 5 < 5.5


def test_a_row_without_card_counts_is_unsetled_not_an_error():
    market = CardsTotals(3.5)

    assert market.settle({"hy": 1, "ay": 1}) is None
    assert market.settle(_row(hr=np.nan)) is None
    assert market.settle(_row(ar="x")) is None
    assert market.settle({}) is None


def test_the_market_declares_two_outcomes_and_no_odds_source_yet():
    market = CardsTotals(4.5)

    assert market.outcomes == ("over", "under")
    assert market.sources == ()
    assert market.name == "cards_over_4.5"
    prices = market.odds(_row())
    assert set(prices) == {"over", "under"}
    assert all(math.isnan(price) for price in prices.values())


@pytest.mark.parametrize("line", [0, -1, np.nan, np.inf, True, "3.5"])
def test_the_line_must_be_a_finite_positive_number(line):
    with pytest.raises(ValueError, match="line"):
        CardsTotals(line)


def test_a_line_specific_market_is_not_registered_under_a_generic_name():
    """Registry keys are market names; this one is identity = name + line, so
    it joins the registry when the odds source gives a config a line to state."""
    assert "cards_over_3.5" not in REGISTRY
    assert set(REGISTRY) == {"match_winner"}
