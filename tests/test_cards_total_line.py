"""Tests for markets/cards_totals.py::CardsTotalLine — the push-aware market.

Part B gives a card total a *per-match* line, and a whole-number line can be hit
exactly: that is a **push** (void), not an under. These tests pin the three
settlement states, the per-row line, the odds columns it prices off, and that it
is the one cards market in the registry (the fixed-line step-1 market stays
unregistered and untouched).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bet_engine.markets import REGISTRY, CardsTotalLine, get_market
from bet_engine.markets.base import PUSH
from bet_engine.markets.cards_totals import CARDS_TOTAL, UNDER_ODDS_COLUMN


def _row(hy=1, ay=1, hr=1, ar=1, *, line=3.5, **extra):
    row = {"hy": hy, "ay": ay, "hr": hr, "ar": ar, "line": line}
    row.update(extra)
    return row


def test_over_and_under_are_read_against_the_rows_own_line():
    market = CardsTotalLine()

    assert market.settle(_row(2, 2, 1, 0, line=3.5)) == "over"  # 5 > 3.5
    assert market.settle(_row(1, 1, 0, 1, line=3.5)) == "under"  # 3 < 3.5
    # The same counts settle the other way at a different line.
    assert market.settle(_row(2, 2, 1, 0, line=5.5)) == "under"  # 5 < 5.5


def test_an_exact_hit_on_a_whole_line_is_a_push_not_an_under():
    market = CardsTotalLine()

    assert market.settle(_row(2, 1, 1, 0, line=4)) == PUSH  # 4 == 4, void
    assert market.settle(_row(2, 1, 1, 1, line=4)) == "over"  # 5 > 4
    assert market.settle(_row(1, 1, 1, 0, line=4)) == "under"  # 3 < 4


def test_a_half_line_never_pushes():
    market = CardsTotalLine()

    assert market.settle(_row(4, 0, 0, 0, line=4.5)) == "under"
    assert market.settle(_row(5, 0, 0, 0, line=4.5)) == "over"
    assert market.settle(_row(3, 0, 0, 0, line=3.5)) == "under"


def test_the_total_is_the_four_published_card_counts():
    market = CardsTotalLine()

    # 3 + 2 + 1 + 0 = 6, and 6 > 5.5 — referee cards (hr/ar) count like any other.
    assert market.settle(_row(3, 2, 1, 0, line=5.5)) == "over"
    assert market.settle(_row(3, 2, 0, 0, line=5.5)) == "under"  # 5 < 5.5


@pytest.mark.parametrize(
    "row",
    [
        {"hy": 1, "ay": 1, "line": 3.5},  # no hr/ar
        _row(hr=np.nan),
        _row(ar="x"),
        {"hy": 1, "ay": 1, "hr": 1, "ar": 1},  # no line at all
        _row(line=np.nan),
        _row(line="x"),
        {},
    ],
)
def test_a_row_without_counts_or_a_line_is_unsettled_not_an_error(row):
    assert CardsTotalLine().settle(row) is None


def test_the_market_declares_two_bettable_outcomes_and_the_cards_odds_source():
    market = CardsTotalLine()

    assert market.outcomes == ("over", "under")
    assert market.name == "cards_total"
    # Prices come from the joined odds columns, labelled so a report can say
    # which quotation a row used.
    (source,) = market.sources
    assert source.columns == {"over": "over_odds", "under": UNDER_ODDS_COLUMN}


def test_odds_reads_the_over_and_under_price_columns():
    market = CardsTotalLine()

    prices = market.odds({"over_odds": 1.9, "under_odds": 2.0})
    assert prices == {"over": 1.9, "under": 2.0}

    missing = market.odds({})
    assert all(math.isnan(price) for price in missing.values())


def test_the_line_market_is_registered_as_cards_total():
    assert get_market("cards_total") is CARDS_TOTAL
    assert isinstance(CARDS_TOTAL, CardsTotalLine)
    assert set(REGISTRY) == {"match_winner", "cards_total"}
    # The frozen fixed-line market is still not a registry key.
    assert "cards_over_4.5" not in REGISTRY
