"""Tests for markets/base.py and markets/match_winner.py.

Rows come from the real fixture CSVs through the real normaliser, because the
odds priority chain exists precisely to survive seasons where football-data
drops columns — a hand-built dict would never catch a renamed column. Nothing
here touches the network: `_fetch` is monkeypatched to raise.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bet_engine.data import ingest
from bet_engine.markets import get_market
from bet_engine.markets.base import (
    DEFAULT_ODDS_POLICY,
    ODDS_POLICIES,
    Market,
    OddsSource,
    attach_odds,
    resolve_odds,
    valid_price,
)
from bet_engine.markets.match_winner import MATCH_WINNER, MatchWinner

FIXTURES = Path(__file__).parent / "fixtures"

ALL_ODDS_COLUMNS = (
    "PSCH", "PSCD", "PSCA",
    "PSH", "PSD", "PSA",
    "B365H", "B365D", "B365A",
)


def _forbid_network(url: str, dst: Path) -> None:
    raise AssertionError(f"unit test must not hit the network: {url}")


def _load(monkeypatch, tmp_path, *names: str) -> pd.DataFrame:
    """Load named fixture files as one canonical league frame."""
    seasons = []
    for name in names:
        shutil.copyfile(FIXTURES / name, tmp_path / name)
        seasons.append(name.removeprefix("E0_").removesuffix(".csv"))
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)
    return ingest.load_league("E0", seasons, tmp_path)


def _row(df: pd.DataFrame, home: str) -> pd.Series:
    matches = df.loc[df["home"] == home]
    assert len(matches) == 1, f"expected exactly one row for {home!r}"
    return matches.iloc[0]


def _without(row: pd.Series, *columns: str) -> pd.Series:
    """A copy of `row` with `columns` present but unpriced (NaN)."""
    row = row.copy()
    row[list(columns)] = np.nan
    return row


# --- settlement -------------------------------------------------------------

def test_settle_returns_the_ftr_outcome_for_every_fixture_row(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    # The fixture must really cover all three outcomes or these tests prove
    # only part of the contract.
    assert set(df["ftr"]) == {"H", "D", "A"}

    for record in df.to_dict("records"):
        assert MATCH_WINNER.settle(record) == record["ftr"]


def test_settle_reads_series_rows(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")

    assert MATCH_WINNER.settle(_row(df, "Manchester City")) == "H"
    assert MATCH_WINNER.settle(_row(df, "Aston Villa")) == "D"
    assert MATCH_WINNER.settle(_row(df, "Norwich")) == "A"


def test_settle_returns_none_without_a_result(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")

    assert MATCH_WINNER.settle({"ftr": np.nan}) is None
    assert MATCH_WINNER.settle({"ftr": None}) is None
    assert MATCH_WINNER.settle({}) is None  # column absent at all

    unplayed = df.iloc[[0]].copy()
    unplayed["ftr"] = np.nan
    assert MATCH_WINNER.settle(unplayed.iloc[0]) is None


def test_settle_tolerates_case_and_whitespace():
    assert MATCH_WINNER.settle({"ftr": " h "}) == "H"
    assert MATCH_WINNER.settle({"ftr": "d"}) == "D"
    assert MATCH_WINNER.settle({"ftr": "A\n"}) == "A"


def test_settle_rejects_unknown_result_codes():
    assert MATCH_WINNER.settle({"ftr": "X"}) is None
    assert MATCH_WINNER.settle({"ftr": 1}) is None
    assert MATCH_WINNER.settle({"ftr": ""}) is None


# --- odds -------------------------------------------------------------------

def test_odds_prefer_pinnacle_closing(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _row(df, "Norwich")

    odds, source = MATCH_WINNER.odds_with_source(row)

    assert source == "pinnacle_close"
    assert odds == {"H": 1.2, "D": 7.0, "A": 14.0}
    # `odds` is the same vector without the provenance label.
    assert MATCH_WINNER.odds(row) == odds


def test_odds_fall_back_to_pinnacle_opening(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _without(_row(df, "Norwich"), "PSCH", "PSCD", "PSCA")

    odds, source = MATCH_WINNER.odds_with_source(row)

    assert source == "pinnacle"
    assert odds == {"H": 1.22, "D": 6.5, "A": 13.0}


def test_default_policy_refuses_bet365_only_rows(monkeypatch, tmp_path):
    """pinnacle_only skips a row only Bet365 can price: that is the policy."""
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _without(
        _row(df, "Norwich"), "PSCH", "PSCD", "PSCA", "PSH", "PSD", "PSA"
    )

    odds, source = MATCH_WINNER.odds_with_source(row)

    assert source is None
    assert all(np.isnan(price) for price in odds.values())


def test_fallback_policy_prices_bet365_rows(monkeypatch, tmp_path):
    """Opting into `fallback` buys coverage at the cost of provenance noise."""
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _without(
        _row(df, "Norwich"), "PSCH", "PSCD", "PSCA", "PSH", "PSD", "PSA"
    )

    odds, source = MATCH_WINNER.odds_with_source(row, policy="fallback")

    assert source == "b365"
    assert odds == {"H": 1.28, "D": 5.5, "A": 11.0}


def test_odds_are_nan_when_no_source_is_complete(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _row(df, "Norwich")

    nothing_left = _without(row, *ALL_ODDS_COLUMNS)
    all_partial = _without(row, "PSCD", "PSCA", "PSD", "B365A")

    # `fallback` so the loop really exercises every book, not just Pinnacle.
    for incomplete in (nothing_left, all_partial):
        odds, source = MATCH_WINNER.odds_with_source(
            incomplete, policy="fallback"
        )
        assert source is None
        assert set(odds) == {"H", "D", "A"}
        assert all(np.isnan(price) for price in odds.values())
        assert all(
            np.isnan(price)
            for price in MATCH_WINNER.odds(incomplete, policy="fallback").values()
        )


def test_partial_price_vector_never_mixes_books(monkeypatch, tmp_path):
    """A half-populated source is skipped, not completed from another book."""
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    row = _without(_row(df, "Norwich"), "PSCD", "PSCA")  # closing has H only

    odds, source = MATCH_WINNER.odds_with_source(row)

    assert source == "pinnacle"
    assert odds == {"H": 1.22, "D": 6.5, "A": 13.0}  # all three from Pinnacle


def test_absent_odds_columns_behave_like_missing_prices(monkeypatch, tmp_path):
    """The 1819 fixture has no PSC* columns at all — the real-world case."""
    df = _load(monkeypatch, tmp_path, "E0_1819.csv")
    assert "PSCH" not in df.columns
    row = _row(df, "Arsenal")

    odds, source = MATCH_WINNER.odds_with_source(row)
    assert source == "pinnacle"
    assert odds == {"H": 2.05, "D": 3.5, "A": 3.8}

    no_pinnacle = _without(row, "PSH", "PSD", "PSA")

    # Under the default policy the row is skipped, not priced off Bet365.
    unpriced, no_source = MATCH_WINNER.odds_with_source(no_pinnacle)
    assert no_source is None
    assert all(np.isnan(price) for price in unpriced.values())

    # The same row under `fallback` takes the Bet365 price that does exist.
    b365, b365_source = MATCH_WINNER.odds_with_source(
        no_pinnacle, policy="fallback"
    )
    assert b365_source == "b365"
    assert b365 == {"H": 2.1, "D": 3.4, "A": 3.6}


def test_odds_are_identical_for_series_and_record_rows(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv")
    record = df.loc[df["home"] == "Brighton"].to_dict("records")[0]

    assert MATCH_WINNER.odds_with_source(record) == MATCH_WINNER.odds_with_source(
        _row(df, "Brighton")
    )


@pytest.mark.parametrize(
    "price, expected",
    [
        (1.5, True),
        (1.01, True),
        (1.0, False),            # even money cannot exist with margin
        (0.9, False),            # corrupt, not a bargain
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_valid_price(price, expected):
    assert valid_price(price) is expected


# --- odds policy -------------------------------------------------------------

def test_policy_constants_nail_down_the_default():
    assert ODDS_POLICIES == ("pinnacle_only", "fallback")
    assert DEFAULT_ODDS_POLICY == "pinnacle_only"


def test_unknown_policy_is_rejected_not_silently_ignored():
    """A typo like "pinnacle" must not quietly widen (or narrow) a run."""
    with pytest.raises(ValueError, match="unknown odds policy"):
        MATCH_WINNER.odds({}, policy="pinnacle")

    with pytest.raises(ValueError, match="unknown odds policy"):
        resolve_odds({}, ("H",), (), policy="everything_goes")


def test_pinnacle_sources_share_one_bookmaker():
    """PSC and PS are one book to policy: the chain PSC -> PS stays inside it."""
    assert {source.bookmaker for source in MATCH_WINNER.sources} == {
        "pinnacle",
        "b365",
    }
    closing, opening, _ = MATCH_WINNER.sources
    assert closing.bookmaker == opening.bookmaker == "pinnacle"


def test_attach_odds_adds_prices_and_provenance(monkeypatch, tmp_path):
    df = _load(monkeypatch, tmp_path, "E0_1920.csv").head(3)

    priced = attach_odds(df, MATCH_WINNER)

    assert list(priced.columns) == list(df.columns) + [
        "odds_source",
        "odds_H",
        "odds_D",
        "odds_A",
    ]
    assert set(priced["odds_source"]) == {"pinnacle_close"}
    assert (priced["odds_H"] > 1).all()
    # a copy: the caller's frame keeps its own columns
    assert "odds_source" not in df.columns


def test_attach_odds_keeps_rows_the_policy_cannot_price(monkeypatch, tmp_path):
    """Skipped rows stay in the frame so the caller can count them."""
    df = _load(monkeypatch, tmp_path, "E0_1819.csv").head(5).copy()
    df[["PSH", "PSD", "PSA"]] = np.nan  # 1819 has no PSC* columns either

    priced = attach_odds(df, MATCH_WINNER)

    assert len(priced) == len(df)
    assert priced["odds_source"].isna().all()  # the skip mask
    assert priced["odds_H"].isna().all()

    recovered = attach_odds(df, MATCH_WINNER, policy="fallback")
    assert recovered["odds_source"].eq("b365").all()


def test_attach_odds_on_an_empty_frame_only_adds_columns():
    priced = attach_odds(pd.DataFrame({"match_id": []}), MATCH_WINNER)

    assert priced.empty
    assert list(priced.columns) == [
        "match_id",
        "odds_source",
        "odds_H",
        "odds_D",
        "odds_A",
    ]


def test_attach_odds_validates_its_arguments():
    with pytest.raises(TypeError, match="DataFrame"):
        attach_odds([{"PSH": 2.0}], MATCH_WINNER)
    with pytest.raises(TypeError, match="Market protocol"):
        attach_odds(pd.DataFrame({"a": []}), object())


# --- contract ---------------------------------------------------------------

def test_market_sources_must_cover_every_outcome():
    with pytest.raises(ValueError, match="pinnacle_close"):

        class Incomplete(MatchWinner):
            sources = (
                OddsSource("pinnacle_close", "pinnacle", {"H": "PSCH"}),
            )

        Incomplete()


def test_duplicate_source_labels_are_rejected():
    with pytest.raises(ValueError, match="duplicate odds source label"):

        class Duplicated(MatchWinner):
            sources = (
                OddsSource(
                    "b365", "b365", {"H": "B365H", "D": "B365D", "A": "B365A"}
                ),
                OddsSource(
                    "b365", "b365", {"H": "PSCH", "D": "PSCD", "A": "PSCA"}
                ),
            )

        Duplicated()


def test_source_without_a_bookmaker_is_rejected():
    """Policy reasons about the book, so a blank bookmaker is a broken source."""
    with pytest.raises(ValueError, match="blank bookmaker"):

        class NoBook(MatchWinner):
            sources = (
                OddsSource(
                    "pinnacle_close", " ", {"H": "PSCH", "D": "PSCD", "A": "PSCA"}
                ),
            )

        NoBook()


def test_market_protocol_and_registry():
    assert isinstance(MATCH_WINNER, Market)
    assert MATCH_WINNER.name == "match_winner"
    assert MATCH_WINNER.outcomes == ("H", "D", "A")
    assert "MatchWinner" in repr(MATCH_WINNER)

    assert get_market("match_winner") is MATCH_WINNER
    with pytest.raises(KeyError, match="unknown market"):
        get_market("over_under")
