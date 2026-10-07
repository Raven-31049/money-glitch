"""Tests for data/ingest.py.

WHY these tests never touch the network: ingest caches into a data_dir that the
tests point at tmp_path, and _fetch is monkeypatched to raise. A test that
silently spawned a download would make the suite slow, flaky, and dependent on
football-data.co.uk being reachable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest  # pyright: ignore[reportMissingImports]

from bet_engine.data import ingest  # pyright: ignore[reportMissingImports]

FIXTURES = Path(__file__).parent / "fixtures"


def _prime(tmp_path: Path, *names: str) -> None:
    """Copy named fixture CSVs into tmp_path as the download cache."""
    for name in names:
        shutil.copyfile(FIXTURES / name, tmp_path / name)


def _forbid_network(url: str, dst: Path) -> None:
    raise AssertionError(f"unit test must not hit the network: {url}")


def test_existing_cache_file_is_used_not_downloaded(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1819.csv")
    calls = []
    def deny(url, dst):
        calls.append(url)
        raise AssertionError("existing cache file should suppress all downloads")
    monkeypatch.setattr(ingest, "_fetch", deny)

    path = ingest.download_season("E0", "1819", tmp_path)

    assert path == tmp_path / "E0_1819.csv"
    assert calls == []


def test_download_happens_once_then_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, dst):
        calls.append(url)
        shutil.copyfile(FIXTURES / "E0_1819.csv", dst)

    monkeypatch.setattr(ingest, "_fetch", fake_fetch)
    first = ingest.download_season("E0", "1819", tmp_path)
    second = ingest.download_season("E0", "1819", tmp_path)

    assert len(calls) == 1
    assert first == second == tmp_path / "E0_1819.csv"


def test_both_date_formats_parse(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1920"], tmp_path)

    assert {"2019-08-10", "2020-06-16", "2020-06-17", "2020-07-18"} == set(
        df["date"].dt.strftime("%Y-%m-%d")
    )
    # dd/mm/yy must land in 2020, not 1920/2018.
    assert (df["date"] == pd.Timestamp("2020-07-18")).any()
    assert df["date"].isna().sum() == 0


def test_trailing_fully_empty_rows_are_dropped(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1920"], tmp_path)

    assert len(df) == 4  # fixture holds 4 matches plus 2 blank rows


def test_missing_odds_column_becomes_nan_not_crash(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1819.csv", "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1819", "1920"], tmp_path)

    assert "PSCH" in df.columns                      # present in 1920
    assert df.loc[df["season"] == "1819", "PSCH"].isna().all()  # absent there -> NaN
    assert df.loc[df["season"] == "1920", "PSCH"].notna().all()


def test_canonical_base_columns_always_present(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1819.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1819"], tmp_path)

    expected = [
        "match_id", "league", "season", "date", "home", "away",
        "fthg", "ftag", "ftr", "referee", "hy", "ay", "hr", "ar",
        "hc", "ac", "hs", "as_", "hst", "ast",
    ]
    assert list(df.columns[: len(expected)]) == expected
    assert "covid_affected" in df.columns


def test_covid_affected_boundary_dates(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1819.csv", "E0_1920.csv", "E0_2021.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1819", "1920", "2021"], tmp_path)

    # 1819: nothing was pandemic-affected.
    assert not df.loc[df["season"] == "1819", "covid_affected"].any()
    # 1920: restart boundary — 16 June is NOT affected, 17 June and 18 July ARE.
    pre = df["date"] == pd.Timestamp("2020-06-16")
    post = df["date"] == pd.Timestamp("2020-06-17")
    july = df["date"] == pd.Timestamp("2020-07-18")
    assert not df.loc[(df["season"] == "1920") & pre, "covid_affected"].any()
    assert df.loc[(df["season"] == "1920") & post, "covid_affected"].all()
    assert df.loc[(df["season"] == "1920") & july, "covid_affected"].all()
    # 2021: every match behind closed doors.
    assert df.loc[df["season"] == "2021", "covid_affected"].all()


def test_load_league_sorts_by_date_then_match_id(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1819.csv", "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1819", "1920"], tmp_path)

    assert df["date"].is_monotonic_increasing
    for _, group in df.groupby("date"):
        assert group["match_id"].is_monotonic_increasing


# --- referee corrections (configs/referee_corrections.yaml) -------------------

REPO_CORRECTIONS = Path(__file__).parents[1] / "configs" / "referee_corrections.yaml"


def _write_csv(tmp_path: Path, rows: str) -> None:
    header = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee\n"
    (tmp_path / "E0_1819.csv").write_text(header + rows, encoding="utf-8")


def test_whitespace_is_stripped_from_text_columns(monkeypatch, tmp_path):
    _write_csv(tmp_path, "E0,01/09/2018,Arsenal,Chelsea,1,0,H,  J Gillett  \n")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    df = ingest.load_league("E0", ["1819"], tmp_path)

    assert df.loc[0, "referee"] == "J Gillett"
    assert df.loc[0, "home"] == "Arsenal"
    assert df.loc[0, "away"] == "Chelsea"
    # The padding must not leak into the match_id either.
    assert df.loc[0, "match_id"] == "1819_20180901_arsenal_chelsea"


def test_match_specific_correction_changes_only_that_match(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)
    target = "1920_20190810_norwich_liverpool"
    corrections = tmp_path / "corrections.yaml"
    corrections.write_text(
        "match_corrections:\n"
        f"  {target}:\n"
        "    referee: Michael Oliver\n"
        "    to: Test Referee\n"
        "    evidence: unit test\n"
        "aliases: {}\n",
        encoding="utf-8",
    )

    df = ingest.load_league(
        "E0", ["1920"], tmp_path, referee_corrections=corrections
    )

    changed = df.loc[df["match_id"] == target, "referee"]
    assert list(changed) == ["Test Referee"]
    # Every other row keeps its original referee, including the other
    # Michael Oliver row — the correction is match-scoped, not a name alias.
    untouched = df.loc[df["match_id"] != target, "referee"]
    assert set(untouched) == {"Michael Oliver", "Jonathan Moss", "Simon Hooper"}
    assert (untouched == "Michael Oliver").sum() == 1


def test_alias_applies_to_every_row_with_that_name(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)
    corrections = tmp_path / "corrections.yaml"
    corrections.write_text(
        "match_corrections: {}\n"
        "aliases:\n"
        "  Michael Oliver: M. Oliver\n",
        encoding="utf-8",
    )

    df = ingest.load_league(
        "E0", ["1920"], tmp_path, referee_corrections=corrections
    )

    assert (df.loc[df["referee"] == "M. Oliver", "home"] == ["Norwich", "Aston Villa"]).all()
    assert (df["referee"] == "Michael Oliver").sum() == 0
    # Non-aliased referees are untouched.
    assert set(df.loc[df["referee"] != "M. Oliver", "referee"]) == {
        "Jonathan Moss",
        "Simon Hooper",
    }


def test_raw_csv_is_unchanged_by_load(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)
    csv_path = tmp_path / "E0_1920.csv"
    before = csv_path.read_bytes()

    ingest.load_league("E0", ["1920"], tmp_path)

    assert csv_path.read_bytes() == before


def test_missing_corrections_file_raises(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)

    with pytest.raises(FileNotFoundError):
        ingest.load_league(
            "E0",
            ["1920"],
            tmp_path,
            referee_corrections=tmp_path / "does_not_exist.yaml",
        )


def test_repo_corrections_file_ships_approved_entries():
    import yaml

    config = yaml.safe_load(REPO_CORRECTIONS.read_text(encoding="utf-8"))

    assert set(config["match_corrections"]) == {
        "1920_20200118_man_city_crystal_palace",
        "2021_20210516_everton_sheffield_united",
        "2425_20250214_brighton_chelsea",
        "2425_20250414_bournemouth_fulham",
    }
    for spec in config["match_corrections"].values():
        assert spec["referee"] and spec["to"] and spec["evidence"]
    assert config["aliases"] == {"S Singh": "S Singh Gill"}


def test_validate_flags_referees_with_two_or_fewer_matches(monkeypatch, tmp_path):
    _prime(tmp_path, "E0_1920.csv")
    monkeypatch.setattr(ingest, "_fetch", _forbid_network)
    df = ingest.load_league("E0", ["1920"], tmp_path)
    # Duplicate the fixture so Michael Oliver has 4 matches while the two
    # single-appearance referees stay at 2 (also <= 2).
    doubled = pd.concat(
        [df, df.assign(match_id=df["match_id"] + "_b")], ignore_index=True
    )

    summary = ingest.validate(doubled)

    rare = summary["rare_referees"]
    assert rare.to_dict() == {"Jonathan Moss": 2, "Simon Hooper": 2}
    assert "Michael Oliver" not in rare.index