"""Provider client and normaliser for football-data.co.uk.

The site publishes one CSV per league per season at
https://www.football-data.co.uk/mmz4281/{season}/{div}.csv, where "SEASON" is a
four-character code like "1920" (2019-20) and "DIV" is a league code like "E0"
(English Premier League).

WHY this module exists beyond "download a CSV": the site's files vary by season
in ways that break naive readers — date formats flip between dd/mm/yy and
dd/mm/yyyy, 1X2 odds columns appear and disappear, and files sometimes end with
fully empty rows. Every later stage (walk-forward folds, EV, Kelly) is only as
trustworthy as the season/date boundary it is handed, so normalising provenance
here, once, keeps that boundary honest across all seasons.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import requests

_BASE_URL = "https://www.football-data.co.uk/mmz4281"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT_S = 30.0

# The 2019-20 EPL resumed on 17 June 2020 behind closed doors. Matches on or
# after that date, and every 2020-21 match, were played without crowds. A fixed
# date keeps covid_affected reproducible and independent of when the code runs.
_COVID_RESTART = pd.Timestamp("2020-06-17")

# Canonical column layout. 1X2 odds columns are appended after these, only for
# bookmakers actually present in a season's file.
_BASE_ORDER: Sequence[str] = [
    "match_id",
    "league",
    "season",
    "date",
    "home",
    "away",
    "fthg",
    "ftag",
    "ftr",
    "referee",
    "hy",
    "ay",
    "hr",
    "ar",
    "hc",
    "ac",
    "hs",
    "as_",
    "hst",
    "ast",
]

# Raw column name -> canonical name. Columns missing from a season become NaN;
# that is a guarantee of the normaliser, not a hope, so every frame is shaped
# identically regardless of the source file's omissions.
_RENAMES: Mapping[str, str] = {
    "HomeTeam": "home",
    "AwayTeam": "away",
    "Referee": "referee",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "ftr",
    "HY": "hy",
    "AY": "ay",
    "HR": "hr",
    "AR": "ar",
    "HC": "hc",
    "AC": "ac",
    "HS": "hs",
    # Canonical name is `as_`, not `as`, which is a Python keyword (df.as would
    # bind to the DataFrame method instead of the column).
    "AS": "as_",
    "HST": "hst",
    "AST": "ast",
}

# 1X2 bookmaker prefixes. A source column is a 1X2 price iff its leading prefix
# is in this set AND what follows is exactly H/D/A. This deliberately excludes
# Asian-handicap and over/under lines (AHh, BbAHh, O2.5, BbOUh ...) whose
# suffixes end in different letters.
_ODDS_1X2_PREFIXES = frozenset(
    {
        "B365", "BA", "BF", "BR", "BS", "BW", "Avg", "BbAv", "BbMx",
        "IW", "LB", "Max", "PS", "PSC", "SB", "SI", "SO", "VC", "WH",
    }
)


def _slug(value: object) -> str:
    """Deterministic id fragment from a team name or a raw date string."""
    if pd.isna(value):
        return "na"
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or "x"


def _is_1x2_odds(column: str) -> bool:
    """True when `column` is a bookmaker 1X2 price (a H/D/A trio member)."""
    for prefix in _ODDS_1X2_PREFIXES:
        if column.startswith(prefix) and column[len(prefix):] in ("H", "D", "A"):
            return True
    return False


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse football-data dates, tolerating dd/mm/yy and dd/mm/yyyy variants.

    Both formats are tried on every value; whichever parses wins. Note the yy
    form is intentionally NOT expanded to 19xx — "18/07/20" must become 2020-07-18.
    """
    text = values.astype("string").str.strip()
    four_digit = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    two_digit = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
    return four_digit.fillna(two_digit)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a fixture CSV, tolerating a UTF-8 BOM and/or latin-1 encoding.

    utf-8-sig is tried first (it strips BOMs and covers the plain-ASCII files
    the site ships); latin-1 is the fallback because it can never fail to
    decode, so a single unknown byte can never kill a whole season load.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _default_data_dir() -> Path:
    """Repo-level cache: <repo root>/data/raw (gitignored)."""
    return Path(__file__).resolve().parents[3] / "data" / "raw"


def _fetch(url: str, dst: Path) -> None:
    """Download `url` into `dst` atomically (temp file, then rename).

    A destination file is never left half-written, so a crash mid-download
    cannot be mistaken later for a valid cache entry.
    """
    response = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT_S,
        stream=True,
    )
    response.raise_for_status()
    tmp = dst.with_suffix(dst.suffix + ".part")
    with open(tmp, "wb") as handle:
        for chunk in response.iter_content(chunk_size=65536):
            handle.write(chunk)
    os.replace(tmp, dst)


def download_season(div: str, season: str, data_dir: Path | None = None) -> Path:
    """Ensure the cached CSV for a season exists and return its path.

    The cache file is data_dir/{div}_{season}.csv. If it already exists — e.g.
    because it was placed there by hand — it is used as-is and no download
    happens. Downloads are written atomically so concurrent runs never see a
    partial file as a valid cache entry.
    """
    data_dir = data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / f"{div}_{season}.csv"
    if dst.exists():
        return dst
    url = f"{_BASE_URL}/{season}/{div}.csv"
    _fetch(url, dst)
    return dst


def _covid_affected(season: str, dates: pd.Series) -> pd.Series:
    """Flag matches played behind closed doors for a single season.

    Season codes are on football-data's convention ("1920" = 2019-20). Only the
    two pandemic seasons are ever True: every 2020-21 match, and 2019-20 matches
    from the 17 June 2020 restart (COVID_RESTART) onwards. NaT dates cannot be
    flagged because the comparison yields False.
    """
    if season == "2021":
        return pd.Series(True, index=dates.index)
    if season == "1920":
        return dates >= _COVID_RESTART
    return pd.Series(False, index=dates.index)


def _read_canonical(div: str, season: str, path: Path) -> pd.DataFrame:
    """Turn one season's raw fixture file into a canonical single-season frame."""
    raw = _read_csv(path)
    # Some files end with fully empty rows; they carry no match and no signal.
    raw = raw.dropna(how="all")
    if "Date" not in raw.columns:
        raise ValueError(
            f"{path.name}: no 'Date' column — is this a football-data CSV?"
        )

    odds_cols = [col for col in raw.columns if _is_1x2_odds(col)]
    present = [col for col in _RENAMES if col in raw.columns]
    frame = raw[present].rename(columns=_RENAMES)
    for col in _BASE_ORDER:
        if col not in ("match_id", "league", "season", "date") and col not in frame.columns:
            frame[col] = pd.NA
    for col in odds_cols:
        frame[col] = raw[col]

    frame["league"] = div
    frame["season"] = season
    frame["date"] = _parse_dates(raw["Date"])
    frame["covid_affected"] = _covid_affected(season, frame["date"])

    tokens = []
    for raw_date, date, home, away in zip(
        raw["Date"], frame["date"], frame["home"], frame["away"]
    ):
        token = date.strftime("%Y%m%d") if pd.notna(date) else _slug(raw_date)
        tokens.append(f"{season}_{token}_{_slug(home)}_{_slug(away)}")
    frame["match_id"] = tokens
    return frame


def load_league(div: str, seasons: Sequence[str], data_dir: Path | None = None) -> pd.DataFrame:
    """Load one league across seasons as a single canonical DataFrame.

    Each season is read from its cache (downloading once if needed), then all
    seasons are joined on the union of columns — a bookmaker absent from one
    season is NaN there, never a crash. Returns rows sorted by date then
    match_id, with the canonical column order above.
    """
    frames = []
    for season in seasons:
        path = download_season(div, season, data_dir)
        frames.append(_read_canonical(div, season, path))
    if not frames:
        raise ValueError("load_league needs at least one season")

    df = pd.concat(frames, join="outer", ignore_index=True)
    odds = sorted(c for c in df.columns if c not in _BASE_ORDER and c != "covid_affected")
    df = df[[*_BASE_ORDER, *odds, "covid_affected"]]
    # mergesort is stable, so within an equal date the season/id order is the
    # one we built, not an arbitrary quicksort scramble.
    return df.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)


def validate(df: pd.DataFrame) -> dict:
    """Print and return a data-quality summary of a canonical league frame.

    Checks: rows per season, % missing per key column, date range, duplicate
    match_ids, and the covid_affected population (total and per season).
    """
    season_counts = df.groupby("season", dropna=False).size()
    key_cols = ["date", "home", "away", "fthg", "ftag", "ftr", "referee"]
    missing_pct = (100.0 * df[key_cols].isna().mean()).round(2)
    covid_by_season = df.groupby("season")["covid_affected"].sum().astype(int)

    summary = {
        "rows": int(len(df)),
        "season_counts": season_counts.astype(int),
        "missing_pct": missing_pct,
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "duplicate_match_ids": int(df["match_id"].duplicated().sum()),
        "covid_affected_total": int(df["covid_affected"].sum()),
        "covid_affected_by_season": covid_by_season,
    }

    print("Rows per season:")
    print(season_counts.to_string())
    print("\n% missing per key column:")
    print(missing_pct.to_string())
    print(f"\nDate range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Duplicate match_ids: {summary['duplicate_match_ids']}")
    print(f"covid_affected: {summary['covid_affected_total']} matches")
    print("covid_affected by season:")
    print(covid_by_season.to_string())
    return summary


__all__ = ["download_season", "load_league", "validate"]