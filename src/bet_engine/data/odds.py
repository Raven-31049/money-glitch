"""Canonical odds table and its join onto our matches (Part B).

The cards over/under market settles at a *per-match* line, and the only place
that line (and its two prices) comes from is the odds source. This module is the
one boundary that turns an outside odds file into the columns the rest of the
system expects, and then attaches them to our matches.

WHY a canonical schema instead of reading a source's columns directly: every
later stage (the market, the model, EV, the report) reads the same column names,
so swapping the source is a change here and nowhere else. The schema is the
market-neutral over/under contract:

    match_id, date, home, away, line, over_odds, under_odds, source, is_closing

WHY the join is by date AND team names, with a mapping file: odds feeds spell
clubs differently from football-data ("Manchester United" vs "Man United"), and
a date alone does not identify a match. ``configs/team_names.yaml`` maps the
odds feed's names onto football-data's canonical ones — the same
corrections-file pattern as ``referee_corrections.yaml`` — and the join REPORTS
every odds row it could not match and every match it could not price, rather
than dropping either silently. A silent drop would quietly bias every ROI that
follows (docs/IMPLEMENTATION_NOTES.md §5).

There is no network access here: a caller supplies the odds file (or frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

#: The canonical odds schema, in order. ``match_id`` is optional on the source
#: side (our matches own the id) and is carried only for provenance if present.
ODDS_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home",
    "away",
    "line",
    "over_odds",
    "under_odds",
    "source",
    "is_closing",
)

#: Columns that must be present, non-null and usable to price a match.
_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "date",
    "home",
    "away",
    "line",
    "over_odds",
    "under_odds",
)

#: The columns the join keys on: a date alone is not an identity.
_JOIN_KEYS: tuple[str, ...] = ("date", "home", "away")


def _default_team_names_path() -> Path:
    """Repo-level team-name map: <repo root>/configs/team_names.yaml."""
    return Path(__file__).resolve().parents[3] / "configs" / "team_names.yaml"


@dataclass(frozen=True)
class OddsJoinReport:
    """What the join matched, and every row it did not.

    ``unmatched_odds`` are odds rows with no match in our frame;
    ``unmatched_matches`` are our matches with no odds row. Both are returned in
    full so a caller can print them rather than trust a count.
    """

    n_odds: int
    n_matches: int
    n_matched: int
    unmatched_odds: pd.DataFrame
    unmatched_matches: pd.DataFrame

    @property
    def n_unmatched_odds(self) -> int:
        return int(len(self.unmatched_odds))

    @property
    def n_unmatched_matches(self) -> int:
        return int(len(self.unmatched_matches))


def normalise_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw odds frame to the canonical schema, or raise naming the gap.

    Dates become datetime64; ``line``/``over_odds``/``under_odds`` become floats;
    team and source text is stripped; ``is_closing`` becomes a bool (default
    False). Missing required columns, unparseable dates, blank team names,
    negative/non-finite lines or non-positive prices are all errors — an odds
    file is either usable or it is not, and a half-usable one would price some
    matches off garbage.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    missing = [column for column in _REQUIRED_SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"odds table is missing required column(s) {missing}; "
            f"have {list(df.columns)}"
        )
    frame = df.copy()

    frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    bad_dates = frame["date"].isna()
    if bad_dates.any():
        raise ValueError(
            f"odds table has {int(bad_dates.sum())} unparseable date(s); "
            f"first bad row(s): {list(frame.index[bad_dates][:5])}"
        )

    for column in ("home", "away"):
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    for column in ("home", "away"):
        blank = frame[column].isna() | (frame[column].astype("string").str.len() == 0)
        if blank.any():
            raise ValueError(
                f"odds table has {int(blank.sum())} blank {column} value(s); "
                f"the join keys on names"
            )

    for column in ("line", "over_odds", "under_odds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad_line = ~frame["line"].map(lambda v: _finite(v) and float(v) >= 0.0)
    if bad_line.any():
        raise ValueError(
            f"odds line must be finite and >= 0; bad row(s): "
            f"{list(frame.index[bad_line][:5])}"
        )
    for column in ("over_odds", "under_odds"):
        bad_price = ~frame[column].map(lambda v: _finite(v) and float(v) > 1.0)
        if bad_price.any():
            raise ValueError(
                f"odds {column} must be a finite price > 1.0; bad row(s): "
                f"{list(frame.index[bad_price][:5])}"
            )

    if "source" in frame.columns:
        frame["source"] = frame["source"].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    else:
        frame["source"] = pd.NA
    frame["is_closing"] = (
        _to_bool(frame["is_closing"]) if "is_closing" in frame.columns else False
    )
    if "match_id" not in frame.columns:
        frame["match_id"] = pd.NA
    frame["match_id"] = frame["match_id"].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    return frame.loc[:, list(ODDS_COLUMNS)].reset_index(drop=True)


def apply_team_names(
    df: pd.DataFrame, path: str | Path | None = None
) -> pd.DataFrame:
    """Apply ``configs/team_names.yaml`` aliases to ``home``/``away``, in place.

    Mirrors ``data.ingest._apply_referee_corrections``: the file is required to
    exist (a missing map is a configuration error, not a silent no-op), and its
    ``aliases`` mapping renames an odds feed's spelling onto football-data's
    canonical one for both team columns. Raw files are never edited; the rename
    lives only in the returned frame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    mapping_path = Path(path) if path is not None else _default_team_names_path()
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"team-name map not found: {mapping_path} — it ships with the repo "
            "and must not be skipped"
        )
    config = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    aliases = config.get("aliases") or {}
    frame = df.copy()
    for column in ("home", "away"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: aliases.get(value, value))
    return frame


def load_odds_table(
    path: str | Path,
    *,
    team_names: str | Path | None = None,
) -> pd.DataFrame:
    """Read an odds CSV, normalise it, and apply the team-name map.

    ``team_names`` overrides the default mapping file (used by tests). The
    result is in the canonical :data:`ODDS_COLUMNS` order.
    """
    frame = pd.read_csv(path)
    frame = normalise_odds(frame)
    return apply_team_names(frame, team_names)


def select_odds(odds: pd.DataFrame, *, prefer_closing: bool = True) -> pd.DataFrame:
    """One odds row per (date, home, away), preferring the closing price.

    A source may quote opening *and* closing prices for a match. Closing is the
    least noisy estimate of the true line, so it wins by default; ties are
    otherwise broken by keeping the first row of the (stable) sort, so the
    choice is deterministic and reruns are reproducible.
    """
    frame = odds.copy()
    if prefer_closing:
        frame = frame.sort_values("is_closing", ascending=False, kind="mergesort")
    return frame.drop_duplicates(subset=list(_JOIN_KEYS), keep="first").reset_index(
        drop=True
    )


def join_odds(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    prefer_closing: bool = True,
) -> tuple[pd.DataFrame, OddsJoinReport]:
    """Left-join ``odds`` onto ``matches`` by date and team names.

    Returns ``(merged, report)``. ``merged`` is ``matches`` plus the odds
    columns (``line``, ``over_odds``, ``under_odds``, ``is_closing``) and the
    odds ``source`` and ``match_id`` (the latter renamed ``odds_match_id`` so it
    does not collide with our own). A match with no odds row keeps NaN prices
    and a NaN line — which the market then treats as unpriced, exactly as it
    treats a row the odds policy cannot price.

    ``report`` names every odds row with no matching match AND every match with
    no matching odds, in full, so nothing is dropped silently.
    """
    for name, frame in (("matches", matches), ("odds", odds)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame, got {type(frame).__name__}")
    missing_matches = [column for column in _JOIN_KEYS if column not in matches.columns]
    if missing_matches:
        raise ValueError(
            f"matches is missing join column(s) {missing_matches}; "
            f"have {list(matches.columns)}"
        )
    missing_odds = [column for column in ODDS_COLUMNS if column not in odds.columns]
    if missing_odds:
        raise ValueError(
            f"odds is missing canonical column(s) {missing_odds}; normalise it "
            f"with normalise_odds() first"
        )

    left = matches.copy()
    left["date"] = pd.to_datetime(left["date"], format="mixed", errors="coerce")
    right = select_odds(odds, prefer_closing=prefer_closing)

    odds_side = right.rename(columns={"match_id": "odds_match_id"})
    odds_side = odds_side[
        [*_JOIN_KEYS, "line", "over_odds", "under_odds", "source", "is_closing", "odds_match_id"]
    ]
    merged = left.merge(
        odds_side,
        on=list(_JOIN_KEYS),
        how="left",
        validate="many_to_one",
        suffixes=("", "_odds"),
    )

    match_keys = left[list(_JOIN_KEYS)].drop_duplicates()
    odds_keys = right[list(_JOIN_KEYS)].drop_duplicates()
    tagged_odds = right.merge(match_keys.assign(_matched=1), on=list(_JOIN_KEYS), how="left")
    unmatched_odds = tagged_odds.loc[tagged_odds["_matched"].isna()].drop(
        columns="_matched"
    )
    tagged_matches = left.merge(odds_keys.assign(_matched=1), on=list(_JOIN_KEYS), how="left")
    unmatched_matches = tagged_matches.loc[tagged_matches["_matched"].isna()].drop(
        columns="_matched"
    )

    n_matched = int(
        merged["line"].notna().sum() if "line" in merged.columns else 0
    )
    report = OddsJoinReport(
        n_odds=int(len(right)),
        n_matches=int(len(left)),
        n_matched=n_matched,
        unmatched_odds=unmatched_odds.reset_index(drop=True),
        unmatched_matches=unmatched_matches.reset_index(drop=True),
    )
    return merged, report


def _finite(value: object) -> bool:
    """True for a real, finite number; False for None/NaN/NaN-like/odd types."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _to_bool(values: pd.Series) -> pd.Series:
    """Coerce a column to bool, reading common textual truth values."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    truthy = {"1", "true", "yes", "y", "closing", "close"}

    def convert(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, int, float)):
            return bool(value)
        return str(value).strip().lower() in truthy

    return values.map(convert).astype(bool)


__all__ = [
    "ODDS_COLUMNS",
    "OddsJoinReport",
    "apply_team_names",
    "join_odds",
    "load_odds_table",
    "normalise_odds",
    "select_odds",
]
