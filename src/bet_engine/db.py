"""SQLite storage: connection lifecycle, schema, and run/prediction/bet helpers.

Storage is deliberately boring — stdlib ``sqlite3`` and one plain SQL file, no
ORM — so the only interesting part of any row is its provenance: which run,
which market, which config.

WHY schema.sql rather than inline DDL: the schema is the contract every report
is built on, and keeping it as reviewable SQL next to the code that reads and
writes it puts a column change in the same diff as the query that would break.

Every table repeats ``market``. The engine is market-agnostic (invariant 8) and
stores several markets in one database, so "which market does this row belong
to" must be answerable from the row itself rather than from a join a caller has
to remember.

Each write helper commits before returning, so a crash mid-run never discards
the rows already written, and readers always see the last committed state.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Resolved relative to this module, so it works from a source checkout and
#: from an installed wheel alike (packaging includes it beside db.py).
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TABLES: tuple[str, ...] = ("runs", "predictions", "bets", "skipped")

# Table columns in insertion order: the single source of truth for writes, so
# a schema change breaks exactly one place instead of scattered SQL strings.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "runs": (
        "run_id",
        "variant_name",
        "market",
        "config_json",
        "created_at",
        "stop_on_breach",
    ),
    "predictions": (
        "run_id",
        "match_id",
        "date",
        "league",
        "market",
        "outcome",
        "model_prob",
        "market_prob",
        "odds",
        "odds_source",
    ),
    "bets": (
        "run_id",
        "match_id",
        "date",
        "league",
        "market",
        "outcome",
        "odds",
        "odds_source",
        "stake",
        "pnl",
        "bankroll_after",
        "seq_no",
    ),
    "skipped": (
        "run_id",
        "match_id",
        "date",
        "league",
        "market",
        "reason",
    ),
}

# Columns added after the first schema shipped. SQLite cannot add a NOT NULL
# column without a default, so a migrated column is nullable and its old rows
# read as NULL — reported as "(unknown)" odds source rather than dropped, since
# pretending those bets had a source would be a fabrication.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("predictions", "odds_source", "ALTER TABLE predictions ADD COLUMN odds_source TEXT"),
    ("bets", "odds_source", "ALTER TABLE bets ADD COLUMN odds_source TEXT"),
)


@dataclass(frozen=True)
class Run:
    """One executed backtest variant: its identity and the config behind it."""

    run_id: str
    variant_name: str
    market: str
    config: dict[str, Any]
    created_at: str
    stop_on_breach: bool


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection, apply the schema, and switch foreign keys on.

    ``foreign_keys`` is a per-connection setting in SQLite, not a database
    property, so it is set here rather than in schema.sql — otherwise every
    caller would have to remember it and one forgotten call would let orphaned
    predictions outlive their run.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql and pending column migrations. Safe to call repeatedly.

    Both halves are idempotent: the DDL is all IF NOT EXISTS, and each
    migration only runs while its column is missing. A database created before
    ``odds_source`` existed therefore upgrades in place instead of needing a
    hand-written ALTER by whoever happens to open it first.
    """
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    migrate(conn)
    conn.commit()


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply pending column migrations, returning the DDL statements run.

    Checked per column through PRAGMA rather than by version number: there is
    no migration ledger to get out of sync, and "does this column exist" is
    the only question that matters. Commits only when something changed, so a
    no-op call leaves any in-flight transaction alone.
    """
    applied: list[str] = []
    for table, column, ddl in _MIGRATIONS:
        columns = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        # An empty set means the table itself is absent, which cannot happen
        # once schema.sql has run; skipping is still the safe answer.
        if columns and column not in columns:
            conn.execute(ddl)
            applied.append(ddl)
    if applied:
        conn.commit()
    return applied


def write_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    variant_name: str,
    market: str,
    config: Mapping[str, Any] | str,
    created_at: datetime | str | None = None,
    stop_on_breach: bool = False,
) -> Run:
    """Insert a run, commit, and return it as written.

    ``config`` is the variant's full configuration: a mapping is serialised
    with sorted keys so the same config always produces byte-identical JSON
    (and therefore the same hash), while a pre-serialised string is validated
    as a JSON object before it is stored. Unparseable provenance is refused at
    write time rather than discovered while reading a report.
    """
    for label, value in (
        ("run_id", run_id),
        ("variant_name", variant_name),
        ("market", market),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-blank string, got {value!r}")

    config_json = config if isinstance(config, str) else json.dumps(config, sort_keys=True)
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"run {run_id!r}: config is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"run {run_id!r}: config must be a JSON object, got "
            f"{type(parsed).__name__}"
        )

    stamp = _utc_now() if created_at is None else _iso(created_at)
    conn.execute(
        "INSERT INTO runs (run_id, variant_name, market, config_json, "
        "created_at, stop_on_breach) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, variant_name, market, config_json, stamp, int(bool(stop_on_breach))),
    )
    conn.commit()
    return Run(
        run_id=run_id,
        variant_name=variant_name,
        market=market,
        config=parsed,
        created_at=stamp,
        stop_on_breach=bool(stop_on_breach),
    )


def read_run(conn: sqlite3.Connection, run_id: str) -> Run | None:
    """One run by id, or None when no such run was written."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return None if row is None else _run_from_row(row)


def list_runs(conn: sqlite3.Connection) -> list[Run]:
    """Every run, oldest first by creation time, then by id for determinism."""
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at, run_id"
    ).fetchall()
    return [_run_from_row(row) for row in rows]


def write_predictions(
    conn: sqlite3.Connection, records: Iterable[Mapping[str, Any]]
) -> int:
    """Insert prediction rows and return the count.

    Records may be ``DataFrame.to_dict("records")`` output: keys must be a
    subset of the predictions columns and every record must carry the same
    keys, so a half-populated batch cannot be written by accident.
    """
    return _insert(conn, "predictions", records)


def read_predictions(
    conn: sqlite3.Connection, run_id: str, *, market: str | None = None
) -> list[dict[str, Any]]:
    """A run's prediction rows in insertion order, optionally one market only."""
    return _read(conn, "predictions", run_id, market, order="rowid")


def write_bets(
    conn: sqlite3.Connection, records: Iterable[Mapping[str, Any]]
) -> int:
    """Insert bet rows and return the count; same record rules as predictions."""
    return _insert(conn, "bets", records)


def read_bets(
    conn: sqlite3.Connection, run_id: str, *, market: str | None = None
) -> list[dict[str, Any]]:
    """A run's bets in ``seq_no`` order — the order the simulation placed them."""
    return _read(conn, "bets", run_id, market, order="seq_no")


def write_skipped(
    conn: sqlite3.Connection, records: Iterable[Mapping[str, Any]]
) -> int:
    """Insert skipped-match rows and return the count.

    Same record rules as predictions and bets. ``reason`` is free text on
    purpose: the set of reasons grows with the system (policy, price sanity,
    settlement) and a CHECK constraint that enumerated them would need a
    migration every time one was added.
    """
    return _insert(conn, "skipped", records)


def read_skipped(
    conn: sqlite3.Connection, run_id: str, *, market: str | None = None
) -> list[dict[str, Any]]:
    """A run's skipped matches in insertion order, optionally one market only."""
    return _read(conn, "skipped", run_id, market, order="rowid")


def _read(
    conn: sqlite3.Connection,
    table: str,
    run_id: str,
    market: str | None,
    *,
    order: str,
) -> list[dict[str, Any]]:
    if market is None:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order}",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ? AND market = ? "
            f"ORDER BY {order}",
            (run_id, market),
        ).fetchall()
    return [dict(row) for row in rows]


def _insert(
    conn: sqlite3.Connection,
    table: str,
    records: Iterable[Mapping[str, Any]],
) -> int:
    """Validate a homogeneous batch of records and insert it in one statement."""
    rows = list(records)
    if not rows:
        return 0

    allowed = _COLUMNS[table]
    shapes: set[frozenset[str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(
                f"{table}: expected mapping records, got {type(row).__name__}"
            )
        unknown = sorted(key for key in row if key not in allowed)
        if unknown:
            raise KeyError(
                f"{table}: unknown column(s) {unknown}; allowed: {list(allowed)}"
            )
        shapes.add(frozenset(row))

    if len(shapes) != 1:
        raise ValueError(
            f"{table}: records do not share one set of columns: "
            f"{[sorted(shape) for shape in shapes]}"
        )
    keys = shapes.pop()
    if not keys:
        raise ValueError(f"{table}: records carry no columns")

    ordered = tuple(column for column in allowed if column in keys)
    placeholders = ", ".join("?" for _ in ordered)
    sql = (
        f"INSERT INTO {table} ({', '.join(ordered)}) VALUES ({placeholders})"
    )
    conn.executemany(
        sql,
        [
            tuple(_normalise(column, row[column]) for column in ordered)
            for row in rows
        ],
    )
    conn.commit()
    return len(rows)


def _run_from_row(row: sqlite3.Row) -> Run:
    try:
        config = json.loads(row["config_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"run {row['run_id']!r}: config_json is not valid JSON"
        ) from exc
    return Run(
        run_id=row["run_id"],
        variant_name=row["variant_name"],
        market=row["market"],
        config=config,
        created_at=row["created_at"],
        stop_on_breach=bool(row["stop_on_breach"]),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _normalise(column: str, value: object) -> Any:
    """Bind ``value``, normalising the columns that need it first."""
    normaliser = _NORMALISERS.get(column)
    return _bind(value) if normaliser is None else normaliser(value)


def _source_text(value: object) -> str | None:
    """Odds provenance for storage: no source is ``''``, never NULL.

    The column is NOT NULL because every prediction and bet must say which book
    priced it; a row the odds policy skipped legitimately has no book, and ``''``
    — not a constraint error, and not a silent NULL — is how that is recorded.
    Report readers fold ``''`` and NULL into "(unknown)" anyway, so pre-migration
    rows (which can only be NULL) read the same way.
    """
    if value is None:
        return ""
    bound = _bind(value)
    if not isinstance(bound, str):
        raise TypeError(
            f"odds_source must be a string or None, got {type(value).__name__}"
        )
    return bound


def _date_text(value: object) -> str | None:
    """Normalise a date to 'YYYY-MM-DD' text, or None when there is no date.

    Dates arrive as ``datetime``/``Timestamp`` from frames, as ``NaT`` when a
    match has no date, and as strings from hand-built records. Normalising
    once, here, means a report never has to compare '2019-08-10' against
    '10/08/2019', and that day-block grouping sees one format only.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value != value:  # NaT compares unequal to itself; datetimes do not
            return None
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        return text or None
    bound = _bind(value)  # NaN and numpy scalars; unknown types raise here
    return None if bound is None else str(bound)


#: Columns whose values are normalised on the way in instead of bound as-is:
#: `date` is format-sensitive and `odds_source` maps "no source" to ''.
_NORMALISERS = {"date": _date_text, "odds_source": _source_text}


def _bind(value: object) -> Any:
    """Coerce a caller's value into something SQLite can store.

    Records built from frames carry numpy scalars, ``Timestamp``/``NaT`` and
    NaN, none of which SQLite accepts as-is (a ``Timestamp`` is rejected even
    though it subclasses ``datetime``). Unrecognised types raise instead of
    being stringified, so a typo cannot end up as text inside a report.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value != value:  # NaT
            return None
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    item = getattr(value, "item", None)  # numpy scalar -> Python scalar
    if callable(item):
        try:
            return _bind(item())
        except (TypeError, ValueError):
            pass
    raise TypeError(
        f"cannot store {type(value).__name__} in SQLite; pass a str, int, "
        f"float, bool, date or None (value: {value!r})"
    )


__all__ = [
    "SCHEMA_PATH",
    "TABLES",
    "Run",
    "connect",
    "init_schema",
    "list_runs",
    "migrate",
    "read_bets",
    "read_predictions",
    "read_run",
    "read_skipped",
    "write_bets",
    "write_predictions",
    "write_run",
    "write_skipped",
]
