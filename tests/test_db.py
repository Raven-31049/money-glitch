"""Tests for db.py: schema shape and full write/read roundtrips.

The run roundtrip reopens the file with a fresh connection, because a helper
that only works on the connection that wrote it would pass an in-memory test
and still lose data on disk. Every test here is offline — SQLite in a tmp dir.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from bet_engine import db

RUN_KWARGS = {"variant_name": "epl_mw_naive", "market": "match_winner"}
DEFAULT_CONFIG = {"staking": {"kelly_fraction": 0.25}}


def _run(conn: sqlite3.Connection, run_id: str = "run-1", **overrides) -> db.Run:
    kwargs = {**RUN_KWARGS, **overrides}
    config = kwargs.pop("config", DEFAULT_CONFIG)
    return db.write_run(conn, run_id, config=config, **kwargs)


# --- schema -----------------------------------------------------------------

def test_schema_creates_every_table_with_a_market_column(tmp_path):
    conn = db.connect(tmp_path / "schema.sqlite3")
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert set(db.TABLES) <= tables

        for table in db.TABLES:
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert "market" in columns, f"{table} has no market column"

        runs = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert runs >= {
            "run_id", "variant_name", "market", "config_json",
            "created_at", "stop_on_breach",
        }

        predictions = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        assert predictions >= {
            "run_id", "match_id", "date", "league", "market", "outcome",
            "model_prob", "market_prob", "odds", "odds_source",
        }

        bets = {row[1] for row in conn.execute("PRAGMA table_info(bets)")}
        assert bets >= {
            "run_id", "match_id", "date", "league", "market", "outcome",
            "odds", "odds_source", "stake", "pnl", "bankroll_after", "seq_no",
        }

        skipped = {row[1] for row in conn.execute("PRAGMA table_info(skipped)")}
        assert skipped >= {"run_id", "match_id", "date", "league", "market", "reason"}
    finally:
        conn.close()


def test_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "schema.sqlite3")
    try:
        db.init_schema(conn)  # applying it twice must not raise or duplicate
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert set(db.TABLES) <= tables
    finally:
        conn.close()


# --- runs -------------------------------------------------------------------

def test_run_roundtrip_reopens_the_file(tmp_path):
    path = tmp_path / "runs.sqlite3"
    conn = db.connect(path)
    config = {"staking": {"ev_threshold": 0.05}, "seasons": ["1819", "1920"]}
    written = db.write_run(
        conn,
        "run-1",
        variant_name="epl_mw_naive",
        market="match_winner",
        config=config,
        stop_on_breach=True,
    )
    conn.close()

    reopened = db.connect(path)
    try:
        got = db.read_run(reopened, "run-1")

        assert got == written
        assert got.config == config
        assert got.variant_name == "epl_mw_naive"
        assert got.market == "match_winner"
        assert got.stop_on_breach is True
        datetime.fromisoformat(got.created_at)  # parseable, not a repr()

        assert db.read_run(reopened, "no-such-run") is None
        assert db.list_runs(reopened) == [written]
    finally:
        reopened.close()


def test_stop_on_breach_defaults_to_false():
    conn = db.connect()
    try:
        written = _run(conn)
        assert written.stop_on_breach is False
        assert db.read_run(conn, "run-1").stop_on_breach is False
    finally:
        conn.close()


def test_config_serialisation_is_key_order_independent():
    conn = db.connect()
    try:
        db.write_run(conn, "a", config={"b": 1, "a": 2}, **RUN_KWARGS)
        db.write_run(conn, "b", config={"a": 2, "b": 1}, **RUN_KWARGS)

        stored = [
            row[0]
            for row in conn.execute("SELECT config_json FROM runs ORDER BY run_id")
        ]
        assert stored[0] == stored[1]
        assert json.loads(stored[0]) == {"a": 2, "b": 1}
    finally:
        conn.close()


@pytest.mark.parametrize(
    "config, expected",
    [
        ("not json at all", "not valid JSON"),
        ("[1, 2, 3]", "JSON object"),
    ],
)
def test_unparseable_config_is_refused(config, expected):
    conn = db.connect()
    try:
        with pytest.raises(ValueError, match=expected):
            _run(conn, config=config)
        assert db.read_run(conn, "run-1") is None
    finally:
        conn.close()


@pytest.mark.parametrize("overrides", [{"run_id": ""}, {"variant_name": " "}, {"market": ""}])
def test_blank_provenance_is_refused(overrides):
    conn = db.connect()
    try:
        with pytest.raises(ValueError, match="must be a non-blank string"):
            db.write_run(
                conn,
                overrides.get("run_id", "run-1"),
                config={},
                variant_name=overrides.get("variant_name", "v"),
                market=overrides.get("market", "match_winner"),
            )
        assert db.list_runs(conn) == []
    finally:
        conn.close()


def test_duplicate_run_id_is_refused():
    conn = db.connect()
    try:
        _run(conn)
        with pytest.raises(sqlite3.IntegrityError):
            _run(conn)
    finally:
        conn.close()


# --- predictions ------------------------------------------------------------

def _prediction(run_id: str, match_id: str, outcome: str, **overrides) -> dict:
    record = {
        "run_id": run_id,
        "match_id": match_id,
        "date": pd.Timestamp("2019-08-10"),
        "league": "E0",
        "market": "match_winner",
        "outcome": outcome,
        "model_prob": 0.5,
        "market_prob": 0.45,
        "odds": 2.1,
        "odds_source": "pinnacle_close",
    }
    record.update(overrides)
    return record


def test_prediction_roundtrip():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        written = [
            _prediction("run-1", "m1", "H", model_prob=np.float64(0.5)),
            _prediction(
                "run-1", "m1", "D", odds=np.nan, market_prob=None,
                odds_source="",
            ),
            _prediction("run-1", "m1", "A", date=None, odds_source=None),
        ]

        assert db.write_predictions(conn, written) == 3
        rows = db.read_predictions(conn, "run-1")

        assert [row["outcome"] for row in rows] == ["H", "D", "A"]
        assert {row["market"] for row in rows} == {"match_winner"}
        assert rows[0]["date"] == "2019-08-10"
        assert rows[0]["model_prob"] == 0.5
        assert rows[0]["odds_source"] == "pinnacle_close"
        assert rows[1]["odds"] is None          # NaN is stored as NULL
        assert rows[1]["market_prob"] is None
        assert rows[1]["odds_source"] == ""     # unpriced: no book to record
        assert rows[2]["date"] is None
        # None means "no source" in memory; storage records it as '' because
        # the column is NOT NULL (see db._source_text).
        assert rows[2]["odds_source"] == ""
    finally:
        conn.close()


def test_predictions_are_separable_by_market():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        db.write_predictions(
            conn,
            [
                _prediction("run-1", "m1", "H"),
                _prediction("run-1", "m1", "over", market="totals"),
            ],
        )

        totals = db.read_predictions(conn, "run-1", market="totals")
        assert [row["outcome"] for row in totals] == ["over"]
        assert len(db.read_predictions(conn, "run-1")) == 2
        assert db.read_predictions(conn, "other-run") == []
    finally:
        conn.close()


def test_orphan_prediction_is_refused_by_the_foreign_key():
    conn = db.connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.write_predictions(conn, [_prediction("ghost", "m1", "H")])
        conn.rollback()
        assert db.read_predictions(conn, "ghost") == []
    finally:
        conn.close()


def test_duplicate_prediction_is_refused():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        record = _prediction("run-1", "m1", "H")
        db.write_predictions(conn, [record])
        with pytest.raises(sqlite3.IntegrityError):
            db.write_predictions(conn, [record])
        conn.rollback()
        assert len(db.read_predictions(conn, "run-1")) == 1
    finally:
        conn.close()


def test_unknown_and_mixed_columns_are_refused():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        with pytest.raises(KeyError, match="unknown column"):
            db.write_predictions(conn, [_prediction("run-1", "m1", "H", edge=0.1)])
        with pytest.raises(ValueError, match="share one set of columns"):
            thinner = _prediction("run-1", "m1", "D")
            del thinner["market_prob"]
            db.write_predictions(
                conn, [_prediction("run-1", "m1", "H"), thinner]
            )
        with pytest.raises(TypeError, match="mapping records"):
            db.write_predictions(conn, [("run-1", "m1")])
        with pytest.raises(TypeError, match="cannot store"):
            db.write_predictions(conn, [_prediction("run-1", "m1", "H", odds=object())])
        with pytest.raises(TypeError, match="odds_source must be a string"):
            db.write_predictions(
                conn, [_prediction("run-1", "m1", "H", odds_source=2.0)]
            )
    finally:
        conn.close()


def test_empty_and_missing_batches_are_no_ops():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        assert db.write_predictions(conn, []) == 0
        assert db.write_bets(conn, iter([])) == 0
        assert db.read_predictions(conn, "run-1") == []
    finally:
        conn.close()


# --- bets -------------------------------------------------------------------

def _bet(run_id: str, match_id: str, seq_no: int, **overrides) -> dict:
    record = {
        "run_id": run_id,
        "match_id": match_id,
        "date": "2019-08-10",
        "league": "E0",
        "market": "match_winner",
        "outcome": "H",
        "odds": 2.1,
        "odds_source": "pinnacle_close",
        "stake": 10.0,
        "pnl": 11.0,
        "bankroll_after": 1011.0,
        "seq_no": seq_no,
    }
    record.update(overrides)
    return record


def test_bet_roundtrip_is_ordered_by_seq_no():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        # Written out of order on purpose: reads must follow the simulation.
        # seq_no arrives as a numpy integer, the way a frame would hand it over.
        db.write_bets(
            conn,
            [
                _bet("run-1", "m2", np.int64(2), pnl=-5.0,
                     bankroll_after=1006.0, date=None, odds=np.float64(3.4),
                     odds_source="b365"),
                _bet("run-1", "m1", 1),
            ],
        )

        rows = db.read_bets(conn, "run-1")

        assert [row["seq_no"] for row in rows] == [1, 2]
        assert [row["match_id"] for row in rows] == ["m1", "m2"]
        assert rows[1]["pnl"] == -5.0
        assert rows[1]["bankroll_after"] == 1006.0
        assert rows[1]["date"] is None
        assert rows[1]["odds"] == 3.4
        assert [row["odds_source"] for row in rows] == ["pinnacle_close", "b365"]
        assert {row["market"] for row in rows} == {"match_winner"}
    finally:
        conn.close()


def test_duplicate_seq_no_is_refused():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        db.write_bets(conn, [_bet("run-1", "m1", 1)])
        with pytest.raises(sqlite3.IntegrityError):
            db.write_bets(conn, [_bet("run-1", "m2", 1)])
        conn.rollback()
        assert len(db.read_bets(conn, "run-1")) == 1
    finally:
        conn.close()


def test_bets_are_separable_by_market():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        db.write_bets(
            conn,
            [
                _bet("run-1", "m1", 1),
                _bet("run-1", "m1", 2, market="totals", outcome="over"),
            ],
        )
        totals = db.read_bets(conn, "run-1", market="totals")
        assert [row["outcome"] for row in totals] == ["over"]
        assert len(db.read_bets(conn, "run-1")) == 2
    finally:
        conn.close()


# --- skipped -----------------------------------------------------------------

def _skip(run_id: str, match_id: str, **overrides) -> dict:
    record = {
        "run_id": run_id,
        "match_id": match_id,
        "date": "2019-08-10",
        "league": "E0",
        "market": "match_winner",
        "reason": "no complete price under policy pinnacle_only",
    }
    record.update(overrides)
    return record


def test_skipped_roundtrip():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        _run(conn, "run-2")
        written = db.write_skipped(
            conn,
            [
                _skip("run-1", "m1"),
                _skip("run-1", "m2", date=None),
                _skip("run-2", "m3"),  # another run's skips must not bleed in
            ],
        )

        rows = db.read_skipped(conn, "run-1")

        assert written == 3
        assert [row["match_id"] for row in rows] == ["m1", "m2"]
        assert rows[1]["date"] is None
        assert rows[0]["reason"] == "no complete price under policy pinnacle_only"
        assert db.read_skipped(conn, "run-1", market="totals") == []
        assert db.read_skipped(conn, "no-such-run") == []
    finally:
        conn.close()


def test_a_match_is_skipped_once_per_market_not_twice():
    """Double-counting the same skip would overstate the run's coverage."""
    conn = db.connect()
    try:
        _run(conn, "run-1")
        db.write_skipped(conn, [_skip("run-1", "m1")])
        with pytest.raises(sqlite3.IntegrityError):
            db.write_skipped(conn, [_skip("run-1", "m1")])
        conn.rollback()

        db.write_skipped(conn, [_skip("run-1", "m1", market="totals")])
        assert len(db.read_skipped(conn, "run-1")) == 2
    finally:
        conn.close()


def test_orphan_skip_is_refused_by_the_foreign_key():
    conn = db.connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.write_skipped(conn, [_skip("ghost", "m1")])
        conn.rollback()
        assert db.read_skipped(conn, "ghost") == []
    finally:
        conn.close()


# --- schema migration --------------------------------------------------------

#: DDL exactly as the first release shipped it: no odds_source, no skipped.
_LEGACY_SCHEMA = """
CREATE TABLE runs (
    run_id         TEXT PRIMARY KEY,
    variant_name   TEXT NOT NULL,
    market         TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    stop_on_breach INTEGER NOT NULL CHECK (stop_on_breach IN (0, 1))
);
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    match_id      TEXT NOT NULL,
    date          TEXT,
    league        TEXT,
    market        TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    model_prob    REAL,
    market_prob   REAL,
    odds          REAL,
    UNIQUE (run_id, market, match_id, outcome)
);
CREATE TABLE bets (
    bet_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    match_id       TEXT NOT NULL,
    date           TEXT,
    league         TEXT,
    market         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    odds           REAL,
    stake          REAL,
    pnl            REAL,
    bankroll_after REAL,
    seq_no         INTEGER NOT NULL,
    UNIQUE (run_id, seq_no)
);
"""


def test_a_database_from_before_odds_source_upgrades_in_place(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.executescript(_LEGACY_SCHEMA)
    legacy.execute(
        "INSERT INTO runs (run_id, variant_name, market, config_json, "
        "created_at, stop_on_breach) VALUES "
        "('run-1', 'v', 'match_winner', '{}', '2020-01-01T00:00:00+00:00', 0)"
    )
    legacy.execute(
        "INSERT INTO predictions (run_id, match_id, market, outcome, "
        "model_prob, market_prob, odds) VALUES "
        "('run-1', 'm1', 'match_winner', 'H', 0.5, 0.45, 2.0)"
    )
    legacy.execute(
        "INSERT INTO bets (run_id, match_id, market, outcome, odds, stake, "
        "pnl, bankroll_after, seq_no) VALUES "
        "('run-1', 'm1', 'match_winner', 'H', 2.0, 10.0, 10.0, 1010.0, 1)"
    )
    legacy.commit()
    legacy.close()

    conn = db.connect(path)
    try:
        # Old rows survive with no recorded source rather than failing to read.
        prediction = db.read_predictions(conn, "run-1")[0]
        assert prediction["odds"] == 2.0
        assert prediction["odds_source"] is None
        assert db.read_bets(conn, "run-1")[0]["odds_source"] is None

        # New rows write with provenance, and the skipped table now exists.
        assert db.write_predictions(conn, [_prediction("run-1", "m1", "D")]) == 1
        assert db.write_skipped(conn, [_skip("run-1", "m2")]) == 1

        # Migrating again is a no-op, not a duplicate-column error.
        assert db.migrate(conn) == []
        db.init_schema(conn)
        assert db.migrate(conn) == []
    finally:
        conn.close()
