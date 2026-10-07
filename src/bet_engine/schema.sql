-- SQLite schema for bet_engine, applied idempotently by db.init_schema().
--
-- One row in `runs` is one executed backtest variant; `predictions` and `bets`
-- hang off it, and `skipped` records what the run's odds policy refused to
-- price. Every table repeats `market`: the validation engine is
-- market-agnostic (invariant 8) and stores several markets in one database,
-- so a query that forgets to say which market it means is a bug waiting to be
-- written, and the column makes that filter impossible to forget.
--
-- Provenance columns are text on purpose: a report must be able to quote the
-- config JSON and the run's creation time verbatim, and name the bookmaker
-- whose price produced each odds value.

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    variant_name   TEXT NOT NULL,
    market         TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    stop_on_breach INTEGER NOT NULL CHECK (stop_on_breach IN (0, 1))
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    match_id       TEXT NOT NULL,
    date           TEXT,
    league         TEXT,
    market         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    model_prob     REAL,
    market_prob    REAL,
    odds           REAL,
    -- Which odds source produced `odds` (OddsSource.label, or the empty
    -- string for a row the policy could not price). NOT NULL on fresh
    -- databases: provenance is part of the value, and an odds number without
    -- its book is not a number a report can defend.
    odds_source    TEXT NOT NULL,
    -- One point estimate per outcome per match: a duplicate would silently
    -- double-count in calibration, so the database refuses it.
    UNIQUE (run_id, market, match_id, outcome)
);

CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);

CREATE TABLE IF NOT EXISTS bets (
    bet_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    match_id       TEXT NOT NULL,
    date           TEXT,
    league         TEXT,
    market         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    odds           REAL,
    odds_source    TEXT NOT NULL,
    stake          REAL,
    pnl            REAL,
    bankroll_after REAL,
    seq_no         INTEGER NOT NULL,
    -- seq_no is the replay order of the simulation; two bets cannot share a
    -- position in it without making the bankroll curve ambiguous.
    UNIQUE (run_id, seq_no)
);

CREATE INDEX IF NOT EXISTS idx_bets_run ON bets(run_id);

-- Matches the run's odds policy refused to price: no permitted book had a
-- complete, valid vector for the row. Stored per match rather than counted in
-- `runs` so a report can say *which* matches were left out and a reader can
-- check the exclusion against the raw data. UNIQUE stops the same skipped
-- match from being recorded twice — which would overstate how much of the
-- sample the run actually covered.
CREATE TABLE IF NOT EXISTS skipped (
    skipped_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    match_id   TEXT NOT NULL,
    date       TEXT,
    league     TEXT,
    market     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    UNIQUE (run_id, market, match_id)
);

CREATE INDEX IF NOT EXISTS idx_skipped_run ON skipped(run_id);
