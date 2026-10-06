# PLAN.md — Betting Analytics Research System

Status: **Phase 0 is the current phase.** Do not begin Phase 1 until Phase 0 exit
criteria pass. Rules for the agent live in `AGENTS.md`; read it before any task.

---

## 1. Purpose

Build a research system that answers one question honestly:

> Does a model find positive expected value on a specific league and market, after
> vig, and does that edge survive walk-forward evaluation, bankroll simulation, and
> statistical significance testing?

The system is a *research instrument*, not a betting bot. It produces reports and
refuses to produce confident-sounding conclusions from thin evidence.

## 2. Non-negotiable invariants

Summarised here for convenience; `AGENTS.md` is authoritative.

1. Walk-forward folds advance by **calendar day**. Train data is strictly earlier
   than the prediction day. Enforced by a runtime assertion that must never be
   removed or weakened.
2. Every backtest runs in both modes: `stop_on_breach=True` and
   `stop_on_breach=False`. Reports show both, side by side.
3. Bootstrap significance always runs on the **full, untruncated** bet sequence,
   even when the reported backtest halted early on a breach.
4. Every market has a **pure market control** variant: de-vigged odds, model weight
   zero. It must place ~0 bets. Non-zero bets indicate an EV or Kelly bug.
5. Uncertainty percentiles per outcome are **never** renormalised to sum to 1.
6. Probability grids and sampling are vectorised numpy. No per-cell Python loops.
7. One model per league. No pooling unless a task explicitly requires it.
8. The validation engine (backtest, bankroll, bootstrap, calibration, control) is
   market-agnostic. Market logic lives only in `markets/` and `models/`.

## 3. Repository layout

```
src/bet_engine/
  config.py            # pydantic settings; paths, thresholds, league registry
  data/
    schema.sql         # SQLite DDL, versioned
    store.py           # connection handling, migrations, upserts
    providers/         # one module per odds/result source
    calendar.py        # calendar-day boundaries, league timezones
  models/
    base.py            # Model protocol; predict(frame) -> PredictionBatch
    registry.py
  probability/
    devig.py           # margin removal methods
    uncertainty.py     # percentile vectors, joint samplers
    sampling.py        # vectorised scenario grids
  markets/
    base.py            # Market protocol (outcomes, settlement, tick size)
    moneyline.py
    spread.py
    totals.py
  ev/
    expected_value.py
    kelly.py           # fractional Kelly, caps
    filters.py         # edge thresholds, liquidity, staleness
  validation/
    walkforward.py     # calendar-day fold engine + temporal assertion
    backtest.py        # stop_on_breach handling
    bankroll.py
    bootstrap.py       # block bootstrap over days
    calibration.py     # Brier, log loss, reliability bins, ECE
    controls.py        # pure-market and other control variants
    report.py
  cli.py
tests/
docs/
```

## 4. Data model (SQLite)

| Table             | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `leagues`         | id, name, sport, timezone                                        |
| `events`          | league_id, start_time_utc, home, away, event_key (unique)        |
| `markets`         | event_key, market_type, line, tick_size                         |
| `odds_snapshots`  | market_id, book, captured_at, price vector, is_closing          |
| `results`         | event_key, settled_at, outcome scores                            |
| `predictions`     | event_key, market_id, model_name, model_version, code_version,   |
|                   | probability vector, uncertainty percentiles, created_at         |
| `bets`            | backtest_run_id, event_key, market_id, side, odds, stake, edge   |
| `backtest_runs`   | run_id, model, market, league, config hash, stop_on_breach      |

Every prediction row stores provenance. A result without a model version and code
version is not reproducible and is not admissible into a report.

## 5. Core design decisions

**Probability representation.** A prediction stores a point probability vector plus,
per outcome, an uncertainty percentile vector (e.g. p10/p50/p90). These percentiles
are marginals and deliberately do not sum to 1 — that would destroy the information
they encode. Any operation that needs a valid joint distribution (simulation,
expected-value integration) goes through the joint sampler, which normalises
internally. The stored percentiles are never rewritten to sum to 1.

**De-vig.** Implemented once in `probability/devig.py` with at least two independent
methods (proportional and Shin or power). Method choice is recorded per market.
The control variant uses de-vigged odds only, with model weight 0.

**EV and Kelly.** EV is computed on de-vigged odds. Kelly is fractional with a
configurable cap and a hard per-bet stake ceiling. A bet is placed only if it clears
the edge threshold; with model weight 0 the edge is ~0 everywhere, so the pure market
control places no bets. This is the tripwire for EV/Kelly bugs.

**Stop on breach.** `stop_on_breach=True` halts the strategy at the first hard limit
breach (drawdown limit, per-bet cap violation, non-finite edge, stale odds beyond
threshold) and records the breach. `stop_on_breach=False` runs to completion.
Reports present both, because "we stopped" and "we kept going" are different claims.

**Walk-forward.** Folds advance one calendar day at a time. Training data is
everything strictly before the fold's day start, in the league's timezone (UTC by
default, configurable per league). The engine asserts
`max(train.timestamp) < fold.day_start` at runtime.

**Bootstrap.** Block bootstrap resampling whole calendar days, because bets within a
day are correlated through shared conditions. Always computed on the full,
untruncated sequence. Reports state the resampling unit and block size.

**Honesty rules in reporting.** No ROI conclusion from fewer than ~100 bets without
an explicit "sample too small" statement. Every report states the bet count, the
period, the model and code version, and the config hash.

## 6. Phases

Each phase lists deliverables and exit criteria. Exit criteria are pass/fail. Do not
start the next phase until the current one is green.

### Phase 0 — Scaffolding (CURRENT)
- `uv` installed, `pyproject.toml` with Python 3.11+, src layout, pytest config.
- `src/bet_engine/` package skeleton with empty modules and docstrings.
- `config.py` with pydantic settings.
- `.gitignore`, lint/format config, `uv run pytest` green on an empty suite.
- Exit: `uv run pytest` passes; `uv run python -c "import bet_engine"` works.

### Phase 1 — Data layer
- `schema.sql`, `store.py`, migrations, typed upserts.
- Calendar-day utilities with league timezone handling.
- One odds/result provider end to end, with snapshot timestamps.
- Exit: fixtures loaded for one league; `events`/`odds_snapshots` queryable;
  tests cover timezone boundaries on day rollover.

### Phase 2 — Probability core
- De-vig methods with cross-validation on overround sanity.
- Uncertainty percentile container that refuses to renormalise.
- Vectorised joint sampler and scenario grid; tests assert no Python-level
  per-cell iteration (shape and vectorisation checks).
- Exit: unit tests for de-vig, sampler marginals, percentile invariants.

### Phase 3 — Market adapters
- `markets/base.py` protocol; moneyline first.
- Settlement logic, tick sizes, de-vig wiring.
- Exit: moneyline round-trips; a synthetic event settles correctly in tests.

### Phase 4 — EV, Kelly, filters
- `expected_value.py`, `kelly.py`, `filters.py`.
- Exit: **pure market control places zero bets** in a test; edge and stake are
  correct on hand-computed fixtures.

### Phase 5 — Validation engine
- Walk-forward engine with the temporal assertion.
- Backtest with both `stop_on_breach` modes.
- Bankroll simulation, block bootstrap, calibration, control harness.
- Exit: a leakage test that deliberately trains on future data **fails loudly**;
  both breach modes reported; bootstrap runs on the full sequence even when the
  reported backtest halted.

### Phase 6 — First real model, one league
- One league, one market, one model. No pooling.
- Full report with calibration, bankroll curve, bootstrap CI.
- Exit: report generated end to end; if the sample is under ~100 bets the report
  says so in plain language.

### Phase 7 — CLI and reporting
- Reproducible CLI entry points, config hashing, report output.
- Exit: one command reproduces a report from a config file.

### Phase 8 — Expansion
- Additional markets (spread, totals), additional leagues one model at a time.
- Each addition repeats Phases 3–6 for that market/league. No shortcuts through
  the validation engine.

## 7. Per-phase checklist

Before declaring any phase done:

1. Full suite: `uv run pytest` — report results honestly, including failures.
2. New tests added for new behaviour. No test edited or deleted to go green.
3. Invariants 1–8 verified explicitly for the new code.
4. Commit only after the suite passes, so any phase can be rolled back.

## 8. Open questions

- Odds provider choice and rate limits are undecided; the provider interface is
  designed so this is a Phase 1 implementation detail.
- Bankroll starting stake, drawdown limit, and Kelly cap need real values before
  Phase 5 reporting is meaningful. Placeholders will be used until decided, and
  reports must print the values actually used.
- Whether to support multi-book line shopping in v1 is undecided. Default: single
  best price per market, recorded.

## 9. Change log

- Initial plan. Phase 0 scaffold only; no implementation beyond scaffolding until
  Phase 0 exit criteria pass.
