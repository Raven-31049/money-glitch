# Phase 0 gate review

Date: 2026-10-07. Reviewed against `docs/PLAN.md` (Phase 0 section, lines 143–154,
and the per-phase checklist in §7) plus the non-negotiable invariants in §2 and
`AGENTS.md`.

This is a **fresh re-verification** of the 2026-10-06 review. Every citation in
the earlier review was re-checked against the current tree, the suite was re-run,
and every command below was executed for this pass. The verdicts are unchanged;
one citation was corrected (stray `# pyright: ignore` comments appear in
`tests/test_config.py` as well as `tests/test_ingest.py`).

Full suite run for this review:

```
$ uv run pytest -q
........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 57%]
........................................................................ [ 76%]
........................................................................ [ 95%]
..................                                                       [100%]
378 passed in 19.54s
```

## Verdict

**Phase 0 gate: NOT PASSED.** The two Phase 0 *exit criteria* pass, but four
checklist items fail or are only partially met: lint/format config (a declared
Remaining item), invariants 5–8 under §7.3, the "no test edited to go green"
clause under §7.2, and §7.4 (no commit exists to roll back to). Details and the
exact gaps are below.

| # | Item | Verdict |
| --- | --- | --- |
| B1 | uv 0.12.x, pyproject `>=3.11`, src layout, pytest config | PASS |
| B2 | Package skeleton: docstring + TODO modules, **no logic** | **NOT passed** |
| B3 | `.gitignore` (Python + `data/` + `reports/`), tracked `.gitkeep` | PASS |
| B4 | All declared dependencies installed; `uv.lock` committed | PASS |
| R1 | `config.py` pydantic settings | PASS |
| R2 | lint/format config | **NOT passed** |
| E1 | `uv run pytest` passes | PASS |
| E2 | `uv run python -c "import bet_engine"` works | PASS |
| 7.1 | Full suite run; results reported honestly | PASS |
| 7.2a | New tests added for new behaviour | PASS |
| 7.2b | No test edited or deleted to go green | **NOT passed** (3 same-session corrections) |
| 7.3 | Invariants 1–8 verified explicitly for the new code | **NOT passed** (1–4 pass; 5–8 fail) |
| 7.4 | Commit only after the suite passes, so a phase can be rolled back | **NOT passed** |

---

## Phase 0 "Built"

### B1 — uv / pyproject / src layout / pytest config — PASS
- `uv --version` → `uv 0.12.23 (46b84fd0b 2026-10-03 x86_64-pc-windows-msvc)`
  (0.12.x as required).
- `pyproject.toml:6` → `requires-python = ">=3.11"`.
- src layout: `pyproject.toml:25-26` → `[tool.hatch.build.targets.wheel]`
  `packages = ["src/bet_engine"]`.
- pytest config: `pyproject.toml:28-29` → `[tool.pytest.ini_options]`
  `testpaths = ["tests"]`.

### B2 — Package skeleton, docstring + TODO modules, no logic — NOT passed
The structure exists and matches §3 exactly (`src/bet_engine/` with `config.py`,
`db.py`, `data/`, `markets/`, `models/`, `backtest/`, `eval/`, plus `run.py` and
`schema.sql`), but the second half of the item — "docstring + TODO modules,
**no logic**" — is no longer true: those modules contain full implementations
(walk-forward, staking, bankroll, bootstrap, calibration, report, the run
pipeline, two models). The scaffold deliverable was completed (change log,
`PLAN.md:225`) and then superseded by work built ahead of phase (Shortcut S1).
*What's missing:* nothing structural; the item as written can only be cleared by
accepting the superseded state or re-scoping the wording in `docs/PLAN.md`.
Flagged rather than silently passed because the same divergence makes the PLAN
status line ("Phase 0 is the current phase") and the "never build ahead of the
current phase" rule untrue as written.

### B3 — `.gitignore` + tracked `.gitkeep` — PASS
- `.gitignore` contains the Python block (`__pycache__/`, `*.py[cod]`, `.venv/`,
  `.pytest_cache/`, …) and `data/raw/*`, `data/processed/*`, `reports/*` with
  `!data/raw/.gitkeep`, `!data/processed/.gitkeep`, `!reports/.gitkeep`
  negations.
- `git ls-files "*.gitkeep"` → `configs/.gitkeep`, `data/processed/.gitkeep`,
  `data/raw/.gitkeep`, `reports/.gitkeep` (all four tracked).

### B4 — Dependencies installed; `uv.lock` committed — PASS
- `uv run python -c "import scipy, statsmodels, requests, pandas, numpy,
  pydantic, yaml"` → `deps OK` (all seven declared dependencies import in the
  locked environment).
- `git ls-files uv.lock` → `uv.lock` (tracked in the repository).

---

## Phase 0 "Remaining"

### R1 — `config.py` pydantic settings — PASS
Implemented: `VariantConfig` with per-section validation (`StakingConfig`,
`BankrollConfig`, `BacktestConfig`, `BootstrapConfig`, `ModelSpec`),
`BootstrapConfig` (`n >= 1`, defaults 10000/42), `load_variant` / `load_all`
with location-aware errors (`config.py:201`, `:227`). Proven by 13 tests in
`tests/test_config.py` (lines 60–222), including
`test_load_variant_parses_all_fields` (`:60`), `test_validation_failures`
(`:205`), `test_missing_section_reports_locations` (`:151`),
`test_example_configs_load` (`:212`), and
`test_example_configs_state_their_odds_policy_explicitly` (`:222`). (`PLAN.md:151`
still says "still a stub" — stale; see Shortcut S9.)

### R2 — lint/format config — NOT passed
No lint or format configuration exists anywhere in the repository:
`pyproject.toml` has no `[tool.ruff]`/`[tool.black]`/`[tool.isort]`/`[tool.mypy]`
section, and a recursive search finds no `.ruff.toml`, `ruff.toml`, `setup.cfg`,
`.pre-commit-config.yaml`, `pyrightconfig.json` or `.editorconfig`. *What's
missing:* any linter/formatter config plus a documented command to run it.
(Note: `# pyright: ignore[...]` comments appear in `tests/test_config.py:11,12,14`
and `tests/test_ingest.py:15,17` although no pyright config exists — an
editor-integration leftover, see Shortcut S7.)

---

## Phase 0 "Exit"

### E1 — `uv run pytest` passes — PASS
`378 passed in 19.54s`, zero failures, zero skips, zero xfails, zero deselected
(fresh run for this review).

### E2 — `import bet_engine` works — PASS
`uv run python -c "import bet_engine"` → `import OK: bet_engine`.

---

## §7 Per-phase checklist

### 7.1 Full suite run, results reported honestly — PASS
`378 passed` reported above; no test was skipped, xfailed, or deselected.

### 7.2a New tests added for new behaviour — PASS
New test modules exist for every new behaviour and are untracked-new relative to
HEAD (`a3f21f0`): `tests/test_bankroll.py`, `test_bootstrap.py`,
`test_calibration.py`, `test_db.py`, `test_markets.py`, `test_models_base.py`,
`test_naive_frequency.py`, `test_pure_market.py`, `test_report.py`,
`test_run.py`, `test_staking.py`, `test_walkforward.py` (12 modules; only
`test_config.py`, `test_ingest.py`, `test_smoke.py` are committed). `tests/test_run.py`
carries 7 end-to-end tests (control zero-bets, tripwire, both ledgers, report
sections, skipped matches, no-folds, CLI). The earlier review's exact delta
"35 tests added (343 → 378)" was not independently re-derived in this pass
(re-running the committed baseline would require stashing), but the behaviour
coverage claimed is present.

### 7.2b No test edited or deleted to go green — NOT passed
Strictly, three edits happened; all were to tests written minutes earlier in the
same session and never green before, and each corrected an objectively wrong
expectation rather than weakening a production-code check:
1. `tests/test_pure_market.py` — expected `match_id == ["m1"]`; `devig_long`
   correctly emits one row per outcome, so the expectation became
   `["m1", "m1", "m1"]`.
2. `tests/test_naive_frequency.py` — asserted pivot row order `[H, D, A]`;
   `pivot` orders columns alphabetically, so the assertion became label-based.
3. `tests/test_run.py` — removed a duplicated assertion (the rendered-text check
   was asserted twice).

Verified against git: no committed test was weakened. `tests/test_config.py`
diff is additive only (+2 new tests); `tests/test_ingest.py` diff touches only
import lines (added `# pyright: ignore` comments — no assertion or test logic
changed). *What's missing:* your explicit acceptance of these three corrections;
I did not treat them as free.

### 7.3 Invariants 1–8 verified explicitly — NOT passed (1–4 pass, 5–8 fail)

| Inv | Verdict | Evidence |
| --- | --- | --- |
| 1 — calendar-day folds, strictly earlier training, runtime assertion | **PASS** | `tests/test_walkforward.py:141` `test_no_fold_ever_trains_on_or_after_its_prediction_day`; `:203` `test_a_leaky_split_is_refused_before_the_first_fit` (leaky split fails loudly); `:276` `test_the_assertion_refuses_training_inside_the_prediction_day`; `:323` `test_the_leak_check_survives_running_under_python_dash_O` (`-O` proof; `walkforward.py:106,119` raise `AssertionError`, never `assert`). |
| 2 — both `stop_on_breach` modes side by side | **PASS** | `tests/test_bankroll.py:127` `test_stop_mode_settles_the_whole_breach_day_then_halts`, `:144` `test_playing_on_records_the_same_breach_and_places_every_bet`, `:168` `test_the_stop_flag_changes_the_final_bankroll`; `tests/test_run.py:98` `test_naive_places_bets_and_persists_both_ledgers`; `tests/test_report.py:453-454` assert both column headers. Real report shows both: `reports/epl_mw_naive_20261006T181645387988.md` → `Bets placed | 2789 | 19`. |
| 3 — bootstrap on the full, untruncated sequence | **PASS** | `tests/test_bootstrap.py:135` `test_a_stopped_runs_ledger_is_refused_as_truncated`, `:145` `test_the_truncation_flag_alone_is_enough_to_refuse_a_ledger`; guard at `eval/bootstrap.py:138-147`. Report proves it end to end: `reports/epl_mw_naive_20261006T181645387988.md` → stopped ledger = 19 bets, bootstrap `Bets resampled (n_bets): 2789`. |
| 4 — pure-market control places ~0 bets | **PASS** | `tests/test_staking.py:255` `test_pure_market_control_places_no_bets`; `tests/test_pure_market.py:127` `test_control_edge_is_never_positive`; `tests/test_run.py:64` `test_control_places_no_bets_in_either_stop_mode` + `:81` `test_tripwire_raises_when_the_control_somehow_selects_a_bet` (runtime guard at `run.py:130-136`). Full EPL report `reports/epl_mw_control_20261006T181620135047.md`: **`Bets placed | 0 | 0`**. |
| 5 — uncertainty percentiles never renormalised | **NOT passed** | No percentile container exists (Phase 2 not built); `uncertainty_percentile` is only a validated config field (`tests/test_config.py:175-176`). The nearest test, `tests/test_calibration.py:230` (`...refused_not_renormalised`), covers probability vectors, not percentiles. *Missing:* the Phase 2 container plus a test asserting it refuses renormalisation. |
| 6 — vectorised numpy, no per-cell Python loops | **NOT passed** | Zero hits for "vectori"/"iteration"/"per-cell" anywhere in `tests/`. The source reviewed is vectorised (`pure_market.devig_long`, `bootstrap._draws` batches), but nothing asserts it. *Missing:* shape/vectorisation tests as required by PLAN Phase 2 exit criteria ("tests assert no Python-level per-cell iteration"). |
| 7 — one model per league, no pooling | **NOT passed** | Structurally satisfied (one `league` per config, one model factory per run), but no test verifies the rule (grep for `pool`/`one model` finds only fixture CSV text). *Missing:* an explicit test. |
| 8 — validation engine market-agnostic | **NOT passed** | Only one market exists (`match_winner`); the engine imports contract-level symbols only, but no test exercises it against a second market type, so agnosticism is structural, not verified. *Missing:* an explicit test. |

### 7.4 Commit only after the suite passes — NOT passed
The suite is green (378 passed), but nothing has been committed: `git status`
shows 20 modified paths and 17 untracked paths (16 excluding this review file),
including `src/bet_engine/run.py`, both new models, `schema.sql`, and 12 test
modules. `git log` ends at `a3f21f0 Add typed YAML variant configs`. There is no
rollback point for the current phase. *What's missing:* one commit made against
this green suite — not done because changes are only committed on explicit
instruction.

---

## Shortcuts and assumptions made during Phase 0 (and the work since)

- **S1 — Built ahead of the phase gates.** `AGENTS.md` says "Never build ahead of
  the current phase" and `PLAN.md:3` says "Do not begin Phase 1 until Phase 0
  exit criteria pass", yet the tree now contains Phase 1–7-equivalent code
  (ingest, db, markets, walk-forward, staking, bankroll, bootstrap, calibration,
  report, CLI run pipeline) while PLAN.md still declares Phase 0 current. This
  was done on explicit instruction to wire the whole system into one command.
  The PLAN status line, the "Remaining" list, and the change log are all stale.
- **S2 — Bootstrap deviates from PLAN §5 (most important technical shortcut).**
  PLAN §5 mandates a *day-block* bootstrap ("bets within a day are correlated
  through shared conditions") and requires reports to state the block size. The
  implementation resamples **single bets** (`eval/bootstrap.py:10-13`, `_draws`
  at `:200-216`), and the report even cites PLAN §5 while printing
  "Resampling unit: **single bet**" (`eval/report.py:404`). Consequence:
  confidence intervals likely too narrow because intra-day correlation is
  ignored. The deviation is neither implemented nor recorded in the PLAN change
  log — it needs a decision: implement day blocks, or amend PLAN §5 and the
  report wording.
- **S3 — De-vig has one method, not two.** PLAN §5 requires at least two
  independent de-vig methods (proportional + Shin/power); only proportional
  (`staking.devig` / `pure_market.devig_long`) exists. Phase 2 scope; method
  choice is not yet recorded per market as PLAN §5 also asks.
- **S4 — Uncertainty percentiles are not implemented at all** (see invariant 5),
  so the "point probability + p10/p50/p90 storage" of PLAN §4/§5 exists only in
  config validation, not in predictions or the DB schema.
- **S5 — EV/Kelly basis is an interpretation choice.** Selection runs on fair
  odds (`run.py:128`, `candidates["odds"] = 1.0 / candidates["market_prob"]`)
  while Kelly and settlement use raw odds (`run.py:305`); selections where
  fractional Kelly stakes zero are filtered out and reported as a run note. The
  invariant-4 tripwire fires before that filter. This satisfies the control
  requirement but differs from a literal reading of "EV computed on de-vigged
  odds" (PLAN §5).
- **S6 — Test corrections.** Three assertions in same-session tests were fixed
  after their first run (details in 7.2b). No committed test was modified.
- **S7 — No lint/format tooling at all** (R2), although stray
  `# pyright: ignore` comments exist in `tests/test_config.py` and
  `tests/test_ingest.py` with no pyright config in the repo.
- **S8 — End-to-end tests run on a 10-row fixture**, not on real data; full-EPL
  verification (control 0/0, naive 2789/19) was performed manually via
  `uv run python -m bet_engine.run ...` and is not asserted in CI.
- **S9 — PLAN.md is stale** in several places relied on by this review: the
  Phase 0 "Remaining" list (`config.py` is done), the status line, and the
  change log (no entry for Phases 1+ or the run pipeline).
- **S10 — Data coverage.** 170 of 3040 loaded matches are skipped by the
  `pinnacle_only` policy (the documented 2025/26 gap, `docs/DATA_NOTES.md`);
  reports state this explicitly as a note.
- **S11 — Reports and the SQLite DB are gitignored artifacts.** Paths cited in
  this review (`reports/epl_mw_control_20261006T181620135047.md`,
  `reports/epl_mw_naive_20261006T181645387988.md`,
  `data/processed/bet_engine.sqlite3`) exist locally only and are reproducible
  from the configs.
- **S12 — Implementation choices PLAN does not specify:** run ids carry a
  microsecond timestamp; the stopped run is stored as a derived row
  (`*_stop`, `derived_from` in `config_json`); `config_hash` is the first 12 hex
  characters of SHA-256 over sorted config JSON (`eval/report.py:503-512`);
  small-sample threshold is 100 bets (`SMALL_SAMPLE_BETS`, `eval/report.py:39`).

## To clear the gate

1. Add lint/format config (Remaining item R2) and document the command.
2. Resolve S2: implement day-block bootstrap per PLAN §5, or amend §5 and the
   report wording (it currently mis-cites the section it deviates from).
3. Add explicit tests for invariants 5–8 (at minimum: vectorisation, one-model-
   per-league, market-agnosticism; invariant 5 additionally needs the Phase 2
   percentile container), or formally defer them with PLAN wording that says so.
4. Sign off on the three test corrections in 7.2b (or ask for re-review).
5. Commit the green suite (378 passed) to create the rollback point (7.4).
6. Update PLAN.md: status line, Remaining list, change log — so the next review
   measures a plan that matches the tree.
