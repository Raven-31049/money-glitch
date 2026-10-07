# Phase 0 gate review

Date: 2026-10-07. Scope: **only the four items in the Phase 0 "Gate checklist"**
in `docs/PLAN.md` (Phase 0, lines 31–36). Everything outside those four items is
listed under [Deferred / later phases](#deferred--later-phases), tagged with the
phase it belongs to. The rest of PLAN.md (build lists, lessons learned) and
IMPLEMENTATION_NOTES.md are not part of this grade.

Full suite run for this review:

```
$ uv run pytest -q
380 passed in 10.20s
```

## Verdict

**Phase 0 four-item gate: PASSED.** All four Phase 0 "Gate checklist" items are
met by the current tree.

| # | Phase 0 Gate checklist item (PLAN.md, verbatim) | Verdict | Proof |
| --- | --- | --- | --- |
| 1 | No-lookahead assertion present and tested against a deliberately-broken case | **PASS** | `walkforward.py:79` `assert_train_before_predict`; `tests/test_walkforward.py:203` `test_a_leaky_split_is_refused_before_the_first_fit` runs a deliberately-leaky split and asserts it raises **before any model is constructed** (`factory.calls == 0`); `tests/test_walkforward.py:323` proves the check survives `python -O` |
| 2 | Both stop_on_breach modes implemented and produce different results on a toy case | **PASS** | `tests/test_bankroll.py:168` `test_the_stop_flag_changes_the_final_bankroll` runs the same toy frame both ways: stopped `800` vs played-on `1382.4`; the flag is a parameter of `simulate(..., stop_on_breach=...)` in `src/bet_engine/backtest/bankroll.py` |
| 3 | Bootstrap checker runs on a synthetic known-edge and known-no-edge sequence, correctly distinguishing them | **PASS** | `tests/test_bootstrap.py:56` 55% wins -> `fraction_profitable > 0.95`; `:68` 50% (no edge) -> `0.4 < fraction_profitable < 0.6`; `:76` 45% -> `fraction_profitable < 0.05` |
| 4 | Control-variant pattern works for at least one market before building a second | **PASS** | `tests/test_run.py:64` `test_control_places_no_bets_in_either_stop_mode` (0 bets in both modes, both simulated); `tests/test_staking.py:277` `test_pure_market_control_places_no_bets`; only `match_winner` exists, so "before building a second" holds |

### Item evidence detail

**1 - no-lookahead assertion, deliberately-broken case.** The check is
`assert_train_before_predict` (`src/bet_engine/backtest/walkforward.py:79`),
called by `walk_forward` on every fold. A deliberately-leaky index splitter is
refused before the first fit (`tests/test_walkforward.py:203`, `factory.calls == 0`,
"temporal leak" in the message); the guard also survives `python -O` because it
raises `AssertionError` explicitly instead of using an `assert` statement
(`tests/test_walkforward.py:323`).

**2 - both stop modes, different toy result.** `simulate` takes `stop_on_breach`
as a parameter (`src/bet_engine/backtest/bankroll.py`). On one toy frame the two
modes diverge: stopped closes at `800` after the breaching day, played-on
finishes at `1382.4` (`tests/test_bankroll.py:168`), while the breach itself is
recorded identically in both (`tests/test_bankroll.py:144`).

**3 - bootstrap known-edge vs known-no-edge.** The bootstrap consumes synthetic
exact-count sequences (`tests/test_bootstrap.py:25`): a 55%-win (positive-edge)
sequence is profitable in >95% of resamples (`:56`); a 50%-win (zero-edge)
sequence lands between 0.4 and 0.6 (`:68`); a 45%-win (negative-edge) sequence
profits in <5% (`:76`). The checker separates the three by the sign of the edge,
not by the implementation.

**4 - control-variant for one market.** The `pure_market` control (zero model
weight, de-vigged odds) places 0 bets in both stop modes end to end
(`tests/test_run.py:64`) and at the staking layer (`tests/test_staking.py:277`);
a tripwire raises if it ever selects one (`tests/test_run.py:81`). Only
`match_winner` exists, which is exactly the "at least one market before building
a second" the item asks for.

### Phase 0 exit criteria (recorded for context)

- `uv run pytest` passes - **PASS**: `380 passed in 10.20s`.
- `uv run python -c "import bet_engine"` - **PASS**: `import OK: bet_engine`.

---

## Deferred / later phases

Each item is tagged with the IMPLEMENTATION_NOTES.md phase it belongs to.
Nothing here is part of the four-item grade above.

### Phase 0 — Remaining (still open)
- **Lint / format config (R2).** None exists: no `[tool.ruff]`/`[tool.black]`/
  `[tool.isort]`/`[tool.mypy]` in `pyproject.toml`; no `.ruff.toml`,
  `setup.cfg`, `.pre-commit-config.yaml`, `pyrightconfig.json` or
  `.editorconfig`. IMPLEMENTATION_NOTES.md §6 lists this under Phase 0 "Remaining", so it is a
  Phase 0 item, not a later phase. *(Stray `# pyright: ignore` comments exist in
  `tests/test_config.py` and `tests/test_ingest.py` with no pyright config.)*
- **`config.py` pydantic settings.** DONE: `VariantConfig` + per-section
  validation in `config.py`, 13 tests in `tests/test_config.py`. IMPLEMENTATION_NOTES.md:157
  still says "still a stub" — stale.

### Phase 2 — Probability core
- **Invariant 5 — uncertainty percentiles never renormalised.** No percentile
  container exists; `uncertainty_percentile` is only a validated config field
  (`tests/test_config.py:175-176`). Nearest test,
  `tests/test_calibration.py:230`, covers probability vectors, not percentiles.
- **Invariant 6 — vectorised numpy, no per-cell Python loops.** Zero
  "vectori"/"iteration"/"per-cell" assertions in `tests/`. Source is vectorised
  (`pure_market.devig_long`, `bootstrap._draws`) but nothing pins it. IMPLEMENTATION_NOTES.md Phase 2
  exit criteria require the test.
- **S3 — de-vig has one method, not two.** Only proportional
  (`staking.devig` / `pure_market.devig_long`); IMPLEMENTATION_NOTES.md §5 asks for two independent
  methods and per-market method recording. Phase 2.
- **S4 — percentile storage** in predictions/DB rests on invariant 5; Phase 2.

### Phase 4 — EV, Kelly, filters
- **S5 — EV basis (FIXED in this pass).** IMPLEMENTATION_NOTES.md §5 now defines EV on the raw odds
  actually paid (`model_prob * raw_odds - 1`); de-vigged odds only produce the
  market probability. Selection in `run.py` switched to raw odds. Regression
  tests: `tests/test_staking.py::test_ev_uses_the_paid_odds_not_the_de_vigged_fair_odds`
  (fair edge +0.10, raw edge -0.01 → not selected) and
  `tests/test_run.py::test_selection_uses_the_raw_odds_actually_paid`.
  Re-run: control **0/0**, naive **2771/19** (was 2789/19).

### Phase 5 — Validation engine
- **Invariant 8 — validation engine market-agnostic.** Only `match_winner`
  exists; agnosticism is structural, not verified by a second-market test.
- **IMPLEMENTATION_NOTES.md §7.3 invariant verification** for the current code (invariants 1–4 are
  proven; 5–8 are deferred above).

### Phase 6 — First real model, one league
- **Invariant 7 — one model per league, no pooling.** Structurally satisfied
  (one `league` per config, one model factory per run); no explicit test.

### Bootstrap — RESOLVED in this pass
- **S2 — bootstrap unit.** IMPLEMENTATION_NOTES.md §5 has been amended from day-block to
  **single-bet** resampling (the code already did single-bet). `eval/bootstrap.py`
  and `eval/report.py:404` cite §5 only for the duty to state the unit; no text
  anywhere claims the plan requires day blocks (grep confirms). `db.py:406`
  "day-block grouping" refers to date parsing, not the bootstrap.

### IMPLEMENTATION_NOTES.md §7 per-phase checklist (applies to every phase)
- **IMPLEMENTATION_NOTES.md §7.1 full suite run** — done this pass (`380 passed`).
- **IMPLEMENTATION_NOTES.md §7.2 no test edited or deleted to go green** — three same-session
  corrections remain (see S6); need explicit sign-off.
- **IMPLEMENTATION_NOTES.md §7.4 commit after the suite passes** — done: checkpoint `a2f1925`, fixes
  commit to follow.

### Other shortcuts / assumptions (S-items)
- **S1 — built ahead of the phase gates.** Phases 1–7-equivalent code exists
  while IMPLEMENTATION_NOTES.md once declared Phase 0 current. This is what makes checkbox 2 fail.
- **S6 — three test corrections** in same-session tests (`test_pure_market.py`,
  `test_naive_frequency.py`, `test_run.py`); no committed test was weakened.
- **S8 — end-to-end tests run on a 10-row fixture**, not real data; the full-EPL
  runs (0/0, 2771/19) are manual, not asserted in CI.
- **S9 — IMPLEMENTATION_NOTES.md drift.** The Phase 0 status line and change log were stale;
  §5 is amended for S2/S5, but the "CURRENT" marker and "Remaining" list may
  still need a pass.
- **S10 — data coverage.** 170 of 3040 loaded matches are skipped by the
  `pinnacle_only` policy (`docs/DATA_NOTES.md`); reports state this.
- **S11 — reports/DB are gitignored artifacts**, reproducible from configs.
- **S12 — unspecified implementation choices:** microsecond run ids; stopped run
  stored as a derived row (`*_stop`, `derived_from`); `config_hash` = first 12
  hex chars of SHA-256 over sorted config JSON (`eval/report.py:503-512`);
   small-sample threshold 100 bets (`SMALL_SAMPLE_BETS`).
