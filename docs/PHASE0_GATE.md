# Phase 0 gate review

Date: 2026-10-07. Scope: **only the four Phase 0 "Built" checkboxes** in
`docs/PLAN.md` §6 (Phase 0, lines 144–148). Everything outside those four items
is listed under [Deferred / later phases](#deferred--later-phases), tagged with
the PLAN.md phase it belongs to. The non-negotiable invariants (§2) and the §7
per-phase checklist are not part of this grade; they are cross-cutting rules and
appear in the deferred section.

Full suite run for this review:

```
$ uv run pytest -q
380 passed in 10.20s        (378 before this pass + 2 new tests)
```

## Verdict

**Phase 0 four-checkbox gate: NOT PASSED.** Checkboxes 1, 3 and 4 pass.
Checkbox 2 ("docstring + TODO modules, **no logic**") is not met by the current
tree: the scaffold was built (commit `1f923e2`) and then superseded by working
modules built ahead of the phase gates (see S1 in the deferred section). Nothing
is structurally missing; the item as literally written can only be cleared by
accepting the superseded state or re-wording PLAN.md.

| # | Phase 0 checkbox (PLAN.md §6) | Verdict | Proof |
| --- | --- | --- | --- |
| 1 | `uv` installed (0.12.x); `pyproject.toml` with Python 3.11+; src layout; pytest config | **PASS** | `uv --version` → `0.12.23`; `pyproject.toml:6` `requires-python = ">=3.11"`; `pyproject.toml:25-26` `packages = ["src/bet_engine"]`; `pyproject.toml:28-29` `testpaths = ["tests"]` |
| 2 | `src/bet_engine/` package skeleton — docstring + TODO modules, **no logic** | **NOT passed** | Skeleton exists and matches §3, but modules are full implementations (`run.py`, `walkforward.py`, `bankroll.py`, `staking.py`, `bootstrap.py`, `calibration.py`, `report.py`, `naive_frequency.py`, `pure_market.py`). See S1 |
| 3 | `.gitignore` (Python + `data/` + `reports/`); tracked `.gitkeep` placeholders | **PASS** | `.gitignore` has the Python block and `data/raw/*`/`data/processed/*`/`reports/*` with `!.gitkeep` negations; `git ls-files "*.gitkeep"` → 4 tracked files |
| 4 | All declared dependencies installed; `uv.lock` committed | **PASS** | `uv run python -c "import scipy, statsmodels, requests, pandas, numpy, pydantic, yaml"` → `deps OK`; `git ls-files uv.lock` → `uv.lock` |

### Checkbox evidence detail

**1 — toolchain / layout / pytest config.** `uv --version` →
`uv 0.12.23 (46b84fd0b 2026-10-03 x86_64-pc-windows-msvc)` (0.12.x). src layout
and pytest config as cited above.

**2 — skeleton, no logic.** `src/bet_engine/` contains `config.py`, `db.py`,
`data/`, `markets/`, `models/`, `backtest/`, `eval/`, plus `run.py` and
`schema.sql`, all populated with real implementations and covered by tests. The
"no logic" half of the checkbox no longer holds.

**3 — .gitignore / .gitkeep.** Tracked placeholders: `configs/.gitkeep`,
`data/processed/.gitkeep`, `data/raw/.gitkeep`, `reports/.gitkeep`.

**4 — dependencies / lock.** All seven declared dependencies import in the
locked environment; `uv.lock` is tracked.

### Phase 0 exit criteria (part of Phase 0, recorded for context)

- `uv run pytest` passes — **PASS**: `380 passed in 10.20s`.
- `uv run python -c "import bet_engine"` — **PASS**: `import OK: bet_engine`.

---

## Deferred / later phases

Each item is tagged with the PLAN.md phase it belongs to. Nothing here is part
of the four-checkbox grade above.

### Phase 0 — Remaining (still open)
- **Lint / format config (R2).** None exists: no `[tool.ruff]`/`[tool.black]`/
  `[tool.isort]`/`[tool.mypy]` in `pyproject.toml`; no `.ruff.toml`,
  `setup.cfg`, `.pre-commit-config.yaml`, `pyrightconfig.json` or
  `.editorconfig`. PLAN §6 lists this under Phase 0 "Remaining", so it is a
  Phase 0 item, not a later phase. *(Stray `# pyright: ignore` comments exist in
  `tests/test_config.py` and `tests/test_ingest.py` with no pyright config.)*
- **`config.py` pydantic settings.** DONE: `VariantConfig` + per-section
  validation in `config.py`, 13 tests in `tests/test_config.py`. PLAN.md:151
  still says "still a stub" — stale.

### Phase 2 — Probability core
- **Invariant 5 — uncertainty percentiles never renormalised.** No percentile
  container exists; `uncertainty_percentile` is only a validated config field
  (`tests/test_config.py:175-176`). Nearest test,
  `tests/test_calibration.py:230`, covers probability vectors, not percentiles.
- **Invariant 6 — vectorised numpy, no per-cell Python loops.** Zero
  "vectori"/"iteration"/"per-cell" assertions in `tests/`. Source is vectorised
  (`pure_market.devig_long`, `bootstrap._draws`) but nothing pins it. PLAN Phase 2
  exit criteria require the test.
- **S3 — de-vig has one method, not two.** Only proportional
  (`staking.devig` / `pure_market.devig_long`); PLAN §5 asks for two independent
  methods and per-market method recording. Phase 2.
- **S4 — percentile storage** in predictions/DB rests on invariant 5; Phase 2.

### Phase 4 — EV, Kelly, filters
- **S5 — EV basis (FIXED in this pass).** PLAN §5 now defines EV on the raw odds
  actually paid (`model_prob * raw_odds - 1`); de-vigged odds only produce the
  market probability. Selection in `run.py` switched to raw odds. Regression
  tests: `tests/test_staking.py::test_ev_uses_the_paid_odds_not_the_de_vigged_fair_odds`
  (fair edge +0.10, raw edge -0.01 → not selected) and
  `tests/test_run.py::test_selection_uses_the_raw_odds_actually_paid`.
  Re-run: control **0/0**, naive **2771/19** (was 2789/19).

### Phase 5 — Validation engine
- **Invariant 8 — validation engine market-agnostic.** Only `match_winner`
  exists; agnosticism is structural, not verified by a second-market test.
- **§7.3 invariant verification** for the current code (invariants 1–4 are
  proven; 5–8 are deferred above).

### Phase 6 — First real model, one league
- **Invariant 7 — one model per league, no pooling.** Structurally satisfied
  (one `league` per config, one model factory per run); no explicit test.

### Bootstrap — RESOLVED in this pass
- **S2 — bootstrap unit.** PLAN §5 has been amended from day-block to
  **single-bet** resampling (the code already did single-bet). `eval/bootstrap.py`
  and `eval/report.py:404` cite §5 only for the duty to state the unit; no text
  anywhere claims the plan requires day blocks (grep confirms). `db.py:406`
  "day-block grouping" refers to date parsing, not the bootstrap.

### §7 per-phase checklist (applies to every phase)
- **§7.1 full suite run** — done this pass (`380 passed`).
- **§7.2 no test edited or deleted to go green** — three same-session
  corrections remain (see S6); need explicit sign-off.
- **§7.4 commit after the suite passes** — done: checkpoint `a2f1925`, fixes
  commit to follow.

### Other shortcuts / assumptions (S-items)
- **S1 — built ahead of the phase gates.** Phases 1–7-equivalent code exists
  while PLAN once declared Phase 0 current. This is what makes checkbox 2 fail.
- **S6 — three test corrections** in same-session tests (`test_pure_market.py`,
  `test_naive_frequency.py`, `test_run.py`); no committed test was weakened.
- **S8 — end-to-end tests run on a 10-row fixture**, not real data; the full-EPL
  runs (0/0, 2771/19) are manual, not asserted in CI.
- **S9 — PLAN.md drift.** The Phase 0 status line and change log were stale;
  §5 is amended for S2/S5, but the "CURRENT" marker and "Remaining" list may
  still need a pass.
- **S10 — data coverage.** 170 of 3040 loaded matches are skipped by the
  `pinnacle_only` policy (`docs/DATA_NOTES.md`); reports state this.
- **S11 — reports/DB are gitignored artifacts**, reproducible from configs.
- **S12 — unspecified implementation choices:** microsecond run ids; stopped run
  stored as a derived row (`*_stop`, `derived_from`); `config_hash` = first 12
  hex chars of SHA-256 over sorted config JSON (`eval/report.py:503-512`);
  small-sample threshold 100 bets (`SMALL_SAMPLE_BETS`).

## To clear the Phase 0 checkbox gate

1. Resolve checkbox 2: accept the superseded "no logic" state, or re-word
   PLAN.md §6 so the Phase 0 deliverable is what actually shipped.
2. Add lint/format config (Phase 0 Remaining, R2) and document the command.
3. Sign off the three same-session test corrections (§7.2).
