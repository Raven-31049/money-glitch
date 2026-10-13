Technical notes written during the build. NOT the roadmap. docs/PLAN.md is the roadmap and wins any conflict.

# IMPLEMENTATION_NOTES.md — Betting Analytics Research System

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

### Built — Phase 0 scaffold

```
src/bet_engine/
  __init__.py
  config.py            # run settings, thresholds, league registry (stub only)
  db.py                # SQLite connection, schema, migrations
  data/
    ingest.py          # provider client, raw landing, normalization
  markets/
    base.py            # market contract: outcomes, settlement, tick size, de-vig
    match_winner.py    # three-way home/draw/away — first market
  models/
    base.py            # model contract; one per league; provenance + percentiles
  backtest/
    walkforward.py     # calendar-day fold engine + temporal assertion
    staking.py         # edge threshold, fractional Kelly, per-bet caps
    bankroll.py        # bankroll, drawdown, stop_on_breach handling
  eval/
    bootstrap.py       # single-bet bootstrap on the full bet sequence
    calibration.py     # Brier, log loss, reliability bins, ECE
    report.py          # both breach modes, significance, sample-size warning
tests/                 # pytest
configs/               # YAML variant configs
data/raw/              # gitignored payloads (.gitkeep tracked)
data/processed/        # gitignored payloads (.gitkeep tracked)
reports/               # gitignored output (.gitkeep tracked)
docs/
```

Modules are placeholders (docstring + TODO) until their phase. Nothing below
exists yet; it is created when its phase starts, never before.

### Built — Phase 1 step 1 (cards, team-only)

```
src/bet_engine/
  models/cards_poisson.py   # causal shrunk card rates (team + referee), team/team+ref Poisson, league-average control
  markets/cards_totals.py   # hy+ay+hr+ar vs line settlement (not registered: no config `line` yet)
scripts/phase1_step1_cards.py  # walk-forward calibration report — no odds, no bets
```

### Built — Phase 1 step 2 (referee variant)

```
configs/referee_corrections.yaml      # load-time referee name corrections (match-level + aliases)
scripts/phase1_step2_referee.py       # 3 models on identical matches: scores, paired bootstrap, verdict
```

### Added later, by phase

| Phase | Module | Purpose |
| ----- | ------ | ------- |
| 1 | `data/calendar.py`, `data/providers/` | day boundaries per league timezone; one module per source |
| 2 | `probability/` → `devig.py`, `uncertainty.py`, `sampling.py` | de-vig, percentile containers, vectorised samplers |
| 4 | `ev/` → `expected_value.py`, `filters.py` | EV and bet filters; Kelly sizing stays in `backtest/staking.py` |
| 5 | `eval/controls.py` | pure-market control harness |
| 6 | `models/registry.py` | league → model lookup |
| 7 | `cli.py` | entry points, config hashing |
| 8 | `markets/spread.py`, `markets/totals.py` | expansion, one market at a time |

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

**EV and Kelly.** EV is computed on the odds actually paid (the raw book odds):
`ev = model_prob * raw_odds - 1`. De-vigged odds are used only to produce the
market's probability — the pure-market control's prediction and the report's
`market_prob` column — never for EV or bet selection. Kelly is fractional with a
configurable cap and a hard per-bet stake ceiling. A bet is placed only if it clears
the edge threshold; with model weight 0 the raw-odds edge is negative wherever the
book prices above fair, so the pure market control places no bets. This is the
tripwire for EV/Kelly bugs.

**Stop on breach.** `stop_on_breach=True` halts the strategy at the first hard limit
breach (drawdown limit, per-bet cap violation, non-finite edge, stale odds beyond
threshold) and records the breach. `stop_on_breach=False` runs to completion.
Reports present both, because "we stopped" and "we kept going" are different claims.

**Walk-forward.** Folds advance one calendar day at a time. Training data is
everything strictly before the fold's day start, in the league's timezone (UTC by
default, configurable per league). The engine asserts
`max(train.timestamp) < fold.day_start` at runtime.

**Bootstrap.** Resamples individual bet P&Ls, drawn with replacement; the single bet
is the resampling unit. Always computed on the full, untruncated sequence. Reports
state the resampling unit.

**Holdout (fixed at Phase 1 step 3c).** Seasons **2425 and 2526 are the final
test set**. Every model decision from step 3c onwards — adoption, tuning,
feature choice, calibration — is made on seasons **1819-2324 only**. The holdout
is touched once, at the end, when the user asks for a final check; reporting it
during development is a leak that voids it. This mirrors `AGENTS.md` rule 9 and
is enforced in `scripts/phase1_step3c_diagnostics.py`, which refuses to load a
holdout season.

**Cards rate recency (adopted at Phase 1 step 3c).** The Phase 1 card rates are
time-weighted with an exponential half-life of **365 calendar days**, one value
applied identically to the team, referee and league-mean sums
(`add_card_features` / `add_referee_feature`, `half_life_days=365`). It was
adopted by a pre-registered rule on 1819-2324 only: the last-two-season mean
|gap| had to at least halve, and walk-forward log loss at lines 3.5/4.5/5.5 had
to worsen by no more than +0.001, in both datasets. Both held (|gap| halved in
both; every log-loss delta was negative), so recency replaced the cumulative
history for the cards model. The feature builders keep `half_life_days=None`
(cumulative) as their default so the frozen steps 1-3b remain reproducible; the
adopted cards recipe passes `365`. A second, more invasive fix (walk-forward
affine recalibration of the expected total) is frozen but **shelved** — see §9.

**Reliability tables are always per-line.** A P(over) reliability table must be
built from *that line's* probabilities: the diagnostic once reused the single
line-4.5 walk-forward output for the 3.5 and 5.5 tables, producing identical
predictions across lines. Since a higher line is strictly harder to clear,
`scripts/phase1_step3c_diagnostics.py` now rebuilds each line from the captured
per-side means and asserts both the line-4.5 reconstruction and
P(over 3.5) > P(over 4.5) > P(over 5.5) on every match. Any future per-line
reliability readout must do the same.

**Honesty rules in reporting.** No ROI conclusion from fewer than ~100 bets without
an explicit "sample too small" statement. Every report states the bet count, the
period, the model and code version, and the config hash.

## 6. Phases

Each phase lists deliverables and exit criteria. Exit criteria are pass/fail. Do not
start the next phase until the current one is green.

### Phase 0 — Scaffolding (CURRENT)
Built:
- `uv` installed (0.12.x), `pyproject.toml` with Python 3.11+, src layout, pytest config.
- `src/bet_engine/` package skeleton — docstring + TODO modules, no logic.
- `.gitignore` (Python + `data/` + `reports/`), tracked `.gitkeep` placeholders.
- All declared dependencies installed; `uv.lock` committed.

Remaining before exit criteria can be called complete:
- `config.py` pydantic settings (still a stub — no logic was added at Phase 0).
- lint/format config.

Exit: `uv run pytest` passes; `uv run python -c "import bet_engine"` works.

### Phase 1 — Data layer
- `db.py`: schema, migrations, typed upserts.
- `data/calendar.py` — calendar-day utilities with league timezone handling.
- `data/providers/` — one odds/result provider end to end, with snapshot timestamps.
- Exit: fixtures loaded for one league; `events`/`odds_snapshots` queryable;
  tests cover timezone boundaries on day rollover.

### Phase 2 — Probability core
- De-vig methods with cross-validation on overround sanity.
- Uncertainty percentile container that refuses to renormalise.
- Vectorised joint sampler and scenario grid; tests assert no Python-level
  per-cell iteration (shape and vectorisation checks).
- Exit: unit tests for de-vig, sampler marginals, percentile invariants.

### Phase 3 — Market adapters
- `markets/base.py` contract; `markets/match_winner.py` first.
- Settlement logic, tick sizes, de-vig wiring.
- Exit: match_winner round-trips; a synthetic event settles correctly in tests.

### Phase 4 — EV, Kelly, filters
- `ev/expected_value.py`, `ev/filters.py`; edge threshold, fractional Kelly and
  caps in `backtest/staking.py`.
- Exit: **pure market control places zero bets** in a test; edge and stake are
  correct on hand-computed fixtures.

### Phase 5 — Validation engine
- Walk-forward engine with the temporal assertion.
- Backtest with both `stop_on_breach` modes.
- Bankroll simulation, bootstrap, calibration, control harness.
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

- Phase 1 step 3c — diagnostics, a reliability bug fix, and Fix A adopted.
  The holdout rule (`AGENTS.md` rule 9, §5) is unchanged: every decision here
  uses **1819-2324 only** in both datasets.
  **(1) Reliability bug fix.** An earlier revision of
  `scripts/phase1_step3c_diagnostics.py` read the P(over) table for *every* line
  off the single walk-forward run at line 4.5, so lines 3.5/4.5/5.5 showed
  identical predictions (bins n=351/1258/511/78, pred 0.172/0.250/0.339/0.429) —
  which is impossible, since a higher line is strictly harder to clear. The
  earlier "mis-shaped tails" reading was an artifact of that bug and is
  **withdrawn**. Each line is now rebuilt from the captured per-side means
  (`probabilities_for_line` + `total_pmf`, step 3b's pattern);
  `assert_reconstruction` pins the rebuilt line-4.5 pmf to the walk-forward's own
  output and `assert_line_ordering` pins
  P(over 3.5) > P(over 4.5) > P(over 5.5) for every match — both raise
  `AssertionError`, not `assert`, so `-O` cannot disable them. New
  `tests/test_step3c_diagnostics.py` covers the rebuild, the two guards and
  `over_reliability`'s per-line signature. Corrected per-line numbers (e.g.
  `all`): line 3.5 low bin 0.2-0.3 predicts 0.28 vs 0.37 actual, high 0.6-0.7
  predicts 0.63 vs 0.62; line 4.5 low 0.1-0.2 predicts 0.17 vs 0.19, high
  0.4-0.5 predicts 0.43 vs 0.41; line 5.5 low 0.0-0.1 predicts 0.08 vs 0.08,
  high 0.2-0.3 predicts 0.23 vs 0.26. The surviving, reproducible failure is not
  a uniformly hot tail but **drift**: league mean total climbs 3.30→4.32 across
  the seasons and the newest season (2324) is under-predicted by +0.53 (`all`) /
  +0.38 (`no_covid`), largest at the season start (Sep `all` +0.70) and
  reversing late (Apr/May negative). Thin history is not the driver (rare
  referees ~11% of the top and middle, 2.7% of the `all` bottom).
  **(2) Fix A — adopted.** `add_card_features`/`add_referee_feature` gained
  `half_life_days` (config parameter `model.params.half_life_days`, default
  `None` = the unchanged cumulative incumbent): each earlier match's contribution
  to the team, referee and league-mean sums is weighted
  `0.5 ** (days_before / half_life_days)`. Vectorised with a per-group base date
  so the inverse-decay factors stay in float range; the prior is a per-group
  `shift(1)` rather than `cumsum - current`, so a match's own (probe-perturbed)
  counts cannot move the rows at or before it by a float epsilon the exact
  causality probe would flag. Evaluated under its frozen rule (half-life fixed
  at **365 days, no tuning**) on 1819-2324 in both datasets. **Result: PASS —
  ADOPT recency-weighted history.** Condition (1) last-two-season mean |gap|
  halves: `all` 0.116 vs 0.370, `no_covid` 0.067 vs 0.202. Condition (2) every
  line's walk-forward log loss improves: `all` 3.5 −0.0047, 4.5 −0.0060, 5.5
  −0.0039; `no_covid` −0.0023, −0.0044, −0.0018 — all deltas negative, well
  inside +0.001. The incumbent-vs-recency drift tables show the recency build
  tracking the newest season (2324 gap +0.53 → +0.13 `all`, +0.38 → +0.13
  `no_covid`) without loosening the early seasons. Adopted as the Phase 1 cards
  recipe (`half_life_days=365`); the library default stays cumulative (`None`)
  so steps 1-3b's reports, produced under the cumulative recipe, stay
  reproducible. See the new §5 paragraph.
  **(3) Fix B — shelved.** Walk-forward affine recalibration `mu' = a + b*mu`
  (refit strictly per fold) is proposed and frozen with its own pre-registered
  rule, but **not implemented**; it is recorded here as shelved, pending the
  corrected per-line tables. `scripts/phase1_step3c_diagnostics.py` now loads
  only 1819-2324 (a guard refuses a holdout season) and, in one run per dataset,
  builds the incumbent features, runs `poisson_team_ref`, asserts the rebuild
  and ordering, then runs the identical recipe on recency-weighted features and
  reports extreme groups, the decile gradient, group month composition, a
  read-only thin-history diagnostic, per-line P(over) reliability, per-season
  and per-month drift for both builds side by side, and Fix A's two conditions
  with the PASS/FAIL. 7 new tests (3 recency in `tests/test_cards_poisson.py`,
  4 in `tests/test_step3c_diagnostics.py`); all pass.
- Phase 1 step 3b — total-target negative binomial and the frozen adoption
  rule. `models/cards_poisson.py` adds `CardsTotalNegativeBinomial`, the one
  new model step 3b was allowed to build: one row per match, target = match
  total, design = `const + log home rate + log away rate + log referee rate`
  (the same k=6 features as `CardsTeamRefPoisson`, so the difference from it is
  the target and the likelihood, not the information). The dispersion alpha is
  estimated by the same MLE-on-Poisson-optimum path as step 3 (the stabilising
  fix), and over/under is read straight off the single negative-binomial pmf of
  the total (`side_pmf` over counts 0..94) — no convolution, which is exactly
  the per-side two-step this model does without. Registered as
  `cards_total_nbinom`. `scripts/phase1_step3b_total_nbinom.py` first measures
  the home/away card correlation — raw, and the out-of-sample per-side
  **residual** correlation of the incumbent — per season and overall, then runs
  incumbent (`poisson_team_ref`) vs challenger (`cards_total_nbinom`) on
  identical matches and folds (asserted at runtime) across lines 3.5/4.5/5.5
  and datasets `all`/`no_covid`: Brier, log loss, paired bootstrap (10,000
  resamples, full sequence), per-line P(over) reliability, the count-frequency
  table 0..10+, OOS residual dispersion, and the fitted alpha per season. The
  decision rule was fixed before the run, applied mechanically, and quoted in
  the report: adopt ONLY IF the challenger's log loss wins ≥2 of 3 lines in
  BOTH datasets AND its combined L1 tail distance over counts {0,1,2,8,9,10+}
  is smaller in BOTH datasets. **Result: KEEP `poisson_team_ref`.** The
  challenger's total-level alpha is genuinely positive (per-season 2e-07–0.04,
  matching total var/mean 1.0–1.2), so the NB is not degenerate — but it wins
  log loss on only 1 of 3 lines (`all`) and 0 of 3 (`no_covid`), the bootstrap
  separates zero in 0 of 6 Brier comparisons, and its tail L1 is worse in both
  datasets (0.0486 vs 0.0469, 0.0413 vs 0.0385): the extra total dispersion
  widens the upper tail (10+ predicted 0.0070 vs actual 0.0037) when the actual
  tail is *thinner* than the incumbent already predicts. The residual
  home/away correlation stays ~+0.19 (raw +0.196 → residual +0.186), i.e. the
  incumbent's shared features explain almost none of the home/away dependence,
  but the total NB's mean-variance form is the wrong tool for it. 13 new tests
  (492 total).
- Phase 1 step 3 — negative-binomial card model and the four-model
  comparison. New `models/count_distributions.py` does the one piece of
  arithmetic step 3 adds: the match total is `home + away`, so its pmf is the
  convolution of the two per-side pmfs, built as one cached 0/1-selector matrix
  product over the batch (`side_pmf`, `convolve_rows`, `total_pmf`,
  `SIDE_MAX_COUNT = 48`); vectorised, per invariant 6. `models/cards_poisson.py`
  now predicts every family through that convolution (for Poisson it is the
  same numbers the old direct cdf produced), records `predicted_sides`
  (home/away means, not just the total) and the fitted `alpha_`, splits `fit`
  into `_fit_family`, and adds `_NegativeBinomialFit` plus
  `CardsNegativeBinomial` / `CardsTeamRefNegativeBinomial` — same designs as
  their Poisson twins, one likelihood swapped in. **A negative-binomial MLE is
  started from the Poisson GLM optimum of the same design**: left to its own
  default start, statsmodels' Newton search diverged on one collinear fold and
  returned `const = 21`, per-side means in the hundreds and a pmf that lost
  ~2.5% of its mass; from the Poisson start the search stays on the alpha → 0
  boundary and reproduces Poisson exactly. Registered as `cards_nbinom` /
  `cards_team_ref_nbinom`. `scripts/phase1_step3_nbinom.py` runs four models
  (Poisson team, Poisson team+ref, NB team, NB team+ref) over 2 datasets with a
  single walk-forward each (the fit does not depend on the line, so all three
  lines are read off the same total pmf), asserts identical match sets, and
  reports scores, deltas, a paired NB-vs-Poisson bootstrap (10,000 resamples,
  fixed seed, full sequence per invariant 3), the predicted-vs-actual count
  distribution 0..10+, the per-season dispersion, a referee-rate-quintile
  diagnostic and a fit-health table. **Result: the negative binomial does not
  fix the tails.** The estimated per-side alpha is ~1e-7 in every season
  (per-side var/mean 0.91–0.99, i.e. *under*-dispersed), so NB coincides with
  Poisson (ΔBrier 0.0000; the bootstrap excludes zero in 0 of 12 comparisons),
  and the predicted-vs-actual tables are unchanged — the total's slight
  overdispersion (var/mean ~1.0–1.2) comes from the home/away correlation, not
  from per-side dispersion a per-side NB could capture. The referee term helps
  across the whole rate range, most for the strictest referees (Q5 ΔBrier
  −0.0062 vs Q1 −0.0020), unlike step 2's aggregate which hid that gradient.
  20 new tests (479 total).
- Phase 1 step 2 — referee variant and comparison. Gate first: load-time
  referee name corrections (`configs/referee_corrections.yaml`, applied by
  `data/ingest.py` — whitespace stripped from text columns, four one-off
  match-level typo fixes, one global alias; raw CSVs untouched, and a listed
  match whose referee does not match the expected value raises rather than
  silently rewriting). 46 → 41 distinct names, 0 remaining bad values;
  `validate()` now also prints referees with ≤ 2 matches. Every kept-distinct
  pair (Madley, Smith) and every fix is evidence-checked — see DATA_NOTES.
  `models/cards_poisson.py` gained `add_referee_feature`: the referee's total
  cards per match shrunk with the same `k` and the same strictly-before-date
  rule (league mean when no history), covered by the same causality probe,
  which now takes `requires`/`watch` so one probe serves both features.
  `CardsTeamRefPoisson` is the team design plus exactly one column,
  `log_ref_rate`, registered as `cards_team_ref`; the team-only model and its
  probe watch are untouched. `coefficient_table()` reports each design term's
  coefficient, GLM standard error and 95% CI. `scripts/phase1_step2_referee.py`
  runs 18 walk-forwards (2 datasets × 3 models × 3 lines) on identical match
  sets (asserted at runtime), no tuning (`k=6` for teams and referees), with a
  paired match-level bootstrap (10,000 resamples, fixed seed, full match
  sequence per invariant 3) and an auto-generated plain verdict. Result:
  team+ref beats team-only 6/6 on Brier and log loss (mean ΔBrier −0.0032),
  but the Brier 95% interval excludes zero in only 3/6 — improvement
  consistent, not statistically convincing overall; referee coefficient
  +0.55 [+0.39, +0.72] (`all`) and +0.45 [+0.28, +0.63] (`no_covid`),
  clearly positive; 8.9% / 10.5% of scored matches have a referee with fewer
  than 10 prior matches. 17 new tests (459 total).
- Phase 1 step 1 — team-only cards model. `models/cards_poisson.py` builds
  per-team rates shrunk toward the league mean *strictly before each date*
  (weight `k`, a config param), and enforces causality at runtime:
  `assert_features_are_causal` perturbs a probe date's card counts and refuses
  any feature function whose output moves on or before that date — tested
  against two deliberately broken functions (same-day self-read, final-matchday
  read). `CardsPoisson` fits a Poisson GLM on team-match rows (log own rate,
  log opponent rate, home flag) and reports over/under lines off the summed
  expectation; `LeagueAveragePoisson` is the same regression minus the rate
  columns — the control that isolates team information. Registered as
  `cards_poisson` / `cards_league_average`. Settlement lives in
  `markets/cards_totals.py`, unregistered because a config has no `line` field
  until an odds source lands (the counting rule is provisional — DATA_NOTES).
  `scripts/phase1_step1_cards.py` runs the walk-forward 12 times (2 datasets ×
  2 models × 3 lines) and writes `reports/phase1_step1_cards_*.md`: Brier, log
  loss, 10-bin reliability tables, widest-bin gap, and dispersion, with and
  without `covid_affected`. 55 new tests.
- Phase 0 gate fixes. §5: EV is now defined on the raw odds actually paid
  (`model_prob * raw_odds - 1`); de-vigged odds only produce the market
  probability (control prediction and reporting). §5 Bootstrap corrected to
  single-bet resampling — the code already did this; the old day-block wording
  was never implemented. §3 layout comment and §6 Phase 5 wording updated to
  match. `run.py` selection changed from fair odds to raw odds; a regression test
  pins it.
- Phase 0 scaffold built (26 files). Section 3 rewritten to match the tree that
  actually exists; stale names (`store.py`, `schema.sql`, `validation/`,
  `moneyline.py`) replaced with `db.py`, `eval/` + `backtest/`, `match_winner.py`.
  Future-only modules (`probability/`, `ev/`, `cli.py`, registry) are now listed as
  phase-scoped additions instead of implying they already exist.
- Phase 0 split into Built / Remaining: pydantic settings in `config.py` and the
  lint/format config are not done — Phase 0 was scaffold-only, no logic.
- Initial plan. Phase 0 scaffold only; no implementation beyond scaffolding until
  Phase 0 exit criteria pass.
