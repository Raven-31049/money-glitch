#!/usr/bin/env python
"""Phase 1 gate 1 — the calibration check for the adopted cards model.

Run::

    uv run python scripts/phase1_gate1_calibration.py

``docs/PLAN.md``'s first Phase 1 gate item is "Calibration check passes
(predicted over/under frequencies match actual)". This script is that check, and
nothing else. It loads the adopted model from ``configs/phase1_cards.yaml`` (the
single source every later step reads), runs one walk-forward per dataset, and
measures its **point** P(over) probabilities on seasons **1819-2324 only**, in
both datasets (``all`` and ``no_covid``), at lines **3.5, 4.5 and 5.5**.

The pass criterion is fixed in advance in ``docs/IMPLEMENTATION_NOTES.md`` §5
(recorded there before this script was ever run). A line passes only if all
three hold:

* (a) calibration-in-the-large: ``|mean predicted - actual over rate| <= 0.02``;
* (b) calibration slope (logistic regression of the outcome on
  ``logit(predicted)``) in ``[0.8, 1.2]``;
* (c) no reliability bin with ``n >= 100`` has ``|predicted - actual| > 0.05``.

The gate passes only if every line passes in both datasets. If a line fails, the
script diagnoses level (a), spread (b) or a local gap (c) and records at most two
**proposed** fixes with their decision rules frozen — it never implements one.
Any proposed fix must keep ``P(over 3.5) > P(over 4.5) > P(over 5.5)`` for every
match. One candidate is the shelved step-3c Fix B.

HOLDOUT (AGENTS.md rule 9). Seasons 2425 and 2526 are the final test set. This
script loads the seasons the config names — development seasons only — and
refuses a holdout season by name. Nothing here reports 2425-2526.

DIAGNOSTIC ONLY. No model, feature or threshold is changed.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
)

from bet_engine.backtest import walk_forward
from bet_engine.config import VariantConfig, load_variant
from bet_engine.data.ingest import load_league
from bet_engine.eval.calibration import calibration_slope, log_loss
from bet_engine.models import (
    SAMPLE_LINES,
    add_card_features,
    add_referee_feature,
    create_model,
    total_pmf,
)

#: The final test set. Never loaded here — see the holdout rule in AGENTS.md.
HOLDOUT_SEASONS: tuple[str, ...] = ("2425", "2526")

#: Where the adopted model lives. Every later step reads this file, not a copy.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "phase1_cards.yaml"

#: The lines the gate is read at; the config's own set must match (asserted).
LINES: tuple[float, ...] = SAMPLE_LINES

#: The line the single walk-forward runs at — the config's own scalar ``line``.
RUN_LINE = 4.5

#: The frozen pass criterion (see docs/IMPLEMENTATION_NOTES.md §5).
N_BINS = 10
CAL_IN_LARGE_TOL = 0.02
SLOPE_LOW = 0.8
SLOPE_HIGH = 1.2
BIN_MIN_N = 100
BIN_TOL = 0.05

#: The count-frequency table's top bucket: 0..9 shown, 10+ pooled.
MAX_COUNT = 9
COUNT_BUCKETS = MAX_COUNT + 1  # indices 0..9 plus the "10+" bucket

#: Up to this many fixes are proposed (never implemented) if a line fails.
MAX_FIXES = 2

#: Pre-registered fix rules. Verbatim in the report and IMPLEMENTATION_NOTES, so
#: the test cannot be chosen after seeing whether it passes.
FIX_B_NAME = "walk-forward affine recalibration of the expected total (shelved step-3c Fix B)"
FIX_B_RULE = (
    "Adopt a walk-forward affine recalibration of the expected total, mu' = a + "
    "b*mu with b > 0 (the intercept a and slope b refit strictly before each "
    "walk-forward fold day from nothing but that fold's history, b clipped "
    "positive so the map is strictly increasing and the line ordering survives) "
    "ONLY IF, on 1819-2324 in BOTH datasets, (1) at every line 3.5/4.5/5.5 the "
    "calibration slope lies in [0.8, 1.2]; (2) every line's calibration-in-the-"
    "large |mean predicted - actual over rate| is <= 0.02; (3) no reliability bin "
    "with n >= 100 has |predicted - actual| > 0.05; (4) walk-forward log loss at "
    "each line worsens by no more than +0.001 versus the raw model; and (5) "
    "P(over 3.5) > P(over 4.5) > P(over 5.5) still holds for every match. "
    "Otherwise keep the raw expected total. a and b are the least-squares fit on "
    "the strictly-earlier history — one rule, no tuning."
)
FIX_C_NAME = "walk-forward monotone (isotonic) recalibration of the expected total"
FIX_C_RULE = (
    "Adopt a walk-forward monotone (isotonic, non-decreasing) recalibration from "
    "the expected total to a corrected expected total, fit strictly before each "
    "walk-forward fold day on nothing but that fold's history, so the correction "
    "is order-preserving and P(over 3.5) > P(over 4.5) > P(over 5.5) survives on "
    "every match ONLY IF, on 1819-2324 in BOTH datasets, (1) at every line the "
    "worst n >= 100 reliability-bin |predicted - actual| falls to <= 0.05 while no "
    "other n >= 100 bin rises above 0.05; (2) every line's calibration-in-the-"
    "large stays <= 0.02; (3) log loss at each line worsens by no more than "
    "+0.001 versus the raw model; and (4) the line ordering still holds on every "
    "match. Otherwise keep the raw expected total. The knot count is the isotonic "
    "fit's own, not a tuned parameter."
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# config load (the adopted model is not re-stated here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdoptedModel:
    """The resolved contents of ``configs/phase1_cards.yaml``.

    ``params`` is what reaches the model constructor; ``half_life_days`` (the
    feature-build recency parameter) and ``lines`` (the quoted set the gate is
    read at) have been split out, and the scalar ``line`` the constructor
    requires is derived as the middle of ``lines``.
    """

    config: VariantConfig
    model_type: str
    params: dict[str, Any]
    half_life_days: float | None
    line: float
    lines: tuple[float, ...]


def load_adopted_model(path: Path = CONFIG_PATH) -> AdoptedModel:
    """Read the adopted variant and split its non-constructor params out."""
    config = load_variant(path)
    params = dict(config.model.params)
    half_life = params.pop("half_life_days", None)
    half_life_days = float(half_life) if half_life is not None else None
    raw_lines = params.pop("lines", None)
    if raw_lines is None:
        lines = tuple(LINES)
    else:
        lines = tuple(float(value) for value in raw_lines)
    if not lines:
        raise ValueError("config lines must not be empty")
    line = float(params.pop("line", lines[len(lines) // 2]))
    if line not in lines:
        raise ValueError(
            f"config line {line:g} is not one of the quoted lines {lines}"
        )
    params["line"] = line
    return AdoptedModel(
        config=config,
        model_type=config.model.type,
        params=params,
        half_life_days=half_life_days,
        line=line,
        lines=lines,
    )


def build_features(frame: pd.DataFrame, adopted: AdoptedModel, k: float) -> pd.DataFrame:
    """The model's input columns, built with the adopted recency half-life."""
    return add_referee_feature(
        add_card_features(frame, k, half_life_days=adopted.half_life_days),
        k,
        half_life_days=adopted.half_life_days,
    )


# ---------------------------------------------------------------------------
# walk-forward capture (point predictions only)
# ---------------------------------------------------------------------------


def run_model(
    features: pd.DataFrame, adopted: AdoptedModel
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """One walk-forward of the adopted model; capture prediction and sides.

    The model's own ``predict`` settles at the config's scalar line, but the
    design and the expected total do not depend on the line, so every other
    line's P(over) is rebuilt from the captured per-side means. ``sides`` maps
    match_id -> (mu_home, mu_away) for exactly the matches the walk-forward
    scored.
    """
    instances: list[Any] = []

    def factory() -> Any:
        model = create_model(adopted.model_type, adopted.params, ("over", "under"))
        instances.append(model)
        return model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        prediction = walk_forward(
            features,
            factory,
            min_train_days=adopted.config.backtest.min_train_days,
            refit_every_days=adopted.config.backtest.refit_every_days,
        )

    sides: dict[str, tuple[float, float]] = {}
    for model in instances:
        for match_id, pair in getattr(model, "predicted_sides", {}).items():
            sides[str(match_id)] = (float(pair[0]), float(pair[1]))
    return prediction, sides


def point_over_per_line(
    sides: dict[str, tuple[float, float]], ids: np.ndarray, lines: tuple[float, ...]
) -> np.ndarray:
    """Point P(over) at every line, from the captured per-side means.

    The side model is Poisson, so the match total is the convolution of the two
    per-side means. Rebuilding each line from the *same* sides the point output
    came from is what keeps a line's reliability readable at the line it claims
    (step 3c's rule).
    """
    record = np.array([sides[str(match_id)] for match_id in ids], dtype=float)
    counts, pmf = total_pmf(record[:, 0], record[:, 1], 0.0)
    return np.column_stack(
        [pmf[:, counts > line].sum(axis=1) for line in lines]
    )


def assert_reconstruction(
    prediction: pd.DataFrame,
    sides: dict[str, tuple[float, float]],
    ids: np.ndarray,
    line: float,
) -> None:
    """The rebuilt line-``line`` pmf must reproduce the walk-forward's output.

    Raises AssertionError (not ``assert``) so ``python -O`` cannot disable the
    check the whole report rests on.
    """
    direct = (
        prediction.pivot(index="match_id", columns="outcome", values="model_prob")
        .astype(float)[["over", "under"]]
        .loc[ids]
    )
    record = np.array([sides[str(match_id)] for match_id in ids], dtype=float)
    counts, pmf = total_pmf(record[:, 0], record[:, 1], 0.0)
    over = pmf[:, counts > line].sum(axis=1)
    rebuilt = pd.DataFrame(
        {"over": over, "under": 1.0 - over}, index=direct.index
    )
    if not np.allclose(
        direct.to_numpy(dtype=float), rebuilt.to_numpy(dtype=float), atol=1e-9
    ):
        raise AssertionError(
            f"rebuilt line-{line:g} probabilities do not match the walk-forward "
            "output; the captured per-side means are not the ones behind the "
            "prediction"
        )


def assert_line_ordering(
    sides: dict[str, tuple[float, float]],
    ids: np.ndarray,
    lines: tuple[float, ...],
) -> None:
    """P(over lower) > P(over higher) must hold for every match."""
    ordered = sorted(lines)
    over = {
        line: point_over_per_line(sides, ids, (line,))[:, 0] for line in ordered
    }
    for lower, higher in zip(ordered, ordered[1:]):
        wins = over[lower] > over[higher]
        if not wins.all():
            offenders = [str(match_id) for match_id in ids[~wins][:3]]
            raise AssertionError(
                f"P(over {lower:g}) must exceed P(over {higher:g}) for every "
                f"match; {int((~wins).sum())} violate it, e.g. {offenders}"
            )


def build_table(
    features: pd.DataFrame,
    sides: dict[str, tuple[float, float]],
    ids: np.ndarray,
) -> pd.DataFrame:
    """Per-scored-match frame: actual total and each line's point P(over)."""
    frame = features.set_index(features["match_id"].astype(str)).loc[ids]
    points = point_over_per_line(sides, ids, LINES)
    data: dict[str, Any] = {
        "match_id": [str(match_id) for match_id in ids],
        "season": frame["season"].to_numpy(),
        "actual_total": frame["total_cards"].to_numpy(dtype=float),
    }
    for position, line in enumerate(LINES):
        data[f"point_{_line_tag(line)}"] = points[:, position]
    table = pd.DataFrame(data)
    if table.isna().any().any():
        raise ValueError("gate table has NaNs")
    return table


def _line_tag(line: float) -> str:
    return f"{line:g}".replace(".", "_")


# ---------------------------------------------------------------------------
# per-line calibration
# ---------------------------------------------------------------------------


def over_reliability(p_over: np.ndarray, y_over: np.ndarray) -> list[dict]:
    """Ten equal-width probability bins, mean prediction vs observed frequency."""
    p = np.asarray(p_over, dtype=float)
    y = np.asarray(y_over, dtype=float)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, N_BINS - 1)
    rows = []
    for position in range(N_BINS):
        mask = bucket == position
        n = int(mask.sum())
        rows.append(
            {
                "low": float(edges[position]),
                "high": float(edges[position + 1]),
                "n": n,
                "predicted": float(p[mask].mean()) if n else float("nan"),
                "actual": float(y[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def evaluate_line(table: pd.DataFrame, line: float) -> dict:
    """Every frozen criterion for one line, plus PASS/FAIL per condition."""
    tag = _line_tag(line)
    p_over = table[f"point_{tag}"].to_numpy(dtype=float)
    y_over = (table["actual_total"].to_numpy(dtype=float) > line).astype(float)

    mean_predicted = float(p_over.mean())
    actual_rate = float(y_over.mean())
    gap = mean_predicted - actual_rate
    pass_large = abs(gap) <= CAL_IN_LARGE_TOL

    fit = calibration_slope(p_over, y_over)
    pass_slope = SLOPE_LOW <= fit.slope <= SLOPE_HIGH

    reliability = over_reliability(p_over, y_over)
    failing_bins = [
        row
        for row in reliability
        if row["n"] >= BIN_MIN_N and abs(row["actual"] - row["predicted"]) > BIN_TOL
    ]
    pass_bins = not failing_bins

    wide = pd.DataFrame({"over": p_over, "under": 1.0 - p_over})
    settled = np.where(y_over > 0.0, "over", "under")
    loss = float(log_loss(wide, settled))

    return {
        "line": line,
        "n": int(len(table)),
        "mean_predicted": mean_predicted,
        "actual_rate": actual_rate,
        "gap": gap,
        "pass_large": pass_large,
        "slope": fit.slope,
        "slope_low": fit.slope_low,
        "slope_high": fit.slope_high,
        "intercept": fit.intercept,
        "intercept_low": fit.intercept_low,
        "intercept_high": fit.intercept_high,
        "pass_slope": pass_slope,
        "reliability": reliability,
        "failing_bins": failing_bins,
        "pass_bins": pass_bins,
        "log_loss": loss,
        "passed": pass_large and pass_slope and pass_bins,
    }


def worst_bin(rows: list[dict]) -> dict | None:
    """The n >= BIN_MIN_N bin whose |actual - predicted| is largest, or None."""
    quoted = [row for row in rows if row["n"] >= BIN_MIN_N]
    if not quoted:
        return None
    return max(quoted, key=lambda row: abs(row["actual"] - row["predicted"]))


def count_frequency(
    sides: dict[str, tuple[float, float]], ids: np.ndarray, actual_total: np.ndarray
) -> list[dict]:
    """Predicted vs actual distribution of the match total, 0..9 and 10+.

    ``predicted`` is the model's own average total pmf (the convolution of the
    two per-side means) and ``actual`` the observed share of matches at each
    count. Both are shares of the scored matches, so they sum to 1.
    """
    record = np.array([sides[str(match_id)] for match_id in ids], dtype=float)
    counts, pmf = total_pmf(record[:, 0], record[:, 1], 0.0)
    bucket = np.minimum(counts, COUNT_BUCKETS - 1)
    n_matches = len(ids)
    predicted = np.zeros(COUNT_BUCKETS)
    for index in range(COUNT_BUCKETS):
        columns = bucket == index
        if columns.any():
            predicted[index] = pmf[:, columns].sum()
    predicted /= n_matches

    actual_counts = np.minimum(
        np.asarray(actual_total, dtype=int), COUNT_BUCKETS - 1
    )
    actual = np.bincount(actual_counts, minlength=COUNT_BUCKETS) / n_matches

    rows = []
    for index in range(COUNT_BUCKETS):
        label = str(index) if index < MAX_COUNT else f"{MAX_COUNT}+"
        rows.append(
            {
                "count": label,
                "predicted": float(predicted[index]),
                "actual": float(actual[index]),
                "gap": float(predicted[index] - actual[index]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# diagnosis and fix proposals
# ---------------------------------------------------------------------------


def diagnose(result: dict) -> list[str]:
    """Which frozen condition(s) a failing line broke, in plain words."""
    reasons = []
    if not result["pass_large"]:
        direction = "over-predicts" if result["gap"] > 0 else "under-predicts"
        reasons.append(
            f"level (a): {direction} by {abs(result['gap']):.3f} "
            f"(tol {CAL_IN_LARGE_TOL:g})"
        )
    if not result["pass_slope"]:
        if result["slope"] < SLOPE_LOW:
            kind = "too extreme (over-confident)"
        else:
            kind = "too flat (under-confident)"
        reasons.append(
            f"spread (b): slope {result['slope']:.2f} is {kind}, outside "
            f"[{SLOPE_LOW:g}, {SLOPE_HIGH:g}]"
        )
    if not result["pass_bins"]:
        worst = worst_bin(result["failing_bins"])
        if worst is not None:
            reasons.append(
                f"local gap (c): bin {worst['low']:.1f}-{worst['high']:.1f} "
                f"(n={worst['n']}) predicts {worst['predicted']:.2f} vs "
                f"{worst['actual']:.2f} actual"
            )
    return reasons


def propose_fixes(failures: list[dict]) -> list[dict]:
    """At most ``MAX_FIXES`` proposals, chosen by which conditions failed.

    The rules are frozen constants (above), written before any result was seen.
    The shelved Fix B is always the first candidate; the monotone Fix C is added
    only when a purely local gap (c) is among the failures, since an affine
    recalibration of the total cannot fix a bin-local shape error.
    """
    any_failure = bool(failures)
    if not any_failure:
        return []
    proposals = [{"name": FIX_B_NAME, "rule": FIX_B_RULE}]
    local_failure = any(
        not result["pass_bins"] and result["pass_large"] and result["pass_slope"]
        for result in failures
    )
    if local_failure and len(proposals) < MAX_FIXES:
        proposals.append({"name": FIX_C_NAME, "rule": FIX_C_RULE})
    return proposals[:MAX_FIXES]


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------


def findings_sentences(
    results: dict[str, dict[float, dict]],
    counts: dict[str, list[dict]],
    match_counts: dict[str, int],
    lines: tuple[float, ...],
) -> list[str]:
    out: list[str] = []
    for dataset, by_line in results.items():
        for line in lines:
            result = by_line[line]
            verdict = "PASS" if result["passed"] else "FAIL"
            out.append(
                f"**{dataset} — line {line:g}: {verdict}.** "
                f"n={result['n']}; mean predicted {result['mean_predicted']:.3f} vs "
                f"actual {result['actual_rate']:.3f} "
                f"(gap {result['gap']:+.3f}, condition (a) "
                f"{'holds' if result['pass_large'] else 'FAILS'}); calibration "
                f"slope {result['slope']:.2f} "
                f"[{result['slope_low']:.2f}, {result['slope_high']:.2f}] with "
                f"intercept {result['intercept']:+.2f} "
                f"[{result['intercept_low']:+.2f}, {result['intercept_high']:+.2f}] "
                f"(condition (b) {'holds' if result['pass_slope'] else 'FAILS'}); "
                + (
                    "no n>=100 bin exceeds 0.05"
                    if result["pass_bins"]
                    else f"{len(result['failing_bins'])} n>=100 bin(s) exceed 0.05"
                )
                + f" (condition (c)); point log loss {result['log_loss']:.4f}."
            )
    for dataset, rows in counts.items():
        n_matches = match_counts[dataset]
        # Share of matches at each count; the largest gap names the shape error.
        biggest = max(rows, key=lambda row: abs(row["gap"]))
        out.append(
            f"**{dataset} — total-count distribution.** Predicted vs actual share "
            f"of matches (n={n_matches}) at each count 0..9 and 10+. The largest "
            f"gap is at {biggest['count']} cards: predicted "
            f"{biggest['predicted']:.3f} vs actual {biggest['actual']:.3f} "
            f"(gap {biggest['gap']:+.3f})."
        )
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_report(payload: dict) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 gate 1 — calibration check (adopted cards model)")
    add("")
    add(f"- Generated: {payload['generated']}")
    add(
        f"- Model: `{payload['model_type']}` loaded from "
        f"`configs/phase1_cards.yaml` (the single source every later step reads). "
        f"k={payload['params'].get('k')}, line={payload['line']:g}, "
        f"lines={list(payload['lines'])}, half_life_days={payload['half_life_days']}."
    )
    add(
        "- Data: E0 seasons "
        + ", ".join(payload["seasons"])
        + f" ({payload['n_matches']} matches). Holdout seasons "
        + ", ".join(HOLDOUT_SEASONS)
        + " are NOT loaded or reported (AGENTS.md rule 9)."
    )
    add(
        f"- Walk-forward: min_train_days={payload['min_train_days']}, "
        f"refit_every_days={payload['refit_every_days']} (train strictly before "
        f"each fold day). One run at line {RUN_LINE:g}; every other line's P(over) "
        "is rebuilt from the captured per-side means."
    )
    add(
        "- DIAGNOSTIC ONLY: no model, feature or threshold is changed, and no fix "
        "is implemented."
    )
    add("")
    add(
        f"- **Gate verdict: {'PASSED' if payload['gate_passed'] else 'FAILED'}** "
        f"(every line, both datasets)."
    )
    add("")

    add("## Pass criterion (fixed in advance)")
    add("")
    add(
        "Recorded in `docs/IMPLEMENTATION_NOTES.md` §5 before this check was run. "
        "Per line, in both datasets, on 1819-2324 only, the adopted model's point "
        "P(over): (a) `|mean predicted - actual over rate| <= "
        f"{CAL_IN_LARGE_TOL:g}`; (b) calibration slope in "
        f"[{SLOPE_LOW:g}, {SLOPE_HIGH:g}]; (c) no reliability bin with "
        f"`n >= {BIN_MIN_N}` has `|predicted - actual| > {BIN_TOL:g}`."
    )
    add("")

    add("## Findings")
    add("")
    for sentence in payload["findings"]:
        add(sentence)
        add("")

    add("## Condition summary")
    add("")
    add("| dataset | line | n | (a) gap | (a) | slope [95% CI] | (b) | worst n>=100 bin | (c) | verdict |")
    add("|---|---:|---:|---:|:--:|---|:--:|---|:--:|:--:|")
    for dataset, by_line in payload["results"].items():
        for line in payload["lines"]:
            r = by_line[line]
            worst = worst_bin(r["reliability"])
            worst_text = (
                "`-`"
                if worst is None
                else f"{worst['low']:.1f}-{worst['high']:.1f} "
                f"({worst['predicted']:.2f}/{worst['actual']:.2f})"
            )
            add(
                f"| {dataset} | {line:g} | {r['n']} | {r['gap']:+.3f} | "
                f"{'ok' if r['pass_large'] else 'NO'} | "
                f"{r['slope']:.2f} [{r['slope_low']:.2f}, {r['slope_high']:.2f}] | "
                f"{'ok' if r['pass_slope'] else 'NO'} | {worst_text} | "
                f"{'ok' if r['pass_bins'] else 'NO'} | "
                f"{'PASS' if r['passed'] else 'FAIL'} |"
            )
    add("")

    add("## Reliability tables (point P(over))")
    add("")
    add(
        f"Ten equal-width probability bins. Bins with `n >= {BIN_MIN_N}` are the "
        "ones condition (c) judges; `gap` is predicted minus actual."
    )
    add("")
    for dataset, by_line in payload["results"].items():
        add(f"### {dataset}")
        add("")
        for line in payload["lines"]:
            add(f"line {line:g}")
            add("")
            add(f"| P(over) bin | n | predicted | actual | gap | n >= {BIN_MIN_N} | quote |")
            add("|---|---:|---:|---:|---:|:--:|:--:|")
            for row in by_line[line]["reliability"]:
                if not np.isfinite(row["predicted"]):
                    add(f"| {row['low']:.1f}-{row['high']:.1f} | 0 | nan | nan | nan | no | |")
                    continue
                quoted = row["n"] >= BIN_MIN_N
                flag = "QUOTED" if quoted else ""
                if quoted and abs(row["actual"] - row["predicted"]) > BIN_TOL:
                    flag = "**FAIL**"
                add(
                    f"| {row['low']:.1f}-{row['high']:.1f} | {row['n']} | "
                    f"{row['predicted']:.3f} | {row['actual']:.3f} | "
                    f"{row['actual'] - row['predicted']:+.3f} | "
                    f"{'yes' if quoted else 'no'} | {flag} |"
                )
            add("")

    add("## Predicted vs actual count frequency")
    add("")
    add(
        "The adopted model's own average total-count pmf (per-side means "
        "convolved) against the observed count share; `gap` is predicted minus "
        "actual. Counts are `0..9` with `10+` pooled."
    )
    add("")
    for dataset, rows in payload["counts"].items():
        add(f"### {dataset}")
        add("")
        add("| cards | predicted share | actual share | gap |")
        add("|---:|---:|---:|---:|")
        for row in rows:
            add(
                f"| {row['count']} | {row['predicted']:.4f} | "
                f"{row['actual']:.4f} | {row['gap']:+.4f} |"
            )
        add("")

    add("## Diagnosis and proposed fixes")
    add("")
    if payload["gate_passed"]:
        add(
            "**Gate passed — no fix is proposed.** The shelved step-3c Fix B "
            "remains frozen and unimplemented; nothing in `models/` recalibrates "
            "anything."
        )
        add("")
    else:
        add("**Diagnosis per failing line.**")
        add("")
        for dataset, by_line in payload["results"].items():
            for line in payload["lines"]:
                r = by_line[line]
                if r["passed"]:
                    continue
                add(f"- {dataset} line {line:g}: " + "; ".join(r["diagnosis"]) + ".")
        add("")
        add(
            "**Proposed fixes** (frozen rules, NOT implemented; the gate does not "
            "adopt anything). Each preserves "
            "`P(over 3.5) > P(over 4.5) > P(over 5.5)` for every match."
        )
        add("")
        for proposal in payload["fixes"]:
            add(f"### {proposal['name']}")
            add("")
            add(f"> {proposal['rule']}")
            add("")

    add("## Notes")
    add("")
    for note in payload["notes"]:
        add(f"- {note}")
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _guard_holdout(seasons: list[str]) -> None:
    """Refuse to load a holdout season — the holdout rule, made mechanical."""
    leaked = sorted(set(seasons) & set(HOLDOUT_SEASONS))
    if leaked:
        raise ValueError(
            f"refusing to load holdout season(s) {leaked}: seasons "
            f"{', '.join(HOLDOUT_SEASONS)} are the final test set (AGENTS.md "
            "rule 9) and must not be used for any model decision"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    started = time.time()
    adopted = load_adopted_model()
    if adopted.lines != LINES:
        raise ValueError(
            f"config lines {adopted.lines} differ from the gate's lines {LINES}"
        )
    seasons = list(adopted.config.seasons)
    _guard_holdout(seasons)
    k = float(adopted.params.get("k", 6.0))

    raw = load_league(adopted.config.league, seasons)
    if raw["covid_affected"].isna().any():
        raise ValueError("covid_affected has NaNs; cannot split the datasets")
    covid = raw["covid_affected"].astype(bool)
    if set(raw["season"]) & set(HOLDOUT_SEASONS):
        raise AssertionError("holdout season present in loaded frame")

    datasets = ["all", "no_covid"]
    frames = {"all": raw, "no_covid": raw.loc[~covid]}

    results: dict[str, dict[float, dict]] = {}
    counts: dict[str, list[dict]] = {}
    match_counts: dict[str, int] = {}
    failures: list[dict] = []

    for dataset_name in datasets:
        frame = frames[dataset_name]
        features = build_features(frame, adopted, k)
        prediction, sides = run_model(features, adopted)
        ids = np.asarray(prediction["match_id"].drop_duplicates())

        # The whole report rests on these two: the captured sides reproduce the
        # walk-forward's own line, and the lines are strictly ordered.
        assert_reconstruction(prediction, sides, ids, RUN_LINE)
        assert_line_ordering(sides, ids, LINES)

        table = build_table(features, sides, ids)
        by_line: dict[float, dict] = {}
        for line in LINES:
            result = evaluate_line(table, line)
            result["diagnosis"] = diagnose(result)
            by_line[line] = result
            if not result["passed"]:
                failures.append(result)
        results[dataset_name] = by_line
        match_counts[dataset_name] = len(table)
        counts[dataset_name] = count_frequency(
            sides, ids, table["actual_total"].to_numpy(dtype=float)
        )
        verdicts = " ".join(
            f"{line:g}:{'PASS' if by_line[line]['passed'] else 'FAIL'}"
            for line in LINES
        )
        print(
            f"[{dataset_name}] {len(table)} scored matches; {verdicts}",
            flush=True,
        )

    gate_passed = not failures
    fixes = propose_fixes(failures)
    findings = findings_sentences(results, counts, match_counts, LINES)

    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model_type": adopted.model_type,
        "params": adopted.params,
        "line": adopted.line,
        "lines": adopted.lines,
        "half_life_days": adopted.half_life_days,
        "seasons": seasons,
        "n_matches": len(raw),
        "min_train_days": adopted.config.backtest.min_train_days,
        "refit_every_days": adopted.config.backtest.refit_every_days,
        "datasets": datasets,
        "results": results,
        "counts": counts,
        "gate_passed": gate_passed,
        "failures": failures,
        "fixes": fixes,
        "findings": findings,
        "notes": [
            "The adopted model is loaded from `configs/phase1_cards.yaml`, not "
            "re-stated here: the config is the single source every later step "
            "reads. `model.params.half_life_days` (feature build) and "
            "`model.params.lines` (report set) are popped before create_model; "
            "the scalar `line` the constructor needs is the middle of `lines`.",
            "The pass criterion is frozen in `docs/IMPLEMENTATION_NOTES.md` §5 "
            "before this script was run.",
            "Only the model's POINT predictions are checked (the gate is about "
            "frequencies matching, not about interval coverage). Sampling "
            "uncertainty is step 4's concern.",
            "Per-line P(over) is rebuilt from each fold's captured per-side "
            "means, so each reliability table is read at the line it claims; the "
            "rebuilt line-4.5 pmf is asserted to reproduce the walk-forward's "
            "output and P(over 3.5) > P(over 4.5) > P(over 5.5) is asserted for "
            "every match.",
            "DIAGNOSTIC ONLY: no model, feature or threshold is changed, and any "
            "fix is proposed with its decision rule frozen — never implemented "
            "at the gate.",
            "HOLDOUT: seasons " + ", ".join(HOLDOUT_SEASONS) + " are the final "
            "test set and are not loaded (AGENTS.md rule 9).",
        ],
    }

    report = render_report(context)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = REPORTS_DIR / f"phase1_gate1_calibration_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print("\n".join(findings), flush=True)
    verdict = "PASSED" if gate_passed else "FAILED"
    print(f"\ngate 1: {verdict}; wrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
