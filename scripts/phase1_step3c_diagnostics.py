#!/usr/bin/env python
"""Phase 1 step 3c — diagnostic only: where the incumbent's extremes go wrong.

Run::

    uv run python scripts/phase1_step3c_diagnostics.py

Step 3b froze `poisson_team_ref` (Poisson team+referee) as the Phase 1 cards
model. Its calibration tables suggested the predictions were too *extreme*: in
the highest P(over) bins the model looked to over-predict and in the lowest bin
to under-predict — exactly where an EV filter would select bets. This script
does **not** change the model. It measures the failure on 1819-2324 alone and
reports the numbers a fix would have to move, then proposes at most two fixes
with decision rules fixed in advance.

The run found the premise only half true once the holdout is removed. The
high-P(over) over-prediction was largely a holdout-season effect: on 1819-2324
the line-4.5 high bin is calibrated and lines 3.5/5.5 are mis-shaped in
*opposite* directions, not uniformly hot. The dominant, reproducible failure is
drift — a league-wide card rate that rises across the seasons faster than the
model's strictly-before-date cumulative history follows.

Two rules frame the whole step:

* HOLDOUT (AGENTS.md rule 9, IMPLEMENTATION_NOTES §5). Seasons 2425 and 2526
  are the final test set. This script loads seasons 1819-2324 **only** and
  refuses a holdout season by name — nothing here reports 2425-2526.
* DIAGNOSTIC ONLY. No model, feature or threshold is changed. The fixes at the
  end are proposals; their decision rules are frozen here, before anything is
  implemented, and are testable on 1819-2324 alone.

Questions answered, per dataset (`all`, `no_covid`):

1. Extremes. Split the walk-forward-scored matches by their *predicted expected
   total* into the bottom 10%, middle 80% and top 10%. For each group: actual
   vs predicted mean total; the share whose referee had fewer than 10 prior
   matches; the share where either team had fewer than 10 prior matches; the
   month-of-season composition; and the mean prior-match counts feeding the team
   and referee rates. A decile table shows the same gradient at finer grain.
2. Drift. Per season and per season-month: actual mean total vs the model's mean
   prediction. Does the league-wide card rate move between seasons, and does the
   model's strictly-before-date history lag behind it?
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
)

from bet_engine.backtest import walk_forward
from bet_engine.data.ingest import load_league
from bet_engine.models import (
    CardsTeamRefPoisson,
    add_card_features,
    add_referee_feature,
)

#: The final test set. Never loaded here — see the holdout rule in AGENTS.md.
HOLDOUT_SEASONS: tuple[str, ...] = ("2425", "2526")

#: Everything a model decision is allowed to see at this step.
DIAGNOSTIC_SEASONS: tuple[str, ...] = ("1819", "1920", "2021", "2122", "2223", "2324")

#: The incumbent, frozen by step 3b. Nothing in this script fits anything else.
INCUMBENT = "poisson_team_ref"

#: Shrinkage weight and walk-forward cadence, unchanged from steps 1-3.
K = 6.0
MIN_TRAIN_DAYS = 60
REFIT_EVERY_DAYS = 1

#: A group is "thin" when the entity behind a rate has fewer than this many
#: strictly-earlier matches. Same threshold step 2 used for rare referees.
RARE_HISTORY = 10

#: The share at each end that counts as an "extreme" prediction.
TAIL_FRACTION = 0.10

#: The over/under lines the P(over) tail re-check is reported at.
LINES: tuple[float, ...] = (3.5, 4.5, 5.5)

#: A reliability bin is quoted only from this many matches — the step-3b rule.
MIN_RELIABILITY_BIN = 50

#: The rows the report quotes verbatim, frozen before any fix is built. Both
#: rules are evaluated on 1819-2324 only, in both datasets; see render_fixes().
FIX_A_NAME = "recency-weighted rate history (stop lagging a rising card rate)"
FIX_A_RULE = (
    "Adopt an exponentially time-weighted rate history — one half-life, applied "
    "identically to the team, referee and league-mean sums, no other change — ONLY "
    "IF, on 1819-2324 in BOTH datasets, (1) the mean absolute per-season gap "
    "|actual mean - predicted mean| over the last two seasons (2223 and 2324) "
    "shrinks by at least half versus the cumulative-history incumbent, AND (2) "
    "walk-forward log loss at lines 3.5, 4.5 and 5.5 does not worsen by more than "
    "+0.001 in either dataset. Otherwise keep the cumulative history."
)

FIX_B_NAME = "walk-forward affine recalibration of the expected total"
FIX_B_RULE = (
    "Adopt an affine recalibration mu' = a + b*mu (a, b refit strictly before "
    "each walk-forward fold day from nothing but that fold's history) ONLY IF, on "
    "1819-2324 in BOTH datasets, (1) the mean absolute gap over all scored matches "
    "and the top-decile |predicted mean - actual mean| both shrink versus the raw "
    "model, AND (2) Brier and log loss at 3.5, 4.5 and 5.5 each improve or worsen "
    "by no more than +0.0005, AND (3) a paired per-match bootstrap of log loss at "
    "line 4.5 does not separate zero against the recalibrated model. Otherwise "
    "keep the raw expected total."
)

#: Season-month ordering: August is month 1, May is month 10.
_MONTH_TO_SEASON_INDEX: dict[int, int] = {
    8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
}
_MONTH_NAMES: dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# feature-level diagnostics (read-only; the model is not touched)
# ---------------------------------------------------------------------------


def add_team_prior_counts(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus each side's count of strictly-earlier usable team-matches.

    This mirrors the grouping ``_add_card_features`` uses to build the team
    rates (aggregate per ``(team, date)``, cumulative sum minus the current
    row's own contribution), so the counts reported here are exactly the
    denominators behind the model's own ``home_rate``/``away_rate``. It is a
    diagnostic column, not a model input: nothing in models/ reads it.
    """
    out = df.copy()
    hy = pd.to_numeric(out["hy"], errors="coerce")
    hr = pd.to_numeric(out["hr"], errors="coerce")
    ay = pd.to_numeric(out["ay"], errors="coerce")
    ar = pd.to_numeric(out["ar"], errors="coerce")
    home_side = pd.DataFrame(
        {
            "match_id": out["match_id"].to_numpy(),
            "date": out["date"].to_numpy(),
            "team": out["home"].to_numpy(),
            "side": "home",
            "own_cards": (hy + hr).to_numpy(),
        }
    )
    away_side = pd.DataFrame(
        {
            "match_id": out["match_id"].to_numpy(),
            "date": out["date"].to_numpy(),
            "team": out["away"].to_numpy(),
            "side": "away",
            "own_cards": (ay + ar).to_numpy(),
        }
    )
    long = pd.concat([home_side, away_side], ignore_index=True)
    usable = long["date"].notna() & long["own_cards"].notna()
    history = (
        long.loc[usable]
        .groupby(["team", "date"], sort=True)["own_cards"]
        .size()
        .rename("appearances")
        .reset_index()
        .sort_values(["team", "date"], kind="mergesort")
    )
    history["prior"] = (
        history.groupby("team")["appearances"].cumsum() - history["appearances"]
    )
    merged = long.merge(history[["team", "date", "prior"]], on=["team", "date"], how="left")
    home_prior = merged.loc[merged["side"] == "home"].set_index("match_id")["prior"]
    away_prior = merged.loc[merged["side"] == "away"].set_index("match_id")["prior"]
    out["home_prior"] = out["match_id"].map(home_prior).astype("float64")
    out["away_prior"] = out["match_id"].map(away_prior).astype("float64")
    return out


# ---------------------------------------------------------------------------
# walk-forward capture (step 3b's incumbent, unchanged)
# ---------------------------------------------------------------------------


def run_incumbent(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """One walk-forward of the incumbent; return its predictions and expected totals.

    The design does not depend on the line, so a single line is enough to
    capture every match's expected total and its over/under probabilities (step
    2/3 established that the folds, design and totals are line-independent and
    assert it). Nothing is tuned and no other model is fitted.
    """
    instances: list[object] = []

    def factory() -> object:
        model = CardsTeamRefPoisson(outcomes=("over", "under"), line=4.5, k=K)
        instances.append(model)
        return model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        prediction = walk_forward(
            features,
            factory,
            min_train_days=MIN_TRAIN_DAYS,
            refit_every_days=REFIT_EVERY_DAYS,
        )

    totals: dict[str, float] = {}
    for model in instances:
        totals.update(
            {str(k): float(v) for k, v in model.predicted_totals.items()}  # type: ignore[attr-defined]
        )
    return prediction, totals


def build_analysis(features: pd.DataFrame, totals: dict[str, float]) -> pd.DataFrame:
    """Per-scored-match diagnostic frame: expected vs actual total and history.

    Only matches the walk-forward actually scored are kept (the incumbent emits
    a total for exactly those), so the extreme split is over out-of-sample
    predictions — never over rows the model trained on.
    """
    ids = pd.Index([str(match_id) for match_id in totals], name="match_id")
    frame = features.set_index(features["match_id"].astype(str)).loc[ids]
    analysis = pd.DataFrame(
        {
            "match_id": ids.to_numpy(),
            "date": frame["date"].to_numpy(),
            "season": frame["season"].to_numpy(),
            "mu": [totals[str(match_id)] for match_id in ids],
            "actual": frame["total_cards"].to_numpy(dtype=float),
            "ref_prior": frame["ref_past_count"].to_numpy(dtype=float),
            "home_prior": frame["home_prior"].to_numpy(dtype=float),
            "away_prior": frame["away_prior"].to_numpy(dtype=float),
        }
    )
    if analysis[["mu", "actual", "ref_prior", "home_prior", "away_prior"]].isna().any().any():
        raise ValueError("analysis frame has NaNs; the extreme split would be biased")
    analysis["team_prior_min"] = analysis[["home_prior", "away_prior"]].min(axis=1)
    analysis["team_prior_mean"] = analysis[["home_prior", "away_prior"]].mean(axis=1)
    analysis["ref_rare"] = analysis["ref_prior"] < RARE_HISTORY
    analysis["team_rare"] = analysis["team_prior_min"] < RARE_HISTORY
    analysis["month"] = analysis["date"].dt.month
    analysis["season_month"] = analysis["month"].map(_MONTH_TO_SEASON_INDEX)
    analysis["month_name"] = analysis["month"].map(_MONTH_NAMES)
    lo = analysis["mu"].quantile(TAIL_FRACTION)
    hi = analysis["mu"].quantile(1.0 - TAIL_FRACTION)
    analysis["group"] = np.select(
        [analysis["mu"] <= lo, analysis["mu"] >= hi],
        ["bottom", "top"],
        default="middle",
    )
    return analysis


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def over_reliability(
    prediction: pd.DataFrame, actual_total: pd.Series, line: float
) -> list[dict]:
    """Bin P(over) and compare mean prediction with the actual over-frequency.

    This re-checks, on 1819-2324 only, the step-3b pattern the diagnostic was
    prompted by: over-prediction in the highest P(over) bins and
    under-prediction in the lowest. ``actual_total`` is the match total keyed by
    match_id; "over" is ``total > line``, the market's own rule.
    """
    wide = (
        prediction.pivot(index="match_id", columns="outcome", values="model_prob")
        .astype(float)[["over", "under"]]
    )
    total = actual_total.reindex(wide.index).to_numpy(dtype=float)
    over_happened = (total > line).astype(float)
    p_over = wide["over"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    bucket = np.clip(np.searchsorted(edges, p_over, side="right") - 1, 0, 9)
    rows = []
    for position in range(10):
        mask = bucket == position
        n = int(mask.sum())
        rows.append(
            {
                "low": float(edges[position]),
                "high": float(edges[position + 1]),
                "n": n,
                "predicted": float(p_over[mask].mean()) if n else float("nan"),
                "actual": float(over_happened[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def _group_row(label: str, sub: pd.DataFrame) -> dict:
    return {
        "group": label,
        "n": int(len(sub)),
        "predicted_mean": float(sub["mu"].mean()),
        "actual_mean": float(sub["actual"].mean()),
        "gap": float(sub["actual"].mean() - sub["mu"].mean()),
        "ref_rare_share": float(sub["ref_rare"].mean()),
        "team_rare_share": float(sub["team_rare"].mean()),
        "ref_prior_mean": float(sub["ref_prior"].mean()),
        "team_prior_mean": float(sub["team_prior_mean"].mean()),
        "team_prior_min_mean": float(sub["team_prior_min"].mean()),
    }


def group_table(analysis: pd.DataFrame) -> list[dict]:
    """Bottom 10% / middle 80% / top 10% of the predicted expected total."""
    order = ["bottom", "middle", "top"]
    labels = {
        "bottom": f"bottom {int(TAIL_FRACTION * 100)}%",
        "middle": f"middle {int((1 - 2 * TAIL_FRACTION) * 100)}%",
        "top": f"top {int(TAIL_FRACTION * 100)}%",
    }
    return [
        _group_row(labels[name], analysis[analysis["group"] == name]) for name in order
    ]


def decile_table(analysis: pd.DataFrame) -> list[dict]:
    """Ten equal-count bins of the predicted expected total, low to high."""
    decile = pd.qcut(analysis["mu"], 10, labels=False, duplicates="drop")
    rows = []
    for position in sorted(pd.unique(decile.dropna())):
        sub = analysis[decile == position]
        rows.append(
            {
                "decile": int(position) + 1,
                "n": int(len(sub)),
                "predicted_mean": float(sub["mu"].mean()),
                "actual_mean": float(sub["actual"].mean()),
                "gap": float(sub["actual"].mean() - sub["mu"].mean()),
            }
        )
    return rows


def month_composition(analysis: pd.DataFrame) -> list[dict]:
    """Share of each extreme group falling in each season-month."""
    rows = []
    for index in sorted(_MONTH_TO_SEASON_INDEX.values()):
        sub = analysis[analysis["season_month"] == index]
        if sub.empty:
            continue
        name = _MONTH_NAMES[int(sub["month"].mode().iloc[0])]
        total = len(sub)
        rows.append(
            {
                "season_month": index,
                "month": name,
                "n": int(total),
                "bottom_share": float((sub["group"] == "bottom").mean()),
                "middle_share": float((sub["group"] == "middle").mean()),
                "top_share": float((sub["group"] == "top").mean()),
            }
        )
    return rows


def season_drift(analysis: pd.DataFrame) -> list[dict]:
    """Per season: actual vs predicted mean total, plus the previous season's actual.

    "Previous" is the previous season actually present in this dataset, so the
    no_covid frame (which drops the 2020-21 season) compares 2122 with 1920
    rather than with a season it never loaded.
    """
    actual_by_season = {
        season: float(sub["actual"].mean())
        for season, sub in analysis.groupby("season")
    }
    present = [season for season in DIAGNOSTIC_SEASONS if season in actual_by_season]
    rows = []
    for position, season in enumerate(present):
        sub = analysis[analysis["season"] == season]
        prev = present[position - 1] if position > 0 else None
        rows.append(
            {
                "season": season,
                "n": int(len(sub)),
                "actual_mean": float(sub["actual"].mean()),
                "predicted_mean": float(sub["mu"].mean()),
                "gap": float(sub["actual"].mean() - sub["mu"].mean()),
                "prev_actual": actual_by_season[prev] if prev else float("nan"),
                "prev_season": prev,
            }
        )
    return rows


def month_drift(analysis: pd.DataFrame) -> list[dict]:
    """Per season-month, pooled across seasons: actual vs predicted mean total."""
    rows = []
    for index in sorted(_MONTH_TO_SEASON_INDEX.values()):
        sub = analysis[analysis["season_month"] == index]
        if sub.empty:
            continue
        name = _MONTH_NAMES[int(sub["month"].mode().iloc[0])]
        rows.append(
            {
                "season_month": index,
                "month": name,
                "n": int(len(sub)),
                "actual_mean": float(sub["actual"].mean()),
                "predicted_mean": float(sub["mu"].mean()),
                "gap": float(sub["actual"].mean() - sub["mu"].mean()),
            }
        )
    return rows


def _finite(value: object) -> bool:
    """True only for a real, non-NaN number; None and NaN are both ``False``."""
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _correlation(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lag_summary(season_rows: list[dict]) -> dict:
    """Does the season prediction track this season's actual or last season's?

    Uses only seasons that have a previous season in the loaded set. Two
    correlations on at most five points — reported as a hint, not a verdict.
    """
    paired = [row for row in season_rows if _finite(row["prev_actual"])]
    predicted = [row["predicted_mean"] for row in paired]
    actual_now = [row["actual_mean"] for row in paired]
    actual_prev = [row["prev_actual"] for row in paired]
    return {
        "n_seasons": len(paired),
        "corr_same": _correlation(predicted, actual_now),
        "corr_prev": _correlation(predicted, actual_prev),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _enrichment(top_share: float, middle_share: float) -> str:
    """Describe the top decile's thin-history share relative to the middle 80%."""
    if top_share > middle_share + 0.02:
        return "more"
    if top_share < middle_share - 0.02:
        return "less"
    return "about as"


def _tail_direction(gap: float) -> str:
    """Name a mean gap: negative over-predicts, positive under-predicts."""
    if gap < -0.02:
        return f"over-predicts by {abs(gap):.2f}"
    if gap > 0.02:
        return f"under-predicts by {gap:.2f}"
    return f"sits about right (gap {gap:+.2f})"


def _reliability_sentence(
    dataset: str, reliability: dict[float, list[dict]], lines: tuple[float, ...]
) -> str:
    """Whether the step-3b P(over) tail pattern survives on 1819-2324."""
    bits = []
    for line in lines:
        populated = [
            row
            for row in reliability[line]
            if row["n"] >= MIN_RELIABILITY_BIN and np.isfinite(row["predicted"])
        ]
        if not populated:
            bits.append(f"line {line:g}: no bin with n >= {MIN_RELIABILITY_BIN}")
            continue
        low, high = populated[0], populated[-1]
        bits.append(
            f"line {line:g}: lowest populated bin "
            f"{low['low']:.1f}-{low['high']:.1f} predicts {low['predicted']:.2f} "
            f"vs {low['actual']:.2f} actual ({_tail_direction(low['actual'] - low['predicted'])}); "
            f"highest {high['low']:.1f}-{high['high']:.1f} predicts "
            f"{high['predicted']:.2f} vs {high['actual']:.2f} actual "
            f"({_tail_direction(high['actual'] - high['predicted'])})"
        )
    return f"**{dataset} — P(over) tails, 1819-2324 only.** " + "; ".join(bits) + "."


def findings_sentences(
    extreme_tables: dict[str, list[dict]],
    decile_tables: dict[str, list[dict]],
    season_rows: dict[str, list[dict]],
    lag: dict[str, dict],
    reliability: dict[str, dict[float, list[dict]]],
    lines: tuple[float, ...],
) -> list[str]:
    """Plain-language reading of the computed tables, per dataset.

    Deliberately describes the *sign* the numbers show rather than the pattern
    the diagnostic was prompted by: the two datasets disagree, and the report's
    job is to say so, not to confirm a prior.
    """
    out: list[str] = []
    for dataset, groups in extreme_tables.items():
        row = {entry["group"]: entry for entry in groups}
        bottom, middle, top = row["bottom 10%"], row["middle 80%"], row["top 10%"]
        deciles = decile_tables[dataset]
        over = sum(1 for entry in deciles if entry["gap"] < -0.05)
        under = sum(1 for entry in deciles if entry["gap"] > 0.05)
        near = len(deciles) - over - under
        out.append(
            f"**{dataset} — extremes.** Bottom 10%: predicted "
            f"{bottom['predicted_mean']:.2f}, actual {bottom['actual_mean']:.2f} "
            f"({_tail_direction(bottom['gap'])}). Middle 80%: predicted "
            f"{middle['predicted_mean']:.2f}, actual {middle['actual_mean']:.2f} "
            f"({_tail_direction(middle['gap'])}). Top 10%: predicted "
            f"{top['predicted_mean']:.2f}, actual {top['actual_mean']:.2f} "
            f"({_tail_direction(top['gap'])}). Of the ten predicted-total deciles, "
            f"{over} over-predict (gap < -0.05), {under} under-predict (gap > +0.05) "
            f"and {near} are within 0.05."
        )
        out.append(
            f"**{dataset} — thin history.** Rare referees (< {RARE_HISTORY} prior) "
            f"are {100 * top['ref_rare_share']:.1f}% of the top decile, "
            f"{100 * middle['ref_rare_share']:.1f}% of the middle 80% and "
            f"{100 * bottom['ref_rare_share']:.1f}% of the bottom; a team with "
            f"< {RARE_HISTORY} prior is present in {100 * top['team_rare_share']:.1f}% "
            f"of the top, {100 * middle['team_rare_share']:.1f}% of the middle and "
            f"{100 * bottom['team_rare_share']:.1f}% of the bottom. Mean prior "
            f"matches — referee: {top['ref_prior_mean']:.0f} top / "
            f"{middle['ref_prior_mean']:.0f} middle / {bottom['ref_prior_mean']:.0f} "
            f"bottom; thinner team side: {top['team_prior_min_mean']:.0f} / "
            f"{middle['team_prior_min_mean']:.0f} / {bottom['team_prior_min_mean']:.0f}. "
            f"Relative to the middle 80%, the top decile is "
            f"{_enrichment(top['ref_rare_share'], middle['ref_rare_share'])} likely to "
            f"involve a rare referee and "
            f"{_enrichment(top['team_rare_share'], middle['team_rare_share'])} likely to "
            f"involve a low-history team; the bottom decile's rare-referee share is "
            f"{100 * bottom['ref_rare_share']:.1f}%."
        )
        out.append(_reliability_sentence(dataset, reliability[dataset], lines))
    out.append(_season_drift_sentence(season_rows, lag))
    return out


def _season_drift_sentence(
    season_rows: dict[str, list[dict]], lag: dict[str, dict]
) -> str:
    bits = []
    for dataset, rows in season_rows.items():
        means = [row["actual_mean"] for row in rows]
        if not means:
            continue
        gaps = [row["gap"] for row in rows]
        latest = rows[-1]
        bits.append(
            f"{dataset}: league actual mean total runs {min(means):.2f}-"
            f"{max(means):.2f} across the seasons present, and the model's "
            f"prediction {min(r['predicted_mean'] for r in rows):.2f}-"
            f"{max(r['predicted_mean'] for r in rows):.2f}; the newest season's "
            f"gap is {latest['gap']:+.2f} ({latest['season']}), and the largest "
            f"|gap| is {max(abs(g) for g in gaps):.2f}"
        )
    return (
        "**Drift.** "
        + "; ".join(bits)
        + ". A positive gap means the league's cards ran above the model's "
        "prediction — the expanding, cumulative history lags a rising rate."
    )


def render_report(payload: dict) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 Step 3c — diagnostic: extreme predictions and drift")
    add("")
    add(f"- Generated: {payload['generated']}")
    add(f"- Incumbent (frozen by step 3b): `{INCUMBENT}`, k={K:g}")
    add(
        "- Data: E0 seasons "
        + ", ".join(DIAGNOSTIC_SEASONS)
        + f" ({payload['n_matches']} matches). Holdout seasons "
        + ", ".join(HOLDOUT_SEASONS)
        + " are NOT loaded or reported (AGENTS.md rule 9)."
    )
    add(
        f"- Walk-forward: min_train_days={MIN_TRAIN_DAYS}, "
        f"refit_every_days={REFIT_EVERY_DAYS} (train strictly before each fold day)"
    )
    add("- Diagnostic only: no model, feature or threshold is changed.")
    add("")

    add("## Findings")
    add("")
    for sentence in payload["findings"]:
        add(sentence)
        add("")

    for dataset in payload["datasets"]:
        groups = payload["extreme_tables"][dataset]
        add(f"## Extreme predictions — {dataset}")
        add("")
        add(
            "Matches are split by the model's own predicted expected total into "
            f"the bottom {int(TAIL_FRACTION * 100)}%, the middle "
            f"{int((1 - 2 * TAIL_FRACTION) * 100)}% and the top "
            f"{int(TAIL_FRACTION * 100)}%. `gap` is actual mean minus predicted "
            "mean; positive means the model under-predicted that group."
        )
        add("")
        add(
            "| group | n | predicted mean | actual mean | gap | ref < 10 prior | "
            "team < 10 prior | mean ref prior | mean team prior (thinner side) |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in groups:
            add(
                f"| {row['group']} | {row['n']} | {row['predicted_mean']:.3f} | "
                f"{row['actual_mean']:.3f} | {row['gap']:+.3f} | "
                f"{100 * row['ref_rare_share']:.1f}% | "
                f"{100 * row['team_rare_share']:.1f}% | "
                f"{row['ref_prior_mean']:.1f} | {row['team_prior_min_mean']:.1f} |"
            )
        add("")

        add("### Decile gradient")
        add("")
        add("| decile (low to high predicted total) | n | predicted mean | actual mean | gap |")
        add("|---|---:|---:|---:|---:|")
        for row in payload["decile_tables"][dataset]:
            add(
                f"| {row['decile']} | {row['n']} | {row['predicted_mean']:.3f} | "
                f"{row['actual_mean']:.3f} | {row['gap']:+.3f} |"
            )
        add("")

        add("### Month-of-season composition of each group")
        add("")
        add(
            "Share of each season-month's scored matches that land in each group "
            "(columns sum to 100% down a month)."
        )
        add("")
        add("| season month | month | n | bottom 10% | middle 80% | top 10% |")
        add("|---|---|---:|---:|---:|---:|")
        for row in payload["month_composition"][dataset]:
            add(
                f"| {row['season_month']} | {row['month']} | {row['n']} | "
                f"{100 * row['bottom_share']:.1f}% | "
                f"{100 * row['middle_share']:.1f}% | "
                f"{100 * row['top_share']:.1f}% |"
            )
        add("")

    for dataset in payload["datasets"]:
        add(f"## Drift — {dataset}")
        add("")
        add("### Per season")
        add("")
        add(
            "| season | n | actual mean | predicted mean | gap | previous season actual |"
        )
        add("|---|---:|---:|---:|---:|---:|")
        for row in payload["season_rows"][dataset]:
            prev = (
                f"{row['prev_actual']:.3f} ({row['prev_season']})"
                if _finite(row["prev_actual"])
                else "`-`"
            )
            add(
                f"| {row['season']} | {row['n']} | {row['actual_mean']:.3f} | "
                f"{row['predicted_mean']:.3f} | {row['gap']:+.3f} | {prev} |"
            )
        info = payload["lag"][dataset]
        add("")
        add(
            f"Model season prediction vs same-season actual: r = "
            f"{info['corr_same']:+.2f}; vs previous-season actual: r = "
            f"{info['corr_prev']:+.2f} (n = {info['n_seasons']} seasons)."
        )
        add("")
        add("### Per season-month (pooled across seasons)")
        add("")
        add("| season month | month | n | actual mean | predicted mean | gap |")
        add("|---|---|---:|---:|---:|---:|")
        for row in payload["month_rows"][dataset]:
            add(
                f"| {row['season_month']} | {row['month']} | {row['n']} | "
                f"{row['actual_mean']:.3f} | {row['predicted_mean']:.3f} | "
                f"{row['gap']:+.3f} |"
            )
        add("")

    add("## P(over) reliability on 1819-2324 only")
    add("")
    add(
        "The step-3b calibration pattern, re-checked without the holdout: each "
        "match binned by its predicted P(over), `pred` the mean prediction in "
        "the bin, `act` the share that went over. Bins with fewer than "
        f"{MIN_RELIABILITY_BIN} matches are shown but not quoted in the findings."
    )
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        for line in LINES:
            add(f"line {line:g}")
            add("")
            add("| P(over) bin | n | pred | act |")
            add("|---|---:|---:|---:|")
            for row in payload["reliability"][dataset][line]:
                pred = "nan" if not np.isfinite(row["predicted"]) else f"{row['predicted']:.3f}"
                act = "nan" if not np.isfinite(row["actual"]) else f"{row['actual']:.3f}"
                add(f"| {row['low']:.1f}-{row['high']:.1f} | {row['n']} | {pred} | {act} |")
            add("")

    add("## Proposed fixes (NOT implemented — decision rules frozen below)")
    add("")
    add(
        "These are proposals only. No fix is in this repository; nothing in "
        "`models/` changed. Each rule is evaluated on 1819-2324 in **both** "
        "datasets, with the incumbent as control, and is written here before any "
        "implementation so the test cannot be chosen after the fact."
    )
    add("")
    add(f"### Fix A — {FIX_A_NAME}")
    add("")
    add(f"> {FIX_A_RULE}")
    add("")
    add(f"### Fix B — {FIX_B_NAME}")
    add("")
    add(f"> {FIX_B_RULE}")
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


def _guard_holdout(seasons: tuple[str, ...]) -> None:
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
    seasons = DIAGNOSTIC_SEASONS
    _guard_holdout(seasons)

    raw = load_league("E0", list(seasons))
    if raw["covid_affected"].isna().any():
        raise ValueError("covid_affected has NaNs; cannot split the datasets")
    covid = raw["covid_affected"].astype(bool)
    if set(raw["season"]) & set(HOLDOUT_SEASONS):
        raise AssertionError("holdout season present in loaded frame")

    datasets = ["all", "no_covid"]
    extreme_tables: dict[str, list[dict]] = {}
    decile_tables: dict[str, list[dict]] = {}
    month_composition_rows: dict[str, list[dict]] = {}
    season_rows: dict[str, list[dict]] = {}
    month_rows: dict[str, list[dict]] = {}
    lag: dict[str, dict] = {}
    reliability: dict[str, dict[float, list[dict]]] = {}

    frames = {"all": raw, "no_covid": raw.loc[~covid]}
    for dataset_name in datasets:
        frame = frames[dataset_name]
        features = add_referee_feature(add_card_features(frame, K), K)
        features = add_team_prior_counts(features)
        prediction, totals = run_incumbent(features)
        analysis = build_analysis(features, totals)

        actual_total = features.set_index("match_id")["total_cards"]
        reliability[dataset_name] = {
            line: over_reliability(prediction, actual_total, line) for line in LINES
        }

        extreme_tables[dataset_name] = group_table(analysis)
        decile_tables[dataset_name] = decile_table(analysis)
        month_composition_rows[dataset_name] = month_composition(analysis)
        season_rows[dataset_name] = season_drift(analysis)
        month_rows[dataset_name] = month_drift(analysis)
        lag[dataset_name] = lag_summary(season_rows[dataset_name])
        print(
            f"[{dataset_name}] {len(analysis)} scored matches; "
            f"top gap {extreme_tables[dataset_name][2]['gap']:+.3f}, "
            f"bottom gap {extreme_tables[dataset_name][0]['gap']:+.3f}",
            flush=True,
        )

    findings = findings_sentences(
        extreme_tables, decile_tables, season_rows, lag, reliability, LINES
    )

    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_matches": len(raw),
        "datasets": datasets,
        "extreme_tables": extreme_tables,
        "decile_tables": decile_tables,
        "month_composition": month_composition_rows,
        "season_rows": season_rows,
        "month_rows": month_rows,
        "lag": lag,
        "reliability": reliability,
        "findings": findings,
        "notes": [
            "Diagnostic only: the model, its features, k and the walk-forward "
            "cadence are unchanged from step 3b. Nothing here is a candidate for "
            "adoption until a fix's frozen rule clears on 1819-2324.",
            "HOLDOUT: seasons " + ", ".join(HOLDOUT_SEASONS) + " are the final "
            "test set and are not loaded (AGENTS.md rule 9). They are reported "
            "once, at the end, when the user asks.",
            "Prior-match counts for teams are diagnostic columns computed in "
            "this script with the same strictly-before-date grouping the model's "
            "rate builder uses; models/ does not read them.",
            "'Expected total' is the incumbent's mu_home + mu_away, captured from "
            "the walk-forward's own fit instances — not re-derived.",
            "The proposed fixes and their decision rules are frozen before any "
            "implementation and are evaluated on 1819-2324 only, both datasets.",
        ],
    }

    report = render_report(context)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = REPORTS_DIR / f"phase1_step3c_diagnostics_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print("\n".join(findings), flush=True)
    print(f"\nwrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
