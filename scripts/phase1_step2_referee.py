#!/usr/bin/env python
"""Phase 1 step 2 — does referee information improve the team card model?

Run::

    uv run python scripts/phase1_step2_referee.py [--quick]

Same discipline as step 1 (scripts/phase1_step1_cards.py): walk-forward only,
no odds, no edge, no bets — calibration and a paired significance check. The
question here is narrower: on *identical* matches, does the design team+referee
beat the team-only design, and is any difference statistically convincing?

Design points worth stating once:

* three models on the same folds and the same rows — league-average baseline,
  team-only, team+referee — so the score differences measure exactly one
  thing each: team information (step 1's comparison) and referee information
  (this step's);
* no tuning anywhere: k=6 for the team rates and k=6 for the referee rate,
  chosen before looking, so the referee version cannot be flattered by a
  sweep; its only difference from the team model is one design column
  (log_ref_rate);
* the referee feature is built by add_referee_feature, which runs the same
  causality probe as the team features: the referee's rate for a match may
  only use matches strictly before that match's date;
* significance is a paired bootstrap on per-match score differences,
  resampling whole matches (invariant 3: the full, untruncated match
  sequence, never a subsequence), with a fixed seed so the numbers in the
  report are reproducible;
* every model is asserted to be scored on the identical match set before any
  difference is reported — a delta between two different match sets would be
  a bug, not a result.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bet_engine.backtest import walk_forward
from bet_engine.data.ingest import load_league
from bet_engine.eval.calibration import brier_score, log_loss, reliability_table
from bet_engine.markets.cards_totals import CardsTotals
from bet_engine.models import (
    CardsPoisson,
    CardsTeamRefPoisson,
    LeagueAveragePoisson,
    add_card_features,
    add_referee_feature,
)

SEASONS = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")
QUICK_SEASONS = ("2425", "2526")
LINES = (3.5, 4.5, 5.5)
QUICK_LINES = (3.5,)

#: Shrinkage weight in pseudo-matches, FIXED for teams and referees alike
#: (models/cards_poisson.py). Step 2's whole point is an untuned comparison:
#: choosing k here — or separately for referees — would be tuning to flatter.
K = 6.0

MIN_TRAIN_DAYS = 60
REFIT_EVERY_DAYS = 1
N_BINS = 10

#: A reliability gap is only reported from a bin that saw this many pairs —
#: a two-decimal gap out of 12 observations is noise wearing a decimal point.
MIN_BIN_COUNT = 50

#: Paired bootstrap: resample whole matches with replacement, this many times,
#: with this seed (each comparison gets seed + its row number, so a rerun
#: reproduces the report exactly).
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_007
BOOTSTRAP_CHUNK = 500

#: A referee with fewer than this many strictly-earlier matches has a rate
#: carried mostly by the shrinkage prior, not by history.
RARE_REFEREE_MATCHES = 10

MODELS: tuple[tuple[str, type], ...] = (
    ("league_average", LeagueAveragePoisson),
    ("team", CardsPoisson),
    ("team_ref", CardsTeamRefPoisson),
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def settle(df: pd.DataFrame, line: float) -> pd.Series:
    """Settled over/under label per match, from the market's own rule."""
    market = CardsTotals(line)
    labels = [market.settle(row) for row in df.to_dict("records")]
    settled = pd.Series(labels, index=df["match_id"].to_numpy(), dtype=object)
    missing = settled[settled.isna()]
    if len(missing):
        raise ValueError(
            f"{len(missing)} match(es) could not be settled at line {line}, "
            f"e.g. {list(missing.index[:3])} — cards should be present for "
            "every ingested row"
        )
    return settled


def run_walk_forward(
    features: pd.DataFrame, model_cls: type, line: float
) -> tuple[pd.DataFrame, dict[str, float], list[object]]:
    """One walk-forward run; returns the long prediction frame, every match's
    expected total (captured from the factory's instances), and the instances
    themselves — the last one carries the final fold's coefficients."""
    instances: list[object] = []

    def factory() -> object:
        model = model_cls(outcomes=("over", "under"), line=line, k=K)
        instances.append(model)
        return model

    prediction = walk_forward(
        features,
        factory,
        min_train_days=MIN_TRAIN_DAYS,
        refit_every_days=REFIT_EVERY_DAYS,
    )
    mu = {
        match_id: total
        for model in instances
        for match_id, total in model.predicted_totals.items()  # type: ignore[attr-defined]
    }
    return prediction, mu, instances


def score(prediction: pd.DataFrame, settled: pd.Series) -> dict:
    """Brier, log loss, reliability bins and the largest reportable gap."""
    wide = prediction.pivot(
        index="match_id", columns="outcome", values="model_prob"
    ).astype(float)
    actual = settled.loc[wide.index]
    bins = reliability_table(wide, actual, n_bins=N_BINS)

    reportable = [
        (bin_, abs(bin_.predicted_mean - bin_.actual_frequency))
        for bin_ in bins
        if bin_.count >= MIN_BIN_COUNT
    ]
    widest = max(reportable, key=lambda pair: pair[1]) if reportable else None
    return {
        "n_matches": len(wide),
        "brier": brier_score(wide, actual),
        "log_loss": log_loss(wide, actual),
        "wide": wide,
        "actual": actual,
        "widest": widest,
    }


def per_match_scores(scores: dict) -> pd.DataFrame:
    """One row per match: the exact per-match values the aggregate scores
    average, so the bootstrap resamples the same quantities the tables quote.

    Brier is the vector form (sum over outcomes of (p - y)^2) matching
    brier_score; log loss is -log(p) of the settled outcome. A settled
    probability of zero would be an infinite log loss and is refused here
    exactly as log_loss refuses it — clipping would hide a broken model.
    """
    wide: pd.DataFrame = scores["wide"]
    actual: pd.Series = scores["actual"]
    probabilities = wide.to_numpy(dtype=float)
    positions = wide.columns.get_indexer(actual.to_numpy())
    if (positions < 0).any():
        raise ValueError("settled outcome not among the predicted columns")
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(probabilities)), positions] = 1.0
    settled_probability = probabilities[np.arange(len(probabilities)), positions]
    if (settled_probability <= 0.0).any():
        raise ValueError(
            "per-match log loss is undefined where the settled outcome got "
            "probability 0; the model declared the observed result impossible"
        )
    return pd.DataFrame(
        {
            "brier": np.square(probabilities - truth).sum(axis=1),
            "log_loss": -np.log(settled_probability),
        },
        index=pd.Index(wide.index, name="match_id"),
    )


def paired_bootstrap(diff: pd.Series, seed: int) -> dict:
    """Match-level paired bootstrap of the mean score difference.

    ``diff`` is team+referee minus team-only, one value per match, on the full
    match sequence. Each resample draws n matches with replacement and takes
    the mean of the same draws for both models (paired: a match's difficulty
    cancels in the difference). Reported: the point estimate (the plain mean),
    the 95% percentile interval, and the fraction of resamples in which
    team+referee scored *better* (difference < 0 — lower is better for both
    Brier and log loss).
    """
    values = diff.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("paired bootstrap needs at least one match")
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_RESAMPLES)
        draws = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n_matches": n,
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "frac_better": float((means < 0.0).mean()),
    }


def dispersion(values: pd.Series) -> dict:
    """mean, sample variance and the variance/mean ratio of a card count."""
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    return {"n": len(values), "mean": mean, "variance": variance, "ratio": variance / mean}


def residual_dispersion(observed: pd.Series, mu: dict[str, float]) -> dict:
    """How under/over-dispersed the model's totals are.

    Two numbers on purpose:

    * ``pearson`` = var(observed - mu) / mean(mu), the dispersion the Poisson
      assumption actually implies (ratio 1 = right, >1 = under-dispersed);
    * ``mean_residual``, because var of residuals alone is undefined
      information — a model whose residuals average 3 cards is broken even if
      their variance looks perfect.
    """
    index = pd.Index(list(mu), name="match_id")
    predicted = pd.Series([mu[key] for key in mu], index=index, dtype=float)
    residual = observed.loc[index] - predicted
    return {
        "n": len(index),
        "mean_residual": float(residual.mean()),
        "pearson": float(residual.var(ddof=1) / predicted.mean()),
        "mean_mu": float(predicted.mean()),
    }


def gap_label(widest: tuple | None) -> str:
    if widest is None:
        return f"n/a (no bin with >= {MIN_BIN_COUNT} pairs)"
    bin_, gap = widest
    direction = "over-predicts" if bin_.predicted_mean > bin_.actual_frequency else "under-predicts"
    return (
        f"{gap:.3f} ({direction}, {bin_.low:.1f}-{bin_.high:.1f} bin, n={bin_.count})"
    )


def render_verdict(delta_rows: list[dict], boot_rows: list[dict]) -> list[str]:
    """The plain-language answer, computed from the numbers — not written to
    flatter them. It counts comparisons first, then leans on the bootstrap:
    a consistent but interval-overlapping improvement is reported as what it
    is (an improvement that is not statistically convincing)."""
    out: list[str] = []
    brier = [row["delta_brier"] for row in delta_rows]
    logloss = [row["delta_log_loss"] for row in delta_rows]
    brier_wins = sum(delta < 0 for delta in brier)
    logloss_wins = sum(delta < 0 for delta in logloss)
    mean_brier = float(np.mean(brier))
    mean_logloss = float(np.mean(logloss))

    out.append(
        f"Across the {len(brier)} dataset x line comparisons, team+referee "
        f"beat team-only on Brier in {brier_wins} of {len(brier)} (mean delta "
        f"{mean_brier:+.4f}) and on log loss in {logloss_wins} of "
        f"{len(logloss)} (mean delta {mean_logloss:+.4f}); negative favours "
        "team+referee."
    )

    brier_boot = [row for row in boot_rows if row["score"] == "brier"]
    excluding = [
        row
        for row in brier_boot
        if row["ci_high"] < 0.0 or row["ci_low"] > 0.0
    ]
    excluding_favouring = [row for row in excluding if row["mean"] < 0.0]
    if not excluding:
        out.append(
            "The paired match-level bootstrap ("
            f"{BOOTSTRAP_RESAMPLES:,} resamples of the full match sequence, "
            "95% percentile interval) excludes zero in NONE of the "
            f"{len(brier_boot)} Brier comparisons: every interval contains "
            "zero, so a difference this size could easily be noise."
        )
        out.append(
            "Plainly: the referee version does NOT improve on team-only in a "
            "statistically convincing way. The point differences "
            + ("favour the referee version, but "
               if brier_wins > len(brier) / 2 else "")
            + "are small relative to match-to-match variation, and no "
            "comparison separates them from zero."
            if brier_wins >= len(brier) / 2
            else "Plainly: the referee version does not improve on team-only: "
            "it loses most comparisons and no bootstrap interval separates "
            "it from zero."
        )
    elif len(excluding_favouring) == len(brier_boot):
        out.append(
            "The paired match-level bootstrap ("
            f"{BOOTSTRAP_RESAMPLES:,} resamples of the full match sequence, "
            "95% percentile interval) has the entire interval below zero in "
            f"ALL {len(brier_boot)} Brier comparisons — every one in "
            "team+referee's favour."
        )
        out.append(
            "Plainly: the referee version improves on team-only, and the "
            "improvement is statistically convincing at this sample size."
        )
    else:
        out.append(
            "The paired match-level bootstrap ("
            f"{BOOTSTRAP_RESAMPLES:,} resamples of the full match sequence, "
            "95% percentile interval) excludes zero in "
            f"{len(excluding)} of {len(brier_boot)} Brier comparisons "
            f"({len(excluding_favouring)} of them favouring team+referee)."
        )
        out.append(
            "Plainly: the referee version "
            + ("improves on team-only on the point estimates, but the "
               "improvement is NOT statistically convincing: most intervals "
               "still contain zero."
               if brier_wins > len(brier) / 2 and len(excluding_favouring) < len(brier_boot)
               else "does not consistently improve on team-only, and only "
               "some comparisons separate from zero.")
        )
    return out


def render_report(
    context: dict,
    rows: list[dict],
    delta_rows: list[dict],
    boot_rows: list[dict],
    coefficient_rows: list[dict],
    coverage_rows: list[dict],
    dispersion_rows: list[dict],
    dataset_stats: list[dict],
    verdict: list[str],
) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 Step 2 — team+referee vs team-only vs league average")
    add("")
    add(f"- Generated: {context['generated']}")
    add(f"- Data: {context['data']}")
    add(
        f"- Walk-forward: min_train_days={MIN_TRAIN_DAYS}, "
        f"refit_every_days={REFIT_EVERY_DAYS} (train strictly before each fold day)"
    )
    add(
        "- League average: Poisson GLM, design = intercept + home flag only"
    )
    add(
        "- Team: design = log own rate + log opponent rate + home flag, "
        f"rates shrunk with k={K:g}"
    )
    add(
        "- Team+referee: the team design plus ONE column, log referee rate — "
        f"the referee's total cards per match shrunk with the same k={K:g}, "
        "strictly-before-date history, league mean when no history. "
        "No parameter is tuned for it."
    )
    add("- Settlement: hy+ay+hr+ar vs the line (provisional counting rule)")
    add("- No odds and no betting in this step: calibration and significance only.")
    add("")

    add("## Verdict")
    add("")
    for sentence in verdict:
        add(sentence)
        add("")

    add("## Datasets")
    add("")
    add("| dataset | matches | covid_affected matches | first date | last date |")
    add("|---|---:|---:|---|---|")
    for stats in dataset_stats:
        add(
            f"| {stats['name']} | {stats['matches']} | {stats['covid']} | "
            f"{stats['first']} | {stats['last']} |"
        )
    add("")

    add("## Scores (all runs, identical match sets — asserted at runtime)")
    add("")
    add("| dataset | model | line | matches | brier | log loss | widest reliability gap |")
    add("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        add(
            f"| {row['dataset']} | {row['model']} | {row['line']} | "
            f"{row['n_matches']} | {row['brier']:.4f} | {row['log_loss']:.4f} | "
            f"{row['gap']} |"
        )
    add("")

    add("## Score deltas (same dataset, same line, same matches)")
    add("")
    add(
        "| dataset | line | team vs league: delta brier | team+ref vs team: "
        "delta brier | team+ref vs team: delta log loss |"
    )
    add("|---|---:|---:|---:|---:|")
    for row in delta_rows:
        add(
            f"| {row['dataset']} | {row['line']} | {row['delta_team_brier']:+.4f} | "
            f"{row['delta_brier']:+.4f} | {row['delta_log_loss']:+.4f} |"
        )
    add("")
    add(
        "Negative favours the second-named model. The team vs league column "
        "is step 1's comparison, repeated here so both information sources "
        "sit in one table."
    )
    add("")

    add(
        f"## Paired bootstrap: team+referee minus team-only "
        f"({BOOTSTRAP_RESAMPLES:,} resamples, whole matches, 95% interval)"
    )
    add("")
    add("| dataset | line | score | mean diff | 95% CI | resamples where ref better |")
    add("|---|---:|---|---:|---|---:|")
    for row in boot_rows:
        add(
            f"| {row['dataset']} | {row['line']} | {row['score']} | "
            f"{row['mean']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{100.0 * row['frac_better']:.1f}% |"
        )
    add("")
    add(
        "Resamples draw whole matches with replacement from the full "
        "sequence, both models on the same draws (paired), so a match's "
        "difficulty cancels in the difference. A 95% interval containing "
        "zero means the observed difference is not separated from noise at "
        "this sample size; a 'ref better' share near 50% says the same."
    )
    add("")

    add("## Referee coefficient (log_ref_rate)")
    add("")
    add("| dataset | coef | std error | 95% CI | training rows (final fold) |")
    add("|---|---:|---:|---|---:|")
    for row in coefficient_rows:
        add(
            f"| {row['dataset']} | {row['coef']:+.4f} | {row['se']:.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['n_training_rows']} |"
        )
    add("")
    add(
        "The coefficient of the referee term in the final fold's GLM (the "
        "fold trained on the longest window), with the GLM's own standard "
        "error. The design and the folds do not depend on the line, so this "
        "value is identical across the three line runs of a dataset — "
        "asserted at runtime. A positive coefficient would mean card-happy "
        "referees raise expected cards after controlling for the teams; an "
        "interval containing zero means referee identity adds no "
        "distinguishable signal beyond the teams in this fit."
    )
    add("")

    add("## Referee history behind the scored matches")
    add("")
    add(
        f"| dataset | scored matches | referee with < {RARE_REFEREE_MATCHES} "
        "prior matches | share | referee with zero prior |"
    )
    add("|---|---:|---:|---:|---:|")
    for row in coverage_rows:
        add(
            f"| {row['dataset']} | {row['n_matches']} | {row['rare']} | "
            f"{row['share']:.1%} | {row['zero']} |"
        )
    add("")
    add(
        "A prior match is one officiated by that referee strictly before the "
        "scored match's date. Matches in the 'rare' bucket get a referee "
        "rate carried mostly by the k-weighted league-mean prior."
    )
    add("")

    add("## Dispersion of the card totals")
    add("")
    add("| dataset | quantity | value |")
    add("|---|---|---:|")
    for row in dispersion_rows:
        add(f"| {row['dataset']} | {row['quantity']} | {row['value']} |")
    add("")
    add(
        "Observed ratio is var(total cards)/mean(total cards) with ddof=1 "
        "(Poisson = 1). Residual rows are the model's, computed once per "
        "dataset/model: the expected total does not depend on the line, so "
        "it is identical across the three line runs."
    )
    add("")

    add("## Notes")
    add("")
    for note in context["notes"]:
        add(f"- {note}")
    add("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="two seasons and one line, for a fast end-to-end check",
    )
    args = parser.parse_args(argv)
    seasons = QUICK_SEASONS if args.quick else SEASONS
    lines = QUICK_LINES if args.quick else LINES

    started = time.time()
    raw = load_league("E0", list(seasons))
    if raw["covid_affected"].isna().any():
        raise ValueError("covid_affected has NaNs; cannot split the datasets")
    covid = raw["covid_affected"].astype(bool)
    datasets = {
        "all": raw,
        "no_covid": raw.loc[~covid],
    }

    dataset_stats = [
        {
            "name": name,
            "matches": len(frame),
            "covid": int(frame["covid_affected"].astype(bool).sum()),
            "first": frame["date"].min().date(),
            "last": frame["date"].max().date(),
        }
        for name, frame in datasets.items()
    ]

    rows: list[dict] = []
    delta_rows: list[dict] = []
    boot_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    coverage_rows: list[dict] = []
    dispersion_rows: list[dict] = []
    total_runs = len(datasets) * len(MODELS) * len(lines)
    run_no = 0
    bootstrap_seed = BOOTSTRAP_SEED

    for dataset_name, frame in datasets.items():
        features = add_referee_feature(add_card_features(frame, K), K)
        observed = features.set_index("match_id")["total_cards"]

        observed_stats = dispersion(observed)
        dispersion_rows.append(
            {
                "dataset": dataset_name,
                "quantity": "observed total cards: mean",
                "value": f"{observed_stats['mean']:.3f}",
            }
        )
        dispersion_rows.append(
            {
                "dataset": dataset_name,
                "quantity": "observed total cards: var/mean",
                "value": f"{observed_stats['ratio']:.3f}",
            }
        )

        model_results: dict[str, dict[float, dict]] = {
            model_name: {} for model_name, _ in MODELS
        }
        mu_by_model: dict[str, dict[str, float]] = {}
        coefficient_tables: list[pd.DataFrame] = []

        for model_name, model_cls in MODELS:
            for line in lines:
                run_no += 1
                prediction, mu, instances = run_walk_forward(features, model_cls, line)
                settled = settle(frame, line)
                scores = score(prediction, settled)
                model_results[model_name][line] = scores
                if model_name == "team_ref":
                    coefficient_tables.append(
                        instances[-1].coefficient_table()  # type: ignore[attr-defined]
                    )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "line": line,
                        **{key: scores[key] for key in ("n_matches", "brier", "log_loss")},
                        "gap": gap_label(scores["widest"]),
                    }
                )
                print(
                    f"[{run_no}/{total_runs}] {dataset_name} {model_name} line {line}: "
                    f"{scores['n_matches']} matches, brier={scores['brier']:.4f}, "
                    f"logloss={scores['log_loss']:.4f}",
                    flush=True,
                )
            mu_by_model[model_name] = mu

        # Identical match sets are the precondition for every delta below:
        # score differences between different match sets would be a bug.
        reference_ids = set(model_results["league_average"][lines[0]]["wide"].index)
        for model_name, _ in MODELS:
            for line in lines:
                ids = set(model_results[model_name][line]["wide"].index)
                if ids != reference_ids:
                    raise AssertionError(
                        f"{dataset_name} {model_name} line {line} scored "
                        f"{len(ids)} matches, not the same {len(reference_ids)} "
                        "as the other models — deltas would be meaningless"
                    )

        # Coefficients must not depend on the line (folds and design do not);
        # assert it rather than claim it in the report.
        for table in coefficient_tables[1:]:
            if not np.allclose(
                table["coef"].to_numpy(), coefficient_tables[0]["coef"].to_numpy()
            ):
                raise AssertionError(
                    f"{dataset_name}: referee coefficient changed across line "
                    "runs — the report's claim of line-independence is false"
                )
        final_table = coefficient_tables[0].set_index("term")
        ref_row = final_table.loc["log_ref_rate"]
        coefficient_rows.append(
            {
                "dataset": dataset_name,
                "coef": float(ref_row["coef"]),
                "se": float(ref_row["se"]),
                "ci_low": float(ref_row["ci_low"]),
                "ci_high": float(ref_row["ci_high"]),
                "n_training_rows": int(instances[-1].n_training_rows),  # type: ignore[attr-defined]
            }
        )

        # Referee history coverage over exactly the matches that were scored.
        scored = features["match_id"].isin(reference_ids)
        ref_past = features.loc[scored, "ref_past_count"]
        rare = int((ref_past < RARE_REFEREE_MATCHES).sum())
        coverage_rows.append(
            {
                "dataset": dataset_name,
                "n_matches": len(reference_ids),
                "rare": rare,
                "share": rare / len(reference_ids),
                "zero": int((ref_past == 0).sum()),
            }
        )

        for line in lines:
            league_scores = model_results["league_average"][line]
            team_scores = model_results["team"][line]
            ref_scores = model_results["team_ref"][line]
            delta_rows.append(
                {
                    "dataset": dataset_name,
                    "line": line,
                    "delta_team_brier": team_scores["brier"] - league_scores["brier"],
                    "delta_brier": ref_scores["brier"] - team_scores["brier"],
                    "delta_log_loss": ref_scores["log_loss"] - team_scores["log_loss"],
                }
            )
            team_per_match = per_match_scores(team_scores)
            ref_per_match = per_match_scores(ref_scores)
            if not team_per_match.index.equals(ref_per_match.index):
                raise AssertionError(
                    f"{dataset_name} line {line}: per-match scores are not "
                    "aligned — the paired bootstrap needs the same matches"
                )
            for score_name in ("brier", "log_loss"):
                diff = ref_per_match[score_name] - team_per_match[score_name]
                boot = paired_bootstrap(diff, seed=bootstrap_seed)
                bootstrap_seed += 1
                boot_rows.append(
                    {
                        "dataset": dataset_name,
                        "line": line,
                        "score": score_name,
                        **boot,
                    }
                )

        for model_name, _ in MODELS:
            residual = residual_dispersion(observed, mu_by_model[model_name])
            dispersion_rows.append(
                {
                    "dataset": dataset_name,
                    "quantity": f"{model_name}: mean residual (observed - expected)",
                    "value": f"{residual['mean_residual']:+.3f}",
                }
            )
            dispersion_rows.append(
                {
                    "dataset": dataset_name,
                    "quantity": f"{model_name}: residual var / mean expected",
                    "value": f"{residual['pearson']:.3f}",
                }
            )

    verdict = render_verdict(delta_rows, boot_rows)
    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": (
            f"E0 seasons {', '.join(seasons)} ({len(raw)} matches), "
            f"lines {', '.join(str(line) for line in lines)}, k={K:g} for "
            "teams and referees"
        ),
        "notes": [
            "PROVISIONAL counting rule: a match total is hy+ay+hr+ar exactly as "
            "football-data publishes it. A bookmaker's own rule (second yellow "
            "counted once or twice, cards to staff) is unconfirmed — see "
            "docs/DATA_NOTES.md. Any comparison to a real line stays provisional.",
            "covid_affected is the ingest flag: matches played behind closed "
            "doors / in the disrupted calendar. Dropping them changes both the "
            "evaluated matches and the history every rate is built from.",
            "No odds, no edge, no bets: these are calibration numbers for a "
            "probabilistic forecast, not a betting result.",
            "No tuning: k=6 for team rates and k=6 for the referee rate, fixed "
            "before looking; the referee version differs from the team model by "
            "exactly one design column.",
            "Reliability gaps are only read from bins with at least "
            f"{MIN_BIN_COUNT} pairs; emptier bins are shown but not quoted.",
            "Bootstrap runs on the full, untruncated match sequence "
            "(AGENTS.md invariant 3), paired by match, fixed seed "
            f"{BOOTSTRAP_SEED}.",
        ],
    }

    report = render_report(
        context,
        rows,
        delta_rows,
        boot_rows,
        coefficient_rows,
        coverage_rows,
        dispersion_rows,
        dataset_stats,
        verdict,
    )
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    suffix = "_quick" if args.quick else ""
    path = REPORTS_DIR / f"phase1_step2_referee{suffix}_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nwrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
