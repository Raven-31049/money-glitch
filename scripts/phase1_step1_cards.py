#!/usr/bin/env python
"""Phase 1 step 1 — team-only Poisson card model, scored honestly.

Run::

    uv run python scripts/phase1_step1_cards.py [--quick]

What this deliberately does NOT do: read an odd, price a line, or talk about
edge (docs/PLAN.md, Phase 1 step 1 — no odds and no betting yet). The only
question here is calibration: does a model that knows the two teams emit card
over/under probabilities that match what happened, and is it any better than
the same regression stripped of team information?

Design points worth stating once:

* every run is a full walk-forward (train strictly before each fold day,
  min 60 days, refit daily), so no probability in the report was ever fitted
  on its own match;
* the three lines are three separate walk-forward runs rather than three tail
  evaluations of one run, because a config picks one line and the pipeline
  should be exercised exactly as it will be used;
* features are rebuilt per dataset, because the ``no_covid`` world has a
  different history behind every rate than the full one;
* the baseline is a real fitted model on the same rows and the same folds —
  the only difference is two dropped columns, which is what makes the
  comparison measure team information and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from bet_engine.backtest import walk_forward
from bet_engine.data.ingest import load_league
from bet_engine.eval.calibration import brier_score, log_loss, reliability_table
from bet_engine.markets.cards_totals import CardsTotals
from bet_engine.models import CardsPoisson, LeagueAveragePoisson, add_card_features

SEASONS = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")
QUICK_SEASONS = ("2425", "2526")
LINES = (3.5, 4.5, 5.5)
QUICK_LINES = (3.5,)

#: Shrinkage weight in pseudo-matches (models/cards_poisson.py). Fixed here
#: rather than swept: choosing k is a later, separate question.
K = 6.0

MIN_TRAIN_DAYS = 60
REFIT_EVERY_DAYS = 1
N_BINS = 10

#: A reliability gap is only reported from a bin that saw this many pairs —
#: a two-decimal gap out of 12 observations is noise wearing a decimal point.
MIN_BIN_COUNT = 50

MODELS: tuple[tuple[str, type], ...] = (
    ("team", CardsPoisson),
    ("league_average", LeagueAveragePoisson),
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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One walk-forward run; returns the long prediction frame and every
    match's expected total, captured from the instances the factory built."""
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
    return prediction, mu


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
        "bins": bins,
        "widest": widest,
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


def render_report(context: dict, rows: list[dict], dispersion_rows: list[dict],
                  calibration: dict, dataset_stats: list[dict]) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 Step 1 — team-only cards model vs league-average baseline")
    add("")
    add(f"- Generated: {context['generated']}")
    add(f"- Data: {context['data']}")
    add(
        f"- Walk-forward: min_train_days={MIN_TRAIN_DAYS}, "
        f"refit_every_days={REFIT_EVERY_DAYS} (train strictly before each fold day)"
    )
    add(
        "- Team model: Poisson GLM on team-match rows, design = "
        f"log own rate + log opponent rate + home flag, rates shrunk with k={K:g}"
    )
    add("- Baseline: same regression and rows, design = intercept + home flag only")
    add("- Settlement: hy+ay+hr+ar vs the line (provisional counting rule, see notes)")
    add("- No odds and no betting in this step: calibration only.")
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

    add("## Scores (all runs)")
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

    add("## Team model minus baseline (same dataset, same line)")
    add("")
    add("| dataset | line | delta brier | delta log loss |")
    add("|---|---:|---:|---:|")
    lookup = {(row["dataset"], row["line"], row["model"]): row for row in rows}
    for dataset in context["datasets"]:
        for line in context["lines"]:
            team = lookup[(dataset, line, "team")]
            base = lookup[(dataset, line, "league_average")]
            add(
                f"| {dataset} | {line} | {team['brier'] - base['brier']:+.4f} | "
                f"{team['log_loss'] - base['log_loss']:+.4f} |"
            )
    add("")
    add("Negative deltas mean the team model beat the baseline on that score.")
    add("")

    add("## Reliability (10 equal-width bins, all pairs)")
    add("")
    for key, bins in calibration.items():
        add(f"### {key}")
        add("")
        add("| bin | n | predicted mean | actual frequency | gap |")
        add("|---|---:|---:|---:|---:|")
        for bin_ in bins:
            if bin_.count == 0:
                continue
            add(
                f"| {bin_.low:.1f}-{bin_.high:.1f} | {bin_.count} | "
                f"{bin_.predicted_mean:.3f} | {bin_.actual_frequency:.3f} | "
                f"{bin_.predicted_mean - bin_.actual_frequency:+.3f} |"
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
        "dataset/model: the expected total does not depend on the line, so it "
        "is identical across the three line runs."
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
    dispersion_rows: list[dict] = []
    calibration: dict = {}
    total_runs = len(datasets) * len(MODELS) * len(lines)
    run_no = 0

    for dataset_name, frame in datasets.items():
        features = add_card_features(frame, K)
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

        for model_name, model_cls in MODELS:
            mu_by_line: dict[float, dict[str, float]] = {}
            for line in lines:
                run_no += 1
                prediction, mu = run_walk_forward(features, model_cls, line)
                mu_by_line[line] = mu
                settled = settle(frame, line)
                scores = score(prediction, settled)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "line": line,
                        **{key: scores[key] for key in ("n_matches", "brier", "log_loss")},
                        "gap": gap_label(scores["widest"]),
                    }
                )
                calibration[f"{dataset_name} / {model_name} / line {line:g}"] = scores["bins"]
                print(
                    f"[{run_no}/{total_runs}] {dataset_name} {model_name} line {line}: "
                    f"{scores['n_matches']} matches, brier={scores['brier']:.4f}, "
                    f"logloss={scores['log_loss']:.4f}",
                    flush=True,
                )

            residual = residual_dispersion(observed, mu_by_line[lines[0]])
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

    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": (
            f"E0 seasons {', '.join(seasons)} ({len(raw)} matches), "
            f"lines {', '.join(str(line) for line in lines)}, k={K:g}"
        ),
        "datasets": list(datasets),
        "lines": list(lines),
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
            "Runs score "
            + ", ".join(
                f"{count} matches ({name})"
                for name, count in sorted(
                    {row["dataset"]: row["n_matches"] for row in rows}.items()
                )
            )
            + ", above the ~100 threshold where calibration gaps become "
            "meaningful.",
            "Reliability gaps are only read from bins with at least "
            f"{MIN_BIN_COUNT} pairs; emptier bins are shown but not quoted.",
        ],
    }

    report = render_report(context, rows, dispersion_rows, calibration, dataset_stats)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    suffix = "_quick" if args.quick else ""
    path = REPORTS_DIR / f"phase1_step1_cards{suffix}_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nwrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
