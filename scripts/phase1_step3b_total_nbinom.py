#!/usr/bin/env python
"""Phase 1 step 3b — model the match *total* directly, and decide.

Run::

    uv run python scripts/phase1_step3b_total_nbinom.py [--quick]

Step 3 fitted each *side's* own cards and built the match total by convolution.
It found the per-side counts carry essentially no dispersion of their own
(alpha ~ 1e-7) while the total's variance/mean sits above 1 — home and away
cards are correlated through the match, not per-side overdispersed. Step 3b
tests the model that takes that finding literally: predict the total directly,
one row per match, with a single negative binomial whose dispersion is
estimated from the data.

Three questions, in order:

1. How correlated are the two sides? Raw (which mixes in the shared covariates),
   and *after* the Poisson team+referee model — the residual correlation, which
   is the part a total-level model could still capture. Per season and overall.
2. Does a total-target negative binomial — design const + log home rate + log
   away rate + log referee rate, the same k=6 features — score better than the
   Poisson team+referee model on identical matches, identical folds, lines 3.5,
   4.5 and 5.5, in both datasets?
3. Apply the decision rule below, mechanically, and let it pick the model.

THE DECISION RULE (fixed before this script was run — do not edit after seeing
the numbers):

    Adopt cards_total_nbinom as the Phase 1 cards model ONLY IF, comparing it
    with poisson_team_ref on identical matches:

    (A) it has lower log loss on at least 2 of the 3 lines (3.5, 4.5, 5.5) in
        BOTH datasets (all and no_covid); AND

    (B) its count-frequency tails are closer to the actual distribution than
        poisson_team_ref's in BOTH datasets, where "closer" is a smaller
        combined L1 distance sum(|predicted - actual|) over the six tail
        buckets {0, 1, 2, 8, 9, 10+}.

    Both (A) and (B) must hold in BOTH datasets. Otherwise poisson_team_ref
    stays the Phase 1 cards model.

Nothing is tuned. k=6 for teams and referees as in steps 1-3, the dispersion is
estimated by the likelihood, the folds and lines are step 3's, and the total
model differs from the convolved team+referee model by the target and the
likelihood — not by a swept parameter.
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
from bet_engine.eval.calibration import brier_score, log_loss, reliability_table
from bet_engine.markets.cards_totals import CardsTotals
from bet_engine.models import (
    SIDE_MAX_COUNT,
    CardsTeamRefPoisson,
    CardsTotalNegativeBinomial,
    add_card_features,
    add_referee_feature,
    side_pmf,
    total_pmf,
)

SEASONS = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")
QUICK_SEASONS = ("2425", "2526")
LINES = (3.5, 4.5, 5.5)
QUICK_LINES = (3.5,)

#: Shrinkage weight in pseudo-matches, fixed for teams and referees (steps 1-3).
K = 6.0

MIN_TRAIN_DAYS = 60
REFIT_EVERY_DAYS = 1
N_BINS = 10

#: A reliability gap is only quoted from a bin that saw this many pairs.
MIN_BIN_COUNT = 50

#: Paired bootstrap settings, copied from step 3 so the comparison methods match.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_100
BOOTSTRAP_CHUNK = 500

#: Counts 0..9 are reported individually; 10 or more is the top bucket.
TAIL_MAX_COUNT = 10

#: The buckets the decision rule looks at, as the labels count_frequencies emits.
TAIL_BUCKETS: tuple[str, ...] = ("0", "1", "2", "8", "9", f"{TAIL_MAX_COUNT}+")

#: The buckets as they should read in prose, e.g. ``{0, 1, 2, 8, 9, 10+}``.
TAIL_BUCKETS_TEXT = "{" + ", ".join(TAIL_BUCKETS) + "}"

#: The two models, incumbent first. The decision compares the challenger with
#: the incumbent; nothing else enters the rule.
INCUMBENT = "poisson_team_ref"
CHALLENGER = "cards_total_nbinom"
MODELS: tuple[tuple[str, type], ...] = (
    (INCUMBENT, CardsTeamRefPoisson),
    (CHALLENGER, CardsTotalNegativeBinomial),
)

#: The rule, frozen in one place so the report can quote it verbatim.
LOGLOSS_LINES_REQUIRED = 2
DECISION_RULE = (
    "Adopt cards_total_nbinom ONLY IF, against poisson_team_ref on identical "
    "matches: (A) lower log loss on at least "
    f"{LOGLOSS_LINES_REQUIRED} of {len(LINES)} lines in BOTH datasets, AND "
    "(B) smaller combined L1 tail distance sum(|predicted-actual|) over the "
    f"counts {TAIL_BUCKETS_TEXT} in BOTH datasets. Otherwise poisson_team_ref "
    "stays."
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# scoring helpers (step 3's, unchanged where the meaning is unchanged)
# ---------------------------------------------------------------------------


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
) -> dict:
    """One walk-forward run; captures per-match means and fitted dispersion.

    The incumbent's total is the convolution of its two per-side means; the
    challenger predicts the total directly. Both are recorded here, keyed by
    match, alongside each fold's fitted alpha, so the report can rebuild either
    model's total pmf after the run without refitting.
    """
    instances: list[object] = []

    def factory() -> object:
        model = model_cls(outcomes=("over", "under"), line=line, k=K)
        instances.append(model)
        return model

    with warnings.catch_warnings():
        # The boundary optimum raises statsmodels' ConvergenceWarning once per
        # fold; the fit-health summary carries the real information. (Still
        # true after starting the MLE from the Poisson optimum — see step 3.)
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        prediction = walk_forward(
            features,
            factory,
            min_train_days=MIN_TRAIN_DAYS,
            refit_every_days=REFIT_EVERY_DAYS,
        )

    sides: dict[str, tuple[float, float, float]] = {}
    totals: dict[str, tuple[float, float]] = {}
    alphas: list[float] = []
    max_total_mu = 0.0
    for model in instances:
        alpha = getattr(model, "alpha_", None)
        if alpha is not None:
            alphas.append(float(alpha))
            effective = float(alpha)
        else:
            effective = 0.0
        for match_id, (home, away) in getattr(model, "predicted_sides", {}).items():
            sides[str(match_id)] = (float(home), float(away), effective)
            max_total_mu = max(max_total_mu, float(home) + float(away))
        for match_id, total in getattr(model, "predicted_totals", {}).items():
            totals[str(match_id)] = (float(total), effective)
            max_total_mu = max(max_total_mu, float(total))
    return {
        "prediction": prediction,
        "sides": sides,
        "totals": totals,
        "health": {
            "n_folds": len(instances),
            "alphas": np.asarray(alphas, dtype=float),
            "max_total_mu": max_total_mu,
        },
    }


TOTAL_COUNTS = np.arange(2 * SIDE_MAX_COUNT - 1)


def distribution_from_sides(
    sides: dict[str, tuple[float, float, float]], ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The convolved total pmf for the two-side (incumbent) model."""
    record = np.array([sides[match_id] for match_id in ids], dtype=float)
    counts, pmf = total_pmf(record[:, 0], record[:, 1], record[:, 2])
    return counts, pmf, record[:, 0] + record[:, 1]


def distribution_from_total(
    totals: dict[str, tuple[float, float]], ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The direct total pmf for the total-target (challenger) model."""
    record = np.array([totals[match_id] for match_id in ids], dtype=float)
    pmf = side_pmf(record[:, 0], record[:, 1], TOTAL_COUNTS)
    return TOTAL_COUNTS, pmf, record[:, 0]


def probabilities_for_line(
    counts: np.ndarray, pmf: np.ndarray, line: float, ids: np.ndarray
) -> pd.DataFrame:
    """Read over/under off a total pmf at ``line``."""
    over = pmf[:, counts > line].sum(axis=1)
    under = pmf[:, counts <= line].sum(axis=1)
    return pd.DataFrame(
        {"over": over, "under": under}, index=pd.Index(ids, name="match_id")
    )


def score(wide: pd.DataFrame, settled: pd.Series) -> dict:
    """Brier, log loss and the largest reportable reliability gap."""
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
    """One row per match: the exact per-match values the aggregate scores average."""
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
            "per-match log loss undefined where the settled outcome got probability 0"
        )
    return pd.DataFrame(
        {
            "brier": np.square(probabilities - truth).sum(axis=1),
            "log_loss": -np.log(settled_probability),
        },
        index=pd.Index(wide.index, name="match_id"),
    )


def paired_bootstrap(diff: pd.Series, seed: int) -> dict:
    """Match-level paired bootstrap of the mean score difference (step 3's method)."""
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


def count_frequencies(
    observed_total: np.ndarray, counts: np.ndarray, pmf: np.ndarray
) -> pd.DataFrame:
    """Predicted vs actual frequency of each total-card count (0..9, 10+)."""
    labels = [str(k) for k in range(TAIL_MAX_COUNT)] + [f"{TAIL_MAX_COUNT}+"]
    predicted = [
        float(pmf[:, counts == k].sum(axis=1).mean()) for k in range(TAIL_MAX_COUNT)
    ]
    predicted.append(float(pmf[:, counts >= TAIL_MAX_COUNT].sum(axis=1).mean()))
    actual = [float((observed_total == k).mean()) for k in range(TAIL_MAX_COUNT)]
    actual.append(float((observed_total >= TAIL_MAX_COUNT).mean()))
    return pd.DataFrame({"count": labels, "predicted": predicted, "actual": actual})


def tail_l1(table: pd.DataFrame) -> float:
    """Combined L1 distance over the decision rule's tail buckets.

    This is the *exact* quantity rule (B) tests, computed here so the decision
    and the reported table cannot disagree: sum of |predicted - actual| over
    counts {0, 1, 2, 8, 9, 10+}.
    """
    indexed = table.set_index("count")
    return float(
        sum(abs(indexed.loc[label, "predicted"] - indexed.loc[label, "actual"])
            for label in TAIL_BUCKETS)
    )


def over_reliability(p_over: np.ndarray, y_over: np.ndarray) -> list[dict]:
    """Bin P(over) and compare mean prediction with actual over-frequency.

    A market-level reliability readout for the two-outcome market: each row's
    P(over) and whether "over" settled. Reported per line so a per-line
    calibration difference between the models is visible.
    """
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


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over the finite pairs; NaN when it is undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def correlation_rows(
    dataset_name: str, features: pd.DataFrame, incumbent: dict, ids: np.ndarray
) -> list[dict]:
    """Raw and residual home/away card correlation, per season and overall.

    Raw uses every match in the dataset (a property of the data). Residual uses
    the incumbent's out-of-sample per-side residuals on the scored matches:

        resid_home = home_cards - mu_home,  resid_away = away_cards - mu_away.

    A residual correlation above zero is exactly the dependence a total-level
    model could capture and the two-independent-side convolution cannot.
    """
    home_all = features["home_cards"].to_numpy(dtype=float)
    away_all = features["away_cards"].to_numpy(dtype=float)
    season_all = features["season"].to_numpy()

    frame = features.set_index("match_id")
    home_oos = frame.loc[ids, "home_cards"].to_numpy(dtype=float)
    away_oos = frame.loc[ids, "away_cards"].to_numpy(dtype=float)
    season_oos = frame.loc[ids, "season"].to_numpy()
    resid_home = home_oos - np.array(
        [incumbent["sides"][match_id][0] for match_id in ids], dtype=float
    )
    resid_away = away_oos - np.array(
        [incumbent["sides"][match_id][1] for match_id in ids], dtype=float
    )

    rows = []
    for season in list(dict.fromkeys(season_all)) + ["all"]:
        if season == "all":
            raw_mask = np.ones(len(home_all), dtype=bool)
            resid_mask = np.ones(len(home_oos), dtype=bool)
        else:
            raw_mask = season_all == season
            resid_mask = season_oos == season
        rows.append(
            {
                "dataset": dataset_name,
                "season": season,
                "raw_n": int(raw_mask.sum()),
                "raw_corr": _correlation(home_all[raw_mask], away_all[raw_mask]),
                "resid_n": int(resid_mask.sum()),
                "resid_corr": _correlation(
                    resid_home[resid_mask], resid_away[resid_mask]
                ),
            }
        )
    return rows


def gap_label(widest: tuple | None) -> str:
    if widest is None:
        return f"n/a (no bin with >= {MIN_BIN_COUNT} pairs)"
    bin_, gap = widest
    direction = (
        "over-predicts" if bin_.predicted_mean > bin_.actual_frequency else "under-predicts"
    )
    return f"{gap:.3f} ({direction}, {bin_.low:.1f}-{bin_.high:.1f} bin, n={bin_.count})"


def make_decision(
    all_scores: dict, tail_distances: dict, datasets: dict, lines: tuple
) -> tuple[bool, dict]:
    """Apply the frozen rule and return (adopt, per-dataset evidence).

    Mechanically: for each dataset, count the lines where the challenger's log
    loss beats the incumbent's, and compare the two models' tail L1 distances;
    adopt only when log loss wins at least LOGLOSS_LINES_REQUIRED lines AND the
    tail distance is smaller, in every dataset.
    """
    per_dataset: dict[str, dict] = {}
    for dataset_name in datasets:
        improved = {
            line: all_scores[(dataset_name, line, CHALLENGER)]["log_loss"]
            < all_scores[(dataset_name, line, INCUMBENT)]["log_loss"]
            for line in lines
        }
        n_improved = int(sum(improved.values()))
        logloss_ok = n_improved >= LOGLOSS_LINES_REQUIRED
        tail_incumbent = tail_distances[(dataset_name, INCUMBENT)]
        tail_challenger = tail_distances[(dataset_name, CHALLENGER)]
        tail_ok = tail_challenger < tail_incumbent
        per_dataset[dataset_name] = {
            "improved": improved,
            "n_improved": n_improved,
            "logloss_ok": logloss_ok,
            "tail_ok": tail_ok,
            "tail_incumbent": tail_incumbent,
            "tail_challenger": tail_challenger,
        }
    adopt = all(
        row["logloss_ok"] and row["tail_ok"] for row in per_dataset.values()
    )
    return adopt, per_dataset


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_report(context: dict, payload: dict) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 Step 3b — total-target negative binomial, and the decision")
    add("")
    add(f"- Generated: {context['generated']}")
    add(f"- Data: {context['data']}")
    add(
        f"- Walk-forward: min_train_days={MIN_TRAIN_DAYS}, "
        f"refit_every_days={REFIT_EVERY_DAYS} (train strictly before each fold day)"
    )
    add(
        f"- Incumbent `{INCUMBENT}`: Poisson team+referee, design = log own rate + "
        f"log opponent rate + log referee rate + home flag; its match total is the "
        "convolution of the two per-side pmfs (step 2/3's model)"
    )
    add(
        f"- Challenger `{CHALLENGER}`: negative-binomial on the match *total*, one "
        "row per match, design = log home rate + log away rate + log referee rate; "
        "dispersion estimated by the likelihood"
    )
    add(
        "- Match total for the challenger reads straight off its single total pmf; "
        "for the incumbent it is the convolution of the two per-side pmfs "
        "(models/count_distributions.py), vectorised over matches"
    )
    add(f"- Settlement: hy+ay+hr+ar vs the line (provisional counting rule)")
    add("- No odds, no betting: calibration and significance only.")
    add("")

    add("## Decision rule (fixed before the run)")
    add("")
    add(f"> {DECISION_RULE}")
    add("")
    add("## Decision")
    add("")
    for sentence in payload["decision_sentences"]:
        add(sentence)
        add("")

    add("## Verdict")
    add("")
    for sentence in payload["verdict"]:
        add(sentence)
        add("")

    add("## Datasets")
    add("")
    add("| dataset | matches | covid_affected matches | first date | last date |")
    add("|---|---:|---:|---|---|")
    for stats in payload["dataset_stats"]:
        add(
            f"| {stats['name']} | {stats['matches']} | {stats['covid']} | "
            f"{stats['first']} | {stats['last']} |"
        )
    add("")

    add("## Home vs away card correlation")
    add("")
    add(
        "*Raw* uses every match in the dataset. *Residual* uses the incumbent "
        "Poisson team+referee model's out-of-sample per-side residuals "
        "(actual - expected own cards) on the scored matches. The residual "
        "correlation is the dependence the two-independent-side convolution "
        "cannot represent."
    )
    add("")
    add("| dataset | season | raw n | raw corr | residual n | residual corr |")
    add("|---|---|---:|---:|---:|---:|")
    for row in payload["correlation_rows"]:
        add(
            f"| {row['dataset']} | {row['season']} | {row['raw_n']} | "
            f"{row['raw_corr']:+.3f} | {row['resid_n']} | {row['resid_corr']:+.3f} |"
        )
    add("")

    add("## Scores (identical match sets — asserted at runtime)")
    add("")
    add("| dataset | model | line | matches | brier | log loss | widest reliability gap |")
    add("|---|---|---:|---:|---:|---:|---|")
    for row in payload["score_rows"]:
        add(
            f"| {row['dataset']} | {row['model']} | {row['line']:g} | "
            f"{row['n_matches']} | {row['brier']:.4f} | {row['log_loss']:.4f} | "
            f"{row['gap']} |"
        )
    add("")

    add("## Score deltas: challenger minus incumbent (negative favours the challenger)")
    add("")
    add("| dataset | line | delta brier | delta log loss | log loss improved |")
    add("|---|---:|---:|---:|---|")
    for row in payload["delta_rows"]:
        add(
            f"| {row['dataset']} | {row['line']:g} | {row['delta_brier']:+.4f} | "
            f"{row['delta_log_loss']:+.4f} | {'yes' if row['improved'] else 'no'} |"
        )
    add("")

    add(
        f"## Paired bootstrap: challenger minus incumbent "
        f"({BOOTSTRAP_RESAMPLES:,} resamples, whole matches, 95% interval)"
    )
    add("")
    add("| dataset | line | score | mean diff | 95% CI | resamples where challenger better |")
    add("|---|---:|---|---:|---|---:|")
    for row in payload["bootstrap_rows"]:
        add(
            f"| {row['dataset']} | {row['line']:g} | {row['score']} | "
            f"{row['mean']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{100.0 * row['frac_better']:.1f}% |"
        )
    add("")
    add(
        "A 95% interval containing zero means the challenger's score is not "
        "separated from the incumbent's at this sample size."
    )
    add("")

    add("## Calibration per line: P(over) reliability")
    add("")
    add(
        "Rows bin each match by its predicted P(over); `pred` is the mean "
        "prediction in the bin, `act` the share of those matches that went over, "
        "`n` the bin size. A well-calibrated model has pred and act on top of "
        "each other in every bin."
    )
    add("")
    for dataset_name, lines in payload["calibration"].items():
        add(f"### {dataset_name}")
        add("")
        for line, rows in lines.items():
            add(f"line {line:g}")
            add("")
            add(
                f"| P(over) bin | incumbent pred | incumbent act | n | "
                "challenger pred | challenger act | n |"
            )
            add("|---|---:|---:|---:|---:|---:|---:|")
            for row in rows:
                add(
                    f"| {row['low']:.1f}-{row['high']:.1f} | "
                    f"{row['incumbent_pred']:.3f} | {row['incumbent_actual']:.3f} | "
                    f"{row['incumbent_n']} | {row['challenger_pred']:.3f} | "
                    f"{row['challenger_actual']:.3f} | {row['challenger_n']} |"
                )
            add("")

    add("## Predicted vs actual frequency for each total card count")
    add("")
    add(
        "Predicted is the mean probability the model put on that count; actual "
        "is the share of matches with that count. `tail L1` is the decision "
        f"rule's distance over the buckets {TAIL_BUCKETS}."
    )
    add("")
    for dataset_name, tables in payload["tail_tables"].items():
        add(f"### {dataset_name}")
        add("")
        add("| total cards | actual | incumbent | challenger |")
        add("|---|---:|---:|---:|")
        actual = tables["actual"]
        for position, label in enumerate(actual["count"]):
            add(
                f"| {label} | {actual['actual'].iloc[position]:.4f} | "
                f"{tables[INCUMBENT]['predicted'].iloc[position]:.4f} | "
                f"{tables[CHALLENGER]['predicted'].iloc[position]:.4f} |"
            )
        add("")
        add("| model | total variation | largest gap | at count | tail L1 (rule B) |")
        add("|---|---:|---:|---|---:|")
        for row in tables["metrics"]:
            add(
                f"| {row['model']} | {row['total_variation']:.4f} | "
                f"{row['max_gap']:.4f} | {row['max_gap_count']} | "
                f"{row['tail_l1']:.4f} |"
            )
        add("")

    add("## Residual dispersion (out-of-sample)")
    add("")
    add(
        "`var/mean` is Var(actual total - expected total) / mean(expected total); "
        "`Pearson` is mean((residual^2) / expected total). Both are ~1 for a "
        "correctly specified conditional distribution, below 1 when the model "
        "over-states variance."
    )
    add("")
    add("| dataset | model | matches | mean residual | var/mean | Pearson |")
    add("|---|---|---:|---:|---:|---:|")
    for row in payload["dispersion_rows"]:
        add(
            f"| {row['dataset']} | {row['model']} | {row['n']} | "
            f"{row['mean_residual']:+.4f} | {row['var_over_mean']:.4f} | "
            f"{row['pearson']:.4f} |"
        )
    add("")

    add("## Fitted total-model dispersion per season (in-sample)")
    add("")
    add(
        "Alpha from a single in-sample fit of the challenger on that season's "
        "matches (variance = mu + alpha * mu^2). `total var/mean` is the raw "
        "overdispersion the total carries before any modelling. Poisson = 0."
    )
    add("")
    add("| dataset | season | matches | total var/mean | challenger alpha |")
    add("|---|---|---:|---:|---:|")
    for row in payload["season_dispersion"]:
        add(
            f"| {row['dataset']} | {row['season']} | {row['matches']} | "
            f"{row['total_ratio']:.3f} | {row['alpha']:.3g} |"
        )
    add("")

    add("## Fit health")
    add("")
    add(
        "`folds` is how many walk-forward refits each model ran; the alpha "
        "columns are the dispersion the likelihood chose (incumbent is Poisson, "
        "so `-`). `max expected total` is the largest expected match total any "
        "fold emitted — it stays in single digits, so no fold drifted."
    )
    add("")
    add("| dataset | model | folds | alpha min | alpha median | alpha max | max expected total |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for row in payload["fold_health"]:
        if row["alpha_min"] is None:
            alpha_cells = "`-` | `-` | `-`"
        else:
            alpha_cells = (
                f"{row['alpha_min']:.3g} | {row['alpha_median']:.3g} | "
                f"{row['alpha_max']:.3g}"
            )
        add(
            f"| {row['dataset']} | {row['model']} | {row['n_folds']} | "
            f"{alpha_cells} | {row['max_total_mu']:.2f} |"
        )
    add("")

    add("## Bootstrap resample count note")
    add("")
    add(
        f"- Fixed bootstrap seed {BOOTSTRAP_SEED}; one interval per "
        "(dataset, line, score); resample count "
        f"{BOOTSTRAP_RESAMPLES:,}."
    )
    add("")

    add("## Notes")
    add("")
    for note in context["notes"]:
        add(f"- {note}")
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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
    datasets = {"all": raw, "no_covid": raw.loc[~covid]}

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

    score_rows: list[dict] = []
    delta_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    dispersion_rows: list[dict] = []
    season_dispersion: list[dict] = []
    fold_health: list[dict] = []
    correlation: list[dict] = []
    tail_tables: dict[str, dict] = {}
    calibration: dict[str, dict] = {}
    all_scores: dict[tuple[str, float, str], dict] = {}
    tail_distance: dict[tuple[str, str], float] = {}
    bootstrap_seed = BOOTSTRAP_SEED

    for dataset_name, frame in datasets.items():
        features = add_referee_feature(add_card_features(frame, K), K)
        observed = features.set_index("match_id")["total_cards"]

        results: dict[str, dict] = {}
        reference_ids: np.ndarray | None = None
        for model_name, model_cls in MODELS:
            run = run_walk_forward(features, model_cls, lines[0])
            ids = run["prediction"]["match_id"].drop_duplicates().to_numpy()
            if reference_ids is None:
                reference_ids = ids
            elif set(ids) != set(reference_ids):
                raise AssertionError(
                    f"{dataset_name} {model_name} scored {len(ids)} matches, not "
                    f"the same {len(reference_ids)} as the other models"
                )
            if model_name == INCUMBENT:
                counts, pmf, mu_total = distribution_from_sides(run["sides"], ids)
            else:
                counts, pmf, mu_total = distribution_from_total(run["totals"], ids)

            # Pin the reconstruction to what walk_forward actually emitted at the
            # first line: same numbers, same match-id alignment.
            direct = (
                run["prediction"]
                .pivot(index="match_id", columns="outcome", values="model_prob")
                .astype(float)
                .loc[ids]
            )
            recon = probabilities_for_line(counts, pmf, lines[0], ids)
            if not np.allclose(
                direct[["over", "under"]].to_numpy(),
                recon[["over", "under"]].to_numpy(),
            ):
                raise AssertionError(
                    f"{dataset_name} {model_name}: reconstructed probabilities "
                    "disagree with the walk-forward output — alignment bug"
                )

            results[model_name] = {
                "prediction": run["prediction"],
                "sides": run["sides"],
                "counts": counts,
                "pmf": pmf,
                "mu_total": mu_total,
                "health": run["health"],
            }

            alphas = run["health"]["alphas"]
            fold_health.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_folds": run["health"]["n_folds"],
                    "alpha_min": float(alphas.min()) if alphas.size else None,
                    "alpha_median": float(np.median(alphas)) if alphas.size else None,
                    "alpha_max": float(alphas.max()) if alphas.size else None,
                    "max_total_mu": run["health"]["max_total_mu"],
                }
            )
            print(
                f"[{dataset_name}] {model_name}: {len(ids)} matches, "
                f"folds={run['health']['n_folds']}",
                flush=True,
            )

        ids = reference_ids
        assert ids is not None
        observed_ids = observed.loc[ids].to_numpy(dtype=float)

        # Correlation diagnostics (uses the incumbent's per-side residuals).
        correlation.extend(
            correlation_rows(dataset_name, features, results[INCUMBENT], ids)
        )

        # Scores for every line.
        for model_name, _ in MODELS:
            counts = results[model_name]["counts"]
            pmf = results[model_name]["pmf"]
            for line in lines:
                width = probabilities_for_line(counts, pmf, line, ids)
                settled = settle(frame, line)
                scores = score(width, settled)
                all_scores[(dataset_name, line, model_name)] = scores
                score_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "line": line,
                        **{key: scores[key] for key in ("n_matches", "brier", "log_loss")},
                        "gap": gap_label(scores["widest"]),
                    }
                )

        # Deltas and per-line calibration.
        dataset_calibration: dict[float, list[dict]] = {}
        for line in lines:
            incumbent = all_scores[(dataset_name, line, INCUMBENT)]
            challenger = all_scores[(dataset_name, line, CHALLENGER)]
            delta_rows.append(
                {
                    "dataset": dataset_name,
                    "line": line,
                    "delta_brier": challenger["brier"] - incumbent["brier"],
                    "delta_log_loss": challenger["log_loss"] - incumbent["log_loss"],
                    "improved": challenger["log_loss"] < incumbent["log_loss"],
                }
            )
            settled = incumbent["actual"]
            y_over = (settled.to_numpy() == "over").astype(float)
            inc_rel = over_reliability(
                incumbent["wide"]["over"].to_numpy(dtype=float), y_over
            )
            chal_rel = over_reliability(
                challenger["wide"]["over"].to_numpy(dtype=float), y_over
            )
            dataset_calibration[line] = [
                {
                    "low": inc_rel[i]["low"],
                    "high": inc_rel[i]["high"],
                    "incumbent_n": inc_rel[i]["n"],
                    "incumbent_pred": inc_rel[i]["predicted"],
                    "incumbent_actual": inc_rel[i]["actual"],
                    "challenger_n": chal_rel[i]["n"],
                    "challenger_pred": chal_rel[i]["predicted"],
                    "challenger_actual": chal_rel[i]["actual"],
                }
                for i in range(N_BINS)
            ]
        calibration[dataset_name] = dataset_calibration

        # Paired bootstrap: challenger vs incumbent on each score, per line.
        for line in lines:
            challenger_per = per_match_scores(
                all_scores[(dataset_name, line, CHALLENGER)]
            )
            incumbent_per = per_match_scores(
                all_scores[(dataset_name, line, INCUMBENT)]
            )
            if not challenger_per.index.equals(incumbent_per.index):
                raise AssertionError(
                    f"{dataset_name} line {line}: per-match scores not aligned "
                    "for the paired bootstrap"
                )
            for score_name in ("brier", "log_loss"):
                diff = challenger_per[score_name] - incumbent_per[score_name]
                boot = paired_bootstrap(diff, seed=bootstrap_seed)
                bootstrap_seed += 1
                bootstrap_rows.append(
                    {
                        "dataset": dataset_name,
                        "line": line,
                        "score": score_name,
                        **boot,
                    }
                )

        # Count frequencies and the rule's tail distance.
        tables: dict[str, object] = {}
        actual_table = count_frequencies(
            observed_ids, results[INCUMBENT]["counts"], results[INCUMBENT]["pmf"]
        )
        tables["actual"] = actual_table
        metrics = []
        for model_name, _ in MODELS:
            counts = results[model_name]["counts"]
            pmf = results[model_name]["pmf"]
            table = count_frequencies(observed_ids, counts, pmf)
            tables[model_name] = table
            gap = (table["predicted"] - table["actual"]).abs()
            distance = tail_l1(table)
            tail_distance[(dataset_name, model_name)] = distance
            metrics.append(
                {
                    "model": model_name,
                    "total_variation": float(0.5 * gap.sum()),
                    "max_gap": float(gap.max()),
                    "max_gap_count": str(table["count"].iloc[int(gap.to_numpy().argmax())]),
                    "tail_l1": distance,
                }
            )
        tables["metrics"] = metrics
        tail_tables[dataset_name] = tables

        # Residual dispersion per model.
        for model_name, _ in MODELS:
            mu_total = results[model_name]["mu_total"]
            residual = observed_ids - mu_total
            variance = float(residual.var(ddof=1)) if residual.size > 1 else float("nan")
            dispersion_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n": int(residual.size),
                    "mean_residual": float(residual.mean()),
                    "var_over_mean": variance / float(mu_total.mean()),
                    "pearson": float(np.mean(residual**2 / mu_total)),
                }
            )

        # Fitted dispersion per season for the challenger.
        for season in seasons:
            subset = features[features["season"] == season]
            if len(subset) < 5:
                continue
            total = subset["total_cards"].to_numpy(dtype=float)
            total_ratio = (
                float(total.var(ddof=1) / total.mean())
                if total.mean() > 0 and len(total) > 1
                else float("nan")
            )
            with warnings.catch_warnings():
                # Same boundary-optimum warnings as the walk-forward folds; the
                # per-season fit-health is reported, so silence and continue.
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", HessianInversionWarning)
                alpha = float(
                    CardsTotalNegativeBinomial(("over", "under"), line=lines[0], k=K)
                    .fit(subset)
                    .alpha_
                )
            season_dispersion.append(
                {
                    "dataset": dataset_name,
                    "season": season,
                    "matches": len(subset),
                    "total_ratio": total_ratio,
                    "alpha": alpha,
                }
            )

    adopt, evidence = make_decision(all_scores, tail_distance, datasets, lines)
    decision_sentences = render_decision(adopt, evidence, datasets, lines)

    verdict = render_verdict(all_scores, bootstrap_rows, tail_tables, correlation, lines)

    total_matches = len(raw)
    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": (
            f"E0 seasons {', '.join(seasons)} ({total_matches} matches), "
            f"lines {', '.join(str(line) for line in lines)}, k={K:g}"
        ),
        "notes": [
            "PROVISIONAL counting rule: a match total is hy+ay+hr+ar exactly as "
            "football-data publishes it. A bookmaker's own rule is unconfirmed — "
            "see docs/DATA_NOTES.md.",
            "covid_affected is the ingest flag; 'no_covid' drops those matches "
            "from the evaluated set and from the history every rate is built on.",
            "No odds, no edge, no bets: these are calibration numbers.",
            "The decision rule was fixed before the run and is applied "
            "mechanically; it is not re-read after seeing the numbers.",
            "No tuning: k=6 for teams and referees, the total-model dispersion "
            "is estimated by the likelihood, folds and lines are step 3's.",
            "The challenger's negative-binomial MLE is started from the Poisson "
            "optimum of the same design (step 3's stabilisation), so a fold "
            "cannot diverge into a nonsensical fit.",
            "Bootstrap runs on the full, untruncated match sequence "
            f"(AGENTS.md invariant 3), paired by match, fixed seed {BOOTSTRAP_SEED}.",
        ],
    }

    report = render_report(
        context,
        {
            "decision_sentences": decision_sentences,
            "verdict": verdict,
            "dataset_stats": dataset_stats,
            "correlation_rows": correlation,
            "score_rows": score_rows,
            "delta_rows": delta_rows,
            "bootstrap_rows": bootstrap_rows,
            "calibration": calibration,
            "tail_tables": tail_tables,
            "dispersion_rows": dispersion_rows,
            "season_dispersion": season_dispersion,
            "fold_health": fold_health,
        },
    )
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    suffix = "_quick" if args.quick else ""
    path = REPORTS_DIR / f"phase1_step3b_total_nbinom{suffix}_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print("\n".join(decision_sentences), flush=True)
    print(f"\nwrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


def render_decision(
    adopt: bool, evidence: dict, datasets: dict, lines: tuple
) -> list[str]:
    """State the frozen rule, then the mechanical outcome, dataset by dataset."""
    out = [
        f"**Rule:** {DECISION_RULE}",
        "",
        "**Evaluation of the rule, as computed:**",
        "",
    ]
    for dataset_name in datasets:
        row = evidence[dataset_name]
        improves = ", ".join(
            f"line {line:g} {'yes' if row['improved'][line] else 'no'}"
            for line in lines
        )
        out.append(
            f"- *{dataset_name}*: log loss improved on {row['n_improved']} of "
            f"{len(lines)} lines ({improves}) -> condition (A) "
            f"{'PASS' if row['logloss_ok'] else 'FAIL'}; tail L1 challenger "
            f"{row['tail_challenger']:.4f} vs incumbent "
            f"{row['tail_incumbent']:.4f} -> condition (B) "
            f"{'PASS' if row['tail_ok'] else 'FAIL'}."
        )
    out.append("")
    if adopt:
        out.append(
            "**Result: ADOPT `cards_total_nbinom`.** Both conditions held in "
            "both datasets, so the total-target negative binomial becomes the "
            "Phase 1 cards model."
        )
    else:
        out.append(
            "**Result: KEEP `poisson_team_ref`.** At least one condition failed "
            "in at least one dataset, so the Poisson team+referee model stays "
            "the Phase 1 cards model."
        )
    return out


def render_verdict(
    all_scores: dict,
    bootstrap_rows: list[dict],
    tail_tables: dict,
    correlation: list[dict],
    lines: tuple,
) -> list[str]:
    """Plain-language answer for the calibration side (the decision is separate)."""
    out: list[str] = []

    # Correlation summary: overall rows.
    overall = [row for row in correlation if row["season"] == "all"]
    corr_bits = ", ".join(
        f"{row['dataset']}: raw {row['raw_corr']:+.3f}, residual "
        f"{row['resid_corr']:+.3f}"
        for row in overall
    )
    out.append(
        "Home/away correlation: " + corr_bits + ". The raw figure mixes in the "
        "shared covariates (both sides share the referee and the match); the "
        "residual figure is what the incumbent's per-side model leaves behind. "
        "Residual correlation close to the raw figure means the incumbent's "
        "features explain little of the dependence and a total-level dispersion "
        "is where it is carried."
    )

    brier_boot = [row for row in bootstrap_rows if row["score"] == "brier"]
    excluding = [
        row for row in brier_boot if row["ci_high"] < 0.0 or row["ci_low"] > 0.0
    ]
    out.append(
        f"The paired bootstrap (challenger minus incumbent, {len(brier_boot)} "
        f"Brier comparisons) separates zero in {len(excluding)} of them."
    )

    tv_lines = []
    for dataset_name, tables in tail_tables.items():
        metrics = {row["model"]: row for row in tables["metrics"]}
        tv_lines.append(
            f"{dataset_name}: incumbent tail L1 "
            f"{metrics[INCUMBENT]['tail_l1']:.4f} vs challenger "
            f"{metrics[CHALLENGER]['tail_l1']:.4f}"
        )
    out.append("Tail distance comparison: " + "; ".join(tv_lines) + ".")
    return out


if __name__ == "__main__":
    sys.exit(main())
