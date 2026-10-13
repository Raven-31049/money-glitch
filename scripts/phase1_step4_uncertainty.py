#!/usr/bin/env python
"""Phase 1 step 4 — uncertainty for the adopted cards model.

Run::

    uv run python scripts/phase1_step4_uncertainty.py

Step 4 does three things and nothing else:

1. It loads the adopted model from ``configs/phase1_cards.yaml`` — the single
   source every later step reads — rather than re-stating k, the line or the
   recency half-life here. The config's ``model.params.half_life_days`` is a
   *feature-build* parameter (it belongs to add_card_features /
   add_referee_feature, not to the model constructor), so the loader pops it and
   passes it to the builders before ``create_model`` sees the rest.
2. It runs one walk-forward of that model and, at every fold, draws probability
   samples with :meth:`CardsPoisson.predict_samples` — coefficient uncertainty
   (N(beta, cov)) and the team/referee rate uncertainty (Gamma posteriors)
   together, in one fit. Percentiles of those draws are marginals and are NEVER
   renormalised to sum to 1 (AGENTS.md rule 5): an uncertain match can have a
   small 10th percentile for over *and* for under at once.
3. It reports width/calibration/timing diagnostics and the adopted model's
   per-line P(over) reliability tables, on 1819-2324 in both datasets.

HOLDOUT (AGENTS.md rule 9). Seasons 2425 and 2526 are the final test set. This
script loads the seasons the config names — development seasons only — and
refuses a holdout season by name. Nothing here reports 2425-2526.

DIAGNOSTIC ONLY. No model, feature or threshold is changed, and no percentile is
chosen: step 4 measures what the uncertainty *is*, it does not adopt a rule that
consumes it.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
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
from bet_engine.eval.calibration import log_loss
from bet_engine.eval.uncertainty import interval_width, marginal_percentiles
from bet_engine.models import (
    SAMPLE_DRAWS,
    SAMPLE_LINES,
    SAMPLE_SEED,
    add_card_features,
    add_referee_feature,
    create_model,
    total_pmf,
)

#: The final test set. Never loaded here — see the holdout rule in AGENTS.md.
HOLDOUT_SEASONS: tuple[str, ...] = ("2425", "2526")

#: Where the adopted model lives. Every later step reads this file, not a copy.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "phase1_cards.yaml"

#: Walk-forward cadence, unchanged from steps 1-3c and read from the config.
#: (Kept as module constants only for the report header.)

#: The lines P(over) is reported at. The config's own line must be one of them.
LINES: tuple[float, ...] = SAMPLE_LINES

#: A group is "thin" when the entity behind a rate has fewer than this many
#: effective prior matches. Same threshold steps 2-3c used for rare referees.
RARE_HISTORY = 10

#: A reliability bin is quoted only from this many matches — the step-3b rule.
MIN_RELIABILITY_BIN = 50

#: The two marginal percentiles whose gap is the quoted "width" of a match.
WIDTH_LOW = 10.0
WIDTH_HIGH = 90.0

#: Number of width-quantile bands in the calibration-by-width table.
WIDTH_BANDS = 5

#: Season-month ordering: August is month 1, May is month 10.
_MONTH_TO_SEASON_INDEX: dict[int, int] = {
    8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
}
_SEASON_INDEX_TO_MONTH: dict[int, int] = {
    index: month for month, index in _MONTH_TO_SEASON_INDEX.items()
}
_MONTH_NAMES: dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# config load (the adopted model is not re-stated here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdoptedModel:
    """The resolved contents of ``configs/phase1_cards.yaml``.

    ``params`` is what reaches the model constructor; ``half_life_days`` and
    ``lines`` have been split out because they are not constructor arguments —
    the split the config's own comments describe. ``half_life_days`` is the
    feature-build recency parameter (or None for the cumulative default);
    ``lines`` is the quoted over/under set the reports are read at, and ``line``
    is the scalar the constructor requires (the middle of ``lines``). In Part B
    the line comes from the odds data per match, not from this file.
    """

    config: VariantConfig
    model_type: str
    params: dict[str, Any]
    half_life_days: float | None
    line: float
    lines: tuple[float, ...]


def load_adopted_model(path: Path = CONFIG_PATH) -> AdoptedModel:
    """Read the adopted variant and separate its feature/line params from params.

    Two config keys are not constructor arguments and are popped here: the
    feature-build ``half_life_days`` (handed to the feature builders) and the
    report ``lines`` (the quoted set). The constructor still needs a scalar
    ``line``, so it is taken from the config if present and otherwise derived as
    the middle of ``lines`` — the same line the model's ``predict()`` settles at.
    """
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
            f"config line {line:g} is not one of the quoted lines {lines}; "
            "the reliability tables would not cover the adopted line"
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


def config_market_line(config: VariantConfig) -> float:
    """The line named by ``market: cards_over_<line>``; a convenience fallback."""
    market = config.market
    prefix = "cards_over_"
    if not market.startswith(prefix):
        raise ValueError(
            f"market {market!r} is not cards_over_<line>; cannot infer the line"
        )
    return float(market[len(prefix):])


def build_features(frame: pd.DataFrame, adopted: AdoptedModel, k: float) -> pd.DataFrame:
    """The model's input columns, built with the adopted recency half-life."""
    with_referee = add_referee_feature(
        add_card_features(frame, k, half_life_days=adopted.half_life_days),
        k,
        half_life_days=adopted.half_life_days,
    )
    return with_referee


# ---------------------------------------------------------------------------
# walk-forward capture with sampling
# ---------------------------------------------------------------------------


@dataclass
class SamplingRecorder:
    """Accumulates per-fold sample blocks, sides and timing across a run."""

    draws: int
    outcomes: tuple[str, ...]
    lines: tuple[float, ...]
    sample_blocks: list[np.ndarray] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    sides: dict[str, tuple[float, float]] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    folds: int = 0

    def add(
        self,
        batch: pd.DataFrame,
        samples: np.ndarray,
        sides: dict[str, tuple[float, float]],
        totals: dict[str, float],
        elapsed: float,
    ) -> None:
        batch_ids = [str(value) for value in batch["match_id"]]
        self.sample_blocks.append(np.asarray(samples, dtype=float))
        self.ids.extend(batch_ids)
        self.sides.update(sides)
        self.totals.update(totals)
        self.seconds += float(elapsed)
        self.folds += 1

    def stacked(self) -> np.ndarray:
        """All folds' draws stacked on the match axis: ``(n, N, lines, outcomes)``."""
        if not self.sample_blocks:
            raise ValueError("no sample blocks recorded; run the walk-forward first")
        return np.concatenate(self.sample_blocks, axis=1)


class _SamplingModel:
    """Wraps the adopted model so walk_forward also draws samples each fold.

    It is a MarketModel (fit + predict) so the engine accepts it; predict()
    returns exactly the inner model's point output, and additionally records
    ``predict_samples`` for the fold's batch and the model's captured per-side
    means. The two are the same fit — sampling never refits anything.
    """

    def __init__(
        self,
        inner: Any,
        recorder: SamplingRecorder,
        seed: int,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._seed = seed

    def fit(self, train_df: pd.DataFrame) -> "_SamplingModel":
        self._inner.fit(train_df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prediction = self._inner.predict(df)
        fold_seed = _fold_seed(self._seed, df["date"].min())
        started = time.perf_counter()
        samples = self._inner.predict_samples(
            df, self._recorder.draws, lines=self._recorder.lines, seed=fold_seed
        )
        elapsed = time.perf_counter() - started
        ids = {str(value) for value in df["match_id"]}
        sides = {
            key: value
            for key, value in getattr(self._inner, "predicted_sides", {}).items()
            if key in ids
        }
        totals = {
            key: value
            for key, value in getattr(self._inner, "predicted_totals", {}).items()
            if key in ids
        }
        self._recorder.add(df, samples, sides, totals, elapsed)
        return prediction


def run_walk_forward(
    features: pd.DataFrame, adopted: AdoptedModel, draws: int, seed: int
) -> tuple[pd.DataFrame, SamplingRecorder]:
    """One walk-forward of the adopted model, capturing samples per fold."""
    outcomes = ("over", "under")
    recorder = SamplingRecorder(draws=draws, outcomes=outcomes, lines=LINES)
    backtest = adopted.config.backtest

    def factory() -> _SamplingModel:
        inner = create_model(adopted.model_type, adopted.params, outcomes)
        return _SamplingModel(inner, recorder, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        prediction = walk_forward(
            features,
            factory,
            min_train_days=backtest.min_train_days,
            refit_every_days=backtest.refit_every_days,
        )
    # Each fold's draws are seeded from the base seed and the fold's day, so the
    # run is deterministic but folds do not share identical random draws.
    return prediction, recorder


def _fold_seed(base: int, day: pd.Timestamp) -> int:
    """A reproducible, fold-varying seed: the base plus the day's ordinal."""
    return int(base) + int(pd.Timestamp(day).toordinal())


# ---------------------------------------------------------------------------
# per-match analysis frame
# ---------------------------------------------------------------------------


def point_over_per_line(
    recorder: SamplingRecorder, ids: list[str]
) -> np.ndarray:
    """Point P(over) at every reported line, from the captured per-side means.

    The side model is Poisson (alpha=0), so the match total is the convolution
    of the two captured per-side means. Rebuilding each line's P(over) from the
    *same* sides the point output came from is what keeps a line's reliability
    readable at the line it claims (step 3c's rule), rather than reusing one
    line's probabilities for all of them.
    """
    record = np.array([recorder.sides[match_id] for match_id in ids], dtype=float)
    counts, pmf = total_pmf(record[:, 0], record[:, 1], 0.0)
    return np.column_stack(
        [pmf[:, counts > line].sum(axis=1) for line in recorder.lines]
    )


def build_match_table(
    features: pd.DataFrame, recorder: SamplingRecorder
) -> pd.DataFrame:
    """Per-scored-match diagnostics: point, sample mean, width and history.

    Only matches the walk-forward actually scored are kept (the recorder has one
    sample block per scored fold), so every width is out-of-sample — never over
    a row the model trained on.
    """
    ids = list(recorder.ids)
    frame = features.set_index(features["match_id"].astype(str)).loc[ids]
    outcomes = list(recorder.outcomes)
    over_index = outcomes.index("over")

    samples = recorder.stacked()  # (n, N, lines, outcomes)
    n_draws = samples.shape[0]
    if samples.shape[1] != len(ids):
        raise AssertionError(
            f"sample blocks cover {samples.shape[1]} matches, ids list has "
            f"{len(ids)}"
        )
    # Marginals per outcome; never renormalised (invariant 5).
    margins = marginal_percentiles(samples, [WIDTH_LOW, 50.0, WIDTH_HIGH])
    low = margins[0, :, :, over_index]     # (N, lines)
    median = margins[1, :, :, over_index]
    high = margins[2, :, :, over_index]
    widths = interval_width(samples, low=WIDTH_LOW, high=WIDTH_HIGH)[..., over_index]
    sample_mean = samples[:, :, :, over_index].mean(axis=0)

    points = point_over_per_line(recorder, ids)

    data: dict[str, Any] = {
        "match_id": ids,
        "date": frame["date"].to_numpy(),
        "season": frame["season"].to_numpy(),
        "mu": [recorder.totals[match_id] for match_id in ids],
        "actual_total": frame["total_cards"].to_numpy(dtype=float),
        "ref_prior": frame["ref_past_count"].to_numpy(dtype=float),
        "home_prior": frame["home_past_matches"].to_numpy(dtype=float),
        "away_prior": frame["away_past_matches"].to_numpy(dtype=float),
        "n_draws": n_draws,
    }
    for position, line in enumerate(recorder.lines):
        tag = _line_tag(line)
        data[f"point_{tag}"] = points[:, position]
        data[f"mean_{tag}"] = sample_mean[:, position]
        data[f"low_{tag}"] = low[:, position]
        data[f"median_{tag}"] = median[:, position]
        data[f"high_{tag}"] = high[:, position]
        data[f"width_{tag}"] = widths[:, position]
    table = pd.DataFrame(data)

    required = ["mu", "actual_total", "ref_prior", "home_prior", "away_prior"]
    if table[required].isna().any().any():
        raise ValueError("match table has NaNs in required columns")
    table["team_prior_min"] = table[["home_prior", "away_prior"]].min(axis=1)
    table["ref_rare"] = table["ref_prior"] < RARE_HISTORY
    table["team_rare"] = table["team_prior_min"] < RARE_HISTORY
    table["month"] = table["date"].dt.month
    table["season_month"] = table["month"].map(_MONTH_TO_SEASON_INDEX)
    table["month_name"] = table["month"].map(_MONTH_NAMES)
    # The model's own draw-level identity must hold: every draw is a distribution.
    if not np.allclose(samples.sum(axis=-1), 1.0):
        raise AssertionError("a draw does not sum to 1 across outcomes")
    return table


def _line_tag(line: float) -> str:
    return f"{line:g}".replace(".", "_")


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def width_summary(table: pd.DataFrame, lines: tuple[float, ...]) -> list[dict]:
    """Distribution of the [10, 90] interval width per line."""
    rows = []
    for line in lines:
        width = table[f"width_{_line_tag(line)}"].to_numpy(dtype=float)
        rows.append(
            {
                "line": line,
                "n": int(len(width)),
                "mean": float(width.mean()),
                "p05": float(np.percentile(width, 5)),
                "p25": float(np.percentile(width, 25)),
                "p50": float(np.percentile(width, 50)),
                "p75": float(np.percentile(width, 75)),
                "p95": float(np.percentile(width, 95)),
                "max": float(width.max()),
            }
        )
    return rows


def mean_vs_point(table: pd.DataFrame, lines: tuple[float, ...]) -> list[dict]:
    """Monte-Carlo sanity: the sample mean must track the point estimate.

    The mean of the draws is a Monte-Carlo estimate of the model's own posterior-
    mean P(over); it differs from the point estimate only by the Jensen gap the
    drawn uncertainty introduces. A large gap would mean the draws are not
    centred on the fit and every percentile below is suspect.
    """
    rows = []
    for line in lines:
        tag = _line_tag(line)
        point = table[f"point_{tag}"].to_numpy(dtype=float)
        mean = table[f"mean_{tag}"].to_numpy(dtype=float)
        rows.append(
            {
                "line": line,
                "mean_point": float(point.mean()),
                "mean_sample": float(mean.mean()),
                "mean_gap": float(mean.mean() - point.mean()),
                "max_abs_gap": float(np.abs(mean - point).max()),
                "mean_abs_gap": float(np.abs(mean - point).mean()),
            }
        )
    return rows


def _width_summary_row(label: str, sub: pd.DataFrame, lines: tuple[float, ...]) -> dict:
    row = {"group": label, "n": int(len(sub)), "ref_prior_mean": float(sub["ref_prior"].mean())}
    for line in lines:
        row[f"width_{_line_tag(line)}"] = float(
            sub[f"width_{_line_tag(line)}"].mean()
        )
    return row


def width_drivers(table: pd.DataFrame, lines: tuple[float, ...]) -> list[dict]:
    """Mean width by the thin-history and calendar groups a widen-expectation has."""
    rows = [
        _width_summary_row("all matches", table, lines),
        _width_summary_row("rare referee (<10 prior)", table[table["ref_rare"]], lines),
        _width_summary_row("known referee (>=10 prior)", table[~table["ref_rare"]], lines),
        _width_summary_row("low-history team (<10)", table[table["team_rare"]], lines),
        _width_summary_row("known team (>=10)", table[~table["team_rare"]], lines),
    ]
    for index in sorted(_SEASON_INDEX_TO_MONTH):
        month = _SEASON_INDEX_TO_MONTH[index]
        sub = table[table["season_month"] == index]
        if sub.empty:
            continue
        rows.append(_width_summary_row(f"month {_MONTH_NAMES[month]}", sub, lines))
    return rows


def _driver_correlation(a: pd.Series, b: pd.Series) -> float:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    if x.size < 3 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def width_history_correlations(table: pd.DataFrame) -> dict[str, float]:
    """Correlation of the line-4.5 width with the effective prior counts.

    A negative correlation is the expected direction: less history, wider
    interval. Reported as a hint at one line, not a law.
    """
    tag = _line_tag(4.5)
    return {
        "ref_prior": _driver_correlation(table["ref_prior"], table[f"width_{tag}"]),
        "team_prior_min": _driver_correlation(
            table["team_prior_min"], table[f"width_{tag}"]
        ),
    }


def width_band_calibration(
    table: pd.DataFrame, line: float, bands: int
) -> list[dict]:
    """Sort matches into width bands; is the wider half also less calibrated?

    For the line in question: per band, the mean width, the mean point and
    sample-mean prediction, the actual over rate, the calibration gap
    (actual - predicted) and the mean absolute error of the sample mean. If the
    model's uncertainty is meaningful, wider bands should carry larger error —
    this measures whether that holds, it does not assume it.
    """
    tag = _line_tag(line)
    width = table[f"width_{tag}"]
    band = pd.qcut(width, bands, labels=False, duplicates="drop")
    actual = (table["actual_total"].to_numpy(dtype=float) > line).astype(float)
    rows = []
    for position in sorted(pd.unique(band.dropna())):
        mask = (band == position).to_numpy()
        sub = table[mask]
        predicted = sub[f"mean_{tag}"].to_numpy(dtype=float)
        point = sub[f"point_{tag}"].to_numpy(dtype=float)
        y = actual[mask]
        rows.append(
            {
                "band": int(position) + 1,
                "n": int(mask.sum()),
                "width_low": float(sub[f"low_{tag}"].mean()),
                "width_high": float(sub[f"high_{tag}"].mean()),
                "mean_width": float(sub[f"width_{tag}"].mean()),
                "mean_point": float(point.mean()),
                "mean_sample": float(predicted.mean()),
                "actual": float(y.mean()),
                "gap": float(y.mean() - predicted.mean()),
                "mae": float(np.abs(predicted - y).mean()),
            }
        )
    return rows


def over_reliability(p_over: np.ndarray, y_over: np.ndarray) -> list[dict]:
    """Bin one line's point P(over) and compare mean prediction with frequency.

    Same decile-of-probability binning step 3b/3c used, so the adopted model's
    corrected per-line tables are directly comparable with the incumbent's.
    """
    p = np.asarray(p_over, dtype=float)
    y = np.asarray(y_over, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    rows = []
    for position in range(10):
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


def reliability_tables(table: pd.DataFrame, lines: tuple[float, ...]) -> dict[float, list[dict]]:
    """The adopted model's per-line P(over) reliability, each line its own pmf."""
    out: dict[float, list[dict]] = {}
    for line in lines:
        tag = _line_tag(line)
        p_over = table[f"point_{tag}"].to_numpy(dtype=float)
        y_over = (table["actual_total"].to_numpy(dtype=float) > line).astype(float)
        out[line] = over_reliability(p_over, y_over)
    return out


def line_log_losses(table: pd.DataFrame, lines: tuple[float, ...]) -> dict[float, float]:
    """Point log loss at each line from that line's own point probabilities."""
    out: dict[float, float] = {}
    for line in lines:
        tag = _line_tag(line)
        wide = pd.DataFrame(
            {
                "over": table[f"point_{tag}"].to_numpy(dtype=float),
                "under": 1.0 - table[f"point_{tag}"].to_numpy(dtype=float),
            }
        )
        settled = pd.Series(
            np.where(table["actual_total"].to_numpy(dtype=float) > line, "over", "under")
        )
        out[line] = float(log_loss(wide, settled))
    return out


def largest_gap_bin(rows: list[dict]) -> dict | None:
    """The populated bin whose |actual - predicted| is largest, or None."""
    populated = [
        row
        for row in rows
        if row["n"] >= MIN_RELIABILITY_BIN and np.isfinite(row["predicted"])
    ]
    if not populated:
        return None
    return max(populated, key=lambda row: abs(row["actual"] - row["predicted"]))


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------


def _direction(gap: float) -> str:
    if gap > 0.02:
        return f"under-predicts by {gap:.2f}"
    if gap < -0.02:
        return f"over-predicts by {abs(gap):.2f}"
    return f"sits about right ({gap:+.2f})"


def findings_sentences(
    tables: dict[str, pd.DataFrame],
    widths: dict[str, list[dict]],
    drivers: dict[str, list[dict]],
    correlations: dict[str, dict[str, float]],
    bands: dict[str, list[dict]],
    reliabilities: dict[str, dict[float, list[dict]]],
    loglosses: dict[str, dict[float, float]],
    lines: tuple[float, ...],
    draws: int,
) -> list[str]:
    out: list[str] = []
    out.append(
        f"**Sampling.** All numbers below come from one walk-forward of the "
        f"adopted model, {draws} draws per match per fold, coefficient and rate "
        "uncertainty together. Percentiles are marginals: the 10th percentile of "
        "P(over) and of P(under) are read independently and are not renormalised "
        "to sum to 1 (AGENTS.md rule 5)."
    )
    for dataset, rows in widths.items():
        by_line = {row["line"]: row for row in rows}
        mid = by_line[4.5]
        out.append(
            f"**{dataset} — width.** The [10, 90] interval width of P(over) at "
            f"line 4.5 averages {mid['mean']:.3f} (median {mid['p50']:.3f}, 5th-95th "
            f"percentile {mid['p05']:.3f}-{mid['p95']:.3f}); at 3.5 it averages "
            f"{by_line[3.5]['mean']:.3f} and at 5.5 {by_line[5.5]['mean']:.3f}."
        )
        driver_rows = {row["group"]: row for row in drivers[dataset]}
        rare = driver_rows["rare referee (<10 prior)"]
        known = driver_rows["known referee (>=10 prior)"]
        team_rare = driver_rows["low-history team (<10)"]
        team_known = driver_rows["known team (>=10)"]
        out.append(
            f"**{dataset} — width drivers.** Mean line-4.5 width is "
            f"{rare['width_4_5']:.3f} for a rare referee vs {known['width_4_5']:.3f} "
            f"for a known one, and {team_rare['width_4_5']:.3f} for a low-history "
            f"team vs {team_known['width_4_5']:.3f} for a known one. The width's "
            f"correlation with the referee's effective prior count is "
            f"{correlations[dataset]['ref_prior']:+.2f} and with the thinner team's "
            f"{correlations[dataset]['team_prior_min']:+.2f} (negative is the "
            "expected direction: less history, wider interval)."
        )
        band_rows = bands[dataset]
        widest = band_rows[-1]
        narrowest = band_rows[0]
        out.append(
            f"**{dataset} — width bands (line 4.5).** The narrowest band "
            f"(width ~{narrowest['mean_width']:.3f}) has mean |sample - actual| "
            f"{narrowest['mae']:.3f} and calibration gap {narrowest['gap']:+.3f}; "
            f"the widest band (width ~{widest['mean_width']:.3f}) has MAE "
            f"{widest['mae']:.3f} and gap {widest['gap']:+.3f}. "
            + (
                "Wider intervals carry more error, so the uncertainty is tracking "
                "where the model is less sure."
                if widest["mae"] > narrowest["mae"]
                else "Wider intervals do NOT carry more error here — the interval "
                "is not a difficulty signal at this line."
            )
        )
        bits = []
        for line in lines:
            row = largest_gap_bin(reliabilities[dataset][line])
            if row is None:
                bits.append(f"line {line:g}: no bin with n >= {MIN_RELIABILITY_BIN}")
                continue
            gap = row["actual"] - row["predicted"]
            bits.append(
                f"line {line:g}: largest gap is bin {row['low']:.1f}-{row['high']:.1f} "
                f"(n={row['n']}) predicting {row['predicted']:.2f} vs "
                f"{row['actual']:.2f} actual ({_direction(gap)}); point log loss "
                f"{loglosses[dataset][line]:.4f}"
            )
        out.append(f"**{dataset} — reliability on 1819-2324 only.** " + "; ".join(bits) + ".")
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_report(payload: dict) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 Step 4 — uncertainty for the adopted cards model")
    add("")
    add(f"- Generated: {payload['generated']}")
    add(
        f"- Model: `{payload['model_type']}` loaded from "
        f"`configs/phase1_cards.yaml` (the single source every later step reads). "
        f"k={payload['params'].get('k')}, line={payload['line']:g}, "
        f"half_life_days={payload['half_life_days']}."
    )
    add(
        "- Sampling: `predict_samples` — coefficient draws N(beta, cov) AND "
        "team/referee rate draws from their Gamma posteriors, in the same fit. "
        f"{payload['draws']} draws per match per fold."
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
        "each fold day). Sampling accumulated across the whole run."
    )
    add(
        "- DIAGNOSTIC ONLY: no model, feature, threshold or percentile is changed."
    )
    add("")

    add("## Findings")
    add("")
    for sentence in payload["findings"]:
        add(sentence)
        add("")

    add("## Timing")
    add("")
    add(
        f"Sampling time across the full walk-forward: "
        f"**{payload['timing']['total_seconds']:.1f}s** over "
        f"{payload['timing']['folds']} folds and "
        f"{payload['timing']['matches']} scored matches "
        f"({payload['timing']['ms_per_match']:.1f} ms/match, "
        f"{payload['timing']['ms_per_fold']:.1f} ms/fold). Point prediction time is "
        "not counted here; this is the marginal cost of the uncertainty draws."
    )
    add("")

    add("## Mean vs point (Monte-Carlo sanity)")
    add("")
    add(
        "The mean of the draws is a Monte-Carlo estimate of the model's own "
        "posterior-mean P(over) per match. A large gap would mean the draws are "
        "not centred on the fit."
    )
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        add("| line | mean point | mean of sample means | mean gap | mean abs gap | max abs gap |")
        add("|---|---:|---:|---:|---:|---:|")
        for row in payload["mean_vs_point"][dataset]:
            add(
                f"| {row['line']:g} | {row['mean_point']:.4f} | "
                f"{row['mean_sample']:.4f} | {row['mean_gap']:+.4f} | "
                f"{row['mean_abs_gap']:.4f} | {row['max_abs_gap']:.4f} |"
            )
        add("")

    add("## Interval width")
    add("")
    add(
        "The [10, 90] marginal interval width of P(over) per line, across scored "
        "matches. Percentiles are per outcome and not renormalised (invariant 5)."
    )
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        add("| line | n | mean | p05 | p25 | median | p75 | p95 | max |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in payload["widths"][dataset]:
            add(
                f"| {row['line']:g} | {row['n']} | {row['mean']:.3f} | "
                f"{row['p05']:.3f} | {row['p25']:.3f} | {row['p50']:.3f} | "
                f"{row['p75']:.3f} | {row['p95']:.3f} | {row['max']:.3f} |"
            )
        add("")

    add("## Width drivers")
    add("")
    add("Mean [10, 90] width of P(over) by group.")
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        add("| group | n | mean ref prior | width 3.5 | width 4.5 | width 5.5 |")
        add("|---|---:|---:|---:|---:|---:|")
        for row in payload["drivers"][dataset]:
            add(
                f"| {row['group']} | {row['n']} | {row['ref_prior_mean']:.1f} | "
                f"{row['width_3_5']:.3f} | {row['width_4_5']:.3f} | "
                f"{row['width_5_5']:.3f} |"
            )
        add("")

    add("## Calibration by width band (line 4.5)")
    add("")
    add(
        "Matches sorted by their own interval width into equal-count bands. "
        "`mae` is mean |sample-mean P(over) - went over|; `gap` is actual over "
        "rate minus mean sample-mean prediction."
    )
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        add("| band | n | mean width | interval low | interval high | mean point | mean sample | actual | gap | mae |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in payload["bands"][dataset]:
            add(
                f"| {row['band']} | {row['n']} | {row['mean_width']:.3f} | "
                f"{row['width_low']:.3f} | {row['width_high']:.3f} | "
                f"{row['mean_point']:.3f} | {row['mean_sample']:.3f} | "
                f"{row['actual']:.3f} | {row['gap']:+.3f} | {row['mae']:.3f} |"
            )
        add("")

    add("## P(over) reliability on 1819-2324 only")
    add("")
    add(
        "The adopted model's per-line P(over) reliability. Each match is binned "
        "by its predicted P(over) at that line (from the per-side means at that "
        "line, so the three tables genuinely differ); `pred` is the mean "
        f"prediction in the bin, `act` the share that went over. Bins with fewer "
        f"than {MIN_RELIABILITY_BIN} matches are shown but not quoted."
    )
    add("")
    for dataset in payload["datasets"]:
        add(f"### {dataset}")
        add("")
        for line in payload["lines"]:
            add(f"line {line:g}")
            add("")
            add("| P(over) bin | n | pred | act |")
            add("|---|---:|---:|---:|")
            for row in payload["reliability"][dataset][line]:
                pred = "nan" if not np.isfinite(row["predicted"]) else f"{row['predicted']:.3f}"
                act = "nan" if not np.isfinite(row["actual"]) else f"{row['actual']:.3f}"
                add(f"| {row['low']:.1f}-{row['high']:.1f} | {row['n']} | {pred} | {act} |")
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
    parser.add_argument(
        "--draws",
        type=int,
        default=SAMPLE_DRAWS,
        help=f"probability draws per match per fold (default {SAMPLE_DRAWS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"base sampling seed (default {SAMPLE_SEED})",
    )
    args = parser.parse_args(argv)
    if args.draws < 1:
        raise ValueError(f"--draws must be >= 1, got {args.draws}")

    started = time.time()
    adopted = load_adopted_model()
    if adopted.lines != LINES:
        raise ValueError(
            f"config lines {adopted.lines} differ from the reported lines {LINES}; "
            "the config is the single source and this script must be read at those lines"
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

    tables: dict[str, pd.DataFrame] = {}
    width_rows: dict[str, list[dict]] = {}
    mean_rows: dict[str, list[dict]] = {}
    driver_rows: dict[str, list[dict]] = {}
    correlations: dict[str, dict[str, float]] = {}
    band_rows: dict[str, list[dict]] = {}
    reliabilities: dict[str, dict[float, list[dict]]] = {}
    loglosses: dict[str, dict[float, float]] = {}
    total_seconds = 0.0
    total_folds = 0
    total_scored = 0

    for dataset_name in datasets:
        frame = frames[dataset_name]
        features = build_features(frame, adopted, k)
        prediction, recorder = run_walk_forward(
            features, adopted, draws=args.draws, seed=args.seed
        )
        table = build_match_table(features, recorder)
        if len(table) != prediction["match_id"].nunique():
            raise AssertionError(
                "sample coverage does not match the walk-forward's scored matches"
            )

        tables[dataset_name] = table
        width_rows[dataset_name] = width_summary(table, LINES)
        mean_rows[dataset_name] = mean_vs_point(table, LINES)
        driver_rows[dataset_name] = width_drivers(table, LINES)
        correlations[dataset_name] = width_history_correlations(table)
        band_rows[dataset_name] = width_band_calibration(table, 4.5, WIDTH_BANDS)
        reliabilities[dataset_name] = reliability_tables(table, LINES)
        loglosses[dataset_name] = line_log_losses(table, LINES)

        total_seconds += recorder.seconds
        total_folds += recorder.folds
        total_scored += len(table)
        print(
            f"[{dataset_name}] {len(table)} scored matches over {recorder.folds} "
            f"folds; sampling {recorder.seconds:.1f}s; mean line-4.5 width "
            f"{width_rows[dataset_name][1]['mean']:.3f}",
            flush=True,
        )

    findings = findings_sentences(
        tables,
        width_rows,
        driver_rows,
        correlations,
        band_rows,
        reliabilities,
        loglosses,
        LINES,
        args.draws,
    )

    context = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model_type": adopted.model_type,
        "params": adopted.params,
        "line": adopted.line,
        "half_life_days": adopted.half_life_days,
        "draws": args.draws,
        "seed": args.seed,
        "seasons": seasons,
        "n_matches": len(raw),
        "min_train_days": adopted.config.backtest.min_train_days,
        "refit_every_days": adopted.config.backtest.refit_every_days,
        "datasets": datasets,
        "lines": LINES,
        "widths": width_rows,
        "mean_vs_point": mean_rows,
        "drivers": driver_rows,
        "correlations": correlations,
        "bands": band_rows,
        "reliability": reliabilities,
        "loglosses": loglosses,
        "timing": {
            "total_seconds": total_seconds,
            "folds": total_folds,
            "matches": total_scored,
            "ms_per_match": 1000.0 * total_seconds / max(total_scored, 1),
            "ms_per_fold": 1000.0 * total_seconds / max(total_folds, 1),
        },
        "findings": findings,
        "notes": [
            "The adopted model is loaded from `configs/phase1_cards.yaml`, not "
            "re-stated here: the config is the single source every later step "
            "reads. `model.params.half_life_days` is a feature-build parameter and "
            "is popped before create_model; the rest reach the constructor.",
            "Both uncertainty sources live in one fit: `predict_samples` draws the "
            "design coefficients from their fitted covariance and the team/referee "
            "rates from their Gamma posteriors. Nothing is refit for sampling.",
            "Percentiles are per-outcome marginals and are never renormalised "
            "(AGENTS.md rule 5); `marginal_percentiles` is the helper, and a test "
            "fails if that ever changes.",
            "Per-line P(over) is rebuilt from each fold's captured per-side means, "
            "so the reliability tables are read at the line they claim (step 3c's "
            "rule), not by reusing one line's probabilities.",
            "DIAGNOSTIC ONLY: no percentile is chosen and no rule consumes the "
            "uncertainty yet; step 4 measures what it is.",
            "HOLDOUT: seasons " + ", ".join(HOLDOUT_SEASONS) + " are the final "
            "test set and are not loaded (AGENTS.md rule 9). They are reported "
            "once, at the end, when the user asks.",
        ],
    }

    report = render_report(context)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = REPORTS_DIR / f"phase1_step4_uncertainty_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print("\n".join(findings), flush=True)
    print(f"\nwrote {path} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
