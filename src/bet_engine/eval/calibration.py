"""Calibration scoring: Brier, log loss, reliability bins.

Market-agnostic scoring (invariant 8): the functions see a probability vector
per row and the settled outcome, never a market object, so a 2-way and a 3-way
market differ only in how many columns the frame carries. That is the whole
trick — every score below is written over the full outcome vector.

Three rules the validations enforce:

* probabilities are never renormalised here. A row that does not sum to 1 is
  rejected with the rows that failed: renormalising would silently repair the
  model's output and report a calibration the model did not earn (the same
  honesty as invariant 5's refusal to squeeze percentiles to sum to 1).
* log loss refuses a zero probability on the settled outcome instead of
  clipping it. Clipping would report a finite score for a model that declared
  the observed outcome impossible.
* the reliability table bins *every* outcome probability with a 0/1 "did it
  happen" indicator, so a 3-way market contributes three pairs per bet and the
  table's shape does not depend on the market.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: A frame of outcome probabilities (one column per outcome, one row per bet)
#: or a plain 2-D array for callers that carry no outcome labels.
Predicted = pd.DataFrame | np.ndarray | Sequence[Sequence[float]]

#: The settled outcome per row: labels naming ``Predicted``'s columns, or
#: integer positions into them. Which one is meant is never guessed — labels
#: only against a labelled frame, integers only as positions.
Actual = pd.Series | np.ndarray | Sequence[str] | Sequence[int]

#: A row's probabilities may miss 1.0 by float noise but not by model error.
_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ReliabilityBin:
    """One width of the reliability diagram: predicted versus observed.

    ``low``/``high`` are the bin's edges on [0, 1]; ``predicted_mean`` is the
    mean probability that landed here, ``actual_frequency`` how often the
    outcome those probabilities referred to actually happened, ``count`` how
    many pairs fell in. An empty bin keeps its edges with ``count`` 0 and NaN
    statistics rather than being dropped, so the table's shape is always
    ``n_bins`` and a report can show where the model never predicted.
    """

    low: float
    high: float
    predicted_mean: float
    actual_frequency: float
    count: int


def brier_score(predicted: Predicted, actual: Actual) -> float:
    """Mean squared error over the whole probability vector, per bet.

    The row score is ``sum over outcomes of (p - y)^2``, so it stays in [0, 2]
    whether the market has two outcomes or three — a 2-way and a 3-way score
    are directly comparable. (For two outcomes it is twice the textbook
    single-probability form; we score the vector the model actually emitted
    rather than a projection of it.)
    """
    probabilities, index = _validated(predicted, actual)
    truth = _one_hot(probabilities, index)
    return float(np.mean(np.square(probabilities - truth).sum(axis=1)))


def log_loss(predicted: Predicted, actual: Actual) -> float:
    """Mean ``-log(p)`` of the settled outcome's probability.

    A model that gave the observed outcome zero probability is rejected, not
    clipped: that model said the result was impossible, and the honest score
    for that claim is infinite, not ``-log(epsilon)``.
    """
    probabilities, index = _validated(predicted, actual)
    settled = probabilities[np.arange(probabilities.shape[0]), index]
    impossible = np.flatnonzero(settled <= 0.0)
    if impossible.size:
        raise ValueError(
            "log loss is undefined where predicted gives the settled outcome "
            f"probability 0; bad row(s): {impossible[:10].tolist()}"
        )
    return float(-np.mean(np.log(settled)))


def reliability_table(
    predicted: Predicted, actual: Actual, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Bin every predicted probability and compare mean to observed frequency.

    Equal-width bins on [0, 1]. Each row of ``predicted`` contributes one pair
    per outcome — (that outcome's probability, 1 if it happened else 0) — so
    calibration is judged on the full vector for any market size.
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise TypeError(f"n_bins must be an integer, got {type(n_bins).__name__}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    probabilities, index = _validated(predicted, actual)
    truth = _one_hot(probabilities, index)

    flat_probabilities = probabilities.ravel()
    flat_truth = truth.ravel()
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # right-side search puts a probability on its upper edge into the bin that
    # ends there (p == 1.0 must land in the last bin, not fall off the table).
    bucket = np.searchsorted(edges, flat_probabilities, side="right") - 1
    np.clip(bucket, 0, int(n_bins) - 1, out=bucket)

    counts = np.bincount(bucket, minlength=int(n_bins))
    weighted_p = np.bincount(bucket, weights=flat_probabilities, minlength=int(n_bins))
    weighted_y = np.bincount(bucket, weights=flat_truth, minlength=int(n_bins))
    divisor = np.where(counts > 0, counts, 1)  # empty bins -> NaN, no /0 warning
    means = np.where(counts > 0, weighted_p / divisor, np.nan)
    frequencies = np.where(counts > 0, weighted_y / divisor, np.nan)

    return tuple(
        ReliabilityBin(
            low=float(edges[position]),
            high=float(edges[position + 1]),
            predicted_mean=float(means[position]),
            actual_frequency=float(frequencies[position]),
            count=int(counts[position]),
        )
        for position in range(int(n_bins))
    )


def _validated(predicted: object, actual: object) -> tuple[np.ndarray, np.ndarray]:
    """Validated probability matrix and settled-outcome positions.

    Every rejection lives here so the three public scores stay arithmetic: a
    frame that reaches them has usable probabilities, one outcome index per
    row, and nothing that any of them would silently repair.
    """
    probabilities, columns = _probabilities(predicted)
    index = _outcome_index(actual, probabilities, columns)
    return probabilities, index


def _one_hot(probabilities: np.ndarray, index: np.ndarray) -> np.ndarray:
    """1.0 at each row's settled outcome, 0.0 everywhere else."""
    truth = np.zeros_like(probabilities)
    truth[np.arange(probabilities.shape[0]), index] = 1.0
    return truth


def _probabilities(predicted: object) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Read ``predicted`` as an (n_bets, n_outcomes) float matrix.

    A DataFrame is taken as outcome-labelled columns (returned as labels for
    the index lookup); anything else must already be a 2-D numeric array and
    therefore carries integer positions instead.
    """
    columns: tuple[str, ...] | None = None
    if isinstance(predicted, pd.DataFrame):
        duplicated = predicted.columns[predicted.columns.duplicated()].tolist()
        if duplicated:
            raise ValueError(
                f"predicted has duplicate outcome column(s) {duplicated}; "
                f"each outcome must appear exactly once"
            )
        columns = tuple(predicted.columns)
        raw = predicted.to_numpy()
    else:
        try:
            raw = np.asarray(predicted)
        except (TypeError, ValueError):
            raise TypeError(
                f"predicted must be a DataFrame or a 2-D array of "
                f"probabilities, got {type(predicted).__name__}"
            ) from None
    if raw.ndim != 2:
        raise ValueError(
            f"predicted must be 2-D (rows = bets, columns = outcomes); "
            f"got shape {raw.shape}"
        )
    try:
        probabilities = raw.astype(float, copy=False)
    except (TypeError, ValueError):
        raise ValueError("predicted must hold numeric probabilities") from None

    n_bets, n_outcomes = probabilities.shape
    if n_bets < 1:
        raise ValueError(f"predicted must have at least one row; got {n_bets}")
    if n_outcomes < 2:
        raise ValueError(
            f"predicted must carry at least two outcome columns; got {n_outcomes}"
        )
    finite = np.isfinite(probabilities)
    if not finite.all():
        raise ValueError(
            f"predicted must be finite on every row; "
            f"bad row(s): {np.flatnonzero(~finite.all(axis=1))[:10].tolist()}"
        )
    out_of_range = (probabilities < 0.0) | (probabilities > 1.0)
    if out_of_range.any():
        raise ValueError(
            f"predicted probabilities must be in [0, 1]; "
            f"bad row(s): {np.flatnonzero(out_of_range.any(axis=1))[:10].tolist()}"
        )
    deviation = np.abs(probabilities.sum(axis=1) - 1.0)
    if deviation.max() > _SUM_TOLERANCE:
        raise ValueError(
            f"predicted probabilities must sum to 1 on every row "
            f"(worst row off by {deviation.max():.3g}); "
            f"bad row(s): {np.flatnonzero(deviation > _SUM_TOLERANCE)[:10].tolist()}; "
            f"rows are not renormalised here because that would hide the bug"
        )
    return probabilities, columns


def _outcome_index(
    actual: object, probabilities: np.ndarray, columns: tuple[str, ...] | None
) -> np.ndarray:
    """Settled outcome per row, as an integer position into the columns.

    Two accepted spellings, and no guessing between them: an integer array is
    always a position, and labels are strings naming a DataFrame's outcome
    columns. Everything else — floats, bools, unknown labels — is refused, so
    a misaligned outcome can never be scored as if it were another one.
    """
    n_bets, n_outcomes = probabilities.shape
    if isinstance(actual, pd.Series):
        values = actual.to_numpy()
    else:
        try:
            values = np.asarray(actual)
        except (TypeError, ValueError):
            raise TypeError(
                f"actual must be a 1-D array of outcome labels or integer "
                f"positions, got {type(actual).__name__}"
            ) from None
    if values.ndim != 1:
        raise ValueError(
            f"actual must be one outcome per bet (1-D); got shape {values.shape}"
        )
    if values.shape[0] != n_bets:
        raise ValueError(
            f"actual has {values.shape[0]} outcome(s); "
            f"predicted has {n_bets} row(s)"
        )

    if np.issubdtype(values.dtype, np.integer):
        index = values.astype(np.intp)
        bad = (index < 0) | (index >= n_outcomes)
        if bad.any():
            raise ValueError(
                f"actual integer positions must be in [0, {n_outcomes}); "
                f"bad row(s): {np.flatnonzero(bad)[:10].tolist()}"
            )
        return index

    if not all(isinstance(value, str) for value in values):
        raise ValueError(
            "actual must be outcome labels (strings naming predicted's "
            "columns) or integer positions"
        )
    if columns is None:
        raise ValueError(
            "actual carries outcome labels, so predicted must be a DataFrame "
            "whose columns name the outcomes; got an array of labels"
        )
    lookup = {label: position for position, label in enumerate(columns)}
    mapped = pd.Series(values, dtype=object).map(lookup)
    if mapped.isna().any():
        unknown = sorted({str(value) for value in values[mapped.isna().to_numpy()]})
        raise ValueError(
            f"actual has outcome label(s) {unknown} not among "
            f"predicted's columns {list(columns)}"
        )
    return mapped.to_numpy(dtype=np.intp)


__all__ = ["Actual", "Predicted", "ReliabilityBin", "brier_score", "log_loss", "reliability_table"]
