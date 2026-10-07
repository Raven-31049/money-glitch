"""Model contract: fit, point-estimate probabilities, optional samples.

One model per league, never pooled (invariant 7). The contract is deliberately
narrow because the validation engine must never need to know what is inside a
model (invariant 8).

``predict`` returns the point estimate in long form with exactly the columns
``[match_id, outcome, prob]`` — one row per outcome per match, the
probabilities within a match summing to 1. :func:`check_predictions` enforces
that shape at the boundary, so a model whose probabilities quietly stop
summing to 1 (or who drops an outcome for some rows) fails where it happens
instead of corrupting EV, calibration and the pure-market control three stages
later.

``predict_samples`` is the optional half of the contract, reserved for the
uncertainty work of Phase 2. Convention: it returns an ndarray of shape
``(n, n_matches, n_outcomes)``, the match axis following ``df``'s rows and the
outcome axis following the market's ``outcomes`` order, with every draw
summing to 1 across outcomes. Percentiles computed *from* those draws are
marginals and are never renormalised to sum to 1 (invariant 5) — that is a
property of the percentiles, not of the draws they summarise.
"""

from __future__ import annotations

from typing import Protocol, Self, Sequence, TypeGuard, runtime_checkable

import numpy as np
import pandas as pd

#: Exact column contract of :meth:`MarketModel.predict`, in order.
PREDICTION_COLUMNS: tuple[str, ...] = ("match_id", "outcome", "prob")

#: Per-match sums must land this close to 1; looser would let drift through,
#: tighter would fail on float32 accumulation in an otherwise correct model.
_SUM_TOLERANCE = 1e-6


@runtime_checkable
class MarketModel(Protocol):
    """The minimum a model must offer to be runnable by the engine."""

    def fit(self, train_df: pd.DataFrame) -> Self:
        """Estimate from training rows; returns self so calls chain."""
        ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point probabilities for every row of ``df``, in long form."""
        ...


@runtime_checkable
class SamplingModel(MarketModel, Protocol):
    """A model that can also draw probability samples for uncertainty."""

    def predict_samples(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """``n`` joint draws with shape ``(n, len(df), n_outcomes)``."""
        ...


def supports_samples(model: object) -> TypeGuard[SamplingModel]:
    """True when a model implements the optional sampling half of the contract."""
    return callable(getattr(model, "predict_samples", None))


def check_predictions(
    pred: pd.DataFrame, outcomes: Sequence[str]
) -> pd.DataFrame:
    """Validate ``pred`` against the prediction contract; return it unchanged.

    Why a runtime check rather than trusting the type hint: every number the
    system reports — edge, EV, Brier score, the pure-market control's bet count
    — is derived from these probabilities, and a malformed frame fails far
    from its source. Checking at the model/engine boundary turns a silent
    poisoning into an error that names the match and the symptom.
    """
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(
            f"predict() must return a DataFrame, got {type(pred).__name__}"
        )
    missing = [column for column in PREDICTION_COLUMNS if column not in pred.columns]
    if missing:
        raise ValueError(
            f"predictions missing column(s) {missing}; have {list(pred.columns)}"
        )
    if pred.empty:
        return pred

    expected = list(outcomes)
    unknown = sorted(set(pred["outcome"]) - set(expected), key=str)
    if unknown:
        raise ValueError(
            f"unknown outcome label(s) {unknown}; market outcomes are {expected}"
        )

    duplicated = pred[pred.duplicated(["match_id", "outcome"], keep=False)]
    if not duplicated.empty:
        examples = [
            f"{match_id}/{outcome}"
            for match_id, outcome in duplicated[["match_id", "outcome"]]
            .head(3)
            .itertuples(index=False)
        ]
        raise ValueError(f"duplicate (match_id, outcome) rows: {examples}")

    counts = pred.groupby("match_id", sort=False, dropna=False).size()
    incomplete = counts[counts != len(expected)]
    if not incomplete.empty:
        examples = list(incomplete.index[:3])
        raise ValueError(
            f"match(es) {examples} have {incomplete.head(3).tolist()} rows, "
            f"expected {len(expected)} (one per outcome)"
        )

    try:
        probs = pred["prob"].to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("prob column must be numeric") from exc

    if not np.isfinite(probs).all():
        bad = pred.loc[~np.isfinite(probs), "match_id"].head(3).tolist()
        raise ValueError(f"non-finite probability for match(es) {bad}")

    out_of_range = (probs < 0.0) | (probs > 1.0)
    if out_of_range.any():
        bad = pred.loc[out_of_range, ["match_id", "outcome"]].head(3)
        raise ValueError(
            f"probability outside [0, 1]: {bad.to_dict('records')}"
        )

    sums = pred.groupby("match_id", sort=False, dropna=False)["prob"].sum()
    off = sums[(sums.to_numpy() - 1.0).__abs__() > _SUM_TOLERANCE]
    if not off.empty:
        examples = list(off.index[:3])
        values = [round(float(value), 6) for value in off.head(3)]
        raise ValueError(
            f"probabilities do not sum to 1 for match(es) {examples}: {values}"
        )

    return pred


__all__ = [
    "MarketModel",
    "PREDICTION_COLUMNS",
    "SamplingModel",
    "check_predictions",
    "supports_samples",
]
