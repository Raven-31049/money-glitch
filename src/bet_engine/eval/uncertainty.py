"""Uncertainty summaries over probability draws (Phase 1 step 4).

The model half of the contract draws ``n`` samples of P(outcome) per match
(``models/base.py``). This module turns those draws into the intervals a report
quotes, and it exists mainly to keep one property explicit and tested:

**Percentiles are marginals and are never renormalised across outcomes**
(AGENTS.md rule 5 / invariant 5). Each outcome's percentiles are computed from
that outcome's own draws alone. They do *not* sum to 1 and must not be made to:
the 10th percentile of P(over) and the 10th percentile of P(under) can both be
small at once (an uncertain match), and forcing them to 1 would fabricate
certainty the samples do not contain. The draws themselves do sum to 1 per
draw; the percentiles taken across draws are a different object.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def marginal_percentiles(
    samples: np.ndarray, percents: Sequence[float]
) -> np.ndarray:
    """Per-outcome percentiles along the draw axis, never renormalised.

    ``samples`` is the array :meth:`~bet_engine.models.cards_poisson.CardsPoisson.predict_samples`
    returns: draw axis first (axis 0), outcomes last, any axes in between
    (matches, lines, ...) carried through. The result has the percent list as a
    new leading axis: ``(len(percents), *samples.shape[1:])``. Each outcome's
    percentile is read off that outcome's own draws and is left exactly as
    computed — see the module docstring for why renormalising would be wrong.
    """
    array = np.asarray(samples, dtype=float)
    if array.ndim < 2:
        raise ValueError(
            "marginal_percentiles needs at least a draw axis and an outcome "
            f"axis, got shape {array.shape}"
        )
    values = np.asarray(list(percents), dtype=float)
    if values.size == 0:
        raise ValueError("marginal_percentiles needs at least one percentile")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 100.0)):
        raise ValueError(f"percentiles must be in [0, 100], got {values.tolist()}")
    # No renormalisation step: np.percentile is per-outcome by construction and
    # the result is returned untouched (invariant 5).
    return np.percentile(array, values, axis=0)


def interval_width(samples: np.ndarray, *, low: float = 10.0, high: float = 90.0) -> np.ndarray:
    """Width of the ``[low, high]`` marginal interval per outcome.

    A thin wrapper over :func:`marginal_percentiles` that returns
    ``high_percentile - low_percentile`` with the percent axis removed, so a
    caller gets one width per outcome (and per match/line). Still never
    renormalised: the width of an outcome's own interval.
    """
    bounds = marginal_percentiles(samples, [low, high])
    return bounds[1] - bounds[0]


def percentile_probabilities(
    samples: np.ndarray,
    percentile: float,
    match_ids: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    """Long-form ``[match_id, outcome, prob]`` at the ``percentile`` of each.

    The table the uncertainty-aware selection reads: for a 2-D-in-outcomes
    ``samples`` array ``(draws, n_matches, n_outcomes)``, the q-th percentile of
    each outcome's own draws, laid out one row per (match, outcome). The match
    axis follows ``samples``; ``outcomes`` labels the last axis in order.

    Each outcome's percentile is read off that outcome's own draws and returned
    untouched — never renormalised to sum to 1 (invariant 5). Two outcomes can
    both come out small (an uncertain match); forcing them to 1 would fabricate
    certainty the draws do not contain.
    """
    value = _percentile_value(percentile)
    array = np.asarray(samples, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "percentile_probabilities needs samples of shape "
            f"(draws, n_matches, n_outcomes), got {array.shape}"
        )
    ids = list(match_ids)
    labels = list(outcomes)
    if len(ids) != array.shape[1]:
        raise ValueError(
            f"match_ids has {len(ids)} entries but samples has {array.shape[1]} "
            f"match(es)"
        )
    if len(labels) != array.shape[2]:
        raise ValueError(
            f"outcomes has {len(labels)} entries but samples has "
            f"{array.shape[2]} outcome(s)"
        )
    values = marginal_percentiles(array, [value])[0]  # (n_matches, n_outcomes)
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.asarray(ids, dtype=object), len(labels)),
            "outcome": np.tile(np.asarray(labels, dtype=object), len(ids)),
            "prob": values.ravel(),
        }
    )


def _percentile_value(percentile: object) -> float:
    """A finite percentile in [0, 100], with bools and text refused."""
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise ValueError(f"percentile must be a number in [0, 100], got {percentile!r}")
    value = float(percentile)
    if not np.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
    return value


__all__ = ["interval_width", "marginal_percentiles", "percentile_probabilities"]
