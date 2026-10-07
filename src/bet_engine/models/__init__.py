"""League-specific models, plus the contract every one of them obeys.

``MODEL_REGISTRY`` maps a variant config's ``model.type`` string to its class,
so a config can only ever name a model that actually exists — an unknown type
fails at construction time with the list of real ones, not halfway through a
walk-forward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    MarketModel,
    PREDICTION_COLUMNS,
    SamplingModel,
    check_predictions,
    supports_samples,
)
from .naive_frequency import NaiveFrequency
from .pure_market import PureMarket, devig_long

#: Registry key (``model.type`` in YAML) -> model class. Every class takes
#: the market's ``outcomes`` plus whatever its own ``params`` are.
MODEL_REGISTRY: dict[str, type] = {
    "pure_market": PureMarket,
    "naive_frequency": NaiveFrequency,
}


def create_model(
    model_type: str,
    params: Mapping[str, Any],
    outcomes: Sequence[str],
) -> MarketModel:
    """Build the model a config names, bound to this market's outcomes.

    ``params`` become constructor keyword arguments, so a parameter the model
    does not accept raises a TypeError naming it — a typo in YAML must not be
    silently ignored as an unused setting. ``outcomes`` is injected by the
    runner from the market registry, not by the config: which outcomes exist
    is the market's business (invariant 8), not the model author's.
    """
    try:
        model_class = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"unknown model type {model_type!r}; "
            f"known types: {sorted(MODEL_REGISTRY)}"
        ) from None
    return model_class(outcomes=tuple(outcomes), **dict(params))


__all__ = [
    "MODEL_REGISTRY",
    "MarketModel",
    "PREDICTION_COLUMNS",
    "NaiveFrequency",
    "PureMarket",
    "SamplingModel",
    "check_predictions",
    "create_model",
    "devig_long",
    "supports_samples",
]
