"""Tests for models/base.py: the model contract and its runtime checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.models import (
    MarketModel,
    PREDICTION_COLUMNS,
    SamplingModel,
    check_predictions,
    supports_samples,
)

OUTCOMES = ("H", "D", "A")


def _uniform(df: pd.DataFrame) -> pd.DataFrame:
    """The smallest legal prediction frame: flat probabilities, one row per outcome."""
    rows = [
        {"match_id": match_id, "outcome": outcome, "prob": 1.0 / len(OUTCOMES)}
        for match_id in df["match_id"]
        for outcome in OUTCOMES
    ]
    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))


class UniformModel:
    """Implements the whole contract, including the optional sampling half."""

    def __init__(self) -> None:
        self.fitted_on: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "UniformModel":
        self.fitted_on = train_df
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return _uniform(df)

    def predict_samples(self, df: pd.DataFrame, n: int) -> np.ndarray:
        return np.full((n, len(df), len(OUTCOMES)), 1.0 / len(OUTCOMES))


class PointOnly:
    """Implements only the required half of the contract."""

    def fit(self, train_df: pd.DataFrame) -> "PointOnly":
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return _uniform(df)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"match_id": ["m1", "m2"]})


def test_protocols_are_structurally_satisfied():
    assert isinstance(UniformModel(), MarketModel)
    assert isinstance(UniformModel(), SamplingModel)

    point_only = PointOnly()
    assert isinstance(point_only, MarketModel)
    assert not isinstance(point_only, SamplingModel)

    assert supports_samples(UniformModel())
    assert not supports_samples(point_only)
    assert not supports_samples(object())


def test_fit_returns_self_so_calls_chain(frame):
    model = UniformModel()
    assert model.fit(frame) is model
    assert model.fitted_on is frame


def test_predict_matches_the_column_contract(frame):
    pred = UniformModel().predict(frame)

    assert list(pred.columns) == list(PREDICTION_COLUMNS)
    # check_predictions returns the frame unchanged so it can be chained.
    assert check_predictions(pred, OUTCOMES) is pred


def _frame(**overrides) -> pd.DataFrame:
    base: dict = {
        "match_id": ["m1", "m1", "m1"],
        "outcome": ["H", "D", "A"],
        "prob": [1 / 3, 1 / 3, 1 / 3],
    }
    base.update(overrides)
    return pd.DataFrame(base)


BAD_FRAMES = [
    (lambda: pd.DataFrame({"match_id": ["m1"], "prob": [0.5]}), "missing column"),
    (lambda: _frame(outcome=["H", "D", "X"]), "unknown outcome"),
    (lambda: _frame(match_id=["m1", "m1"], outcome=["H", "D"], prob=[0.5, 0.5]),
     "expected 3"),
    (lambda: _frame(outcome=["H", "H", "D"], prob=[0.3, 0.3, 0.4]), "duplicate"),
    (lambda: _frame(prob=[np.nan, 0.5, 0.5]), "non-finite"),
    (lambda: _frame(prob=[1.5, 0.0, 0.0]), "outside"),
    (lambda: _frame(prob=[0.5, 0.5, 0.5]), "sum to 1"),
    (lambda: _frame(prob=["high", "mid", "low"]), "numeric"),
]


@pytest.mark.parametrize("build, expected", BAD_FRAMES)
def test_check_predictions_rejects_malformed_frames(build, expected):
    with pytest.raises(ValueError, match=expected):
        check_predictions(build(), OUTCOMES)


def test_check_predictions_rejects_non_frames():
    with pytest.raises(TypeError, match="DataFrame"):
        check_predictions([("m1", "H", 0.5)], OUTCOMES)  # type: ignore[arg-type]


def test_check_predictions_allows_an_empty_frame():
    empty = pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    assert check_predictions(empty, OUTCOMES) is empty
