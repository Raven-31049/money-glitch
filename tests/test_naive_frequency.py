"""Tests for models/naive_frequency.py: the deliberately dumb baseline.

Frequencies must be exactly what the window says — no smoothing, no priors,
unsettled rows never counted — because the baseline's whole job is to be
honestly weak and be scored as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.models import NaiveFrequency
from bet_engine.models.base import check_predictions

OUTCOMES = ("H", "D", "A")


def _train(ftr_values: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(len(ftr_values))],
            "ftr": ftr_values,
        }
    )


def test_frequencies_are_a_hand_counted_relative_frequency():
    model = NaiveFrequency(OUTCOMES)

    model.fit(_train(["H", "H", "A"]))

    assert model.frequencies == pytest.approx({"H": 2 / 3, "D": 0.0, "A": 1 / 3})


def test_unsettled_rows_are_not_counted_as_evidence():
    model = NaiveFrequency(OUTCOMES)

    model.fit(_train(["H", None, "A"]))

    assert model.frequencies == pytest.approx({"H": 0.5, "D": 0.0, "A": 0.5})


def test_case_and_whitespace_in_the_settlement_code_are_tolerated():
    model = NaiveFrequency(OUTCOMES)

    model.fit(_train([" h ", "D", "a"]))

    assert model.frequencies == pytest.approx({"H": 1 / 3, "D": 1 / 3, "A": 1 / 3})


def test_garbled_settlement_is_refused_not_bucketed():
    """An unknown label means the frame bypassed ingest normalisation."""
    model = NaiveFrequency(OUTCOMES)

    with pytest.raises(ValueError, match="no settled rows"):
        model.fit(_train(["X", "Y"]))


def test_fit_without_a_settlement_column_names_what_is_missing():
    model = NaiveFrequency(OUTCOMES)

    with pytest.raises(ValueError, match="'ftr'"):
        model.fit(pd.DataFrame({"match_id": ["m1"]}))


def test_predict_before_fit_is_an_error_not_a_guess():
    model = NaiveFrequency(OUTCOMES)

    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(_train(["H"]))


def test_predict_tiles_one_constant_vector_over_every_row():
    model = NaiveFrequency(OUTCOMES)
    model.fit(_train(["H", "H", "A"]))
    batch = pd.DataFrame({"match_id": ["x1", "x2", "x3"]})

    pred = model.predict(batch)

    assert list(pred.columns) == ["match_id", "outcome", "prob"]
    assert len(pred) == 3 * len(OUTCOMES)
    # check_predictions is the engine's contract check: one row per outcome,
    # probabilities summing to 1 per match.
    check_predictions(pred, list(OUTCOMES))
    vectors = pred.pivot(index="match_id", columns="outcome", values="prob")
    assert (vectors == vectors.iloc[0]).all().all()
    assert vectors.iloc[0]["H"] == pytest.approx(2 / 3)
    assert vectors.iloc[0]["D"] == pytest.approx(0.0)
    assert vectors.iloc[0]["A"] == pytest.approx(1 / 3)


def test_predict_on_an_empty_batch_returns_the_contract_shape():
    model = NaiveFrequency(OUTCOMES)
    model.fit(_train(["A"]))

    pred = model.predict(pd.DataFrame({"match_id": []}))

    assert list(pred.columns) == ["match_id", "outcome", "prob"]
    assert pred.empty
    assert pred["prob"].dtype == np.float64


def test_model_requires_outcomes():
    with pytest.raises(ValueError, match="must not be empty"):
        NaiveFrequency(())
