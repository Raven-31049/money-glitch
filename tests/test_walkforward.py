"""Tests for backtest/walkforward.py: calendar-day folds and the leak check.

The heart of these tests is invariant 1: training data strictly earlier than
the prediction day. A deliberately leaky splitter is injected to prove the
assertion refuses it before any model is fitted, and the same-day cases are
covered from both sides — the engine's own split, and the assertion alone.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import (
    WALK_FORWARD_COLUMNS,
    assert_train_before_predict,
    split_by_day,
    walk_forward,
)
from bet_engine.models import PREDICTION_COLUMNS

OUTCOMES = ("H", "D", "A")


def _uniform(df: pd.DataFrame) -> pd.DataFrame:
    """Flat 1/3 probabilities for every row of ``df``: the smallest legal output."""
    rows = [
        {"match_id": match_id, "outcome": outcome, "prob": 1.0 / len(OUTCOMES)}
        for match_id in df["match_id"]
        for outcome in OUTCOMES
    ]
    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))


def _make_frame(
    *,
    start: str = "2021-08-01",
    days: int = 8,
    hours: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """``len(hours)`` matches per calendar day, with only the columns the engine is promised."""
    dates = pd.date_range(start, periods=days, freq="D")
    rows = [
        {
            "match_id": f"{day.date()}-{slot}",
            "date": day + pd.Timedelta(hours=hour),
            "league": "EPL",
        }
        for day in dates
        for slot, hour in enumerate(hours)
    ]
    return pd.DataFrame(rows)


class Recorder:
    """Records every fit and predict it is asked to perform, across instances.

    The shared events list shows the whole run: which rows each fit saw, and
    which batch each predict served — the provenance the assertions re-check.
    """

    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.train: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "Recorder":
        self.train = train_df
        self.events.append({"kind": "fit", "train": train_df, "model": self})
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.train is not None, "predict() called before fit()"
        self.events.append(
            {"kind": "predict", "train": self.train, "batch": df.copy(), "model": self}
        )
        return _uniform(df)


class Factory:
    """model_factory double: counts constructions (one per refit)."""

    def __init__(self, model_cls: Callable[[], object] | None = None) -> None:
        self.model_cls = model_cls
        self.events: list[dict] = []
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        if self.model_cls is None:
            return Recorder(self.events)
        return self.model_cls()


def _predicts(events: list[dict]) -> list[dict]:
    return [event for event in events if event["kind"] == "predict"]


def _fits(events: list[dict]) -> list[dict]:
    return [event for event in events if event["kind"] == "fit"]


def _fit_days(events: list[dict]) -> list[pd.Timestamp]:
    """The fold day each fit served: with contiguous daily data the training
    rows end the day before the fold the engine fitted for."""
    return [
        event["train"]["date"].max() + pd.Timedelta(days=1)
        for event in _fits(events)
    ]


def _days_per_model(events: list[dict]) -> list[int]:
    """Prediction days served by each model instance, in order (refit cadence)."""
    counts: list[int] = []
    current: object = None
    for event in _predicts(events):
        if event["model"] is not current:
            current = event["model"]
            counts.append(0)
        counts[-1] += 1
    return counts


def _index_based_split_that_leaks(
    df: pd.DataFrame, predict_day: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deliberately broken splitter: trains on the prediction day's own rows.

    Positional cuts pretend row order is time, so half of the matchday it is
    about to predict ends up in its training set. walk_forward must refuse it.
    """
    day_rows = df[df["date"].dt.normalize() == predict_day]
    cut = day_rows.index[len(day_rows) // 2]
    return df[df.index <= cut], day_rows


def test_no_fold_ever_trains_on_or_after_its_prediction_day():
    df = _make_frame(days=8, hours=(12, 18))
    factory = Factory()
    walk_forward(df, factory, min_train_days=2, refit_every_days=3)

    predicts = _predicts(factory.events)
    assert len(predicts) == 6  # 2021-08-03 .. 2021-08-08, one batch per day
    for event in predicts:
        train, batch = event["train"], event["batch"]
        assert len(train) > 0
        assert set(train["match_id"]).isdisjoint(batch["match_id"])
        assert train["date"].max() < batch["date"].min()
        # the whole history sits on earlier *calendar days*, not merely earlier
        # timestamps — invariant 1, not just ordering:
        assert train["date"].dt.normalize().max() < batch["date"].min().normalize()
        assert batch["date"].dt.normalize().nunique() == 1


def test_same_calendar_day_matches_arrive_in_one_batch_and_never_in_training():
    df = _make_frame(days=6, hours=(12, 15, 18))
    factory = Factory()
    walk_forward(df, factory, min_train_days=1, refit_every_days=99)

    predicts = _predicts(factory.events)
    assert len(predicts) == 5  # 2021-08-02 .. 2021-08-06
    for event in predicts:
        assert set(event["train"]["match_id"]).isdisjoint(event["batch"]["match_id"])
        assert event["train"]["date"].max() < event["batch"]["date"].min()

    oneday = df[df["date"].dt.normalize() == pd.Timestamp("2021-08-04")]
    fold = next(
        event
        for event in predicts
        if event["batch"]["date"].min().normalize() == pd.Timestamp("2021-08-04")
    )
    assert len(fold["batch"]) == 3
    assert set(fold["batch"]["match_id"]) == set(oneday["match_id"])


def test_refit_every_three_days_fits_on_the_right_days_and_reuses_between():
    df = _make_frame(days=10)
    factory = Factory()
    walk_forward(df, factory, min_train_days=2, refit_every_days=3)

    assert factory.calls == 3
    assert _fit_days(factory.events) == [
        pd.Timestamp("2021-08-03"),
        pd.Timestamp("2021-08-06"),
        pd.Timestamp("2021-08-09"),
    ]
    assert _days_per_model(factory.events) == [3, 3, 2]  # 8 folds in total


def test_refit_every_day_builds_a_fresh_model_for_each_fold():
    df = _make_frame(days=10)
    factory = Factory()
    walk_forward(df, factory, min_train_days=2, refit_every_days=1)

    assert factory.calls == 8
    assert _days_per_model(factory.events) == [1] * 8


def test_a_leaky_split_is_refused_before_the_first_fit():
    df = _make_frame(days=6, hours=(12, 18))
    factory = Factory()

    with pytest.raises(AssertionError) as exc:
        walk_forward(
            df,
            factory,
            min_train_days=2,
            refit_every_days=3,
            split=_index_based_split_that_leaks,
        )

    message = str(exc.value)
    assert "temporal leak" in message
    assert "train.date.max()" in message
    assert "2021-08-03" in message  # the first eligible fold refused the cut
    assert factory.calls == 0  # refused before any model was constructed


def test_split_by_day_cuts_on_the_calendar_boundary_with_no_overlap():
    df = _make_frame(days=4, hours=(0, 12, 23))
    day = pd.Timestamp("2021-08-03")
    train, predict = split_by_day(df, day)

    assert set(train.index).isdisjoint(predict.index)
    assert set(predict.index) == set(df.index[df["date"].dt.normalize() == day])
    assert not (train["date"].dt.normalize() == day).any()

    last_day = df["date"].max().normalize()
    train, predict = split_by_day(df, last_day)
    assert set(train.index) | set(predict.index) == set(df.index)
    assert set(train.index).isdisjoint(predict.index)


def test_split_by_day_keeps_clock_times_inside_their_own_calendar_day():
    df = _make_frame(days=4, hours=(0, 12, 23))
    train, predict = split_by_day(df, pd.Timestamp("2021-08-03"))

    assert train["date"].max() == pd.Timestamp("2021-08-02 23:00")
    assert sorted(predict["date"]) == [
        pd.Timestamp("2021-08-03"),
        pd.Timestamp("2021-08-03 12:00"),
        pd.Timestamp("2021-08-03 23:00"),
    ]


def test_split_by_day_normalises_any_timestamp_to_the_fold_day():
    df = _make_frame(days=4, hours=(18,))
    train, predict = split_by_day(df, "2021-08-03 15:40")

    assert train["date"].max() == pd.Timestamp("2021-08-02 18:00")
    assert list(predict["date"]) == [pd.Timestamp("2021-08-03 18:00")]


def test_split_by_day_rejects_an_undatable_fold():
    df = _make_frame(days=4)
    with pytest.raises(ValueError, match="real calendar date"):
        split_by_day(df, pd.NaT)


def _side(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def test_the_assertion_accepts_strictly_earlier_history():
    assert_train_before_predict(
        _side(["2021-08-01 18:00", "2021-08-02 12:00"]),
        _side(["2021-08-03 00:30"]),
        context="fold 3",
    )


def test_the_assertion_refuses_training_inside_the_prediction_day():
    train = _side(["2021-08-03 12:00"])  # an earlier kick-off the same day
    predict = _side(["2021-08-03 18:00"])

    with pytest.raises(AssertionError) as exc:
        assert_train_before_predict(train, predict, context="fold probe")

    message = str(exc.value)
    assert "temporal leak" in message
    assert "2021-08-03" in message
    assert "train rows=1" in message
    assert "[fold probe]" in message


def test_the_assertion_refuses_a_training_row_on_the_fold_boundary():
    with pytest.raises(AssertionError, match="temporal leak"):
        assert_train_before_predict(
            _side(["2021-08-03"]), _side(["2021-08-03"])
        )


def test_the_assertion_refuses_backwards_time():
    with pytest.raises(AssertionError, match="temporal leak"):
        assert_train_before_predict(
            _side(["2021-08-05"]), _side(["2021-08-04"])
        )


@pytest.mark.parametrize(
    "train, predict",
    [
        (_side([]), _side(["2021-08-04"])),
        (_side(["2021-08-01"]), _side([])),
    ],
)
def test_the_assertion_refuses_an_empty_side(train, predict):
    with pytest.raises(AssertionError, match="temporal leak"):
        assert_train_before_predict(train, predict)


def test_the_assertion_refuses_rows_that_cannot_be_placed_on_a_calendar():
    with pytest.raises(AssertionError, match="cannot be certified"):
        assert_train_before_predict(
            _side(["2021-08-01", "NaT"]), _side(["2021-08-04"])
        )


def test_the_leak_check_survives_running_under_python_dash_O():
    script = (
        "import pandas as pd\n"
        "from bet_engine.backtest.walkforward import "
        "assert_train_before_predict\n"
        "train = pd.DataFrame({'date': pd.to_datetime(['2021-08-03 12:00'])})\n"
        "predict = pd.DataFrame({'date': pd.to_datetime(['2021-08-03 18:00'])})\n"
        "try:\n"
        "    assert_train_before_predict(train, predict)\n"
        "except AssertionError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_output_is_long_form_chronological_and_row_order_independent():
    df = _make_frame(days=7, hours=(18, 12))
    shuffled = df.sample(frac=1, random_state=7)
    factory = Factory()
    result = walk_forward(shuffled, factory, min_train_days=2, refit_every_days=99)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert result["date"].is_monotonic_increasing
    eligible = set(df.loc[df["date"] >= pd.Timestamp("2021-08-03"), "match_id"])
    assert set(result["match_id"]) == eligible
    assert set(result["outcome"]) == set(OUTCOMES)
    assert set(result["league"]) == {"EPL"}
    assert result.groupby("match_id").size().eq(3).all()
    assert np.allclose(result.groupby("match_id")["model_prob"].sum(), 1.0)

    in_order = walk_forward(
        df, Factory(), min_train_days=2, refit_every_days=99
    )
    pd.testing.assert_frame_equal(result, in_order)


@pytest.mark.parametrize(
    "min_train, first_fold",
    [(1, "2021-08-02"), (2, "2021-08-03"), (5, "2021-08-06")],
)
def test_days_without_enough_history_are_skipped(min_train, first_fold):
    df = _make_frame(days=8)
    factory = Factory()
    result = walk_forward(df, factory, min_train_days=min_train, refit_every_days=99)

    assert result["date"].min() == pd.Timestamp(first_fold)
    assert factory.calls == 1  # one fit: the first fold's, reused afterwards


def test_a_stray_context_column_from_the_model_cannot_corrupt_context():
    class StrayDate:
        def fit(self, train_df: pd.DataFrame) -> "StrayDate":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            pred = _uniform(df)
            pred["date"] = pd.Timestamp("1999-01-01")
            return pred

    df = _make_frame(days=5, hours=(12, 18))
    result = walk_forward(df, Factory(StrayDate), min_train_days=1, refit_every_days=99)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert (result["date"].dt.year == 2021).all()


def test_context_follows_match_id_when_the_model_returns_rows_backwards():
    class ShuffledOrder:
        def fit(self, train_df: pd.DataFrame) -> "ShuffledOrder":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            return _uniform(df).iloc[::-1]

    df = _make_frame(days=5, hours=(12, 18))
    result = walk_forward(
        df, Factory(ShuffledOrder), min_train_days=1, refit_every_days=99
    )

    expected_date = df.set_index("match_id")["date"]
    assert (result["date"] == result["match_id"].map(expected_date)).all()
    assert set(result["league"]) == {"EPL"}


def test_empty_frame_returns_the_typed_empty_result():
    empty = pd.DataFrame(
        {
            "match_id": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "league": pd.Series(dtype="object"),
        }
    )
    factory = Factory()
    result = walk_forward(empty, factory, min_train_days=2, refit_every_days=3)

    assert result.empty
    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["model_prob"].dtype == "float64"
    assert factory.calls == 0


def test_a_frame_shorter_than_the_history_window_returns_no_folds():
    factory = Factory()
    result = walk_forward(
        _make_frame(days=4), factory, min_train_days=99, refit_every_days=1
    )

    assert result.empty
    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert factory.calls == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_train_days": 0},
        {"min_train_days": -1},
        {"min_train_days": 1.5},
        {"min_train_days": True},
        {"refit_every_days": 0},
        {"refit_every_days": "3"},
    ],
)
def test_cadence_parameters_must_be_positive_integers(overrides):
    params = {"min_train_days": 2, "refit_every_days": 3}
    params.update(overrides)
    with pytest.raises(ValueError, match="must be an integer"):
        walk_forward(_make_frame(days=6), Factory(), **params)


def _missing_league() -> pd.DataFrame:
    return _make_frame(days=4).drop(columns=["league"])


def _string_dates() -> pd.DataFrame:
    df = _make_frame(days=4)
    return df.assign(date=df["date"].astype(str))


def _with_a_gap() -> pd.DataFrame:
    df = _make_frame(days=4)
    df.loc[df.index[-1], "date"] = pd.NaT
    return df


def _with_a_duplicate() -> pd.DataFrame:
    df = _make_frame(days=4)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _not_a_frame() -> dict:
    return {"match_id": [], "date": [], "league": []}


BAD_FRAMES = [
    (_missing_league, ValueError, "missing required column"),
    (_string_dates, TypeError, "datetime64"),
    (_with_a_gap, ValueError, "no date"),
    (_with_a_duplicate, ValueError, "duplicate match_id"),
    (_not_a_frame, TypeError, "DataFrame"),
]


@pytest.mark.parametrize("build, error, expected", BAD_FRAMES)
def test_bad_frames_are_rejected_before_any_fitting(build, error, expected):
    factory = Factory()
    with pytest.raises(error, match=expected):
        walk_forward(build(), factory, min_train_days=2, refit_every_days=3)
    assert factory.calls == 0


def test_model_factory_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        walk_forward(_make_frame(days=4), None, min_train_days=2, refit_every_days=3)


def test_model_factory_must_return_a_market_model():
    with pytest.raises(TypeError, match="MarketModel"):
        walk_forward(
            _make_frame(days=4), lambda: object(), min_train_days=2, refit_every_days=3
        )


def test_predict_dropping_a_match_is_refused():
    class DropFirstMatch:
        def fit(self, train_df: pd.DataFrame) -> "DropFirstMatch":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            pred = _uniform(df)
            return pred[pred["match_id"] != df["match_id"].iloc[0]]

    with pytest.raises(ValueError, match="covered"):
        walk_forward(
            _make_frame(days=5, hours=(12, 18)),
            Factory(DropFirstMatch),
            min_train_days=1,
            refit_every_days=99,
        )


def test_predict_inventing_a_match_is_refused():
    class ExtraMatch:
        def fit(self, train_df: pd.DataFrame) -> "ExtraMatch":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            ghost = _uniform(pd.DataFrame({"match_id": ["ghost-match"]}))
            return pd.concat([_uniform(df), ghost], ignore_index=True)

    with pytest.raises(ValueError, match="unexpected"):
        walk_forward(
            _make_frame(days=5, hours=(12, 18)),
            Factory(ExtraMatch),
            min_train_days=1,
            refit_every_days=99,
        )


def test_predict_returning_something_other_than_a_frame_is_refused():
    class ReturnList:
        def fit(self, train_df: pd.DataFrame) -> "ReturnList":
            return self

        def predict(self, df: pd.DataFrame) -> object:
            return [("ghost-match", "H", 1.0)]

    with pytest.raises(TypeError, match="DataFrame"):
        walk_forward(
            _make_frame(days=5),
            Factory(ReturnList),
            min_train_days=1,
            refit_every_days=99,
        )


def test_probabilities_that_do_not_sum_to_one_are_refused():
    class FlatProbs:
        def fit(self, train_df: pd.DataFrame) -> "FlatProbs":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            pred = _uniform(df)
            pred["prob"] = 0.4
            return pred

    with pytest.raises(ValueError, match="sum to 1"):
        walk_forward(
            _make_frame(days=5),
            Factory(FlatProbs),
            min_train_days=1,
            refit_every_days=99,
        )


def test_predict_without_the_contract_columns_is_refused():
    class MissingOutcome:
        def fit(self, train_df: pd.DataFrame) -> "MissingOutcome":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            return _uniform(df).drop(columns=["outcome"])

    with pytest.raises(ValueError) as exc:
        walk_forward(
            _make_frame(days=5),
            Factory(MissingOutcome),
            min_train_days=1,
            refit_every_days=99,
        )
    assert "predict() missing column" in str(exc.value)


def test_outcome_labels_must_not_change_mid_run():
    class ShiftingLabels:
        def __init__(self) -> None:
            self.calls = 0

        def fit(self, train_df: pd.DataFrame) -> "ShiftingLabels":
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
            pred = _uniform(df)
            if self.calls == 1:
                return pred
            pred = pred[pred["outcome"] != "A"].copy()
            pred["prob"] = 0.5
            return pred

    with pytest.raises(ValueError) as exc:
        walk_forward(
            _make_frame(days=6, hours=(12, 18)),
            Factory(ShiftingLabels),
            min_train_days=1,
            refit_every_days=99,
        )
    message = str(exc.value)
    assert "changed mid-run" in message
    assert "2021-08-03" in message  # the second fold is where drift appears
