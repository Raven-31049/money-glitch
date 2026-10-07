"""Calendar-day walk-forward fold engine.

Invariant 1 (AGENTS.md): backtests batch by CALENDAR DAY, and training data is
STRICTLY earlier than the prediction day. Everything here exists to make that
true and to make a violation loud:

* the cut is a calendar-day boundary (``date < day_start`` against the day's
  own rows), so kick-off times inside a day can never straddle it — a match at
  18:00 on the fold's day is not information a model may train on;
* :func:`assert_train_before_predict` runs before every fit and again before
  every predict, and is a raised ``AssertionError`` rather than an ``assert``
  statement, because a leak check that ``python -O`` can switch off is not a
  leak check. The message names both dates and both row spans;
* the split lives in its own function so a test can inject a deliberately
  leaky one and prove the assertion catches it.

The engine is market-agnostic (invariant 8): this module needs only
``match_id``/``date``/``league`` columns and a model factory. It never looks at
outcomes, odds or settlement — those belong to markets/ and to the stages
downstream of this one.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..models import PREDICTION_COLUMNS, MarketModel, check_predictions

#: Columns of the returned frame. Long form: one row per outcome per match,
#: with the day and league that produced it, ready for a caller to stamp with
#: run_id/market and hand to db.write_predictions.
WALK_FORWARD_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "league",
    "outcome",
    "model_prob",
)

#: The minimum a frame must carry: everything else is context this engine does
#: not need.
_REQUIRED_COLUMNS: tuple[str, ...] = ("match_id", "date", "league")

#: How a frame is cut for one prediction day: (training rows, prediction batch).
#: Injectable so leakage tests can substitute a broken splitter — see
#: ``tests/test_walkforward.py``.
SplitFn = Callable[
    [pd.DataFrame, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]
]


def split_by_day(
    df: pd.DataFrame, predict_day: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows strictly before ``predict_day``'s calendar day, and the day's rows.

    WHY a day boundary and not an index or timestamp cut: the frame carries no
    kick-off time, so two matches on one calendar day give no order to learn
    from, and treating row order as time would let a model train on the first
    half of a matchday and predict the second. Both sides of the cut are
    returned with their original index, so ``pd.concat([train, predict])``
    reconstructs the frame — the partition is exact, never overlapping.

    ``predict_day`` may be a Timestamp, a ``date`` or an ISO string; it is
    normalised to midnight, which is the fold's ``day_start``.
    """
    day = pd.Timestamp(predict_day)
    if pd.isna(day):
        raise ValueError(f"predict_day must be a real calendar date, got {predict_day!r}")
    day = day.normalize()
    next_day = day + pd.Timedelta(days=1)
    train = df[df["date"] < day]
    predict = df[(df["date"] >= day) & (df["date"] < next_day)]
    return train, predict


def assert_train_before_predict(
    train: pd.DataFrame, predict: pd.DataFrame, *, context: str = ""
) -> None:
    """Raise AssertionError unless training ends strictly before the prediction day.

    The invariant written out (PLAN.md, invariant 1):
    ``max(train.timestamp) < fold.day_start``, where the fold's day start is
    the earliest prediction date normalised to midnight. Comparing against the
    *day start* rather than the batch's earliest kick-off time is the point:
    a model that saw this calendar day's 12:00 row predicting the same day's
    18:00 row has trained on the matchday, and timestamp order alone would
    wave that through. An empty side, a NaT date, an equality and any
    same-day overlap all fail — there is nothing to certify in those cases,
    and silence would be indistinguishable from a clean result.

    Deliberately ``raise AssertionError`` instead of an ``assert`` statement:
    ``-O``/``-OO`` removes ``assert``, and this is the check the whole system's
    numbers depend on (invariant 1). The message carries both dates and both
    row spans so a failing fold reports itself without a debugger.
    """
    train_max = train["date"].max() if len(train) else pd.NaT
    train_min = train["date"].min() if len(train) else pd.NaT
    predict_min = predict["date"].min() if len(predict) else pd.NaT
    day_start = predict_min.normalize() if not pd.isna(predict_min) else pd.NaT

    undated = int(train["date"].isna().sum()) + int(predict["date"].isna().sum())
    if undated:
        raise AssertionError(
            "temporal leak (invariant 1) cannot be certified: "
            f"{undated} row(s) carry no date, so their position relative to "
            f"the fold is unknown; train rows={len(train)}, predict rows="
            f"{len(predict)}"
            + (f" [{context}]" if context else "")
        )

    if (
        pd.isna(train_max)
        or pd.isna(day_start)
        or not (train_max < day_start)
    ):
        raise AssertionError(
            "temporal leak (invariant 1): training data must end strictly "
            f"before the prediction day, but train.date.max()="
            f"{_stamp(train_max)} is not before the prediction day start="
            f"{_stamp(day_start)} (predict.date.min()="
            f"{_stamp(predict_min)}); train rows={len(train)} spanning "
            f"{_stamp(train_min)}..{_stamp(train_max)}, predict rows="
            f"{len(predict)}"
            + (f" [{context}]" if context else "")
        )


def walk_forward(
    df: pd.DataFrame,
    model_factory: Callable[[], MarketModel],
    min_train_days: int,
    refit_every_days: int,
    *,
    split: SplitFn = split_by_day,
) -> pd.DataFrame:
    """Fit and predict over successive calendar days; return long-form predictions.

    For each calendar day ``d`` in the frame, in order:

    * days are skipped until the history spans at least ``min_train_days``
      calendar days (the frame's first day to ``d``);
    * ``train, batch = split(df, d)`` and the split is asserted strict
      immediately — a leaky split is caught on the fold it first affects, not
      at the next refit;
    * the model is refitted only when ``refit_every_days`` calendar days have
      passed since the last fit; the reused model's training data is older
      still, so it is re-asserted against today's batch before predicting;
    * ``predict(batch)`` runs, and its output is checked against the model
      contract (:func:`~bet_engine.models.check_predictions`) plus coverage of
      the batch's matches — this is the engine/model boundary, where a bad
      frame is still attributable to the model.

    ``model_factory`` builds a *fresh* model at each refit rather than reusing
    one instance: the contract says ``fit`` estimates from the rows it is
    given, not that it resets prior state, so a fresh object is the only way to
    guarantee a refit is a clean refit.

    ``split`` is keyword-only and normally :func:`split_by_day`; it exists as
    an injection point for the leakage tests, which substitute a splitter that
    trains on the prediction day and require the assertion to refuse it.

    Rows come back sorted by ``(date, match_id)``: intra-day order is not
    information the frame carries (there is no kick-off time), so sorting
    makes the result a pure function of the frame's contents — a shuffled
    input produces byte-identical output — while giving callers chronological
    order for free.
    """
    _validate_frame(df)
    min_train_days = _positive_int("min_train_days", min_train_days)
    refit_every_days = _positive_int("refit_every_days", refit_every_days)
    if not callable(model_factory):
        raise TypeError(
            f"model_factory must be callable, got {type(model_factory).__name__}"
        )
    if df.empty:
        return _empty_predictions()

    days = df["date"].dt.normalize().drop_duplicates().sort_values()
    first_day = days.iloc[0]

    outputs: list[pd.DataFrame] = []
    expected_outcomes: frozenset[str] | None = None
    model: MarketModel | None = None
    fit_day: pd.Timestamp | None = None
    fit_train: pd.DataFrame | None = None

    for day in days:
        if (day - first_day).days < min_train_days:
            continue  # not enough history yet: skip the day, never fill it

        train, batch = split(df, day)
        assert_train_before_predict(
            train, batch, context=f"split for prediction day {_stamp(day)}"
        )

        if fit_day is None or (day - fit_day).days >= refit_every_days:
            fitted = model_factory()
            if not isinstance(fitted, MarketModel):
                raise TypeError(
                    "model_factory() must return a MarketModel (fit + predict), "
                    f"got {type(fitted).__name__}"
                )
            fitted.fit(train)
            model, fit_day, fit_train = fitted, day, train

        # The model's provenance, checked literally immediately before the
        # predict call it guards: this model saw only rows from before `day`.
        # `model`/`fit_train` are always set here — the first fold fits.
        assert_train_before_predict(
            fit_train,
            batch,
            context=(
                f"model fitted on {_stamp(fit_day)} predicting {_stamp(day)}"
            ),
        )
        assert model is not None  # keeps the hint honest without a dead branch

        day_pred = _predict_one_day(model, batch)
        outcomes = frozenset(day_pred["outcome"])
        if expected_outcomes is None:
            expected_outcomes = outcomes
        elif outcomes != expected_outcomes:
            raise ValueError(
                f"predict() outcome labels changed mid-run at {_stamp(day)}: "
                f"{sorted(outcomes)} vs {sorted(expected_outcomes)} earlier"
            )
        outputs.append(day_pred)

    if not outputs:
        return _empty_predictions()
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["date", "match_id"]).reset_index(drop=True)


def _predict_one_day(model: MarketModel, batch: pd.DataFrame) -> pd.DataFrame:
    """Predict one batch and return it with ``date``/``league`` attached.

    Context comes from a merge on ``match_id``, never positional assignment:
    a model is free to return rows in any order, and zipping columns by
    position would quietly mislabel predictions with another match's date —
    exactly the class of silent corruption this module exists to prevent.
    """
    pred = model.predict(batch)
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(
            f"predict() must return a DataFrame, got {type(pred).__name__}"
        )
    missing = [column for column in PREDICTION_COLUMNS if column not in pred.columns]
    if missing:
        raise ValueError(
            f"predict() missing column(s) {missing}; have {list(pred.columns)}"
        )
    # The engine knows no market's label list (invariant 8), so the contract
    # check derives labels from the model's own output: every match must carry
    # one row per label, and walk_forward() holds that label set constant
    # across days.
    pred = check_predictions(pred, sorted(set(pred["outcome"])))
    # Contract columns only — date/league come from the batch below, and a
    # stray "date" column from the model would collide in the merge.
    pred = pred.loc[:, list(PREDICTION_COLUMNS)]

    expected_ids = set(batch["match_id"])
    predicted_ids = set(pred["match_id"])
    if predicted_ids != expected_ids:
        raise ValueError(
            f"predict() covered {len(predicted_ids & expected_ids)} of "
            f"{len(expected_ids)} matches in the batch; missing "
            f"{sorted(expected_ids - predicted_ids)[:5]}, unexpected "
            f"{sorted(predicted_ids - expected_ids)[:5]}"
        )

    merged = pred.merge(
        batch[["match_id", "date", "league"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.rename(columns={"prob": "model_prob"})
    return merged.loc[:, list(WALK_FORWARD_COLUMNS)]


def _validate_frame(df: pd.DataFrame) -> None:
    """Reject a frame this engine cannot fold without guessing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing required column(s) {missing}; have {list(df.columns)}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError(
            "df['date'] must be datetime64 — load frames through "
            "data.ingest.load_league"
        )
    undated = df["date"].isna()
    if undated.any():
        examples = df.loc[undated, "match_id"].head(3).tolist()
        raise ValueError(
            f"df has {int(undated.sum())} row(s) with no date; a match that "
            f"cannot be placed on a calendar day cannot be folded "
            f"(e.g. {examples})"
        )
    duplicated = df["match_id"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "match_id"].head(3).tolist()
        raise ValueError(
            f"df has {int(duplicated.sum())} duplicate match_id(s), e.g. "
            f"{examples}; predictions would be ambiguous"
        )


def _positive_int(name: str, value: object) -> int:
    """Validate a cadence parameter: an integer >= 1 (booleans are not counts)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _stamp(value: object) -> str:
    """A timestamp (or its absence) as text for assertion messages."""
    if value is None or pd.isna(value):
        return "<none>"
    return str(value)


def _empty_predictions() -> pd.DataFrame:
    """The result for a frame with no foldable rows: right columns, right dtypes."""
    return pd.DataFrame(
        {
            "match_id": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "league": pd.Series(dtype="object"),
            "outcome": pd.Series(dtype="object"),
            "model_prob": pd.Series(dtype="float64"),
        }
    )


__all__ = [
    "WALK_FORWARD_COLUMNS",
    "SplitFn",
    "assert_train_before_predict",
    "split_by_day",
    "walk_forward",
]
