"""Naive frequency model: historical H/D/A rates from the training window.

The deliberately dumb baseline. Every row gets the same vector — how often
each outcome actually happened in the training data — regardless of who is
playing, where, or at what price. It exists so the system has a model that
is honest about being weak: if THIS beats the market, either the market is
broken or the evaluation is, and if it loses, that is the yardstick a real
model has to beat.

No smoothing, on purpose: add-one priors would make the numbers look
well-behaved at 0 while hiding that the window genuinely never saw an
outcome, and this model's job is to be plainly what it is. Frequencies are
relative frequencies of the settlement column over *settled* training rows;
an outcome the window missed predicts 0, which calibration then honestly
scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class NaiveFrequency:
    """Outcome frequencies from the fit window; constant across predictions."""

    def __init__(self, outcomes: Sequence[str]) -> None:
        outcome_list = tuple(str(outcome) for outcome in outcomes)
        if not outcome_list:
            raise ValueError("outcomes must not be empty")
        self.outcomes = outcome_list
        #: Relative frequency per outcome, set by fit; None until then so a
        #: predict-before-fit is an error instead of a guess.
        self.frequencies: dict[str, float] | None = None

    def fit(self, train_df: pd.DataFrame) -> NaiveFrequency:
        """Estimate relative outcome frequencies from ``train_df``.

        Only settled rows count: an unsettled row is not evidence of
        anything, and counting it as a fourth, unnamed outcome would break
        the probabilities' sum. Unknown labels are refused rather than
        dropped into a bucket, because a garbled settlement code means the
        frame was not normalised by data.ingest.
        """
        if "ftr" not in train_df.columns:
            raise ValueError(
                f"naive_frequency needs the settlement column 'ftr'; "
                f"have {list(train_df.columns)}"
            )
        settled = train_df["ftr"].dropna().astype(str).str.strip().str.upper()
        known = settled[settled.isin(self.outcomes)]
        if known.empty:
            raise ValueError(
                "naive_frequency.fit got no settled rows to count "
                f"({len(train_df)} training row(s) supplied)"
            )
        counts = known.value_counts()
        total = int(counts.sum())
        self.frequencies = {
            outcome: float(counts.get(outcome, 0)) / total
            for outcome in self.outcomes
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """The fit's frequency vector, repeated for every row of ``df``."""
        if self.frequencies is None:
            raise RuntimeError("predict() called before fit()")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
        vector = np.array(
            [self.frequencies[outcome] for outcome in self.outcomes],
            dtype=float,
        )
        count = len(df)
        return pd.DataFrame(
            {
                "match_id": np.repeat(df["match_id"].to_numpy(), len(self.outcomes)),
                "outcome": np.tile(np.asarray(self.outcomes, dtype=object), count),
                "prob": np.tile(vector, count),
            }
        )


__all__ = ["NaiveFrequency"]
