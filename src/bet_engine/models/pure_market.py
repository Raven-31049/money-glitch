"""Pure-market control: the de-vigged market, used as the model itself.

Invariant 4 (AGENTS.md): this "model" must place ~0 bets â€” any bet it places
is an EV/Kelly bug, not an edge. It predicts exactly what the market's own
prices, de-vigged, say. EV is measured on the raw odds actually paid
(IMPLEMENTATION_NOTES.md §5): for every outcome the control's edge is
``market_prob * raw_odds - 1 = 1 / overround - 1 < 0``, because the de-vigged
probability times the book's price is the inverse overround. ``select_bets``'
strict ``>`` therefore refuses every row.

WHY the reference de-vig lives in this module: ``market_prob`` (the run's
column for "what the market says after vig") and this model's prediction are
the same quantity by definition, so the run computes ``market_prob`` by
calling :func:`devig_long` â€” the very function ``predict`` calls â€” and the
two are bit-identical, never merely close. That identity keeps the control's
edge at the single negative value ``1 / overround - 1`` rather than drifting
with a second de-vig: a hand-rolled de-vig anywhere else in the pipeline
would break it.

The run also imports :func:`devig_long` for its ``market_prob`` column, so
the control's identity is structural, not a convention someone must
remember.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

#: Long-form columns every prediction must carry (models.PREDICTION_COLUMNS),
#: spelled out here for the frame this module builds.
_PREDICT_COLUMNS = ("match_id", "outcome", "prob")


def devig_long(
    df: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    value_name: str = "prob",
    columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Proportionally de-vig ``odds_<outcome>`` columns into long form.

    One row per outcome per match: ``[match_id, outcome, value_name]``, with
    probabilities summing to 1 per match. Rows without a complete, usable
    price vector come back with NaN â€” this function is arithmetic and does
    not decide what "unpriced" means; callers do (``run.py`` filters before
    calling, and :meth:`PureMarket.predict` refuses what reaches it).

    ``columns`` overrides the default ``odds_<outcome>`` naming with an explicit
    ``{outcome: column}`` map: the cards over/under market prices off
    ``over_odds``/``under_odds`` (markets/cards_totals.py), so a control bound
    to that market must say which columns its prices are in. Omitted, the
    default naming is used, and an outcome missing from a supplied map is an
    error rather than a silent NaN.

    Vectorised over rows: one ``1 / odds`` pass, one row-sum, one division â€”
    no per-row Python (invariant 6). Row-wise results are independent of the
    other rows in the frame, which is what lets ``predict`` on one fold's
    batch produce the same bits as the run's whole-frame ``market_prob``
    (see the module docstring).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    outcome_list = tuple(outcomes)
    if not outcome_list:
        raise ValueError("outcomes must not be empty")
    if columns is None:
        column_map = {outcome: f"odds_{outcome}" for outcome in outcome_list}
    else:
        column_map = {
            str(outcome): str(column) for outcome, column in columns.items()
        }
        missing_outcomes = [o for o in outcome_list if o not in column_map]
        if missing_outcomes:
            raise ValueError(
                f"columns map is missing outcome(s) {missing_outcomes}; "
                f"have {sorted(column_map)}"
            )
    odds_columns = [column_map[outcome] for outcome in outcome_list]
    missing = [column for column in odds_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing price column(s) {missing}; attach odds first "
            f"(markets.attach_odds)"
        )
    try:
        prices = df[odds_columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError("price columns must be numeric") from None

    implied = 1.0 / prices
    total = implied.sum(axis=1)  # per-row: NaN rows stay NaN, never poison others
    matrix = implied / total[:, None]

    count = len(df)
    return pd.DataFrame(
        {
            "match_id": np.repeat(df["match_id"].to_numpy(), len(outcome_list)),
            "outcome": np.tile(np.asarray(outcome_list, dtype=object), count),
            value_name: matrix.ravel(),
        }
    )


class PureMarket:
    """The control: probabilities are the run's own de-vigged market prices.

    No parameters to estimate, no history to learn â€” ``fit`` exists to
    satisfy the model contract walk-forward calls, not to do anything. That
    is the point of a control: it holds model weight at zero so any bet that
    survives selection is provably a bug in the edge or sizing code, never
    an edge (invariant 4).
    """

    def __init__(
        self,
        outcomes: Sequence[str],
        *,
        odds_columns: Mapping[str, str] | None = None,
    ) -> None:
        outcome_list = tuple(str(outcome) for outcome in outcomes)
        if not outcome_list:
            raise ValueError("outcomes must not be empty")
        if odds_columns is not None:
            odds_columns = {
                str(outcome): str(column)
                for outcome, column in odds_columns.items()
            }
            missing = [o for o in outcome_list if o not in odds_columns]
            if missing:
                raise ValueError(
                    f"odds_columns is missing outcome(s) {missing}; "
                    f"have {sorted(odds_columns)}"
                )
        self.outcomes = outcome_list
        self.odds_columns = odds_columns

    def fit(self, train_df: pd.DataFrame) -> PureMarket:
        """Learn nothing; the market is already priced. Returns self."""
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """De-vigged market probabilities for every row of ``df``.

        A row without a complete price vector raises rather than predicting
        something: the control has no other information to fall back on, and
        a NaN or a guessed probability here would silently flow into EV,
        calibration and the control's own invariant.
        """
        long = devig_long(
            df, self.outcomes, value_name="prob", columns=self.odds_columns
        )
        finite = np.isfinite(long["prob"].to_numpy())
        if not finite.all():
            bad_matches = long.loc[~finite, "match_id"].unique()[:5].tolist()
            raise ValueError(
                f"pure_market cannot price match(es) {bad_matches}: no "
                f"complete, valid odds vector (the run's odds policy "
                f"skipped them upstream)"
            )
        return long.loc[:, list(_PREDICT_COLUMNS)]

    def predict_samples(
        self, df: pd.DataFrame, n: int = 1000
    ) -> np.ndarray:
        """Zero-uncertainty draws: every draw is the de-vigged probability.

        The control has no parameters to be unsure about, so its whole sampled
        distribution is a point mass on the de-vigged probability. That makes
        the two selection paths agree by construction, which is exactly the
        property invariant 4 needs: the q-th percentile of a constant is that
        constant for every q, so :func:`bet_engine.backtest.select_bets_at_percentile`
        places the same (zero) bets the point path does. Return shape follows
        the sampling contract: ``(n, len(df), n_outcomes)``, ``df`` order on the
        match axis and ``self.outcomes`` order on the last.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be an integer >= 1, got {n!r}")
        point = self.predict(df)  # raises on any unpriced row, like predict
        order = list(self.outcomes)
        wide = (
            point.pivot(index="match_id", columns="outcome", values="prob")
            .reindex(index=df["match_id"].tolist(), columns=order)
            .to_numpy(dtype=float)
        )
        return np.broadcast_to(wide, (n, *wide.shape)).copy()


__all__ = ["PureMarket", "devig_long"]
