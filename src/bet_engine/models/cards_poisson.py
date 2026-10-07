"""Team-level Poisson card model for the cards over/under market (Phase 1).

Two models share one implementation here:

* :class:`CardsPoisson` — the team model. Each side's expected own cards come
  from a Poisson regression on its own shrunk card rate, the opponent's rate,
  and a home/away flag. The expected match total is ``home mu + away mu``, and
  over/under probabilities are read off that Poisson total.
* :class:`LeagueAveragePoisson` — the baseline: the same regression with the
  team rates dropped, so it knows only the league's average cards per side plus
  the home/away flag. It is the yardstick the team model has to beat.

WHY the target is each side's *own* cards rather than the match total: the plan
asks for each team's expected cards per match as a function of its own rate,
the opponent's rate and a home/away flag, and the match total is the sum of the
two sides. Fitting team-match rows (two per match) gives both sides from one
fit, which is also what keeps this one model per league (invariant 7).

WHY the features are built *before* fit rather than inside it: ``predict``
receives only the prediction day's rows, which carry no history of their own.
So :func:`add_card_features` builds the feature columns once over the whole
league frame, and the guarantee that every row's rates come from matches
strictly before that row's date is enforced at runtime by
:func:`assert_features_are_causal` — invariant 1 extended to features.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import ClassVar

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

#: History columns :func:`add_card_features` reads. Nothing else is consulted.
HISTORY_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home",
    "away",
    "hy",
    "ay",
    "hr",
    "ar",
)

#: Raw card columns, in the order the causality probe perturbs them.
CARD_COLUMNS: tuple[str, ...] = ("hy", "ay", "hr", "ar")

#: What the models refuse to run without — all produced by add_card_features.
FEATURE_COLUMNS: tuple[str, ...] = (
    "home_rate",
    "away_rate",
    "home_cards",
    "away_cards",
    "total_cards",
)

#: The features the causality probe watches. The ``*_cards`` targets are
#: deliberately absent: perturbing a date *must* change that date's card
#: counts, so watching them would report every probe as a leak.
_PROBE_COLUMNS: tuple[str, ...] = ("home_rate", "away_rate", "league_mean")

#: How many spread-out dates the causality probe exercises.
_PROBE_COUNT = 3

#: Cards added to a probed row's counts. Large enough that a real change can
#: never be mistaken for float noise; the comparison below is exact anyway.
_PROBE_DELTA = 10.0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def add_card_features(df: pd.DataFrame, k: float, *, verify: bool = True) -> pd.DataFrame:
    """``df`` plus causal per-team card rates.

    ``k`` is the shrinkage weight in pseudo-matches: a team's rate is

        (own cards in strictly-earlier matches + k * league mean before this date)
        / (strictly-earlier matches + k)

    so a team with no history (promoted side, season start) gets exactly the
    league average as of that date, and real history dilutes the prior as it
    accumulates. ``k`` is a config parameter (``model.params.k``), not a
    constant, because how fast a rate should be trusted is a modelling choice a
    variant config has to be able to state.

    ``verify`` runs :func:`assert_features_are_causal` on the result. It costs a
    handful of extra feature passes (milliseconds at league scale) and turns
    "features never see their own result" from a claim in a docstring into a
    runtime assertion — the same shape as the walk-forward leak check.
    """
    _positive_float("k", k)
    _require_columns(df, HISTORY_COLUMNS)
    features = _add_card_features(df, k)
    if verify:
        assert_features_are_causal(_add_card_features, df, k, context="add_card_features")
    return features


def _add_card_features(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """The computation itself: no verification, so the probe cannot recurse."""
    out = df.copy()
    hy = pd.to_numeric(out["hy"], errors="coerce")
    hr = pd.to_numeric(out["hr"], errors="coerce")
    ay = pd.to_numeric(out["ay"], errors="coerce")
    ar = pd.to_numeric(out["ar"], errors="coerce")
    out["home_cards"] = hy + hr
    out["away_cards"] = ay + ar
    out["total_cards"] = out["home_cards"] + out["away_cards"]

    # Two rows per match: one per side, each carrying that side's own cards.
    # `side` exists only so the rates can be mapped back onto the match rows.
    home_side = pd.DataFrame(
        {
            "match_id": out["match_id"].to_numpy(),
            "date": out["date"].to_numpy(),
            "team": out["home"].to_numpy(),
            "side": "home",
            "own_cards": out["home_cards"].to_numpy(),
        }
    )
    away_side = pd.DataFrame(
        {
            "match_id": out["match_id"].to_numpy(),
            "date": out["date"].to_numpy(),
            "team": out["away"].to_numpy(),
            "side": "away",
            "own_cards": out["away_cards"].to_numpy(),
        }
    )
    long = pd.concat([home_side, away_side], ignore_index=True)

    # Only dated rows with usable card counts are evidence of anything; a row
    # whose source columns are missing contributes neither history nor a rate.
    usable = long["date"].notna() & long["own_cards"].notna()

    # Aggregate per (team, date) first, so a team playing twice on one date
    # still sees nothing from that date — the calendar day, not the row, is the
    # boundary (same rule the walk-forward fold engine uses).
    history = (
        long.loc[usable]
        .groupby(["team", "date"], sort=True)["own_cards"]
        .agg(past_sum="sum", past_count="count")
        .reset_index()
        .sort_values(["team", "date"], kind="mergesort")
    )
    history["past_sum"] = (
        history.groupby("team")["past_sum"].cumsum() - history["past_sum"]
    )
    history["past_count"] = (
        history.groupby("team")["past_count"].cumsum() - history["past_count"]
    )

    # League mean per team-match row, likewise strictly before each date.
    by_date = (
        long.loc[usable]
        .groupby("date")["own_cards"]
        .agg(day_sum="sum", day_count="count")
        .sort_index()
    )
    past_league_sum = by_date["day_sum"].cumsum() - by_date["day_sum"]
    past_league_count = by_date["day_count"].cumsum() - by_date["day_count"]
    # The first date has no earlier match, so its mean is NaN rather than 0/0:
    # there is genuinely no league average to shrink toward yet.
    league_mean = past_league_sum / past_league_count.replace(0, np.nan)

    merged = long.merge(history, on=["team", "date"], how="left")
    merged["league_mean"] = merged["date"].map(league_mean)
    # past_count stays NaN for unusable rows, so their rate is NaN too and
    # fit() drops them instead of treating missing data as zero cards.
    merged["rate"] = (merged["past_sum"] + k * merged["league_mean"]) / (
        merged["past_count"] + k
    )

    home_rate = merged.loc[merged["side"] == "home"].set_index("match_id")["rate"]
    away_rate = merged.loc[merged["side"] == "away"].set_index("match_id")["rate"]
    out["home_rate"] = out["match_id"].map(home_rate).astype("float64")
    out["away_rate"] = out["match_id"].map(away_rate).astype("float64")
    out["league_mean"] = out["date"].map(league_mean).astype("float64")
    return out


def assert_features_are_causal(
    feature_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    df: pd.DataFrame,
    k: float,
    *,
    probes: int = _PROBE_COUNT,
    context: str = "",
) -> None:
    """Raise AssertionError unless ``feature_fn`` ignores same-day and later rows.

    WHY a perturbation check rather than a recomputation: any recomputation
    would use the same code path as the feature function and could only prove
    the function agrees with itself. The property the correct answer must have
    is stated in terms of the data: changing a match's card counts may change
    features of *later* matches, and must change nothing on or before that
    match. A function that reads its own match, its own matchday or any future
    row is therefore caught red-handed.

    Two probe shapes per probed date, because they catch different leaks:

    * perturb that date's rows -> every row dated **on or before** it must be
      unchanged (catches same-day and self-inclusion);
    * perturb every row **after** it -> those same rows must be unchanged
      (catches a strictly-future leak, which the first shape alone would miss).

    Deliberately raises AssertionError rather than ``assert``: like the
    walk-forward check, this is a guard the whole evaluation rests on and
    ``python -O`` must not be able to switch it off.
    """
    if not callable(feature_fn):
        raise TypeError(
            f"feature_fn must be callable, got {type(feature_fn).__name__}"
        )
    _require_columns(df, HISTORY_COLUMNS)
    if isinstance(probes, bool) or not isinstance(probes, int) or probes < 1:
        raise ValueError(f"probes must be an integer >= 1, got {probes!r}")
    if df.empty:
        return

    base = feature_fn(df, k)
    ran = 0
    for probe in _probe_dates(df, probes):
        on_probe = _perturbed_frame(df, df["date"] == probe)
        if on_probe is not None:
            ran += 1
            _require_unchanged(
                base,
                feature_fn(on_probe, k),
                df["date"] <= probe,
                probe=probe,
                leak="on or after",
                context=context,
            )
        after_probe = df["date"] > probe
        if after_probe.any():
            perturbed = _perturbed_frame(df, after_probe)
            if perturbed is not None:
                ran += 1
                _require_unchanged(
                    base,
                    feature_fn(perturbed, k),
                    df["date"] <= probe,
                    probe=probe,
                    leak="after",
                    context=context,
                )
    if ran == 0:
        # Nothing with card data to perturb: the property is untestable here
        # rather than proven, so say so instead of passing in silence.
        raise AssertionError(
            "causality probe could not run: no row carries dated card counts, "
            f"so there is nothing to perturb ({len(df)} row(s))"
            + (f" [{context}]" if context else "")
        )


def _require_unchanged(
    base: pd.DataFrame,
    probe_result: pd.DataFrame,
    rows_of_interest: pd.Series,
    *,
    probe: pd.Timestamp,
    leak: str,
    context: str,
) -> None:
    """Assert the watched feature columns match on ``rows_of_interest``."""
    checked = base.loc[rows_of_interest, list(_PROBE_COLUMNS)].to_numpy(dtype=float)
    other = probe_result.loc[rows_of_interest, list(_PROBE_COLUMNS)].to_numpy(
        dtype=float
    )
    # NaN == NaN here: "no history yet" is a value, not a difference.
    differs = ~((checked == other) | (np.isnan(checked) & np.isnan(other)))
    if not differs.any():
        return
    hit_rows = np.flatnonzero(differs.any(axis=1))
    ids = base.loc[rows_of_interest, "match_id"].to_numpy()[hit_rows[:3]]
    raise AssertionError(
        "card features are not strictly causal: perturbing matches "
        f"{leak} {pd.Timestamp(probe).date()} changed {len(hit_rows)} feature "
        f"row(s) dated on or before that day, e.g. {list(ids)} — features may "
        "only use matches strictly before each row's date"
        + (f" [{context}]" if context else "")
    )


def _perturbed_frame(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame | None:
    """A copy with every card count under ``mask`` bumped; None if nothing to bump."""
    perturbed = df.copy()
    changed = 0
    for column in CARD_COLUMNS:
        values = pd.to_numeric(perturbed[column], errors="coerce")
        hit = mask & values.notna()
        if hit.any():
            values = values.astype("float64")
            values.loc[hit] = values.loc[hit] + _PROBE_DELTA
            perturbed[column] = values
            changed += int(hit.sum())
    return perturbed if changed else None


def _probe_dates(df: pd.DataFrame, probes: int) -> list[pd.Timestamp]:
    """Up to ``probes`` evenly spread dates that actually carry card counts."""
    usable = pd.Series(True, index=df.index)
    for column in CARD_COLUMNS:
        usable &= pd.to_numeric(df[column], errors="coerce").notna()
    candidates = np.sort(df.loc[usable & df["date"].notna(), "date"].unique())
    if not len(candidates):
        return []
    count = min(int(probes), len(candidates))
    positions = np.unique(
        np.linspace(0, len(candidates) - 1, num=count).round().astype(int)
    )
    return [pd.Timestamp(candidates[position]) for position in positions]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class CardsPoisson:
    """Poisson regression on team card rates, expressed as over/under lines."""

    #: Which design the subclass fits: the team model uses the rates, the
    #: baseline deliberately does not.
    uses_team_rates: ClassVar[bool] = True

    def __init__(self, outcomes: Sequence[str], line: float, k: float = 6.0) -> None:
        labels = tuple(str(outcome) for outcome in outcomes)
        if len(labels) != 2 or set(labels) != {"over", "under"}:
            raise ValueError(
                "cards model needs exactly the outcomes ('over', 'under'); "
                f"got {labels}"
            )
        self.outcomes = labels
        self.line = _positive_float("line", line)
        self.k = _positive_float("k", k)
        #: match_id -> expected total cards from this instance's own predict
        #: calls. NOT part of the MarketModel contract: the dispersion report
        #: needs mu, and walk_forward hands back probabilities only, so the
        #: model records what it predicted. The runner captures instances from
        #: the model factory to collect these.
        self.predicted_totals: dict[str, float] = {}
        self._beta: np.ndarray | None = None
        self.n_training_rows: int = 0
        self.n_dropped_rows: int = 0

    def fit(self, train_df: pd.DataFrame) -> CardsPoisson:
        """Estimate the Poisson coefficients from ``train_df``'s team rows."""
        frame = _require_features(train_df)
        rows = pd.concat(
            [_side_rows(frame, "home"), _side_rows(frame, "away")],
            ignore_index=True,
        )

        if self.uses_team_rates:
            # Checked before the design is built so a zero rate is reported as
            # what it is, rather than surfacing as a bare log(0) in a column.
            rates = rows[["own_rate", "opp_rate"]].to_numpy(dtype=float)
            bad = np.isfinite(rates) & (rates <= 0.0)
            if bad.any():
                raise ValueError(
                    "cards_poisson got a non-positive card rate; shrinkage "
                    f"k={self.k} must keep rates > 0 ({int(bad.sum())} cell(s))"
                )

        design = self._design(rows)
        endog = rows["own_cards"].to_numpy(dtype="float64")

        usable = np.isfinite(endog) & np.isfinite(design.to_numpy(dtype=float)).all(
            axis=1
        )
        self.n_dropped_rows = int((~usable).sum())
        self.n_training_rows = int(usable.sum())
        if not usable.any():
            raise ValueError(
                "cards_poisson.fit got no usable training rows "
                f"({len(rows)} team row(s) supplied, {self.n_dropped_rows} "
                "dropped for missing cards or features)"
            )

        result = sm.GLM(
            endog[usable],
            design.loc[usable],
            family=sm.families.Poisson(),
        ).fit()
        self._beta = result.params.to_numpy(dtype=float)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Over/under probabilities per match, long form, from the Poisson total."""
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        frame = _require_features(df)

        mu_total = self._mu(_side_rows(frame, "home")) + self._mu(
            _side_rows(frame, "away")
        )
        if not np.isfinite(mu_total).all():
            bad = frame.loc[~np.isfinite(mu_total), "match_id"].head(3).tolist()
            raise ValueError(
                f"cards_poisson produced a non-finite expected total for {bad}"
            )
        for match_id, value in zip(frame["match_id"], mu_total):
            self.predicted_totals[str(match_id)] = float(value)

        # over means X > line, under means X < line. ceil(line) - 1 is the
        # largest whole count below the line, so it states the half-line and
        # the integer-line cases with one expression.
        under_count = math.ceil(self.line) - 1
        probabilities = {
            "over": poisson.sf(under_count, mu_total),
            "under": poisson.cdf(under_count, mu_total),
        }
        return pd.DataFrame(
            {
                "match_id": np.repeat(frame["match_id"].to_numpy(), 2),
                "outcome": np.tile(np.asarray(self.outcomes, dtype=object), len(frame)),
                "prob": np.column_stack(
                    [probabilities[outcome] for outcome in self.outcomes]
                ).ravel(),
            }
        )

    def _design(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Regressor matrix. The baseline drops the rate columns, nothing else."""
        columns: dict[str, np.ndarray] = {"const": np.ones(len(rows), dtype=float)}
        if self.uses_team_rates:
            # log of a non-positive rate is -inf/NaN on purpose: fit() rejects
            # those rates before it gets here, and _mu() turns the non-finite
            # column into a ValueError naming the match rather than a silent
            # number, so the warning would only be noise.
            with np.errstate(divide="ignore", invalid="ignore"):
                columns["log_own_rate"] = np.log(rows["own_rate"].to_numpy(dtype=float))
                columns["log_opp_rate"] = np.log(rows["opp_rate"].to_numpy(dtype=float))
        columns["home_flag"] = rows["home_flag"].to_numpy(dtype=float)
        return pd.DataFrame(columns)

    def _mu(self, rows: pd.DataFrame) -> np.ndarray:
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        design = self._design(rows)
        finite = np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
        if not finite.all():
            bad = rows.loc[~finite, "match_id"].head(3).tolist()
            raise ValueError(
                f"cards_poisson got a non-finite feature for match(es) {bad}; "
                "build features with add_card_features(df, k)"
            )
        return np.exp(design.to_numpy(dtype=float) @ self._beta)

    def __repr__(self) -> str:
        kind = "team" if self.uses_team_rates else "league-average"
        return f"<{type(self).__name__} {kind} line={self.line:g} k={self.k:g}>"


class LeagueAveragePoisson(CardsPoisson):
    """The baseline: the same Poisson with every team identifier removed.

    Design is intercept + home/away flag only, fitted to the same team-match
    rows, so it reproduces the league's average own cards per side (with the
    home/away split) and nothing about who is playing. Same machinery, same
    data, same walk-forward — the only difference between it and the team model
    is the two rate columns, which is exactly the comparison the plan asks for.
    """

    uses_team_rates: ClassVar[bool] = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _side_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per match for one side, aligned to ``frame``'s order."""
    if side == "home":
        cards, own, opp, flag = "home_cards", "home_rate", "away_rate", 1.0
    elif side == "away":
        cards, own, opp, flag = "away_cards", "away_rate", "home_rate", 0.0
    else:
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")
    return pd.DataFrame(
        {
            "match_id": frame["match_id"].to_numpy(),
            "own_cards": frame[cards].to_numpy(),
            "own_rate": frame[own].to_numpy(dtype="float64"),
            "opp_rate": frame[opp].to_numpy(dtype="float64"),
            "home_flag": np.full(len(frame), flag, dtype=float),
        }
    )


def _require_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Refuse a frame the model cannot score, naming exactly what is missing."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"cards model is missing feature column(s) {missing}; build them "
            "with models.cards_poisson.add_card_features(df, k) before fit/predict"
        )
    return frame


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"frame is missing column(s) {missing}; have {list(df.columns)}")


def _positive_float(name: str, value: object) -> float:
    """A finite, strictly positive float.

    ``numbers.Real`` rather than ``float`` alone so numpy scalars pass, while
    booleans (which are also ``Real``) and numeric *strings* are refused: a
    config value has to be a number, not something that merely parses as one.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


__all__ = [
    "CARD_COLUMNS",
    "FEATURE_COLUMNS",
    "HISTORY_COLUMNS",
    "CardsPoisson",
    "LeagueAveragePoisson",
    "add_card_features",
    "assert_features_are_causal",
]
