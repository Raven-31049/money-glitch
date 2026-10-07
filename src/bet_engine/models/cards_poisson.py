"""Team-level Poisson card model for the cards over/under market (Phase 1).

Three models share one implementation here:

* :class:`CardsPoisson` — the team model. Each side's expected own cards come
  from a Poisson regression on its own shrunk card rate, the opponent's rate,
  and a home/away flag. The expected match total is ``home mu + away mu``, and
  over/under probabilities are read off that Poisson total.
* :class:`LeagueAveragePoisson` — the baseline: the same regression with the
  team rates dropped, so it knows only the league's average cards per side plus
  the home/away flag. It is the yardstick the team model has to beat.
* :class:`CardsTeamRefPoisson` — the referee variant: the team design plus one
  column, the log of the match referee's own shrunk card rate
  (:func:`add_referee_feature`). No hyperparameter is re-tuned for it — same k,
  same folds, same GLM — so any difference in the scores is attributable to
  referee information alone.

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

#: History columns :func:`add_referee_feature` reads: the team history plus the
#: referee's name, which is the key the shrunk rate is built on.
REFEREE_HISTORY_COLUMNS: tuple[str, ...] = HISTORY_COLUMNS + ("referee",)

#: What the models refuse to run without — all produced by add_card_features.
FEATURE_COLUMNS: tuple[str, ...] = (
    "home_rate",
    "away_rate",
    "home_cards",
    "away_cards",
    "total_cards",
)

#: Columns produced by :func:`add_referee_feature`. Only ``ref_rate`` is a
#: model input; ``ref_past_count`` rides along so reports can count matches
#: whose referee had little or no history without recomputing it.
REFEREE_FEATURE_COLUMNS: tuple[str, ...] = ("ref_rate", "ref_past_count")

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


def add_referee_feature(df: pd.DataFrame, k: float, *, verify: bool = True) -> pd.DataFrame:
    """``df`` plus a causal shrunk card rate for the match's referee.

    The referee sees the whole match, so the rate is total cards per match
    (not per side), shrunk with exactly the same formula and the same ``k``
    as the team rates:

        (referee's cards in strictly-earlier matches + k * league mean before this date)
        / (strictly-earlier matches + k)

    A referee with no history (or none before that date) gets the league mean
    as of the date — the same "no history means league mean" rule the teams
    follow, with no referee-specific tuning. The shrinkage prior is the league
    mean *total* per match, the same quantity being shrunk, so the prior and
    the evidence are on one scale.

    ``ref_past_count`` (strictly-earlier matches for that referee) is returned
    alongside the rate so reports can count matches officiated by a referee
    with little history — a model's referee column is only as good as the
    sample behind it.

    ``verify`` runs the same causality probe as :func:`add_card_features`, on
    the referee columns.
    """
    _positive_float("k", k)
    _require_columns(df, REFEREE_HISTORY_COLUMNS)
    features = _add_referee_feature(df, k)
    if verify:
        assert_features_are_causal(
            _add_referee_feature,
            df,
            k,
            requires=REFEREE_HISTORY_COLUMNS,
            watch=REFEREE_FEATURE_COLUMNS,
            context="add_referee_feature",
        )
    return features


def _add_referee_feature(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """The referee computation itself: no verification, so the probe cannot recurse."""
    out = df.copy()
    hy = pd.to_numeric(out["hy"], errors="coerce")
    hr = pd.to_numeric(out["hr"], errors="coerce")
    ay = pd.to_numeric(out["ay"], errors="coerce")
    ar = pd.to_numeric(out["ar"], errors="coerce")
    total = hy + hr + ay + ar

    matches = pd.DataFrame(
        {
            "match_id": out["match_id"].to_numpy(),
            "date": out["date"].to_numpy(),
            "referee": out["referee"].to_numpy(),
            "total_cards": total.to_numpy(),
        }
    )
    # A row is evidence only if it is dated, counted and attributed: a match
    # with no referee cannot enter anyone's history, and a row missing cards
    # says nothing about how card-happy the referee is.
    usable = (
        matches["date"].notna()
        & matches["total_cards"].notna()
        & matches["referee"].notna()
    )

    # Aggregate per (referee, date) first, so a referee with two matches on
    # one date sees nothing from that date — the same calendar-day boundary
    # the team features and the walk-forward folds use.
    history = (
        matches.loc[usable]
        .groupby(["referee", "date"], sort=True)["total_cards"]
        .agg(past_sum="sum", past_count="count")
        .reset_index()
        .sort_values(["referee", "date"], kind="mergesort")
    )
    history["past_sum"] = (
        history.groupby("referee")["past_sum"].cumsum() - history["past_sum"]
    )
    history["past_count"] = (
        history.groupby("referee")["past_count"].cumsum() - history["past_count"]
    )

    # League mean total per match, strictly before each date. The first date
    # has no earlier match, so its mean is NaN: no league average to shrink
    # toward yet (and no prediction window starts there anyway).
    by_date = (
        matches.loc[usable]
        .groupby("date")["total_cards"]
        .agg(day_sum="sum", day_count="count")
        .sort_index()
    )
    past_league_sum = by_date["day_sum"].cumsum() - by_date["day_sum"]
    past_league_count = by_date["day_count"].cumsum() - by_date["day_count"]
    league_mean_total = past_league_sum / past_league_count.replace(0, np.nan)

    merged = matches.merge(history, on=["referee", "date"], how="left")
    merged["league_mean_total"] = merged["date"].map(league_mean_total)
    merged["ref_rate"] = (merged["past_sum"] + k * merged["league_mean_total"]) / (
        merged["past_count"] + k
    )
    # pandas merges NaN keys with NaN keys, so a match without a referee would
    # otherwise inherit the history of every other referee-less match on the
    # same date. It has no referee: its rate and count are NaN, not borrowed.
    no_referee = merged["referee"].isna()
    merged.loc[no_referee, ["ref_rate", "past_count"]] = np.nan

    out["ref_rate"] = (
        out["match_id"].map(merged.set_index("match_id")["ref_rate"]).astype("float64")
    )
    out["ref_past_count"] = (
        out["match_id"]
        .map(merged.set_index("match_id")["past_count"])
        .astype("float64")
    )
    return out


def assert_features_are_causal(
    feature_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    df: pd.DataFrame,
    k: float,
    *,
    probes: int = _PROBE_COUNT,
    requires: Sequence[str] = HISTORY_COLUMNS,
    watch: Sequence[str] = _PROBE_COLUMNS,
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

    ``requires`` are the columns ``feature_fn`` reads (the team history by
    default, plus the referee column for the referee feature) and ``watch``
    are the output columns compared for change (the team rates by default,
    the referee rate and count for the referee feature). Both are parameters
    rather than globals so one probe implementation serves every feature.
    """
    if not callable(feature_fn):
        raise TypeError(
            f"feature_fn must be callable, got {type(feature_fn).__name__}"
        )
    _require_columns(df, requires)
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
                watch=watch,
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
                    watch=watch,
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
    watch: Sequence[str],
    context: str,
) -> None:
    """Assert the watched feature columns match on ``rows_of_interest``."""
    checked = base.loc[rows_of_interest, list(watch)].to_numpy(dtype=float)
    other = probe_result.loc[rows_of_interest, list(watch)].to_numpy(
        dtype=float
    )
    # NaN == NaN here: "no history yet" is a value, not a difference.
    differs = ~((checked == other) | (np.isnan(checked) & np.isnan(other)))
    if not differs.any():
        return
    hit_rows = np.flatnonzero(differs.any(axis=1))
    ids = base.loc[rows_of_interest, "match_id"].to_numpy()[hit_rows[:3]]
    raise AssertionError(
        "features are not strictly causal: perturbing matches "
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
    #: baseline deliberately does not, and the referee variant adds the
    #: referee's rate on top of the team design.
    uses_team_rates: ClassVar[bool] = True
    uses_referee_rate: ClassVar[bool] = False

    #: Feature columns beyond FEATURE_COLUMNS this design needs. The team
    #: model and the baseline need none; the referee variant needs ref_rate,
    #: which add_referee_feature (not add_card_features) produces.
    requires_features: ClassVar[tuple[str, ...]] = ()

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
        #: Last fit's coefficient and standard-error series, indexed by design
        #: column name — kept so coefficient_table() can report the referee
        #: coefficient with an interval instead of only the point estimate.
        self._fitted_params: pd.Series | None = None
        self._fitted_bse: pd.Series | None = None
        self.n_training_rows: int = 0
        self.n_dropped_rows: int = 0

    def fit(self, train_df: pd.DataFrame) -> CardsPoisson:
        """Estimate the Poisson coefficients from ``train_df``'s team rows."""
        frame = _require_features(train_df, self.requires_features)
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

        if self.uses_referee_rate:
            # Same guard for the referee column: a non-positive rate here
            # means the shrinkage input is broken, not that the referee is.
            referee = rows["ref_rate"].to_numpy(dtype=float)
            bad = np.isfinite(referee) & (referee <= 0.0)
            if bad.any():
                raise ValueError(
                    "cards_poisson got a non-positive referee rate; shrinkage "
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
        self._fitted_params = result.params
        self._fitted_bse = result.bse
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Over/under probabilities per match, long form, from the Poisson total."""
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        frame = _require_features(df, self.requires_features)

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
        """Regressor matrix. The baseline drops the rate columns, nothing else.

        The referee variant appends one column — the log of the referee's
        shrunk per-match rate — so its design is the team design plus exactly
        one term. Adding referee information never removes or rescales a team
        column, which is what makes the team-vs-team+referee score difference
        attributable to the referee term alone.
        """
        columns: dict[str, np.ndarray] = {"const": np.ones(len(rows), dtype=float)}
        if self.uses_team_rates:
            # log of a non-positive rate is -inf/NaN on purpose: fit() rejects
            # those rates before it gets here, and _mu() turns the non-finite
            # column into a ValueError naming the match rather than a silent
            # number, so the warning would only be noise.
            with np.errstate(divide="ignore", invalid="ignore"):
                columns["log_own_rate"] = np.log(rows["own_rate"].to_numpy(dtype=float))
                columns["log_opp_rate"] = np.log(rows["opp_rate"].to_numpy(dtype=float))
        if self.uses_referee_rate:
            with np.errstate(divide="ignore", invalid="ignore"):
                columns["log_ref_rate"] = np.log(rows["ref_rate"].to_numpy(dtype=float))
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
                + (
                    " and add_referee_feature(df, k)"
                    if self.uses_referee_rate
                    else ""
                )
            )
        return np.exp(design.to_numpy(dtype=float) @ self._beta)

    def coefficient_table(self) -> pd.DataFrame:
        """The last fit's coefficients: point estimate, SE and 95% normal CI.

        Rows are the design's own column names, so ``log_ref_rate``'s row is
        the referee coefficient. The interval is the GLM's own standard error
        (cov_params of the final fold's fit), not a bootstrap: it answers
        "is this coefficient distinguishable from zero in the fit that saw the
        most training data", which is the honest question for one regression.
        """
        if self._fitted_params is None or self._fitted_bse is None:
            raise RuntimeError("coefficient_table() called before fit()")
        table = pd.DataFrame(
            {
                "coef": self._fitted_params.to_numpy(dtype=float),
                "se": self._fitted_bse.to_numpy(dtype=float),
            },
            index=pd.Index(self._fitted_params.index, name="term"),
        )
        table["ci_low"] = table["coef"] - 1.96 * table["se"]
        table["ci_high"] = table["coef"] + 1.96 * table["se"]
        return table.reset_index()

    def __repr__(self) -> str:
        if self.uses_referee_rate:
            kind = "team+referee"
        else:
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


class CardsTeamRefPoisson(CardsPoisson):
    """The referee variant: the team design plus the referee's shrunk rate.

    Design is const + log own rate + log opponent rate + log referee rate +
    home flag — the team design with exactly one column added. It shares
    CardsPoisson's fit, predict and walk-forward machinery, so the only thing
    that differs between this and the team model is referee information: same
    k, same folds, same GLM family, no tuning. That is the comparison Phase 1
    step 2 asks for — the score difference between this and the team model is
    the referee term's worth, and nothing else.

    Requires ref_rate, produced by :func:`add_referee_feature`; a frame with
    only add_card_features' columns is refused by name before fit or predict.
    """

    uses_referee_rate: ClassVar[bool] = True
    requires_features: ClassVar[tuple[str, ...]] = ("ref_rate",)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _side_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per match for one side, aligned to ``frame``'s order.

    ref_rate is carried when the frame has it (both sides of a match share
    the same referee, so the column is match-level) and is NaN otherwise —
    the team model never reads it, and the referee variant refuses the frame
    by name in _require_features long before the NaN could reach a log().
    """
    if side == "home":
        cards, own, opp, flag = "home_cards", "home_rate", "away_rate", 1.0
    elif side == "away":
        cards, own, opp, flag = "away_cards", "away_rate", "home_rate", 0.0
    else:
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")
    if "ref_rate" in frame.columns:
        ref_rate = frame["ref_rate"].to_numpy(dtype="float64")
    else:
        ref_rate = np.full(len(frame), np.nan, dtype="float64")
    return pd.DataFrame(
        {
            "match_id": frame["match_id"].to_numpy(),
            "own_cards": frame[cards].to_numpy(),
            "own_rate": frame[own].to_numpy(dtype="float64"),
            "opp_rate": frame[opp].to_numpy(dtype="float64"),
            "ref_rate": ref_rate,
            "home_flag": np.full(len(frame), flag, dtype=float),
        }
    )


def _require_features(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Refuse a frame the model cannot score, naming exactly what is missing."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    missing = [
        column for column in (*FEATURE_COLUMNS, *extra) if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"cards model is missing feature column(s) {missing}; build them "
            "with models.cards_poisson.add_card_features(df, k)"
            + (
                " and models.cards_poisson.add_referee_feature(df, k)"
                if any(column in missing for column in REFEREE_FEATURE_COLUMNS)
                else ""
            )
            + " before fit/predict"
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
    "REFEREE_FEATURE_COLUMNS",
    "REFEREE_HISTORY_COLUMNS",
    "CardsPoisson",
    "CardsTeamRefPoisson",
    "LeagueAveragePoisson",
    "add_card_features",
    "add_referee_feature",
    "assert_features_are_causal",
]
