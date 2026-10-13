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

from .count_distributions import OVER_UNDER_PUSH as _OVER_UNDER_PUSH
from .count_distributions import SIDE_MAX_COUNT as _SIDE_MAX_COUNT
from .count_distributions import over_under_push as _over_under_push
from .count_distributions import side_pmf as _side_pmf
from .count_distributions import total_pmf as _convolved_total_pmf

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
#: model input; ``ref_past_count``, ``ref_past_cards`` and ``league_mean_total``
#: ride along as the referee Gamma posterior's sufficient statistics and so
#: reports can count matches officiated by a low-history referee without
#: recomputing them (Phase 1 step 4).
REFEREE_FEATURE_COLUMNS: tuple[str, ...] = (
    "ref_rate",
    "ref_past_count",
    "ref_past_cards",
    "league_mean_total",
)

#: The team-rate Gamma posterior's sufficient statistics, produced by
#: :func:`add_card_features` (Phase 1 step 4). shape = k*league_mean +
#: past_cards, rate = k + past_matches for each side.
SAMPLE_TEAM_COLUMNS: tuple[str, ...] = (
    "home_past_cards",
    "home_past_matches",
    "away_past_cards",
    "away_past_matches",
    "league_mean",
)

#: The over/under lines the sampling diagnostics are quoted at (Phase 1 step 4).
#: The market's quoted half-lines; half-lines never sit on an integer, so over
#: and under partition the pmf exactly.
SAMPLE_LINES: tuple[float, ...] = (3.5, 4.5, 5.5)

#: The settlement states :meth:`CardsPoisson.over_under` reports, in this order
#: (Part B). Same tuple as count_distributions.OVER_UNDER_PUSH: over and under
#: are the two bettable outcomes, push is the void state a whole-number line
#: can produce.
OVER_UNDER_PUSH: tuple[str, ...] = _OVER_UNDER_PUSH

#: Default draw count and the fixed seed for :meth:`CardsPoisson.predict_samples`.
#: A fixed seed makes a report reproducible; the count is the task's default.
SAMPLE_DRAWS: int = 1000
SAMPLE_SEED: int = 20260101

#: Rows of the flattened (draw x match) batch convolved at a time in the
#: negative-binomial sampling path, so the two-side convolution never
#: materialises a grid as large as draws x matches x SIDE_MAX_COUNT**2.
_SAMPLE_CHUNK: int = 4096

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


def add_card_features(
    df: pd.DataFrame,
    k: float,
    *,
    half_life_days: float | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """``df`` plus causal per-team card rates.

    ``k`` is the shrinkage weight in pseudo-matches: a team's rate is

        (own cards in strictly-earlier matches + k * league mean before this date)
        / (strictly-earlier matches + k)

    so a team with no history (promoted side, season start) gets exactly the
    league average as of that date, and real history dilutes the prior as it
    accumulates. ``k`` is a config parameter (``model.params.k``), not a
    constant, because how fast a rate should be trusted is a modelling choice a
    variant config has to be able to state.

    ``half_life_days`` is the optional recency weight (config parameter
    ``model.params.half_life_days``). Left ``None`` the history is cumulative —
    every earlier match of the season-to-date counts the same, which is the
    incumbent frozen by step 3b, reproduced bit for bit. Set to a number of
    days, each earlier match's contribution to *both* the numerator and the
    denominator is weighted ``0.5 ** (days_before / half_life_days)``, so a
    match one half-life old counts half as much and the rate leans on the
    recent form of a rising or falling card rate. One half-life is applied to
    the team sums, the referee sums and the league-mean sums alike (see
    :func:`add_referee_feature`); nothing else changes.

    ``verify`` runs :func:`assert_features_are_causal` on the result. It costs a
    handful of extra feature passes (milliseconds at league scale) and turns
    "features never see their own result" from a claim in a docstring into a
    runtime assertion — the same shape as the walk-forward leak check. The probe
    is bound to the same ``half_life_days`` the features were built with, so a
    recency build is checked, not an unweighted one.

    Alongside the rates it emits the Gamma posterior's sufficient statistics
    (``home_past_cards``/``home_past_matches``/``away_past_cards``/
    ``away_past_matches``, with ``league_mean``), so
    :meth:`CardsPoisson.predict_samples` can draw each side's rate from the same
    history the point estimate used instead of re-deriving it.
    """
    _positive_float("k", k)
    if half_life_days is not None:
        _positive_float("half_life_days", half_life_days)
    _require_columns(df, HISTORY_COLUMNS)
    features = _add_card_features(df, k, half_life_days)
    if verify:
        assert_features_are_causal(
            lambda frame, weight: _add_card_features(frame, weight, half_life_days),
            df,
            k,
            context="add_card_features",
        )
    return features


def _add_card_features(
    df: pd.DataFrame, k: float, half_life_days: float | None = None
) -> pd.DataFrame:
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
        .agg(value="sum", count="count")
        .reset_index()
    )
    if half_life_days is None:
        history = history.sort_values(["team", "date"], kind="mergesort")
        history["past_sum"] = (
            history.groupby("team")["value"].cumsum() - history["value"]
        )
        history["past_count"] = (
            history.groupby("team")["count"].cumsum() - history["count"]
        )
    else:
        history = _decayed_prior(history, ["team"], half_life_days)

    # League mean per team-match row, likewise strictly before each date.
    by_date = (
        long.loc[usable]
        .groupby("date")["own_cards"]
        .agg(value="sum", count="count")
        .sort_index()
    )
    if half_life_days is None:
        past_league_sum = by_date["value"].cumsum() - by_date["value"]
        past_league_count = by_date["count"].cumsum() - by_date["count"]
    else:
        league_rows = by_date.reset_index()
        # No grouping key: the league is one group. _decayed_prior wants a key,
        # so give it a constant one rather than special-casing the single group.
        league_rows["_league"] = 0
        decayed = _decayed_prior(league_rows, ["_league"], half_life_days).set_index(
            "date"
        )
        past_league_sum = decayed["past_sum"]
        past_league_count = decayed["past_count"]
    # The first date has no earlier match, so its mean is NaN rather than 0/0:
    # there is genuinely no league average to shrink toward yet.
    league_mean = past_league_sum / past_league_count.replace(0, np.nan)

    merged = long.merge(
        history[["team", "date", "past_sum", "past_count"]],
        on=["team", "date"],
        how="left",
    )
    merged["league_mean"] = merged["date"].map(league_mean)
    # past_count stays NaN for unusable rows, so their rate is NaN too and
    # fit() drops them instead of treating missing data as zero cards.
    merged["rate"] = (merged["past_sum"] + k * merged["league_mean"]) / (
        merged["past_count"] + k
    )

    home = merged.loc[merged["side"] == "home"].set_index("match_id")
    away = merged.loc[merged["side"] == "away"].set_index("match_id")
    out["home_rate"] = out["match_id"].map(home["rate"]).astype("float64")
    out["away_rate"] = out["match_id"].map(away["rate"]).astype("float64")
    out["league_mean"] = out["date"].map(league_mean).astype("float64")
    # Sufficient statistics for the Gamma posterior on each side's rate
    # (Phase 1 step 4): the strictly-earlier card sum and match count behind the
    # rate, with the same recency weights. shape = k*league_mean + past_cards,
    # rate = k + past_matches, so the posterior mean is exactly the shrunk rate
    # already in home_rate/away_rate. Exposed as columns so the model samples the
    # rate it actually used without re-deriving the history.
    out["home_past_cards"] = out["match_id"].map(home["past_sum"]).astype("float64")
    out["home_past_matches"] = (
        out["match_id"].map(home["past_count"]).astype("float64")
    )
    out["away_past_cards"] = out["match_id"].map(away["past_sum"]).astype("float64")
    out["away_past_matches"] = (
        out["match_id"].map(away["past_count"]).astype("float64")
    )
    return out


def add_referee_feature(
    df: pd.DataFrame,
    k: float,
    *,
    half_life_days: float | None = None,
    verify: bool = True,
) -> pd.DataFrame:
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

    ``half_life_days`` is the same optional recency weight
    :func:`add_card_features` takes (config parameter
    ``model.params.half_life_days``): ``None`` keeps the cumulative history,
    a number of days applies ``0.5 ** (days_before / half_life_days)`` to the
    referee sums and the league-mean sums. It is deliberately the same value
    the team rates use — one half-life for the whole feature build.

    ``ref_past_count`` (strictly-earlier matches for that referee) is returned
    alongside the rate so reports can count matches officiated by a referee
    with little history — a model's referee column is only as good as the
    sample behind it. Under recency it is the *effective* (decayed) count, not
    an integer. ``ref_past_cards`` and ``league_mean_total`` join it as the
    Gamma posterior's sufficient statistics for :meth:`CardsPoisson.predict_samples`
    (shape = k*league_mean_total + ref_past_cards, rate = k + ref_past_count).

    ``verify`` runs the same causality probe as :func:`add_card_features`, on
    the referee columns, bound to the same ``half_life_days``.
    """
    _positive_float("k", k)
    if half_life_days is not None:
        _positive_float("half_life_days", half_life_days)
    _require_columns(df, REFEREE_HISTORY_COLUMNS)
    features = _add_referee_feature(df, k, half_life_days)
    if verify:
        assert_features_are_causal(
            lambda frame, weight: _add_referee_feature(frame, weight, half_life_days),
            df,
            k,
            requires=REFEREE_HISTORY_COLUMNS,
            watch=REFEREE_FEATURE_COLUMNS,
            context="add_referee_feature",
        )
    return features


def _add_referee_feature(
    df: pd.DataFrame, k: float, half_life_days: float | None = None
) -> pd.DataFrame:
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
        .agg(value="sum", count="count")
        .reset_index()
    )
    if half_life_days is None:
        history = history.sort_values(["referee", "date"], kind="mergesort")
        history["past_sum"] = (
            history.groupby("referee")["value"].cumsum() - history["value"]
        )
        history["past_count"] = (
            history.groupby("referee")["count"].cumsum() - history["count"]
        )
    else:
        history = _decayed_prior(history, ["referee"], half_life_days)

    # League mean total per match, strictly before each date. The first date
    # has no earlier match, so its mean is NaN: no league average to shrink
    # toward yet (and no prediction window starts there anyway).
    by_date = (
        matches.loc[usable]
        .groupby("date")["total_cards"]
        .agg(value="sum", count="count")
        .sort_index()
    )
    if half_life_days is None:
        past_league_sum = by_date["value"].cumsum() - by_date["value"]
        past_league_count = by_date["count"].cumsum() - by_date["count"]
    else:
        league_rows = by_date.reset_index()
        league_rows["_league"] = 0
        decayed = _decayed_prior(league_rows, ["_league"], half_life_days).set_index(
            "date"
        )
        past_league_sum = decayed["past_sum"]
        past_league_count = decayed["past_count"]
    league_mean_total = past_league_sum / past_league_count.replace(0, np.nan)

    merged = matches.merge(
        history[["referee", "date", "past_sum", "past_count"]],
        on=["referee", "date"],
        how="left",
    )
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
    # Sufficient statistics for the Gamma posterior on the referee rate
    # (Phase 1 step 4): past_cards is the decayed card total behind ref_rate and
    # league_mean_total is the per-match prior it shrinks toward, so
    # shape = k*league_mean_total + ref_past_cards, rate = k + ref_past_count.
    out["ref_past_cards"] = (
        out["match_id"].map(merged.set_index("match_id")["past_sum"]).astype("float64")
    )
    out["league_mean_total"] = (
        out["match_id"]
        .map(merged.set_index("match_id")["league_mean_total"])
        .astype("float64")
    )
    return out


def _decayed_prior(
    aggregated: pd.DataFrame,
    group_columns: Sequence[str],
    half_life_days: float,
) -> pd.DataFrame:
    """Decayed prior sum/count per aggregated row, strictly before its date.

    ``aggregated`` carries one row per (group..., date) with columns ``value``
    and ``count`` (that date's summed cards and its number of observations).
    Returns a copy with ``past_sum``/``past_count``: the sum of ``value`` and
    ``count`` over rows of the same group dated *strictly earlier*, each
    weighted ``0.5 ** (days_before / half_life_days)`` — a match one half-life
    old counts half as much. This is the recency counterpart of the
    cumulative ``cumsum() - current`` the unweighted path uses.

    WHY the inverse-decay trick: a literal ``0.5 ** (days_before / half_life)``
    weight would ask for ``0.5 ** (large positive)`` and underflow to zero for
    an old enough match. Instead each row is scaled by
    ``r ** -(days since the group's own first date)`` (``r = 0.5 ** (1/h)``),
    which is bounded by that group's span, prefix-summed, and scaled back by
    ``r ** +(days since the group's first date)`` — the two factors cancel per
    pair, so the weight is exactly the requested one and never touches an
    underflowing exponent.

    The sort is ``mergesort`` not the default: ties (a team playing twice on
    one date) must keep their input order so the strict ``cumsum - current``
    excludes the whole same-date group, the calendar-day boundary the rest of
    the file relies on.
    """
    frame = aggregated.sort_values([*group_columns, "date"], kind="mergesort").copy()
    ratio = 0.5 ** (1.0 / half_life_days)
    base = frame.groupby(list(group_columns), sort=False)["date"].transform("min")
    days = (frame["date"] - base).dt.days.to_numpy(dtype=float)
    inverse = ratio ** (-days)
    frame["_weighted_value"] = frame["value"].to_numpy(dtype=float) * inverse
    frame["_weighted_count"] = frame["count"].to_numpy(dtype=float) * inverse
    grouped = frame.groupby(list(group_columns), sort=False)
    # shift(1) rather than "cumsum - current": the current row is already
    # excluded, so perturbing its own counts leaves the prior bit-identical
    # instead of shifting it by a float-epsilon cancellation (which the
    # causality probe, comparing exactly, would rightly flag). Each (group,
    # date) is one row, so shift(1) is exactly "strictly earlier date".
    keys = [frame[column].to_numpy() for column in group_columns]
    prior_value = grouped["_weighted_value"].cumsum().groupby(keys).shift(1)
    prior_count = grouped["_weighted_count"].cumsum().groupby(keys).shift(1)
    decay = ratio ** days
    frame["past_sum"] = (decay * prior_value).fillna(0.0)
    frame["past_count"] = (decay * prior_count).fillna(0.0)
    return frame.drop(columns=["_weighted_value", "_weighted_count"])


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

    #: Which count likelihood the fit uses. The classes here are Poisson; the
    #: negative-binomial subclasses relabel it so a report cannot confuse the
    #: two families that share this design.
    family_name: ClassVar[str] = "poisson"

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
        #: match_id -> (expected home cards, expected away cards). The total
        #: alone is not enough to rebuild the step-3 distribution: the negative
        #: binomial total is the convolution of the two sides, so each side's
        #: own mean is needed, not just their sum.
        self.predicted_sides: dict[str, tuple[float, float]] = {}
        #: Dispersion estimated by the fit, or None for the Poisson family.
        #: NegativeBinomial sets it to its MLE alpha (variance = mu + alpha*mu^2);
        #: the recruitment (Poisson) family leaves it None, meaning the limit.
        self.alpha_: float | None = None
        self._beta: np.ndarray | None = None
        #: Covariance of the fitted coefficients (rows/cols the design's own
        #: terms, alpha excluded for the negative binomial). The coefficient half
        #: of predict_samples' uncertainty: draws come from a multivariate normal
        #: with this covariance around _beta. None before fit.
        self._cov: np.ndarray | None = None
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

        self._fit_family(design.loc[usable], endog[usable])
        return self

    def _fit_family(self, design: pd.DataFrame, endog: np.ndarray) -> None:
        """Estimate the coefficients from a usable design and target.

        Separate from :meth:`fit` so the negative-binomial subclasses change
        only the likelihood — the design, the rows, the drop rule and every
        downstream probability are shared. The base implementation is the
        Poisson GLM; subclasses override this one method.

        ``alpha_`` is set to None here: the Poisson family has no dispersion
        parameter to estimate, and the convolution treats None as that limit.
        """
        result = sm.GLM(
            endog,
            design,
            family=sm.families.Poisson(),
        ).fit()
        self._beta = result.params.to_numpy(dtype=float)
        self._fitted_params = result.params
        self._fitted_bse = result.bse
        # cov_params() is the GLM's own coefficient covariance; row/col order
        # matches params, so predict_samples can draw coefficients from
        # N(beta, cov) without any further bookkeeping.
        self._cov = np.asarray(
            result.cov_params().loc[self._fitted_params.index, self._fitted_params.index],
            dtype=float,
        )
        self.alpha_ = None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Over/under probabilities per match, long form, from the total pmf.

        The match total is home + away, and for the negative-binomial family
        that total is the *convolution* of the two sides, not a named
        distribution — see models/count_distributions.py. The Poisson family
        goes through the same convolution (sums of independent Poissons are
        Poisson, so it returns the same numbers the direct formula did) so one
        probability path serves every family.
        """
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        frame = _require_features(df, self.requires_features)

        mu_home = self._mu(_side_rows(frame, "home"))
        mu_away = self._mu(_side_rows(frame, "away"))
        mu_total = mu_home + mu_away
        if not np.isfinite(mu_total).all():
            bad = frame.loc[~np.isfinite(mu_total), "match_id"].head(3).tolist()
            raise ValueError(
                f"cards_poisson produced a non-finite expected total for {bad}"
            )
        for match_id, total, home, away in zip(
            frame["match_id"], mu_total, mu_home, mu_away
        ):
            key = str(match_id)
            self.predicted_totals[key] = float(total)
            self.predicted_sides[key] = (float(home), float(away))

        counts, pmf = _convolved_total_pmf(mu_home, mu_away, self.alpha_)
        # over means X > line, under means X <= line — the market's own rule
        # (markets/cards_totals.py). For the half-lines a config can name there
        # is no integer on the line, so the two partition the whole pmf and
        # over + under == the row sum == 1.
        over = pmf[:, counts > self.line].sum(axis=1)
        under = pmf[:, counts <= self.line].sum(axis=1)
        probabilities = {"over": over, "under": under}
        return pd.DataFrame(
            {
                "match_id": np.repeat(frame["match_id"].to_numpy(), 2),
                "outcome": np.tile(np.asarray(self.outcomes, dtype=object), len(frame)),
                "prob": np.column_stack(
                    [probabilities[outcome] for outcome in self.outcomes]
                ).ravel(),
            }
        )

    def predict_samples(
        self,
        df: pd.DataFrame,
        n: int = SAMPLE_DRAWS,
        *,
        lines: Sequence[float] = SAMPLE_LINES,
        seed: int | None = SAMPLE_SEED,
    ) -> np.ndarray:
        """Draw ``n`` joint samples of P(over)/P(under) at each of ``lines``.

        Uncertainty comes from a single fit — nothing is refitted and nothing is
        bootstrapped — by drawing the two parameters the fit is genuinely unsure
        about and propagating them together:

        * **coefficients** from ``N(beta, cov)``, the GLM's own covariance (the
          coefficient half of ``coefficient_table``'s interval);
        * **team and referee rates** from their Gamma posteriors. Each rate is a
          Gamma prior of mean ``league_mean`` (per side) or ``league_mean_total``
          (referee) and weight ``k`` pseudo-matches updated by the strictly
          earlier, recency-weighted card sums, so ``shape = k*prior + past_cards``
          and ``rate = k + past_matches`` — the same shrinkage ``add_card_features``
          already applied, now carried as a distribution. Its mean is exactly the
          point rate, so a larger posterior (more history) draws tighter.

        Returns an array of shape ``(n, len(df), len(lines), n_outcomes)``: the
        draw axis, the match axis (following ``df``'s rows), the line axis, and
        the market's ``outcomes`` order. Every draw sums to 1 across outcomes.
        The percentiles taken *from* those draws by callers are marginals and
        are never renormalised across outcomes (invariant 5) — see
        :mod:`bet_engine.eval.uncertainty`.

        A fixed ``seed`` (the default) makes a report reproducible; pass
        ``seed=None`` for fresh entropy.
        """
        if self._beta is None or self._cov is None:
            raise RuntimeError("predict_samples() called before fit()")
        draws = _positive_int("n", n)
        line_labels = tuple(_positive_float("line", line) for line in lines)
        if not line_labels:
            raise ValueError("predict_samples needs at least one line")
        frame = _require_features(df, (*self.requires_features, *self._sample_columns()))

        rng = np.random.default_rng(seed)
        beta = self._coefficient_draws(rng, draws)
        mu_home, mu_away = self._sampled_mus(frame, beta, rng, draws)
        over = self._over_probabilities(mu_home, mu_away, line_labels)

        # Every draw is a full distribution over the two outcomes (over + under
        # = 1 within a draw). The *percentiles* callers take across draws are
        # marginals and are never renormalised — invariant 5.
        by_outcome = {"over": over, "under": 1.0 - over}
        return np.stack([by_outcome[outcome] for outcome in self.outcomes], axis=-1)

    def over_under(
        self,
        df: pd.DataFrame,
        *,
        line_column: str = "line",
        draws: int | None = None,
        seed: int | None = SAMPLE_SEED,
    ) -> pd.DataFrame | np.ndarray:
        """Over/under/push at each row's OWN line (Part B).

        ``predict`` quotes one fixed line for every match. The odds data does
        not: each quoted over/under carries its own threshold, so a match may
        need P(over)/P(under) at 3.5, at 4.5, or at a whole number like 4 — and
        at a whole number an exact hit is a *push*, not a win for under. This
        method reads that per-row line from ``line_column`` and reports all
        three settlement states.

        Point estimate (``draws is None``): a long-form frame
        ``[match_id, line, outcome, prob]`` with one row per match per state in
        :data:`OVER_UNDER_PUSH` (over, under, push). For a whole line the three
        sum to 1 and ``push`` is nonzero; for a half-line ``push`` is exactly 0
        and over + under == 1.

        Sampled (``draws`` given): an array ``(draws, len(df), 3)`` in the same
        outcome order, each draw summing to 1 — the same coefficient and rate
        uncertainty ``predict_samples`` propagates, just settled at each row's
        own line. Percentiles taken from these draws are marginals and are never
        renormalised (invariant 5).
        """
        frame = _require_features(df, self.requires_features)
        lines = _line_values(frame, line_column)

        if draws is None:
            mu_home = self._mu(_side_rows(frame, "home"))
            mu_away = self._mu(_side_rows(frame, "away"))
            _, pmf = _convolved_total_pmf(mu_home, mu_away, self.alpha_)
            over, under, push = _over_under_push(pmf, lines)
            return pd.DataFrame(
                {
                    "match_id": np.repeat(frame["match_id"].to_numpy(), 3),
                    "line": np.repeat(lines, 3),
                    "outcome": np.tile(
                        np.asarray(_OVER_UNDER_PUSH, dtype=object), len(frame)
                    ),
                    "prob": np.column_stack([over, under, push]).ravel(),
                }
            )

        self._require_sampling()
        count = _positive_int("draws", draws)
        rng = np.random.default_rng(seed)
        beta = self._coefficient_draws(rng, count)
        mu_home, mu_away = self._sampled_mus(frame, beta, rng, count)
        return self._sampled_over_under(mu_home, mu_away, lines, count, frame)

    def _sampled_over_under(
        self,
        mu_home: np.ndarray,
        mu_away: np.ndarray,
        lines: np.ndarray,
        draws: int,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """``(draws, n_matches, 3)`` over/under/push at each row's line.

        Shared by both targets: the side model passes its two per-side means and
        convolves; the total model overrides it to pass the single total mean
        through one pmf. Flattened (draw x match) rows are convolved in chunks
        with a Python loop over chunks but never over a grid cell (invariant 6).
        """
        shape = mu_home.shape  # (draws, n_matches)
        flat_home = mu_home.reshape(-1)
        flat_away = mu_away.reshape(-1)
        flat_lines = np.tile(lines, draws)
        size = flat_home.size
        over = np.empty(size, dtype=float)
        under = np.empty(size, dtype=float)
        push = np.empty(size, dtype=float)
        for start in range(0, size, _SAMPLE_CHUNK):
            stop = start + _SAMPLE_CHUNK
            _, pmf = _convolved_total_pmf(
                flat_home[start:stop], flat_away[start:stop], self.alpha_
            )
            o, u, p = _over_under_push(pmf, flat_lines[start:stop])
            over[start:stop] = o
            under[start:stop] = u
            push[start:stop] = p
        grid = (draws, shape[1])
        return np.stack(
            [over.reshape(grid), under.reshape(grid), push.reshape(grid)], axis=-1
        )

    def _require_sampling(self) -> None:
        """Refuse a sampling call before fit, naming the fix."""
        if self._beta is None or self._cov is None:
            raise RuntimeError("over_under(draws=...) called before fit()")

    def _sample_columns(self) -> tuple[str, ...]:
        """Feature columns :meth:`predict_samples` needs beyond the design's."""
        columns: list[str] = []
        if self.uses_team_rates:
            columns.extend(SAMPLE_TEAM_COLUMNS)
        if self.uses_referee_rate:
            columns.extend(("ref_past_cards", "ref_past_count", "league_mean_total"))
        return tuple(columns)

    def _coefficient_draws(self, rng: np.random.Generator, draws: int) -> np.ndarray:
        """``(draws, n_terms)`` coefficient draws from N(beta, cov).

        Uses an eigendecomposition with negative eigenvalues clipped to zero
        rather than ``rng.multivariate_normal``: a rank-deficient fold (the
        small/collinear cases the walk-forward already meets) yields a
        numerically semi-definite covariance that the direct sampler would warn
        about or reject, while the clipped square root stays a valid, finite
        draw. Non-finite covariance entries (a degenerate fit) become zero, so
        that coefficient simply does not move across draws.
        """
        cov = np.nan_to_num(
            np.asarray(self._cov, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        cov = 0.5 * (cov + cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        noise = rng.standard_normal((draws, self._beta.shape[0]))
        return self._beta[None, :] + noise @ root.T

    def _gamma_rate_draws(
        self,
        rng: np.random.Generator,
        prior_mean: np.ndarray,
        past_cards: np.ndarray,
        past_matches: np.ndarray,
        draws: int,
    ) -> np.ndarray:
        """``(draws, n_matches)`` draws of a shrunk rate from its Gamma posterior.

        ``shape = k*prior_mean + past_cards`` and ``rate = k + past_matches``,
        recency weights already folded into both sums, so the posterior mean is
        the shrunk rate itself. Vectorised over matches and draws in one
        ``rng.gamma`` call — no per-match loop.
        """
        shape = self.k * prior_mean + past_cards
        rate = self.k + past_matches
        bad = (
            ~np.isfinite(shape) | ~np.isfinite(rate) | (shape <= 0.0) | (rate <= 0.0)
        )
        if bad.any():
            raise ValueError(
                "predict_samples got a non-positive or non-finite Gamma "
                f"posterior for {int(bad.sum())} match(es); build features with "
                "add_card_features/add_referee_feature so the sufficient "
                "statistics exist"
            )
        return rng.gamma(shape[None, :], 1.0 / rate[None, :], size=(draws, len(shape)))

    def _sampled_mus(
        self,
        frame: pd.DataFrame,
        beta: np.ndarray,
        rng: np.random.Generator,
        draws: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw expected home and away cards, shape ``(draws, n_matches)``."""
        home_rate = away_rate = ref_rate = None
        if self.uses_team_rates:
            home_rate = self._gamma_rate_draws(
                rng,
                frame["league_mean"].to_numpy(dtype=float),
                frame["home_past_cards"].to_numpy(dtype=float),
                frame["home_past_matches"].to_numpy(dtype=float),
                draws,
            )
            away_rate = self._gamma_rate_draws(
                rng,
                frame["league_mean"].to_numpy(dtype=float),
                frame["away_past_cards"].to_numpy(dtype=float),
                frame["away_past_matches"].to_numpy(dtype=float),
                draws,
            )
        if self.uses_referee_rate:
            ref_rate = self._gamma_rate_draws(
                rng,
                frame["league_mean_total"].to_numpy(dtype=float),
                frame["ref_past_cards"].to_numpy(dtype=float),
                frame["ref_past_count"].to_numpy(dtype=float),
                draws,
            )
        mu_home = self._design_mu(
            beta, len(frame), home_rate, away_rate, ref_rate, 1.0
        )
        mu_away = self._design_mu(
            beta, len(frame), away_rate, home_rate, ref_rate, 0.0
        )
        return mu_home, mu_away

    def _design_mu(
        self,
        beta: np.ndarray,
        n_matches: int,
        own_rate: np.ndarray | None,
        opp_rate: np.ndarray | None,
        ref_rate: np.ndarray | None,
        home_flag: float,
    ) -> np.ndarray:
        """``exp(X beta)`` for one side's rows, X built per term and vectorised.

        ``beta`` is ``(draws, n_terms)`` and the rate arrays are
        ``(draws, n_matches)``; the result is ``(draws, n_matches)``. Terms come
        from the fitted design's own names, so the same code serves the team,
        league-average and referee designs without a branch per model — the
        league-average design has no rate column, so the rates are unused.
        """
        assert self._fitted_params is not None
        mu = np.zeros((beta.shape[0], n_matches))
        for position, term in enumerate(self._fitted_params.index):
            coefficient = beta[:, position][:, None]
            if term == "const":
                column = 1.0
            elif term == "log_own_rate":
                column = np.log(own_rate)
            elif term == "log_opp_rate":
                column = np.log(opp_rate)
            elif term == "log_ref_rate":
                column = np.log(ref_rate)
            elif term == "home_flag":
                column = home_flag
            else:  # pragma: no cover - the design's names are a closed set
                raise ValueError(f"unexpected design term {term!r} in predict_samples")
            mu = mu + coefficient * column
        return np.exp(mu)

    def _over_probabilities(
        self, mu_home: np.ndarray, mu_away: np.ndarray, lines: tuple[float, ...]
    ) -> np.ndarray:
        """``(draws, n_matches, n_lines)`` P(total > line) for each draw and line.

        The Poisson total is Poisson(mu_home + mu_away) (sums of independent
        Poissons), read analytically with the survival function — no grid. The
        negative-binomial total is the convolution of the two side pmfs, so it
        goes through :func:`~bet_engine.models.count_distributions.total_pmf`
        on flattened (draw x match) rows, in chunks, with a Python loop over the
        handful of lines but never over a grid cell.
        """
        if self.alpha_ is None or self.alpha_ <= 0.0:
            total = mu_home + mu_away
            return np.stack(
                [poisson.sf(np.floor(line), total) for line in lines], axis=-1
            )

        shape = mu_home.shape
        flat_home = mu_home.reshape(-1)
        flat_away = mu_away.reshape(-1)
        over = np.empty((flat_home.size, len(lines)), dtype=float)
        for start in range(0, flat_home.size, _SAMPLE_CHUNK):
            stop = start + _SAMPLE_CHUNK
            counts, pmf = _convolved_total_pmf(
                flat_home[start:stop], flat_away[start:stop], self.alpha_
            )
            for position, line in enumerate(lines):
                over[start:stop, position] = pmf[:, counts > line].sum(axis=1)
        return over.reshape(shape + (len(lines),))

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
        return (
            f"<{type(self).__name__} {self.family_name} {kind} "
            f"line={self.line:g} k={self.k:g}>"
        )


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


class _NegativeBinomialFit:
    """Mixin that swaps the Poisson GLM for a negative-binomial MLE.

    statsmodels' discrete :class:`~statsmodels.discrete.discrete_model.NegativeBinomial`
    estimates the dispersion ``alpha`` from the training rows (variance =
    mu + alpha * mu**2), so nothing here is tuned: the same design, the same
    rows and the same folds as the Poisson models feed it, and the only new
    number is the one the likelihood itself reports. ``alpha_`` is stored for
    the report; the design coefficients are dropped out of ``params`` because
    ``alpha`` is not one of them.

    WHY the MLE starts at the Poisson GLM solution: negative binomial nests
    Poisson (alpha = 0), so the Poisson fit is the natural, well-conditioned
    starting point for the same design. Left to its own default start,
    statsmodels' Newton search occasionally diverges on a small or collinear
    fold and returns a wild coefficient vector (per-side means in the
    hundreds), which is a numerical failure rather than a result. Starting from
    the Poisson optimum removes that failure without changing the objective: if
    the data carries no overdispersion the search stays at alpha = 0 and
    reproduces Poisson, and if it does the likelihood still moves the
    coefficients.

    WHY the convolution still matters even though step 3 finds alpha ~ 0: the
    parameter is estimated by the fit, not assumed, so if a league or season
    ever carries real per-side overdispersion this class captures it without a
    code change — the Poisson result is recovered as the alpha -> 0 limit.
    """

    family_name: ClassVar[str] = "negative-binomial"

    def _fit_family(self, design: pd.DataFrame, endog: np.ndarray) -> None:
        # The Poisson optimum of the SAME design is the start (see the class
        # docstring): a numerical initialisation, not a tuned parameter.
        poisson = sm.GLM(endog, design, family=sm.families.Poisson()).fit()
        start = np.concatenate([poisson.params.to_numpy(dtype=float), [0.5]])
        result = sm.NegativeBinomial(endog, design).fit(start_params=start, disp=0)
        params = result.params
        if "alpha" not in params.index:
            raise RuntimeError(
                "negative-binomial fit returned no alpha parameter; "
                f"terms were {list(params.index)}"
            )
        # The likelihood is maximised on the alpha = 0 boundary whenever the
        # training rows carry no overdispersion, which is the expected answer
        # here (per-side counts are not overdispersed). A non-finite alpha is
        # the same statement numerically; both are recorded as the Poisson
        # limit, not as a NaN pmf. Any finite positive alpha is used unchanged.
        # (statsmodels reports converged=False for the boundary optimum even
        # when the coefficients are healthy, so convergence is not consulted.)
        raw_alpha = float(params["alpha"])
        self.alpha_ = raw_alpha if np.isfinite(raw_alpha) and raw_alpha > 0.0 else 0.0
        self._fitted_params = params.drop("alpha")
        self._beta = self._fitted_params.to_numpy(dtype=float)
        # Drop alpha's row/column: predict_samples draws the design
        # coefficients only, and alpha is handled by its own convolution.
        # A degenerate fold (rank-deficient design, e.g. a tiny training
        # window) can leave statsmodels with no covariance at all; that is not
        # a reason to fail the fit, so fall back to a point (zero) covariance —
        # predict_samples then reports coefficient uncertainty as zero for the
        # affected fold rather than raising.
        try:
            self._fitted_bse = result.bse.drop("alpha")
            cov = result.cov_params()
            self._cov = np.asarray(
                cov.loc[self._fitted_params.index, self._fitted_params.index],
                dtype=float,
            )
        except (ValueError, np.linalg.LinAlgError):
            self._fitted_bse = pd.Series(
                np.nan, index=self._fitted_params.index, dtype=float
            )
            self._cov = np.zeros(
                (len(self._fitted_params), len(self._fitted_params)), dtype=float
            )


class CardsNegativeBinomial(_NegativeBinomialFit, CardsPoisson):
    """The team model, fitted by negative binomial instead of Poisson.

    Same design and rows as :class:`CardsPoisson` (log own rate, log opponent
    rate, home flag); the fit estimates the dispersion from the training data.
    The match total is the convolution of the two per-side negative-binomial
    pmfs (models/count_distributions.py), which is where any tail improvement
    over Poisson would show. No parameter is tuned for it.
    """


class CardsTeamRefNegativeBinomial(_NegativeBinomialFit, CardsTeamRefPoisson):
    """The team+referee design, fitted by negative binomial.

    The design is exactly :class:`CardsTeamRefPoisson`'s — const + log own rate
    + log opponent rate + log referee rate + home flag — so the only difference
    from :class:`CardsNegativeBinomial` is the referee column, and the only
    difference from :class:`CardsTeamRefPoisson` is the count likelihood.
    """


class CardsTotalNegativeBinomial(_NegativeBinomialFit, CardsPoisson):
    """Negative-binomial regression on the match *total*, one row per match.

    WHERE this sits among the other card models: every model above predicts each
    *side's* own cards, then builds the match total by convolving the two
    per-side pmfs. The market, though, settles on the total alone. Step 3 found
    the per-side counts carry essentially no dispersion of their own (alpha ~ 0)
    while the *total's* variance/mean sits above 1 — home and away cards are
    correlated through the match and the referee. The direct way to test whether
    that total-level overdispersion is worth modelling is to target the total
    itself: one row per match, one negative-binomial likelihood, one dispersion
    estimated straight from the data instead of assembled from two sides.

    Design is ``const + log home rate + log away rate + log referee rate`` —
    the same k=6 team and referee features as :class:`CardsTeamRefPoisson`, so
    any score difference against it is the target and the likelihood, not a
    change in the information available. (Those are the per-side rates, so the
    total mean is driven by both teams' discipline and the referee, exactly as
    the convolved model's would be, but the dispersion is now a single parameter
    for the total.)

    Over/under is read off this single total pmf — the one-count negative
    binomial in models/count_distributions.side_pmf — so there is no convolution
    here: the two-side step is exactly what this model does without.
    """

    #: Reported by __repr__ so a report cannot confuse the total target with
    #: the per-side models that share this file.
    target_name: ClassVar[str] = "total"

    #: The total design reads the referee rate directly, so it needs the column
    #: add_referee_feature produces — same requirement as the team+referee model.
    requires_features: ClassVar[tuple[str, ...]] = ("ref_rate",)

    def fit(self, train_df: pd.DataFrame) -> CardsTotalNegativeBinomial:
        """Estimate the total's coefficients and dispersion from match rows."""
        frame = _require_features(train_df, self.requires_features)

        # Same guard as the side models: a non-positive rate is a broken
        # shrinkage input, not a small number, and must be named before log().
        rates = frame[["home_rate", "away_rate", "ref_rate"]].to_numpy(dtype=float)
        bad = np.isfinite(rates) & (rates <= 0.0)
        if bad.any():
            raise ValueError(
                "cards_total_nbinom got a non-positive card rate; shrinkage "
                f"k={self.k} must keep rates > 0 ({int(bad.sum())} cell(s))"
            )

        design = self._design(frame)
        endog = frame["total_cards"].to_numpy(dtype="float64")

        usable = np.isfinite(endog) & np.isfinite(design.to_numpy(dtype=float)).all(
            axis=1
        )
        self.n_dropped_rows = int((~usable).sum())
        self.n_training_rows = int(usable.sum())
        if not usable.any():
            raise ValueError(
                "cards_total_nbinom.fit got no usable training rows "
                f"({len(frame)} match row(s) supplied, {self.n_dropped_rows} "
                "dropped for missing cards or features)"
            )

        self._fit_family(design.loc[usable], endog[usable])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Over/under probabilities per match from the single total pmf."""
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        frame = _require_features(df, self.requires_features)

        mu_total = self._mu(frame)
        if not np.isfinite(mu_total).all():
            bad = frame.loc[~np.isfinite(mu_total), "match_id"].head(3).tolist()
            raise ValueError(
                f"cards_total_nbinom produced a non-finite expected total for {bad}"
            )
        for match_id, total in zip(frame["match_id"], mu_total):
            self.predicted_totals[str(match_id)] = float(total)

        # One grid, one distribution: no convolution, because the target is
        # already the match total. side_pmf is a function of a count and its
        # mean, so it serves the total as it serves a side.
        counts = np.arange(2 * _SIDE_MAX_COUNT - 1)
        pmf = _side_pmf(mu_total, self.alpha_, counts)
        over = pmf[:, counts > self.line].sum(axis=1)
        under = pmf[:, counts <= self.line].sum(axis=1)
        probabilities = {"over": over, "under": under}
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
        """Match-level regressors: the two side rates and the referee rate.

        One row per match (never two), because the likelihood is on the match
        total. The columns are the per-side shrunk rates, so the total mean is
        the same kind of function of discipline and referee as the convolved
        model's, and the referee column makes this design a straight swap of the
        team+referee model's target.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame(
                {
                    "const": np.ones(len(rows), dtype=float),
                    "log_home_rate": np.log(rows["home_rate"].to_numpy(dtype=float)),
                    "log_away_rate": np.log(rows["away_rate"].to_numpy(dtype=float)),
                    "log_ref_rate": np.log(rows["ref_rate"].to_numpy(dtype=float)),
                }
            )

    def _mu(self, rows: pd.DataFrame) -> np.ndarray:
        """Expected *total* cards per match, directly (no per-side split)."""
        if self._beta is None:
            raise RuntimeError("predict() called before fit()")
        design = self._design(rows)
        finite = np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
        if not finite.all():
            bad = rows.loc[~finite, "match_id"].head(3).tolist()
            raise ValueError(
                f"cards_total_nbinom got a non-finite feature for match(es) "
                f"{bad}; build features with add_card_features(df, k) and "
                "add_referee_feature(df, k)"
            )
        return np.exp(design.to_numpy(dtype=float) @ self._beta)

    def predict_samples(
        self,
        df: pd.DataFrame,
        n: int = SAMPLE_DRAWS,
        *,
        lines: Sequence[float] = SAMPLE_LINES,
        seed: int | None = SAMPLE_SEED,
    ) -> np.ndarray:
        """Draw ``n`` samples of P(over)/P(under) for the *total* design.

        Same two uncertainty sources as the side models (coefficients from
        N(beta, cov), the three rates from their Gamma posteriors), but the
        single match-level total mean ``exp(X beta)`` feeds one negative-binomial
        count pmf per draw instead of two convolved sides. Return shape matches
        the side models: ``(n, len(df), len(lines), n_outcomes)``.
        """
        if self._beta is None or self._cov is None:
            raise RuntimeError("predict_samples() called before fit()")
        draws = _positive_int("n", n)
        line_labels = tuple(_positive_float("line", line) for line in lines)
        if not line_labels:
            raise ValueError("predict_samples needs at least one line")
        frame = _require_features(df, (*self.requires_features, *self._sample_columns()))

        rng = np.random.default_rng(seed)
        beta = self._coefficient_draws(rng, draws)
        mu_total = self._sampled_total_mu(frame, beta, rng, draws)

        counts = np.arange(2 * _SIDE_MAX_COUNT - 1)
        flat = mu_total.reshape(-1)
        over = np.empty((flat.size, len(line_labels)), dtype=float)
        for start in range(0, flat.size, _SAMPLE_CHUNK):
            stop = start + _SAMPLE_CHUNK
            pmf = _side_pmf(flat[start:stop], self.alpha_, counts)
            for position, line in enumerate(line_labels):
                over[start:stop, position] = pmf[:, counts > line].sum(axis=1)
        over = over.reshape(mu_total.shape + (len(line_labels),))
        by_outcome = {"over": over, "under": 1.0 - over}
        return np.stack([by_outcome[outcome] for outcome in self.outcomes], axis=-1)

    def _sampled_total_mu(
        self,
        frame: pd.DataFrame,
        beta: np.ndarray,
        rng: np.random.Generator,
        draws: int,
    ) -> np.ndarray:
        """``(draws, n_matches)`` per-draw expected *total* cards.

        Coefficients from ``N(beta, cov)`` and the three rates from their Gamma
        posteriors, then ``exp(X beta)`` on the match-level total design. Shared
        by :meth:`predict_samples` and :meth:`over_under` so both draw from the
        same posterior in the same order.
        """
        assert self._fitted_params is not None
        prior = frame["league_mean"].to_numpy(dtype=float)
        total_prior = frame["league_mean_total"].to_numpy(dtype=float)
        home_rate = self._gamma_rate_draws(
            rng, prior,
            frame["home_past_cards"].to_numpy(dtype=float),
            frame["home_past_matches"].to_numpy(dtype=float), draws,
        )
        away_rate = self._gamma_rate_draws(
            rng, prior,
            frame["away_past_cards"].to_numpy(dtype=float),
            frame["away_past_matches"].to_numpy(dtype=float), draws,
        )
        ref_rate = self._gamma_rate_draws(
            rng, total_prior,
            frame["ref_past_cards"].to_numpy(dtype=float),
            frame["ref_past_count"].to_numpy(dtype=float), draws,
        )
        mu_total = np.zeros((draws, len(frame)))
        for position, term in enumerate(self._fitted_params.index):
            coefficient = beta[:, position][:, None]
            if term == "const":
                column = 1.0
            elif term == "log_home_rate":
                column = np.log(home_rate)
            elif term == "log_away_rate":
                column = np.log(away_rate)
            elif term == "log_ref_rate":
                column = np.log(ref_rate)
            else:  # pragma: no cover - the design's names are a closed set
                raise ValueError(f"unexpected design term {term!r} in predict_samples")
            mu_total = mu_total + coefficient * column
        return np.exp(mu_total)

    def over_under(
        self,
        df: pd.DataFrame,
        *,
        line_column: str = "line",
        draws: int | None = None,
        seed: int | None = SAMPLE_SEED,
    ) -> pd.DataFrame | np.ndarray:
        """Per-row-line over/under/push from the single total pmf.

        Same contract as :meth:`CardsPoisson.over_under`, but the total is one
        negative-binomial count pmf (no convolution) because this model targets
        the match total directly.
        """
        counts = np.arange(2 * _SIDE_MAX_COUNT - 1)
        if draws is None:
            frame = _require_features(df, self.requires_features)
            lines = _line_values(frame, line_column)
            mu_total = self._mu(frame)
            pmf = _side_pmf(mu_total, self.alpha_, counts)
            over, under, push = _over_under_push(pmf, lines)
            return pd.DataFrame(
                {
                    "match_id": np.repeat(frame["match_id"].to_numpy(), 3),
                    "line": np.repeat(lines, 3),
                    "outcome": np.tile(
                        np.asarray(_OVER_UNDER_PUSH, dtype=object), len(frame)
                    ),
                    "prob": np.column_stack([over, under, push]).ravel(),
                }
            )

        self._require_sampling()
        frame = _require_features(
            df, (*self.requires_features, *self._sample_columns())
        )
        lines = _line_values(frame, line_column)
        count = _positive_int("draws", draws)
        rng = np.random.default_rng(seed)
        beta = self._coefficient_draws(rng, count)
        mu_total = self._sampled_total_mu(frame, beta, rng, count)
        flat = mu_total.reshape(-1)
        flat_lines = np.tile(lines, count)
        size = flat.size
        over = np.empty(size, dtype=float)
        under = np.empty(size, dtype=float)
        push = np.empty(size, dtype=float)
        for start in range(0, size, _SAMPLE_CHUNK):
            stop = start + _SAMPLE_CHUNK
            pmf = _side_pmf(flat[start:stop], self.alpha_, counts)
            o, u, p = _over_under_push(pmf, flat_lines[start:stop])
            over[start:stop] = o
            under[start:stop] = u
            push[start:stop] = p
        grid = (count, len(frame))
        return np.stack(
            [over.reshape(grid), under.reshape(grid), push.reshape(grid)], axis=-1
        )

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.family_name} {self.target_name} "
            f"line={self.line:g} k={self.k:g}>"
        )


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


def _line_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Per-row over/under lines, as a finite non-negative float array.

    The line a match settles against comes from the odds data, not the model,
    so a missing or unusable line is a join/config problem and is named as one
    rather than silently defaulted. Zero is allowed (a degenerate "over 0" that
    is always over); negatives and non-finite values are not.
    """
    if column not in frame.columns:
        raise ValueError(
            f"over_under needs a per-row line column {column!r}; join an odds "
            f"table that carries it (data.odds.join_odds) before predicting"
        )
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values) | (values < 0.0)
    if bad.any():
        examples = frame.loc[bad, "match_id"].head(3).tolist()
        raise ValueError(
            f"over_under got a non-finite or negative line for match(es) "
            f"{examples}"
        )
    return values


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"frame is missing column(s) {missing}; have {list(df.columns)}")


def _positive_int(name: str, value: object) -> int:
    """A draw count: an integer >= 1 (booleans are not counts)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


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
    "OVER_UNDER_PUSH",
    "REFEREE_FEATURE_COLUMNS",
    "REFEREE_HISTORY_COLUMNS",
    "SAMPLE_DRAWS",
    "SAMPLE_LINES",
    "SAMPLE_SEED",
    "SAMPLE_TEAM_COLUMNS",
    "CardsNegativeBinomial",
    "CardsPoisson",
    "CardsTeamRefNegativeBinomial",
    "CardsTeamRefPoisson",
    "CardsTotalNegativeBinomial",
    "LeagueAveragePoisson",
    "add_card_features",
    "add_referee_feature",
    "assert_features_are_causal",
]
