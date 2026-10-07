"""Tests for models/cards_poisson.py: causal features + the Poisson models.

Three properties carry most of the weight here:

* **Causality.** The card rates — team and referee alike — may only use
  matches strictly before each row's date, so the runtime probe must pass on
  the real feature functions and must *refuse* deliberately broken ones — one
  that reads its own match (caught by the same-day probe) and one that reads
  the final matchday of the frame (only the strictly-future probe can catch
  it, which is why that probe exists).
* **Hand-checkability.** When observed cards equal the fitted rate the team
  model must predict exactly that rate, and with every card count constant the
  baseline must predict exactly the two constants — arithmetic a reader can do
  on paper, not "the number the optimiser returned". The referee shrinkage is
  checked the same way, against the same formula with the same k.
* **Contract.** over + under = 1 per match per line, and the failures
  (predict before fit, missing features, bad params) name what went wrong.
  The referee variant adds exactly one design column; the team-only model must
  not notice the referee column's presence at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import WALK_FORWARD_COLUMNS, walk_forward
from bet_engine.models import (
    CardsPoisson,
    CardsTeamRefPoisson,
    LeagueAveragePoisson,
    create_model,
)
from bet_engine.models.cards_poisson import (
    add_card_features,
    add_referee_feature,
    assert_features_are_causal,
)
from bet_engine.models.base import check_predictions

OUTCOMES = ("over", "under")
LINES = (3.5, 4.5, 5.5)

#: League mean of each date's own-card counts, worked out by hand: the sums
#: are over every team-match row *before* that date (d1 8/4, then +9, +6).
DAY2_MEAN = 8 / 4
DAY3_MEAN = 17 / 8
DAY4_MEAN = 23 / 12

#: League mean *total* cards per match before each date (totals: d1 4+4, then
#: +4+5, +4+2) — the quantity the referee rate shrinks toward.
TOTAL_DAY2_MEAN = 8 / 2
TOTAL_DAY3_MEAN = 17 / 4
TOTAL_DAY4_MEAN = 23 / 6


def _history_frame() -> pd.DataFrame:
    """Eight matches over four weekly dates, cards hand-chosen to be countable.

    Team New first appears on day 3, which is the zero-history case; every
    other rate below is checked against the shrinkage formula with k=6.
    """
    rows = [
        # date, home, away, hy, ay, hr, ar
        ("2021-08-01", "A", "B", 2, 1, 0, 1),
        ("2021-08-01", "C", "D", 4, 0, 0, 0),
        ("2021-08-08", "A", "C", 0, 2, 1, 1),
        ("2021-08-08", "B", "D", 2, 1, 2, 0),
        ("2021-08-15", "New", "A", 1, 3, 0, 0),
        ("2021-08-15", "B", "C", 0, 1, 0, 1),
        ("2021-08-22", "New", "B", 2, 0, 1, 2),
        ("2021-08-22", "A", "D", 1, 2, 1, 0),
    ]
    return pd.DataFrame(
        {
            "match_id": [f"m{i + 1}" for i in range(len(rows))],
            "league": "E0",
            "date": pd.to_datetime([row[0] for row in rows]),
            "home": [row[1] for row in rows],
            "away": [row[2] for row in rows],
            "hy": [row[3] for row in rows],
            "ay": [row[4] for row in rows],
            "hr": [row[5] for row in rows],
            "ar": [row[6] for row in rows],
        }
    )


def _referee_frame() -> pd.DataFrame:
    """The eight history matches plus a referee column, countable by hand.

    R1 takes m1, m3, m5 (returning referee); R2 takes m2 and m6; R3 takes m4
    (debut, no history) and m7; R4 debuts on m8. Match totals are 4, 4, 4, 5,
    4, 2, 5, 4 — so every referee shrinkage case (first date, returning,
    returning after a gap, two different debuts) sits on one league-mean
    sequence: day 1 has no mean, then 8/2, 17/4, 23/6 total cards per match.
    """
    frame = _history_frame()
    frame["referee"] = ["R1", "R2", "R1", "R3", "R1", "R2", "R3", "R4"]
    return frame


def _scored_frame(
    home_cards: list[float],
    away_cards: list[float],
    home_rate: list[float],
    away_rate: list[float],
    match_ids: list[str] | None = None,
) -> pd.DataFrame:
    """A frame that already carries the feature columns.

    The model consumes features, not history, so a hand-check builds them
    directly: same columns add_card_features would emit, no walk back through
    the shrinkage arithmetic.
    """
    n = len(home_cards)
    ids = match_ids or [f"m{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "match_id": ids,
            "league": ["E0"] * n,
            "date": pd.date_range("2021-08-01", periods=n, freq="7D"),
            "home": [f"home{i}" for i in range(n)],
            "away": [f"away{i}" for i in range(n)],
            "home_cards": home_cards,
            "away_cards": away_cards,
            "total_cards": np.add(home_cards, away_cards),
            "home_rate": home_rate,
            "away_rate": away_rate,
            "league_mean": [2.0] * n,
        }
    )


def _rates_equal_cards(n: int = 12) -> pd.DataFrame:
    """Cards that *are* the rates, so the fitted model must predict them back."""
    home_rate = ([1.0, 2.0, 3.0, 4.0] * ((n + 3) // 4))[:n]
    away_rate = ([4.0, 3.0, 2.0, 1.0] * ((n + 3) // 4))[:n]
    return _scored_frame(
        home_cards=home_rate,
        away_cards=away_rate,
        home_rate=home_rate,
        away_rate=away_rate,
    )


def _fit_team(frame: pd.DataFrame, line: float = 3.5) -> CardsPoisson:
    return CardsPoisson(OUTCOMES, line=line, k=6.0).fit(frame)


# ---------------------------------------------------------------------------
# causality of the features
# ---------------------------------------------------------------------------


def test_add_card_features_verifies_itself_by_default():
    """The default path runs the probe: a leak cannot be shipped by omission."""
    features = add_card_features(_history_frame(), 6.0)

    assert "home_rate" in features.columns
    assert "away_rate" in features.columns
    assert "total_cards" in features.columns


def _sees_its_own_match(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """BROKEN: every row's rate is its own card count — a same-day/self leak."""
    out = df.copy()
    own = (df["hy"] + df["hr"]).astype(float)
    out["home_rate"] = own
    out["away_rate"] = own
    out["league_mean"] = own
    return out


def test_a_feature_function_that_reads_its_own_match_is_refused():
    with pytest.raises(AssertionError) as exc:
        assert_features_are_causal(
            _sees_its_own_match, _history_frame(), 6.0, context="leaky test fn"
        )

    message = str(exc.value)
    assert "not strictly causal" in message
    assert "on or after" in message  # the same-day probe is the one that fired
    assert "[leaky test fn]" in message


def _sees_the_final_matchday(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """BROKEN: every row is scored off the last matchday's cards — a future leak.

    Only the *strictly-future* probe can catch this: perturbing the probe
    date's own rows leaves the final matchday untouched, so the same-day probe
    sees no change at all.
    """
    out = df.copy()
    cards = (df["hy"] + df["ay"] + df["hr"] + df["ar"]).astype(float)
    final_mean = float(cards[df["date"] == df["date"].max()].mean())
    out["home_rate"] = final_mean
    out["away_rate"] = final_mean
    out["league_mean"] = final_mean
    return out


def test_a_feature_function_that_reads_the_future_is_refused():
    with pytest.raises(AssertionError) as exc:
        assert_features_are_causal(
            _sees_the_final_matchday, _history_frame(), 6.0, context="future test fn"
        )

    message = str(exc.value)
    assert "not strictly causal" in message
    assert "after 2021-08-01" in message  # the first probe's future probe fired
    assert "[future test fn]" in message


def test_a_probe_with_nothing_to_perturb_says_so_rather_than_passing():
    """Card columns that exist but carry nothing: the property is untestable
    on this frame, and an untested guard must not read as a passed check."""
    frame = _history_frame()
    frame[["hy", "ay", "hr", "ar"]] = np.nan

    with pytest.raises(AssertionError, match="nothing to perturb"):
        assert_features_are_causal(_sees_its_own_match, frame, 6.0)


def test_the_first_match_of_a_frame_has_no_rate_yet():
    features = add_card_features(_history_frame(), 6.0).set_index("match_id")

    # No earlier match exists, so there is no league average to shrink toward:
    # NaN is the honest answer, and fit() drops the row rather than scoring 0.
    assert np.isnan(features.loc["m1", "home_rate"])
    assert np.isnan(features.loc["m1", "away_rate"])
    assert np.isnan(features.loc["m1", "league_mean"])


def test_a_team_with_no_history_gets_the_league_average_of_its_day():
    features = add_card_features(_history_frame(), 6.0).set_index("match_id")

    # New's first match is on day 3, whose league mean is 17/8 cards per side.
    assert features.loc["m5", "home_rate"] == pytest.approx(DAY3_MEAN)


def test_returning_team_rates_match_the_shrinkage_formula_by_hand():
    features = add_card_features(_history_frame(), 6.0).set_index("match_id")

    # k=6: (own cards in earlier matches + 6 * league mean) / (matches + 6).
    assert features.loc["m3", "home_rate"] == pytest.approx(
        (2 + 6 * DAY2_MEAN) / 7
    )
    assert features.loc["m3", "away_rate"] == pytest.approx(
        (4 + 6 * DAY2_MEAN) / 7
    )
    assert features.loc["m5", "away_rate"] == pytest.approx(
        (3 + 6 * DAY3_MEAN) / 8
    )
    assert features.loc["m6", "home_rate"] == pytest.approx(
        (6 + 6 * DAY3_MEAN) / 8
    )
    assert features.loc["m7", "home_rate"] == pytest.approx(
        (1 + 6 * DAY4_MEAN) / 7
    )
    assert features.loc["m8", "away_rate"] == pytest.approx(
        (1 + 6 * DAY4_MEAN) / 8
    )
    assert features.loc["m3", "home_rate"] == pytest.approx(2.0)


def test_two_matches_on_one_calendar_day_never_see_each_other():
    """The calendar day, not the row, is the boundary — same rule as the folds."""
    frame = pd.DataFrame(
        {
            "match_id": ["d1", "d2", "d3"],
            "league": "E0",
            "date": pd.to_datetime(["2021-08-01", "2021-08-08", "2021-08-08"]),
            "home": ["X", "X", "Q"],
            "away": ["P", "Q", "X"],
            "hy": [4, 0, 1],
            "ay": [0, 2, 0],
            "hr": [0, 0, 1],
            "ar": [0, 0, 0],
        }
    )

    features = add_card_features(frame, 6.0).set_index("match_id")

    # X's day-2 rows may only use day 1 (own cards 4, one match); the other
    # day-2 match it plays in must not leak into its own rate.
    day2_rate = (4 + 6 * 2.0) / 7
    assert features.loc["d2", "home_rate"] == pytest.approx(day2_rate)
    assert features.loc["d3", "away_rate"] == pytest.approx(day2_rate)


# ---------------------------------------------------------------------------
# causality and arithmetic of the referee feature
# ---------------------------------------------------------------------------


def test_add_referee_feature_verifies_itself_by_default():
    """The default path runs the probe on the referee columns too."""
    features = add_referee_feature(_referee_frame(), 6.0)

    assert "ref_rate" in features.columns
    assert "ref_past_count" in features.columns


def test_referee_rates_match_the_shrinkage_formula_by_hand():
    """k=6 and the same formula as the teams, checked on paper:

    (referee's total cards in strictly-earlier matches + 6 * league mean total)
    / (strictly-earlier matches + 6).
    """
    features = add_referee_feature(_referee_frame(), 6.0).set_index("match_id")

    # R1 at m3: one earlier match (4 cards), league mean total before day 2.
    assert features.loc["m3", "ref_rate"] == pytest.approx((4 + 6 * TOTAL_DAY2_MEAN) / 7)
    # R1 at m5: two earlier matches (4+4), league mean before day 3.
    assert features.loc["m5", "ref_rate"] == pytest.approx((8 + 6 * TOTAL_DAY3_MEAN) / 8)
    # R2 at m6: one earlier match (m2, 4 cards).
    assert features.loc["m6", "ref_rate"] == pytest.approx((4 + 6 * TOTAL_DAY3_MEAN) / 7)
    # R3 at m7: one earlier match (m4, 5 cards), league mean before day 4.
    assert features.loc["m7", "ref_rate"] == pytest.approx((5 + 6 * TOTAL_DAY4_MEAN) / 7)
    assert features.loc["m7", "ref_past_count"] == 1


def test_a_referee_with_no_history_gets_the_league_mean_total():
    """The referee rule is the team rule: no history means the league mean
    as of that date, with no referee-specific special case."""
    features = add_referee_feature(_referee_frame(), 6.0).set_index("match_id")

    # R3's debut (m4) is on day 2: shrinkage of 0 matches toward day-2 mean.
    assert features.loc["m4", "ref_rate"] == pytest.approx(TOTAL_DAY2_MEAN)
    assert features.loc["m4", "ref_past_count"] == 0
    # R4's debut (m8) is on day 4.
    assert features.loc["m8", "ref_rate"] == pytest.approx(TOTAL_DAY4_MEAN)
    assert features.loc["m8", "ref_past_count"] == 0


def test_the_first_date_has_no_referee_rate_yet():
    """No earlier match means no league mean total to shrink toward: NaN, and
    the row is dropped by fit rather than scored as if it were a zero."""
    features = add_referee_feature(_referee_frame(), 6.0).set_index("match_id")

    assert np.isnan(features.loc["m1", "ref_rate"])
    assert np.isnan(features.loc["m2", "ref_rate"])
    assert features.loc["m1", "ref_past_count"] == 0  # the count is real


def test_a_referee_rate_ignores_its_own_match():
    """Perturbing m3's cards must leave m3's rate untouched (it can only use
    strictly-earlier matches) while moving R1's *later* rate — the referee
    feature is history, not a restatement of the row."""
    base = add_referee_feature(_referee_frame(), 6.0).set_index("match_id")

    touched = _referee_frame()
    touched.loc[2, ["hy", "ay"]] = [5, 5]  # m3: same date, later referee history
    other = add_referee_feature(touched, 6.0).set_index("match_id")

    assert other.loc["m3", "ref_rate"] == base.loc["m3", "ref_rate"]
    assert other.loc["m5", "ref_rate"] != base.loc["m5", "ref_rate"]


def test_two_referee_matches_on_one_calendar_day_never_see_each_other():
    """The calendar-day boundary applies to referees exactly as to teams."""
    frame = pd.DataFrame(
        {
            "match_id": ["r1", "r2", "r3"],
            "league": "E0",
            "date": pd.to_datetime(["2021-08-01", "2021-08-08", "2021-08-08"]),
            "home": ["X", "X", "Q"],
            "away": ["P", "Q", "X"],
            "hy": [4, 0, 1],
            "ay": [0, 2, 0],
            "hr": [0, 0, 1],
            "ar": [0, 0, 0],
            "referee": ["R", "R", "R"],
        }
    )

    features = add_referee_feature(frame, 6.0).set_index("match_id")

    # Day 2: the referee's only history is day 1 (one match, 4 cards) and the
    # day-2 mean total is 4 — each day-2 match must see exactly that, never
    # the other day-2 match's cards.
    expected = (4 + 6 * 4.0) / 7
    assert features.loc["r2", "ref_rate"] == pytest.approx(expected)
    assert features.loc["r3", "ref_rate"] == pytest.approx(expected)
    assert features.loc["r2", "ref_past_count"] == 1


def _ref_sees_its_own_match(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """BROKEN: the referee rate *is* the row's own total — a self leak."""
    out = df.copy()
    total = (df["hy"] + df["hr"] + df["ay"] + df["ar"]).astype(float)
    out["ref_rate"] = total
    out["ref_past_count"] = 0.0
    return out


def test_a_referee_feature_that_reads_its_own_match_is_refused():
    with pytest.raises(AssertionError) as exc:
        assert_features_are_causal(
            _ref_sees_its_own_match,
            _referee_frame(),
            6.0,
            requires=("match_id", "date", "hy", "ay", "hr", "ar", "referee"),
            watch=("ref_rate", "ref_past_count"),
            context="leaky referee test fn",
        )

    message = str(exc.value)
    assert "not strictly causal" in message
    assert "on or after" in message
    assert "[leaky referee test fn]" in message


def test_a_missing_referee_column_is_named():
    with pytest.raises(ValueError, match="referee"):
        add_referee_feature(_history_frame(), 6.0)


def test_a_zero_shrinkage_weight_is_refused_at_the_referee_boundary():
    with pytest.raises(ValueError, match="k"):
        add_referee_feature(_referee_frame(), 0.0)


# ---------------------------------------------------------------------------
# the team model
# ---------------------------------------------------------------------------


def test_the_team_model_predicts_the_rate_when_cards_equal_the_rate():
    """Hand-check: with own cards == own rate, mu must come back as exactly
    that rate, so the expected total is home_rate + away_rate per row."""
    frame = _rates_equal_cards()

    model = _fit_team(frame)
    model.predict(frame)  # populates predicted_totals

    assert set(model.predicted_totals.values())  # captured for the report
    for match_id, expected in zip(
        frame["match_id"], frame["home_rate"] + frame["away_rate"]
    ):
        assert model.predicted_totals[match_id] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("line", LINES)
def test_over_and_under_sum_to_one_for_every_line(line):
    frame = _rates_equal_cards()

    prediction = _fit_team(frame, line=line).predict(frame)

    check_predictions(prediction, list(OUTCOMES))
    assert prediction["prob"].between(0.0, 1.0).all()
    assert (prediction["prob"] > 0.0).all()  # Poisson never says "impossible"


def test_changing_one_rows_rate_moves_only_that_match():
    """The rate columns are real predictors, not decoration: a rate the data
    does not support (cards no longer equal the rate) must move mu."""
    frame = _rates_equal_cards()
    model = _fit_team(frame)

    touched = frame.copy()
    touched.loc[5, "away_rate"] *= 10.0
    moved = model.predict(touched).set_index("match_id")["prob"]
    original = model.predict(frame).set_index("match_id")["prob"]

    disturbed = frame.loc[5, "match_id"]
    assert not np.allclose(moved.loc[disturbed], original.loc[disturbed])
    others = original.index != disturbed
    assert np.allclose(moved.loc[others], original.loc[others], atol=1e-6)


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="before fit"):
        CardsPoisson(OUTCOMES, line=3.5).predict(_rates_equal_cards())


def test_fit_drops_unusable_rows_and_counts_them():
    frame = pd.concat(
        [_rates_equal_cards(), _scored_frame([np.nan], [1.0], [2.0], [2.0], ["bad"])],
        ignore_index=True,
    )

    model = _fit_team(frame)

    # Fit sees two team rows per match, so 13 matches are 26 rows minus the
    # one whose home cards are missing.
    assert model.n_dropped_rows == 1
    assert model.n_training_rows == 2 * len(frame) - 1
    assert model.predict(frame).notna().all().all()


def test_fit_with_nothing_usable_is_refused():
    frame = _scored_frame([np.nan], [np.nan], [np.nan], [np.nan])

    with pytest.raises(ValueError, match="no usable training rows"):
        _fit_team(frame)


def test_a_missing_feature_column_is_named():
    frame = _rates_equal_cards().drop(columns=["home_rate"])

    with pytest.raises(ValueError, match="home_rate"):
        _fit_team(frame)


def test_a_non_finite_feature_names_the_match():
    frame = _rates_equal_cards()
    frame.loc[3, "home_rate"] = np.nan

    with pytest.raises(ValueError) as exc:
        _fit_team(frame).predict(frame)

    assert frame.loc[3, "match_id"] in str(exc.value)


def test_a_non_positive_rate_is_refused_by_fit():
    frame = _rates_equal_cards()
    frame.loc[2, "home_rate"] = 0.0

    with pytest.raises(ValueError, match="non-positive card rate"):
        _fit_team(frame)


@pytest.mark.parametrize("line", [0, -3.5, np.nan, np.inf, True, "4.5"])
def test_the_line_must_be_a_finite_positive_number(line):
    with pytest.raises(ValueError, match="line"):
        CardsPoisson(OUTCOMES, line=line)


@pytest.mark.parametrize("k", [0, -1, np.nan, np.inf, True, "6"])
def test_the_shrinkage_weight_must_be_a_finite_positive_number(k):
    with pytest.raises(ValueError, match="k"):
        CardsPoisson(OUTCOMES, line=3.5, k=k)


@pytest.mark.parametrize(
    "outcomes",
    [("over",), ("over", "under", "push"), ("over", "Over"), ("H", "D", "A"), ()],
)
def test_the_model_only_accepts_a_two_way_over_under_market(outcomes):
    with pytest.raises(ValueError, match="over"):
        CardsPoisson(outcomes, line=3.5)


def test_a_missing_card_column_stops_features_before_anything_else():
    with pytest.raises(ValueError, match="hy"):
        add_card_features(_history_frame().drop(columns=["hy"]), 6.0)


def test_a_zero_shrinkage_weight_is_refused_at_the_feature_boundary():
    with pytest.raises(ValueError, match="k"):
        add_card_features(_history_frame(), 0.0)


# ---------------------------------------------------------------------------
# the league-average baseline
# ---------------------------------------------------------------------------


def test_the_baseline_predicts_the_hand_fit_of_a_constant_card_league():
    """Every side cards exactly 3 at home and 1 away: intercept + home flag
    solves on paper to mu_home = 3, mu_away = 1, so the total is 4 — not a
    number the optimiser happened to land on."""
    frame = _scored_frame(
        home_cards=[3.0] * 8,
        away_cards=[1.0] * 8,
        home_rate=[float(i % 4 + 1) for i in range(8)],
        away_rate=[float(i % 3 + 1) for i in range(8)],
    )

    model = LeagueAveragePoisson(OUTCOMES, line=3.5).fit(frame)
    model.predict(frame)

    for total in model.predicted_totals.values():
        assert total == pytest.approx(4.0, abs=1e-6)


def test_the_baseline_ignores_team_rates_entirely():
    """The control has to be *purely* league-average, or the comparison
    against the team model would not isolate team information."""
    frame = _rates_equal_cards()
    scrambled = frame.copy()
    scrambled["home_rate"] = scrambled["home_rate"] * 17.0 + 5.0
    scrambled["away_rate"] = scrambled["away_rate"] / 3.0

    baseline = LeagueAveragePoisson(OUTCOMES, line=3.5)
    as_built = baseline.fit(frame).predict(frame)
    as_scrambled = LeagueAveragePoisson(OUTCOMES, line=3.5).fit(frame).predict(
        scrambled
    )

    assert np.allclose(as_built["prob"], as_scrambled["prob"])


def test_the_baseline_reports_one_total_per_match_and_the_team_model_differs():
    frame = _rates_equal_cards()

    baseline = LeagueAveragePoisson(OUTCOMES, line=3.5).fit(frame)
    baseline.predict(frame)
    team = _fit_team(frame)

    assert len(set(baseline.predicted_totals.values())) == 1  # one number league-wide
    assert set(team.predicted_totals.values()) != set(
        baseline.predicted_totals.values()
    )


# ---------------------------------------------------------------------------
# the referee variant (team design + log referee rate)
# ---------------------------------------------------------------------------


def _referee_features() -> pd.DataFrame:
    """Both feature sets built on the hand-counted referee frame."""
    return add_referee_feature(add_card_features(_referee_frame(), 6.0), 6.0)


def test_the_referee_variant_adds_exactly_one_design_column():
    """Team design plus log_ref_rate — nothing else moves, which is what makes
    the team-vs-team+referee score difference attributable to referee
    information alone. Terms are read from the public coefficient table."""
    features = _referee_features()

    team_terms = set(
        CardsPoisson(OUTCOMES, line=3.5).fit(features)
        .coefficient_table()["term"]
        .tolist()
    )
    ref_terms = set(
        CardsTeamRefPoisson(OUTCOMES, line=3.5)
        .fit(features)
        .coefficient_table()["term"]
        .tolist()
    )

    assert ref_terms == team_terms | {"log_ref_rate"}


def test_the_team_model_ignores_the_referee_column_entirely():
    """Same property as the baseline ignoring the team rates: a scrambled
    referee column must not move a single team-model probability. Only
    scoreable rows (day 1 onward is excluded: no league mean exists yet, and
    walk-forward never predicts from such rows either)."""
    features = _referee_features().iloc[2:]
    scrambled = features.copy()
    scrambled["ref_rate"] = scrambled["ref_rate"] * 17.0 + 1.0

    as_built = CardsPoisson(OUTCOMES, line=3.5).fit(features).predict(features)
    as_scrambled = (
        CardsPoisson(OUTCOMES, line=3.5).fit(features).predict(scrambled)
    )

    assert np.allclose(as_built["prob"], as_scrambled["prob"])


def test_the_referee_variant_refuses_a_frame_without_ref_rate():
    """The failure names the column *and* the function that produces it, so a
    config wired to cards_team_ref but built with only team features says
    what to do instead of dying in a log()."""
    with pytest.raises(ValueError) as exc:
        CardsTeamRefPoisson(OUTCOMES, line=3.5).fit(_rates_equal_cards())

    message = str(exc.value)
    assert "ref_rate" in message
    assert "add_referee_feature" in message


def test_the_referee_coefficient_is_reported_with_an_interval():
    table = (
        CardsTeamRefPoisson(OUTCOMES, line=3.5)
        .fit(_referee_features())
        .coefficient_table()
        .set_index("term")
    )

    assert "log_ref_rate" in table.index
    row = table.loc["log_ref_rate"]
    assert row["se"] > 0.0
    assert row["ci_low"] < row["coef"] < row["ci_high"]


def test_changing_one_rows_referee_rate_moves_only_that_match():
    """The referee column is a real predictor: scrambling one match's rate
    must move that match's probabilities and no others. Scoreable rows only —
    the frame's first date has no league mean to shrink toward."""
    features = _referee_features().iloc[2:]
    model = CardsTeamRefPoisson(OUTCOMES, line=3.5).fit(features)

    touched = features.copy()
    touched.loc[4, "ref_rate"] *= 10.0
    moved = model.predict(touched).set_index("match_id")["prob"]
    original = model.predict(features).set_index("match_id")["prob"]

    disturbed = features.loc[4, "match_id"]
    assert not np.allclose(moved.loc[disturbed], original.loc[disturbed])
    others = original.index != disturbed
    assert np.allclose(moved.loc[others], original.loc[others], atol=1e-6)


def test_a_non_positive_referee_rate_is_refused_by_fit():
    features = _referee_features()
    features.loc[3, "ref_rate"] = 0.0

    with pytest.raises(ValueError, match="non-positive referee rate"):
        CardsTeamRefPoisson(OUTCOMES, line=3.5).fit(features)


def test_over_and_under_still_sum_to_one_for_the_referee_variant():
    features = _referee_features().iloc[2:]  # scoreable rows, as above

    prediction = (
        CardsTeamRefPoisson(OUTCOMES, line=4.5, k=6.0).fit(features).predict(features)
    )

    check_predictions(prediction, list(OUTCOMES))
    assert prediction["prob"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# registry + walk-forward smoke
# ---------------------------------------------------------------------------


def test_the_registry_builds_both_card_models_with_the_lines_from_config():
    team = create_model("cards_poisson", {"line": 4.5}, OUTCOMES)
    baseline = create_model("cards_league_average", {"line": 3.5}, OUTCOMES)
    team_ref = create_model("cards_team_ref", {"line": 5.5}, OUTCOMES)

    assert type(team) is CardsPoisson
    assert type(baseline) is LeagueAveragePoisson
    assert type(team_ref) is CardsTeamRefPoisson
    assert team.line == 4.5
    assert baseline.line == 3.5
    assert team_ref.line == 5.5
    assert team.outcomes == OUTCOMES
    assert "team+referee" in repr(team_ref)


def test_walk_forward_end_to_end_on_the_history_frame():
    """The whole path in miniature: features -> folds -> probabilities that
    sum to one, with every match's expected total captured for the report."""
    features = add_card_features(_history_frame(), 6.0)
    models: list[CardsPoisson] = []

    def factory() -> CardsPoisson:
        model = CardsPoisson(OUTCOMES, line=3.5, k=6.0)
        models.append(model)
        return model

    # min_train_days=14: weekly dates, so the first fold with two full weeks of
    # history is day 3 — the earliest fold whose training rows have rates at
    # all (day 1 has no history behind it), same reality as a season start.
    result = walk_forward(features, factory, min_train_days=14, refit_every_days=1)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert set(result["outcome"]) == set(OUTCOMES)
    assert result.groupby("match_id").size().eq(2).all()
    assert np.allclose(result.groupby("match_id")["model_prob"].sum(), 1.0)
    assert result["date"].min() > features["date"].min()
    captured = {key for model in models for key in model.predicted_totals}
    assert captured == set(result["match_id"])


def test_walk_forward_end_to_end_with_the_referee_variant():
    """The referee variant through the same full path: features (team +
    referee, both probe-verified), folds strictly by day, probabilities that
    sum to one. Training rows whose referee rate is NaN (the frame's first
    date) are dropped by fit, not rescored."""
    features = _referee_features()
    models: list[CardsTeamRefPoisson] = []

    def factory() -> CardsTeamRefPoisson:
        model = CardsTeamRefPoisson(OUTCOMES, line=3.5, k=6.0)
        models.append(model)
        return model

    result = walk_forward(features, factory, min_train_days=14, refit_every_days=1)

    assert list(result.columns) == list(WALK_FORWARD_COLUMNS)
    assert result.groupby("match_id").size().eq(2).all()
    assert np.allclose(result.groupby("match_id")["model_prob"].sum(), 1.0)
    captured = {key for model in models for key in model.predicted_totals}
    assert captured == set(result["match_id"])
    assert all(model.n_training_rows > 0 for model in models)

