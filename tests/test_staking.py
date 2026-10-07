"""Tests for backtest/staking.py: de-vig, EV, Kelly stakes, edge selection.

Invariant 4 is the spine of these tests: an exactly-fair price must never clear
the threshold, because the pure-market control relies on that to place ~0 bets.
The Kelly cases are hand-calculated so the formula, the cap, and the order of
the two multipliers are pinned to numbers rather than to the implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bet_engine.backtest import devig, ev, kelly_stake, select_bets

COLUMNS = ["match_id", "market", "outcome", "odds", "model_prob"]


def _frame(rows, columns=COLUMNS):
    return pd.DataFrame(rows, columns=list(columns))


# --- devig ------------------------------------------------------------------


def test_devig_three_way_probabilities_match_the_hand_calculation():
    probs, overround = devig({"H": 2.0, "D": 3.4, "A": 4.0})

    assert overround == pytest.approx(1.0441176470588236, rel=1e-12)
    assert probs["H"] == pytest.approx(0.4788732394366197, rel=1e-9)
    assert probs["D"] == pytest.approx(0.28169014084507044, rel=1e-9)
    assert probs["A"] == pytest.approx(0.23943661971830985, rel=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_devig_equal_prices_split_evenly():
    probs, overround = devig({"H": 1.91, "D": 1.91})

    assert probs == {"H": 0.5, "D": 0.5}
    assert overround == pytest.approx(1.0471204188481675, rel=1e-12)


def test_devig_five_way_overround_is_the_sum_of_the_implied_prices():
    probs, overround = devig({"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0, "e": 6.0})

    assert overround == pytest.approx(1.45, rel=1e-12)
    assert sum(probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "odds",
    [
        {"H": 1.5, "D": 4.0},
        {"H": 2.0, "D": 3.4, "A": 4.0},
        {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0, "e": 6.0},
    ],
)
def test_devig_rescales_every_implied_probability_by_the_overround(odds):
    probs, overround = devig(odds)
    implied = {outcome: 1.0 / price for outcome, price in odds.items()}

    assert sum(probs.values()) == pytest.approx(1.0)
    for outcome in odds:
        assert probs[outcome] == pytest.approx(implied[outcome] / overround)


def test_devig_keeps_the_markets_own_ordering():
    probs, _ = devig({"H": 2.0, "D": 3.4, "A": 4.0})

    assert probs["H"] > probs["D"] > probs["A"]


@pytest.mark.parametrize(
    "odds",
    [{}, {"H": 0.5}, {"H": 1.0}, {"H": float("nan")}, {"H": "abc"}, {"H": None}],
)
def test_devig_refuses_a_vector_it_cannot_normalize(odds):
    with pytest.raises(ValueError):
        devig(odds)


# --- ev ---------------------------------------------------------------------


def test_ev_is_bare_arithmetic():
    assert ev(0.5, 2.0) == 0.0
    assert ev(1.0, 3.0) == 2.0
    assert ev(0.6, 2.0) == pytest.approx(0.2)
    assert ev(0.4, 2.0) == pytest.approx(-0.2)


def test_ev_applies_elementwise_to_a_series():
    result = ev(pd.Series([0.5, 0.6, 0.4]), 2.0)

    assert isinstance(result, pd.Series)
    assert list(result) == pytest.approx([0.0, 0.2, -0.2])


# --- kelly_stake ------------------------------------------------------------


@pytest.mark.parametrize(
    "prob, odds, fraction, bankroll, cap, expected",
    [
        # f* = (p*o - 1) / (o - 1), then fraction, then cap, then bankroll.
        (0.55, 2.0, 0.25, 1000.0, 0.05, 25.0),   # f* 0.10 -> 0.025 -> 25
        (0.70, 3.0, 0.25, 200.0, 0.05, 10.0),    # uncapped 27.5 -> cap 0.05 * 200
        (0.50, 2.0, 1.0, 1000.0, 0.05, 0.0),     # break-even edge: no bet
        (0.40, 2.0, 1.0, 1000.0, 0.05, 0.0),     # negative edge: no bet
        (0.60, 2.0, 0.25, 1000.0, 0.05, 50.0),   # fraction lands exactly on the cap
        (0.75, 4.0, 1.0, 1000.0, 0.05, 50.0),    # full Kelly: the cap binds
        (0.55, 2.0, 0.25, 0.0, 0.05, 0.0),       # nothing to stake
    ],
)
def test_kelly_stake_hand_calculated(prob, odds, fraction, bankroll, cap, expected):
    stake = kelly_stake(prob, odds, fraction, bankroll, cap)

    assert stake == pytest.approx(expected)
    assert isinstance(stake, float)


def test_kelly_stake_never_exceeds_the_hard_cap():
    bankroll, cap = 1000.0, 0.05

    for prob in [0.5, 0.6, 0.75, 0.9, 0.99]:
        for odds in [1.5, 2.0, 3.5, 10.0]:
            for fraction in [0.25, 0.5, 1.0]:
                stake = kelly_stake(prob, odds, fraction, bankroll, cap)
                assert 0.0 <= stake <= cap * bankroll


@pytest.mark.parametrize(
    "overrides",
    [
        {"prob": 1.2},
        {"prob": -0.1},
        {"prob": float("nan")},
        {"odds": 1.0},
        {"odds": float("nan")},
        {"odds": "evens"},
        {"fraction": 0.0},
        {"fraction": 1.5},
        {"max_stake_frac": 0.0},
        {"max_stake_frac": 1.1},
    ],
)
def test_kelly_stake_rejects_out_of_range_arguments(overrides):
    params = {
        "prob": 0.55,
        "odds": 2.0,
        "fraction": 0.25,
        "bankroll": 1000.0,
        "max_stake_frac": 0.05,
    }
    params.update(overrides)

    with pytest.raises(ValueError):
        kelly_stake(**params)


@pytest.mark.parametrize("bankroll", [float("nan"), float("inf")])
def test_kelly_stake_rejects_a_non_finite_bankroll(bankroll):
    with pytest.raises(ValueError, match="bankroll"):
        kelly_stake(0.55, 2.0, 0.25, bankroll, 0.05)


def test_kelly_stake_rejects_a_negative_bankroll():
    # A spent bankroll (0) stakes nothing; a negative one is a bug upstream
    # and must not be silently priced as "stake nothing".
    with pytest.raises(ValueError, match="bankroll"):
        kelly_stake(0.55, 2.0, 0.25, -100.0, 0.05)


# --- select_bets ------------------------------------------------------------


def test_selects_one_bet_per_market_of_each_match():
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6},
            {"match_id": "m1", "market": "match_winner", "outcome": "D", "odds": 3.5, "model_prob": 0.2},
            {"match_id": "m1", "market": "match_winner", "outcome": "A", "odds": 4.0, "model_prob": 0.1},
            {"match_id": "m1", "market": "over_under_25", "outcome": "over", "odds": 1.9, "model_prob": 0.6},
            {"match_id": "m1", "market": "over_under_25", "outcome": "under", "odds": 1.9, "model_prob": 0.3},
        ]
    )

    selected = select_bets(frame, 0.0)

    assert selected[["market", "outcome"]].to_dict("records") == [
        {"market": "match_winner", "outcome": "H"},
        {"market": "over_under_25", "outcome": "over"},
    ]


def test_output_keeps_input_columns_and_input_row_order():
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 3.0, "model_prob": 0.2, "date": "2021-08-01"},
            {"match_id": "m2", "market": "match_winner", "outcome": "H", "odds": 2.5, "model_prob": 0.5, "date": "2021-08-01"},
            {"match_id": "m1", "market": "match_winner", "outcome": "A", "odds": 2.0, "model_prob": 0.6, "date": "2021-08-02"},
            {"match_id": "m2", "market": "match_winner", "outcome": "D", "odds": 2.0, "model_prob": 0.4, "date": "2021-08-02"},
        ],
        columns=["date", *COLUMNS],
    )

    selected = select_bets(frame, 0.0)

    assert list(selected.columns) == list(frame.columns)
    # Frame order, not group order: m2's row precedes m1's second row.
    assert selected["match_id"].tolist() == ["m2", "m1"]
    assert selected["date"].tolist() == ["2021-08-01", "2021-08-02"]
    assert selected.index.tolist() == [0, 1]


def test_ties_go_to_the_first_row_in_the_frame():
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6},
            {"match_id": "m1", "market": "match_winner", "outcome": "D", "odds": 2.0, "model_prob": 0.6},
        ]
    )

    assert select_bets(frame, 0.0)["outcome"].tolist() == ["H"]


def test_exactly_fair_prices_never_clear_a_zero_threshold():
    # Invariant 4: EV == 0.0 is what the pure-market control's rows look like.
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.5},
            {"match_id": "m2", "market": "match_winner", "outcome": "D", "odds": 4.0, "model_prob": 0.25},
            {"match_id": "m3", "market": "match_winner", "outcome": "A", "odds": 5.0, "model_prob": 0.2},
        ]
    )

    selected = select_bets(frame, 0.0)

    assert selected.empty
    assert list(selected.columns) == COLUMNS


def test_threshold_comparison_is_strict_not_inclusive():
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.75},
            {"match_id": "m1", "market": "match_winner", "outcome": "D", "odds": 2.0, "model_prob": 0.751},
        ]
    )

    # First row's EV is exactly the threshold; only the second clears it.
    assert select_bets(frame, 0.5)["outcome"].tolist() == ["D"]


def test_ev_uses_the_paid_odds_not_the_de_vigged_fair_odds():
    # market_prob 0.50 implies a fair price of 2.0; the model says 0.55, so the
    # edge on fair odds is 0.55 * 2.0 - 1 = +0.10 and would clear the 0.05
    # threshold. The book only pays 1.8, so the edge on the odds actually paid
    # is 0.55 * 1.8 - 1 = -0.01: the bet must NOT be selected (PLAN.md section 5).
    raw_frame = _frame(
        [
            {
                "match_id": "m1",
                "market": "match_winner",
                "outcome": "H",
                "odds": 1.8,
                "model_prob": 0.55,
            }
        ]
    )
    fair_frame = raw_frame.assign(odds=[2.0])  # 1 / market_prob == 2.0

    assert select_bets(raw_frame, 0.05).empty
    assert select_bets(fair_frame, 0.05)["outcome"].tolist() == ["H"]


def test_pure_market_control_places_no_bets():
    raw = {"H": 2.0, "D": 3.4, "A": 4.0}
    probs, _ = devig(raw)
    # Zero model weight: the model believes the de-vigged market exactly, and
    # it bets at the raw book prices. Every EV is (1 / overround) - 1, which is
    # strictly negative, so the control places nothing.
    frame = _frame(
        [
            {
                "match_id": "m1",
                "market": "match_winner",
                "outcome": outcome,
                "odds": raw[outcome],
                "model_prob": probability,
            }
            for outcome, probability in probs.items()
        ]
    )

    assert select_bets(frame, 0.0).empty


def test_rows_without_a_price_are_never_selected():
    frame = _frame(
        [
            {"match_id": "m1", "market": "match_winner", "outcome": "A", "odds": float("nan"), "model_prob": 0.99},
            {"match_id": "m1", "market": "match_winner", "outcome": "B", "odds": 3.0, "model_prob": 0.4},
            {"match_id": "m2", "market": "match_winner", "outcome": "A", "odds": float("nan"), "model_prob": 0.9},
        ]
    )

    selected = select_bets(frame, 0.0)

    # m1's priced row wins its group; m2 has no price, so no bet at all.
    assert selected[["match_id", "outcome"]].to_dict("records") == [
        {"match_id": "m1", "outcome": "B"}
    ]


@pytest.mark.parametrize("odds", [0.5, 1.0, float("inf")])
def test_a_present_but_invalid_price_is_an_error(odds):
    frame = _frame(
        [{"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": odds, "model_prob": 0.6}]
    )

    with pytest.raises(ValueError, match="odds must be"):
        select_bets(frame, 0.0)


@pytest.mark.parametrize("model_prob", [float("nan"), 1.2, -0.01, "abc"])
def test_an_unusable_model_prob_is_an_error(model_prob):
    frame = _frame(
        [{"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": model_prob}]
    )

    with pytest.raises(ValueError, match="model_prob"):
        select_bets(frame, 0.0)


def test_a_null_identity_is_an_error():
    frame = _frame(
        [{"match_id": None, "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6}]
    )

    with pytest.raises(ValueError, match="null match_id"):
        select_bets(frame, 0.0)


def test_a_missing_required_column_is_an_error():
    frame = _frame(
        [{"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6}]
    ).drop(columns=["odds"])

    with pytest.raises(ValueError, match="missing required column"):
        select_bets(frame, 0.0)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), "generous"])
def test_an_unusable_threshold_is_an_error(threshold):
    frame = _frame(
        [{"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6}]
    )

    with pytest.raises(ValueError, match="ev_threshold"):
        select_bets(frame, threshold)


def test_an_empty_frame_is_a_valid_no_bet_result():
    selected = select_bets(_frame([]), 0.0)

    assert selected.empty
    assert list(selected.columns) == COLUMNS


def test_a_non_dataframe_is_a_type_error():
    with pytest.raises(TypeError, match="DataFrame"):
        select_bets([{"match_id": "m1"}], 0.0)


def test_the_input_frame_is_never_modified():
    frame = _frame(
        [{"match_id": "m1", "market": "match_winner", "outcome": "H", "odds": 2.0, "model_prob": 0.6}]
    )
    before = frame.copy()

    select_bets(frame, 0.0)

    pd.testing.assert_frame_equal(frame, before)
