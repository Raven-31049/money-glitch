"""Tests for backtest/bankroll.py: day-atomic stakes, peak drawdown, breaches.

The toy sequence is hand-calculated (1000 bankroll, 10% per day, odds 2.0, two
bets a day over four days). Its stake list [100, 100, 80, 80, ...] is the point
of the test: every stake is a fraction of the day's OPENING balance, so the
second bet of a day is never re-priced by the first one's result. Both stop
modes run on the same inputs and must record the same breach (invariant 2), and
the peak test pins drawdown to the running peak rather than to the start.

Settlement is day-atomic too: a whole day's bets are placed and settled before
anything is judged, drawdown is measured only on CLOSING balances, and a stop
lands on a day boundary. So a mid-day dip never cuts a day in half, and the
order of bets within a day can never change the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine import db
from bet_engine.backtest import simulate

#: Two losses then six wins: a breach on day 1, then a full recovery.
WON = (False, False, True, True, True, True, True, True)


def _toy_frame(
    won=WON,
    *,
    stake_frac=0.1,
    odds=2.0,
    days=4,
    per_day=2,
    start="2021-08-01",
):
    """``per_day`` identically-priced bets on each of ``days`` calendar days."""
    dates = [
        day
        for day in pd.date_range(start, periods=days, freq="D")
        for _ in range(per_day)
    ]
    return pd.DataFrame(
        [
            {
                "match_id": f"{day.date()}-{position % per_day}",
                "date": day,
                "league": "EPL",
                "market": "match_winner",
                "outcome": "H",
                "odds": odds,
                "odds_source": "pinnacle",
                "won": won[position],
                "stake_frac": stake_frac,
            }
            for position, day in enumerate(dates)
        ]
    )


def _two_day_frame():
    """A day-1 win to a new peak, then a day-2 loss deep enough to breach."""
    return pd.DataFrame(
        [
            {"match_id": "m1", "date": "2021-08-01", "market": "match_winner",
             "outcome": "H", "odds": 2.0, "won": True, "stake_frac": 0.5},
            {"match_id": "m2", "date": "2021-08-02", "market": "match_winner",
             "outcome": "H", "odds": 2.0, "won": False, "stake_frac": 0.5},
        ]
    )


def _day_frame(day_results, *, stake_frac=0.4, odds=2.0, start="2021-08-01"):
    """One row per bet, grouped by day: ``day_results[d][i]`` is the outcome.

    match_ids are m1..mN in build order, so a variant that reorders rows keeps
    each bet's identity, odds, and result attached to it.
    """
    rows = [
        {"date": day, "league": "EPL", "market": "match_winner", "outcome": "H",
         "odds": odds, "odds_source": "pinnacle", "won": won, "stake_frac": stake_frac}
        for day, results in zip(
            pd.date_range(start, periods=len(day_results), freq="D"), day_results
        )
        for won in results
    ]
    return pd.DataFrame(
        [{**row, "match_id": f"m{index + 1}"} for index, row in enumerate(rows)]
    )


# --- day-atomic staking -----------------------------------------------------


def test_toy_sequence_sizes_every_stake_from_the_days_opening_balance():
    sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

    # Day 2 opens at 800, so its stakes are 80 — not 90 (yesterday's bankroll)
    # and not 88 (today's moving bankroll). Nothing re-prices a stake mid-day.
    assert sim.bets["stake"].tolist() == pytest.approx(
        [100, 100, 80, 80, 96, 96, 115.2, 115.2]
    )
    assert sim.bets["pnl"].tolist() == pytest.approx(
        [-100, -100, 80, 80, 96, 96, 115.2, 115.2]
    )
    assert sim.bets["bankroll_after"].tolist() == pytest.approx(
        [900, 800, 880, 960, 1056, 1152, 1267.2, 1382.4]
    )
    assert sim.starting == 1000
    assert sim.final_bankroll == pytest.approx(1382.4)
    assert sim.peak == pytest.approx(1382.4)


def test_stakes_are_numbered_by_placement_order():
    frame = _toy_frame()

    sim = simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert sim.bets["seq_no"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sim.bets["match_id"].tolist() == frame["match_id"].tolist()


# --- both stop modes (invariant 2) ------------------------------------------


def test_stop_mode_settles_the_whole_breach_day_then_halts():
    sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=True)

    # Day 1 closes at 800 (exactly 0.2 down): both of its bets are settled,
    # the breach is booked against the day's LAST seq_no, and the next day
    # is never started.
    assert sim.bets["seq_no"].tolist() == [1, 2]
    assert set(sim.bets["date"]) == {pd.Timestamp("2021-08-01")}
    assert sim.final_bankroll == pytest.approx(800)
    assert sim.breached is True
    assert sim.breach_seq_no == 2
    assert sim.breach_date == pd.Timestamp("2021-08-01")
    assert sim.worst_drawdown >= 0.2
    assert sim.worst_drawdown == pytest.approx(0.2)
    assert sim.stopped is True


def test_playing_on_records_the_same_breach_and_places_every_bet():
    sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert sim.bets["seq_no"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    # Same breach, both modes: "when did it break" does not depend on the flag.
    assert sim.breached is True
    assert sim.breach_seq_no == 2
    assert sim.breach_date == pd.Timestamp("2021-08-01")
    assert sim.stopped is False
    assert sim.final_bankroll == pytest.approx(1382.4)
    assert sim.worst_drawdown == pytest.approx(0.2)


def test_a_breach_on_the_final_bet_cannot_stop_anything():
    sim = simulate(_two_day_frame(), starting=1000, max_drawdown=0.3, stop_on_breach=True)

    assert sim.breach_seq_no == 2
    assert sim.stopped is False
    assert len(sim.bets) == 2


# --- day-atomic settlement ---------------------------------------------------


def test_the_stop_flag_changes_the_final_bankroll():
    # The same toy sequence run both ways (invariant 2): stopping banks 800
    # after the breaching day; playing on finishes at 1382.4. The difference
    # is the whole point of reporting both modes.
    stopped = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=True)
    played_on = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert stopped.final_bankroll == pytest.approx(800)
    assert played_on.final_bankroll == pytest.approx(1382.4)
    assert stopped.final_bankroll != played_on.final_bankroll
    assert len(stopped.bets) == 2
    assert len(played_on.bets) == 8


def test_a_mid_day_dip_never_cuts_the_day_in_half():
    # Day 1 loses 40% in one bet — exactly at the limit — then loses again.
    # Judged mid-day, bet 1 already breached and the old behaviour would have
    # halted with a single row; settled by the day, both bets stand and the
    # measured drawdown is the day's closing 0.8, not the intraday 0.4.
    frame = _day_frame([[False, False], [True, True], [True, True]])

    stopped = simulate(frame, starting=1000, max_drawdown=0.4, stop_on_breach=True)

    assert stopped.bets["seq_no"].tolist() == [1, 2]
    assert set(stopped.bets["date"]) == {pd.Timestamp("2021-08-01")}
    assert stopped.final_bankroll == pytest.approx(200)
    assert stopped.worst_drawdown == pytest.approx(0.8)
    assert stopped.breach_seq_no == 2
    assert stopped.breach_date == pd.Timestamp("2021-08-01")
    assert stopped.stopped is True

    played_on = simulate(frame, starting=1000, max_drawdown=0.4, stop_on_breach=False)

    # Days 2 and 3 open at 200 and 360, so their 80/144 stakes rebuild the
    # run only part way: 200 -> 360 -> 648, still 0.352 under the old peak.
    assert len(played_on.bets) == 6
    assert played_on.final_bankroll == pytest.approx(648)
    assert played_on.worst_drawdown == pytest.approx(0.8)
    assert played_on.breach_seq_no == 2
    assert played_on.breach_date == pd.Timestamp("2021-08-01")
    assert played_on.stopped is False


def test_a_day_that_recovers_by_settlement_has_not_breached():
    # Day 1 dips to exactly the limit after bet 1, then wins it all back —
    # its closing balance equals the start, so there is nothing to breach and
    # stop mode must place every bet of the run.
    frame = _day_frame([[False, True], [True, True]])

    sim = simulate(frame, starting=1000, max_drawdown=0.4, stop_on_breach=True)

    assert len(sim.bets) == 4
    assert sim.breached is False
    assert sim.breach_seq_no is None
    assert sim.breach_date is None
    assert sim.worst_drawdown == 0.0
    assert sim.stopped is False
    assert sim.final_bankroll == pytest.approx(1800)


@pytest.mark.parametrize("stop_on_breach", [True, False])
def test_row_order_within_a_day_cannot_change_the_result(stop_on_breach):
    # Same bets, each day's two rows listed in the opposite order. Under
    # per-bet judging the first variant's day-1 win would lift the peak before
    # the loss (0.286 down) while the reversed order would lose first (0.4
    # down, an instant breach) — results would depend on listing order. Judged
    # on closing balances, both orders must settle to the same day ends.
    ordered = _day_frame([[True, False], [False, False], [True, True]])
    reversed_days = pd.concat(
        [ordered.iloc[start : start + 2].iloc[::-1] for start in (0, 2, 4)]
    ).reset_index(drop=True)

    first = simulate(ordered, starting=1000, max_drawdown=0.4, stop_on_breach=stop_on_breach)
    second = simulate(reversed_days, starting=1000, max_drawdown=0.4, stop_on_breach=stop_on_breach)

    assert (
        first.final_bankroll,
        first.peak,
        first.worst_drawdown,
        first.breach_seq_no,
        first.breach_date,
        first.stopped,
    ) == (
        second.final_bankroll,
        second.peak,
        second.worst_drawdown,
        second.breach_seq_no,
        second.breach_date,
        second.stopped,
    )
    # The per-bet ledger must also agree as an unordered set of bets (seq_no
    # and bankroll_after are placement artifacts and legitimately differ).
    def bets_by_match(sim):
        return {row["match_id"]: (row["stake"], row["pnl"]) for row in sim.bets.to_dict("records")}

    first_bets, second_bets = bets_by_match(first), bets_by_match(second)
    assert first_bets.keys() == second_bets.keys()
    for match_id in first_bets:
        assert first_bets[match_id] == pytest.approx(second_bets[match_id])
    assert len(first.bets) == len(second.bets)

    # Curve under test: day 1 ends where it started, day 2 closes at 200 (a
    # 0.8 breach on its last seq_no), day 3 recovers only part way.
    if stop_on_breach:
        assert first.bets["seq_no"].tolist() == [1, 2, 3, 4]
        assert first.final_bankroll == pytest.approx(200)
        assert first.stopped is True
    else:
        assert len(first.bets) == 6
        assert first.final_bankroll == pytest.approx(360)
        assert first.stopped is False
    assert first.peak == 1000
    assert first.worst_drawdown == pytest.approx(0.8)
    assert first.breach_seq_no == 4
    assert first.breach_date == pd.Timestamp("2021-08-02")
    assert first.breached is True and second.breached is True


# --- drawdown ---------------------------------------------------------------


def test_drawdown_is_measured_from_the_peak_not_from_the_start():
    sim = simulate(_two_day_frame(), starting=1000, max_drawdown=0.3, stop_on_breach=False)

    # The win lifts the run to 1500; the next day's stake (0.5 * 1500) drops it
    # to 750. From the peak that is 0.5 and breaches; from the start it would be
    # 0.25 and never would — which is the definition under test.
    assert sim.peak == 1500
    assert sim.final_bankroll == 750
    assert sim.worst_drawdown == 0.5
    assert sim.breached is True
    assert sim.breach_seq_no == 2


def test_a_recovered_run_can_fall_back_into_breach():
    # A shallow dip, a long recovery to a new peak, then two losses that leave
    # the run ABOVE where it started — yet 2560 -> 1024 is a 0.6 drawdown from
    # the peak. Measured from the start this curve's worst dip was 0.3
    # (1000 -> 700), so the peak is the only thing that can breach here.
    won = (False, True, True, True, True, True, False, False)
    sim = simulate(
        _toy_frame(won, stake_frac=0.3),
        starting=1000,
        max_drawdown=0.4,
        stop_on_breach=False,
    )

    assert sim.breach_seq_no == 8
    assert sim.breach_date == pd.Timestamp("2021-08-04")
    assert sim.peak == 2560
    assert sim.final_bankroll == 1024
    assert sim.worst_drawdown == pytest.approx(0.6)


# --- input validation -------------------------------------------------------


def test_a_non_dataframe_is_a_type_error():
    with pytest.raises(TypeError, match="DataFrame"):
        simulate([{"match_id": "m1"}], starting=1000, max_drawdown=0.2, stop_on_breach=True)


@pytest.mark.parametrize("starting", [0, -5, float("nan"), float("inf"), "abc"])
def test_an_unusable_starting_bankroll_is_refused(starting):
    with pytest.raises(ValueError, match="starting"):
        simulate(_toy_frame(), starting=starting, max_drawdown=0.2, stop_on_breach=True)


@pytest.mark.parametrize("max_drawdown", [0, 1, 1.5, -0.1, float("nan"), "thirty percent"])
def test_an_unusable_max_drawdown_is_refused(max_drawdown):
    with pytest.raises(ValueError, match="max_drawdown"):
        simulate(_toy_frame(), starting=1000, max_drawdown=max_drawdown, stop_on_breach=True)


@pytest.mark.parametrize("stop_on_breach", [1, 0, "yes", None])
def test_stop_on_breach_must_be_a_bool(stop_on_breach):
    with pytest.raises(TypeError, match="stop_on_breach"):
        simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=stop_on_breach)


def test_a_numpy_bool_stop_flag_is_accepted():
    sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=np.bool_(True))

    assert sim.stopped is True


def test_numeric_text_run_parameters_are_coerced_not_refused():
    # YAML authors and hand-written calls pass numbers as text; pydantic would
    # coerce them at load time, so simulate() does the same instead of being
    # stricter than the config layer that feeds it.
    from_text = simulate(_toy_frame(), starting="1000", max_drawdown="0.2", stop_on_breach=False)
    as_numbers = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert from_text.starting == 1000
    assert from_text.final_bankroll == as_numbers.final_bankroll


@pytest.mark.parametrize(
    "column", ["match_id", "date", "market", "outcome", "odds", "won", "stake_frac"]
)
def test_a_missing_required_column_is_refused(column):
    frame = _toy_frame().drop(columns=[column])

    with pytest.raises(ValueError, match="missing required column"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


def test_out_of_order_dates_are_refused():
    # seq_no and the day freeze both assume the caller means replay order.
    reversed_frame = _toy_frame().iloc[::-1]

    with pytest.raises(ValueError, match="date order"):
        simulate(reversed_frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


def test_day_month_order_is_never_guessed():
    frame = _toy_frame()
    frame["date"] = [day.strftime("%d/%m/%Y") for day in frame["date"]]

    with pytest.raises(ValueError, match="day/month"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


def test_iso_text_dates_are_accepted():
    frame = _toy_frame()
    frame["date"] = [day.strftime("%Y-%m-%d") for day in frame["date"]]

    sim = simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert len(sim.bets) == 8
    assert pd.Timestamp(sim.bets["date"].iloc[0]) == pd.Timestamp("2021-08-01")


def test_a_missing_date_is_refused():
    frame = _toy_frame()
    frame.loc[0, "date"] = None

    with pytest.raises(ValueError, match="present and parseable"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


@pytest.mark.parametrize("odds", [0.5, 1.0, float("nan"), float("inf"), "abc"])
def test_an_unusable_price_on_a_placed_bet_is_refused(odds):
    frame = _toy_frame()
    frame["odds"] = odds

    with pytest.raises(ValueError, match="odds must be"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


@pytest.mark.parametrize("won", [None, "yes", float("nan")])
def test_a_non_boolean_won_is_refused(won):
    frame = _toy_frame()
    frame["won"] = won

    with pytest.raises(ValueError, match="won must be"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


@pytest.mark.parametrize("stake_frac", [-0.1, 1.5, float("nan"), "half"])
def test_an_unusable_stake_frac_is_refused(stake_frac):
    frame = _toy_frame()
    frame["stake_frac"] = stake_frac

    with pytest.raises(ValueError, match="stake_frac must be"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


def test_a_day_cannot_stake_more_than_it_opens_with():
    frame = _toy_frame()
    frame["stake_frac"] = 0.6  # two bets a day: 1.2 of the day's opening balance

    with pytest.raises(ValueError, match="stake_frac sums above 1.0"):
        simulate(frame, starting=1000, max_drawdown=0.2, stop_on_breach=True)


def test_a_day_may_stake_exactly_its_opening_balance():
    frame = _toy_frame()
    frame["stake_frac"] = 0.5  # 0.5 + 0.5: at the budget, not over it

    sim = simulate(frame, starting=1000, max_drawdown=0.99, stop_on_breach=False)

    assert len(sim.bets) == 8
    assert sim.bets["bankroll_after"].iloc[1] == pytest.approx(0.0)


# --- ledger -----------------------------------------------------------------


def test_an_empty_frame_is_a_valid_no_op():
    empty = pd.DataFrame(
        columns=["match_id", "date", "market", "outcome", "odds", "won", "stake_frac"]
    )

    sim = simulate(empty, starting=500, max_drawdown=0.2, stop_on_breach=True)

    assert len(sim.bets) == 0
    assert list(sim.bets.columns) == [
        "match_id",
        "date",
        "market",
        "outcome",
        "odds",
        "seq_no",
        "stake",
        "pnl",
        "bankroll_after",
    ]
    assert sim.final_bankroll == 500
    assert sim.peak == 500
    assert sim.worst_drawdown == 0.0
    assert sim.breach_seq_no is None
    assert sim.breach_date is None
    assert sim.stopped is False
    assert sim.breached is False


def test_ledger_keeps_provenance_and_drops_the_simulation_inputs():
    sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

    assert list(sim.bets.columns) == [
        "match_id",
        "date",
        "market",
        "outcome",
        "odds",
        "league",
        "odds_source",
        "seq_no",
        "stake",
        "pnl",
        "bankroll_after",
    ]
    # won/stake_frac are how the bet was priced, not facts about the ledger.
    assert "won" not in sim.bets.columns
    assert "stake_frac" not in sim.bets.columns


def test_ledger_roundtrips_through_the_bets_table(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite3")
    try:
        db.write_run(
            conn,
            "run-1",
            variant_name="epl_mw_naive",
            market="match_winner",
            config={"staking": {"kelly_fraction": 0.25}},
        )
        sim = simulate(_toy_frame(), starting=1000, max_drawdown=0.2, stop_on_breach=False)

        # run_id is the only bets-table column the ledger does not carry itself.
        db.write_bets(
            conn, [{"run_id": "run-1", **row} for row in sim.bets.to_dict("records")]
        )
        rows = db.read_bets(conn, "run-1")

        assert len(rows) == len(sim.bets)
        assert [row["seq_no"] for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert rows[0]["date"] == "2021-08-01"
        assert rows[0]["stake"] == pytest.approx(100.0)
        assert rows[0]["pnl"] == pytest.approx(-100.0)
        assert rows[-1]["bankroll_after"] == pytest.approx(1382.4)
        assert {row["league"] for row in rows} == {"EPL"}
        assert {row["odds_source"] for row in rows} == {"pinnacle"}
        assert "won" not in rows[0]
        assert "stake_frac" not in rows[0]
    finally:
        conn.close()
