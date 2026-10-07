"""Tests for eval/bootstrap.py: known edges, the full-sequence guard, warnings.

The three edge fixtures are synthetic and exact: unit-stake bets at fair odds
(2.0, implied 0.50) where exactly 55%, 50%, or 45% of the bets won — so the
bootstrap's answers are checkable against the sign of the edge rather than
against the implementation. Exact counts rather than a random draw keep the
fixture's mean pinned (order is irrelevant to resampling with replacement, so
the shuffle only makes the sequence read like a run).

Invariant 3 is the other spine: a ledger a stop truncated must be refused,
because an interval over a truncated curve answers a different question than
the played-on curve reported beside it (invariant 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_engine.backtest import simulate
from bet_engine.eval import BootstrapResult, bootstrap_pnl


def _sequence_with_win_rate(win_rate, *, n_bets=1000, odds=2.0):
    """Unit-stake P&Ls with exactly ``win_rate`` of the bets winning."""
    wins = round(win_rate * n_bets)
    pnl = np.array([odds - 1.0] * wins + [-1.0] * (n_bets - wins))
    return pnl[np.random.default_rng(17).permutation(n_bets)]


def _stopped_run(stop=True):
    """A run that breaches on day 1 of 3, so stopping really truncates it."""
    won = (False, False, True, True, True, True)
    days = pd.date_range("2021-08-01", periods=3, freq="D")
    sequence = [
        (day, outcome)
        for day, outcomes in zip(days, (won[:2], won[2:4], won[4:6]))
        for outcome in outcomes
    ]
    frame = pd.DataFrame(
        [
            {"match_id": f"m{position + 1}", "date": day, "market": "match_winner",
             "outcome": "H", "odds": 2.0, "won": outcome, "stake_frac": 0.4}
            for position, (day, outcome) in enumerate(sequence)
        ]
    )
    return simulate(
        frame, starting=1000, max_drawdown=0.4, stop_on_breach=stop
    )


# --- known edges -------------------------------------------------------------


def test_a_positive_edge_is_profitable_in_nearly_every_resample():
    # 55% wins at even odds: +0.10 per unit staked, every resample reflects it.
    result = bootstrap_pnl(_sequence_with_win_rate(0.55), n=4000, seed=11)

    assert isinstance(result, BootstrapResult)
    assert result.n_bets == 1000
    assert result.fraction_profitable > 0.95
    assert result.mean_roi == pytest.approx(0.10, abs=0.02)
    assert result.ci_5 > 0.0
    assert result.ci_5 < result.ci_95


def test_a_zero_edge_is_profitable_about_half_the_time():
    result = bootstrap_pnl(_sequence_with_win_rate(0.50), n=4000, seed=5)

    assert 0.4 < result.fraction_profitable < 0.6
    assert abs(result.mean_roi) < 0.05
    assert result.ci_5 < 0.0 < result.ci_95


def test_a_negative_edge_is_almost_never_profitable():
    result = bootstrap_pnl(_sequence_with_win_rate(0.45), n=4000, seed=8)

    assert result.fraction_profitable < 0.05
    assert result.mean_roi == pytest.approx(-0.10, abs=0.02)
    assert result.ci_95 < 0.0


def test_the_default_resample_count_is_used_when_n_is_omitted():
    result = bootstrap_pnl(np.array([1.0, -1.0]), seed=1)

    assert result.n_bets == 2
    assert 0.0 <= result.fraction_profitable <= 1.0


def test_the_same_seed_reproduces_the_interval_and_another_seed_does_not():
    sequence = _sequence_with_win_rate(0.55, n_bets=200)

    first = bootstrap_pnl(sequence, n=1000, seed=42)
    second = bootstrap_pnl(sequence, n=1000, seed=42)
    other = bootstrap_pnl(sequence, n=1000, seed=43)

    assert first == second
    assert first != other


def test_a_single_bet_sequence_has_an_exact_interval():
    # Every resample of one bet is that bet, so nothing is left to chance.
    result = bootstrap_pnl(np.array([10.0]), n=50, seed=0)

    assert result.n_bets == 1
    assert result.fraction_profitable == 1.0
    assert result.mean_roi == pytest.approx(10.0)
    assert result.ci_5 == pytest.approx(10.0)
    assert result.ci_95 == pytest.approx(10.0)


def test_explicit_stakes_turn_the_interval_into_money_on_money():
    result = bootstrap_pnl(
        np.array([10.0]), n=50, seed=0, stakes=np.array([4.0])
    )

    assert result.mean_roi == pytest.approx(2.5)
    assert result.ci_5 == pytest.approx(2.5)


def test_a_ledger_frames_stake_column_is_the_roi_denominator():
    ledger = pd.DataFrame({"pnl": [10.0], "stake": [4.0]})

    with_stake = bootstrap_pnl(ledger, n=50, seed=0)
    overridden = bootstrap_pnl(ledger, n=50, seed=0, stakes=[1.0])

    assert with_stake.mean_roi == pytest.approx(2.5)
    assert overridden.mean_roi == pytest.approx(10.0)


# --- the full-sequence guard (invariant 3) -----------------------------------


def test_a_stopped_runs_ledger_is_refused_as_truncated():
    stopped = _stopped_run()
    assert stopped.stopped is True

    with pytest.raises(ValueError, match="truncated by a stop"):
        bootstrap_pnl(stopped, seed=1)
    with pytest.raises(ValueError, match="full, untruncated"):
        bootstrap_pnl(stopped.bets, seed=1)


def test_the_truncation_flag_alone_is_enough_to_refuse_a_ledger():
    ledger = pd.DataFrame({"pnl": [1.0, -1.0]})
    ledger.attrs["truncated_by_stop"] = True

    with pytest.raises(ValueError, match="truncated by a stop"):
        bootstrap_pnl(ledger, seed=1)

    ledger.attrs["truncated_by_stop"] = False
    assert bootstrap_pnl(ledger, n=10, seed=1).n_bets == 2


def test_a_played_on_run_is_accepted_and_covers_every_bet():
    played_on = _stopped_run(stop=False)
    assert played_on.stopped is False

    result = bootstrap_pnl(played_on, n=100, seed=1)

    assert result.n_bets == len(played_on.bets) == 6


def test_a_ledger_without_a_pnl_column_is_refused():
    with pytest.raises(ValueError, match="missing its 'pnl' column"):
        bootstrap_pnl(pd.DataFrame({"stake": [1.0]}), seed=1)


# --- small-sample warning ----------------------------------------------------


def test_fewer_than_100_bets_prints_a_warning(capsys):
    bootstrap_pnl(np.array([1.0, -1.0, 0.5]), n=10, seed=1)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "3 bets" in out
    assert "100" in out


def test_a_hundred_bets_prints_no_warning(capsys):
    bootstrap_pnl(np.full(100, 0.1), n=10, seed=1)

    assert capsys.readouterr().out == ""


# --- input validation --------------------------------------------------------


@pytest.mark.parametrize("n", [2.5, "100", True, None])
def test_a_non_integer_resample_count_is_refused(n):
    with pytest.raises(TypeError, match="n must be an integer"):
        bootstrap_pnl(np.array([1.0]), n=n, seed=1)


@pytest.mark.parametrize("n", [0, -1])
def test_a_non_positive_resample_count_is_refused(n):
    with pytest.raises(ValueError, match="n must be"):
        bootstrap_pnl(np.array([1.0]), n=n, seed=1)


@pytest.mark.parametrize("seed", ["abc", 1.5, True])
def test_a_non_integer_seed_is_refused(seed):
    with pytest.raises(TypeError, match="seed must be an int"):
        bootstrap_pnl(np.array([1.0]), n=10, seed=seed)


def test_an_explicit_none_seed_draws_fresh_entropy():
    result = bootstrap_pnl(np.array([1.0, -1.0]), n=10, seed=None)

    assert 0.0 <= result.fraction_profitable <= 1.0


def test_an_empty_sequence_is_refused():
    with pytest.raises(ValueError, match="at least one bet"):
        bootstrap_pnl(np.array([]), seed=1)


def test_a_non_finite_pnl_is_refused():
    with pytest.raises(ValueError, match="pnl must be finite"):
        bootstrap_pnl(np.array([1.0, np.nan, -1.0]), seed=1)


def test_a_two_dimensional_pnl_is_refused():
    with pytest.raises(ValueError, match="one value per bet"):
        bootstrap_pnl(np.array([[1.0, -1.0]]), seed=1)


def test_a_non_numeric_pnl_is_refused():
    with pytest.raises(ValueError, match="pnl must be numeric"):
        bootstrap_pnl(np.array(["ten", "-five"]), seed=1)


def test_stakes_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="one value per bet"):
        bootstrap_pnl(np.array([1.0, -1.0]), seed=1, stakes=np.array([1.0]))


def test_negative_stakes_are_refused():
    with pytest.raises(ValueError, match="stakes must be >= 0"):
        bootstrap_pnl(np.array([1.0, -1.0]), seed=1, stakes=np.array([1.0, -1.0]))


def test_zero_exposure_is_refused():
    with pytest.raises(ValueError, match="positive amount"):
        bootstrap_pnl(np.array([1.0, -1.0]), seed=1, stakes=np.zeros(2))


def test_non_finite_stakes_are_refused():
    with pytest.raises(ValueError, match="stakes must be finite"):
        bootstrap_pnl(np.array([1.0, -1.0]), seed=1, stakes=np.array([1.0, np.nan]))
