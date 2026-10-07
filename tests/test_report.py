"""Tests for eval/report.py: odds-source breakdown, stop-mode summaries,
and the markdown document itself.

Everything runs against the real db helpers so the report reads what a run
actually writes: a summary that worked only against hand-built SQL would not
prove the report and the storage agree with each other.
"""

from __future__ import annotations

import re
import sqlite3

import pandas as pd
import pytest

from bet_engine import db
from bet_engine.backtest import Simulation
from bet_engine.config import VariantConfig
from bet_engine.eval.bootstrap import BootstrapResult
from bet_engine.eval.calibration import ReliabilityBin
from bet_engine.eval.report import (
    SMALL_SAMPLE_BETS,
    UNKNOWN_SOURCE,
    BootstrapSummary,
    CalibrationSummary,
    OddsSourceSummary,
    ReportData,
    SourceStats,
    StopModeSummary,
    config_hash,
    format_odds_source_summary,
    odds_source_summary,
    render_markdown,
    summarise_simulation,
)

RUN_KWARGS = {"variant_name": "epl_mw_naive", "market": "match_winner"}


def _run(conn: sqlite3.Connection, run_id: str = "run-1") -> db.Run:
    return db.write_run(conn, run_id, config={}, **RUN_KWARGS)


def _bet(
    run_id: str, seq_no: int, *, odds_source: str, stake: float = 10.0,
    pnl: float = 0.0, **overrides,
) -> dict:
    record = {
        "run_id": run_id,
        "match_id": f"m{seq_no}",
        "date": "2019-08-10",
        "league": "E0",
        "market": "match_winner",
        "outcome": "H",
        "odds": 2.1,
        "odds_source": odds_source,
        "stake": stake,
        "pnl": pnl,
        "bankroll_after": 1000.0,
        "seq_no": seq_no,
    }
    record.update(overrides)
    return record


def _skip(run_id: str, match_id: str, **overrides) -> dict:
    record = {
        "run_id": run_id,
        "match_id": match_id,
        "date": "2019-08-10",
        "league": "E0",
        "market": "match_winner",
        "reason": "no complete price under policy pinnacle_only",
    }
    record.update(overrides)
    return record


def _stats_by_source(summary) -> dict:
    return {source.odds_source: source for source in summary.sources}


# --- summary -----------------------------------------------------------------

def test_summary_splits_bets_by_odds_source():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(
            conn,
            [
                _bet("run-1", 1, odds_source="pinnacle_close", pnl=11.0),
                _bet("run-1", 2, odds_source="pinnacle_close", pnl=-10.0),
                _bet("run-1", 3, odds_source="b365", stake=5.0, pnl=2.5),
            ],
        )

        summary = odds_source_summary(conn, "run-1")
        stats = _stats_by_source(summary)

        assert summary.run_id == "run-1"
        assert summary.total_bets == 3
        assert summary.matches_skipped == 0

        assert stats["pinnacle_close"].bets == 2
        assert stats["pinnacle_close"].staked == 20.0
        assert stats["pinnacle_close"].pnl == 1.0
        assert stats["pinnacle_close"].roi == pytest.approx(0.05)

        assert stats["b365"].bets == 1
        assert stats["b365"].staked == 5.0
        assert stats["b365"].pnl == 2.5
        assert stats["b365"].roi == pytest.approx(0.5)
    finally:
        conn.close()


def test_summary_counts_skipped_matches_for_this_run_only():
    conn = db.connect()
    try:
        _run(conn, "run-1")
        _run(conn, "run-2")
        db.write_skipped(
            conn,
            [
                _skip("run-1", "m1"),
                _skip("run-1", "m2"),
                _skip("run-2", "m3"),
            ],
        )

        summary = odds_source_summary(conn, "run-1")

        assert summary.matches_skipped == 2
        assert summary.sources == ()
        assert summary.total_bets == 0
    finally:
        conn.close()


def test_unknown_source_rows_are_reported_not_dropped():
    """An unattributed bet must show up, or every ROI denominator shrinks."""
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(
            conn,
            [
                _bet("run-1", 1, odds_source="pinnacle", pnl=1.0),
                _bet("run-1", 2, odds_source="", pnl=-1.0),
            ],
        )

        stats = _stats_by_source(odds_source_summary(conn, "run-1"))

        assert set(stats) == {"pinnacle", UNKNOWN_SOURCE}
        assert stats[UNKNOWN_SOURCE].bets == 1
    finally:
        conn.close()


def test_roi_is_none_when_nothing_was_staked():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(conn, [_bet("run-1", 1, odds_source="pinnacle", stake=0.0)])

        source = odds_source_summary(conn, "run-1").sources[0]

        assert source.staked == 0.0
        assert source.roi is None  # no exposure, no return on exposure
    finally:
        conn.close()


def test_summary_of_an_unknown_run_is_empty():
    conn = db.connect()
    try:
        summary = odds_source_summary(conn, "no-such-run")

        assert summary.sources == ()
        assert summary.matches_skipped == 0
    finally:
        conn.close()


# --- formatting --------------------------------------------------------------

def test_formatter_names_every_source_with_its_sample():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(
            conn,
            [
                _bet("run-1", 1, odds_source="pinnacle_close", pnl=11.0),
                _bet("run-1", 2, odds_source="pinnacle_close", pnl=-10.0),
                _bet("run-1", 3, odds_source="b365", stake=5.0, pnl=-5.0),
            ],
        )
        db.write_skipped(conn, [_skip("run-1", "m9")])

        text = format_odds_source_summary(odds_source_summary(conn, "run-1"))

        # bets / staked / pnl / roi per source, on one line each.
        assert re.search(
            r"pinnacle_close\s+2\s+20\.00\s+1\.00\s+\+5\.0%", text
        )
        assert re.search(r"b365\s+1\s+5\.00\s+-5\.00\s+-100\.0%", text)
        assert "matches skipped by odds policy: 1" in text
        # 3 bets is a small sample, and the report says so rather than
        # letting +5.0% read as an edge.
        assert "too small a sample" in text
    finally:
        conn.close()


def test_formatter_marks_an_unattributed_source():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(conn, [_bet("run-1", 1, odds_source="", stake=0.0)])

        text = format_odds_source_summary(odds_source_summary(conn, "run-1"))

        assert UNKNOWN_SOURCE in text
        assert "n/a" in text  # zero staked means no ROI to print
    finally:
        conn.close()


def test_formatter_warns_when_the_sample_is_too_small():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(
            conn, [_bet("run-1", 1, odds_source="pinnacle", pnl=5.0)]
        )

        text = format_odds_source_summary(odds_source_summary(conn, "run-1"))

        assert "1 bets is too small a sample" in text
        assert str(SMALL_SAMPLE_BETS) in text
    finally:
        conn.close()


def test_formatter_stays_silent_at_the_threshold():
    conn = db.connect()
    try:
        _run(conn)
        db.write_bets(
            conn,
            [
                _bet("run-1", seq, odds_source="pinnacle", pnl=1.0)
                for seq in range(1, SMALL_SAMPLE_BETS + 1)
            ],
        )

        text = format_odds_source_summary(odds_source_summary(conn, "run-1"))

        assert str(SMALL_SAMPLE_BETS) in text       # the count is still printed
        assert "too small a sample" not in text     # just not labelled small
    finally:
        conn.close()


def test_formatter_handles_a_run_with_no_bets_and_no_skips():
    conn = db.connect()
    try:
        _run(conn)

        text = format_odds_source_summary(odds_source_summary(conn, "run-1"))

        assert "(no bets placed)" in text
        assert "matches skipped by odds policy: 0" in text
        assert "WARNING" not in text
    finally:
        conn.close()

# --- stop-mode summaries -----------------------------------------------------

def _simulation(**overrides) -> Simulation:
    defaults = dict(
        bets=pd.DataFrame({"stake": [10.0, 20.0], "pnl": [11.0, -20.0]}),
        starting=1000.0,
        final_bankroll=991.0,
        peak=1011.0,
        worst_drawdown=0.02,
        breach_seq_no=2,
        breach_date=pd.Timestamp("2019-08-12"),
        stopped=True,
    )
    defaults.update(overrides)
    return Simulation(**defaults)


def test_summarise_reduces_a_curve_to_report_numbers():
    summary = summarise_simulation(_simulation(), stop_on_breach=True)

    assert summary.stop_on_breach is True
    assert summary.label == "stop on breach"
    assert summary.bets == 2
    assert summary.staked == 30.0
    assert summary.pnl == -9.0
    assert summary.roi == pytest.approx(-9.0 / 30.0)
    assert summary.starting == 1000.0
    assert summary.final_bankroll == 991.0
    assert summary.peak == 1011.0
    assert summary.max_drawdown == pytest.approx(0.02)
    assert summary.breach_seq_no == 2
    assert summary.breach_date == "2019-08-12"
    assert summary.stopped is True


def test_summarise_of_a_run_without_bets_has_no_roi():
    sim = _simulation(
        bets=pd.DataFrame(columns=["stake", "pnl"]),
        breach_seq_no=None,
        breach_date=None,
        stopped=False,
        final_bankroll=1000.0,
        peak=1000.0,
        worst_drawdown=0.0,
    )

    summary = summarise_simulation(sim, stop_on_breach=False)

    assert summary.label == "play on"
    assert summary.bets == 0
    assert summary.staked == 0.0
    assert summary.roi is None  # no exposure, no return on exposure
    assert summary.breach_seq_no is None
    assert summary.breach_date is None
    assert summary.stopped is False


# --- the markdown document ---------------------------------------------------


def _report_config(model_type: str = "naive_frequency") -> VariantConfig:
    return VariantConfig(
        name="epl_mw_naive",
        market="match_winner",
        league="E0",
        seasons=["1819"],
        model={"type": model_type, "params": {}},
        staking={
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "max_stake_frac": 0.05,
        },
        bankroll={"starting": 1000.0, "max_drawdown": 0.3},
        backtest={"min_train_days": 30},
    )


def _stop_summary(*, stop_on_breach: bool, bets: int) -> StopModeSummary:
    label = "stop on breach" if stop_on_breach else "play on"
    return StopModeSummary(
        stop_on_breach=stop_on_breach,
        label=label,
        bets=bets,
        staked=30.0 * bets,
        pnl=-3.0 * bets,
        roi=-0.1,
        starting=1000.0,
        final_bankroll=1000.0 - 3.0 * bets,
        peak=1010.0,
        max_drawdown=0.02,
        breach_seq_no=None,
        breach_date=None,
        stopped=False,
    )


def _report_data(
    model_type: str = "naive_frequency",
    *,
    bets: int = 3,
    bootstrap: BootstrapSummary | None = None,
    calibration: CalibrationSummary | None = None,
    notes: tuple[str, ...] = (),
) -> ReportData:
    config = _report_config(model_type)
    config_json = '{"name": "epl_mw_naive"}'
    return ReportData(
        config=config,
        config_path="configs/epl_mw_naive.yaml",
        config_hash=config_hash(config_json),
        code_version="0.1.0",
        created_at="2026-10-06T12:00:00+00:00",
        run_id="epl_mw_naive_stamped",
        stopped_run_id="epl_mw_naive_stamped_stop",
        report_path="reports/epl_mw_naive_stamped.md",
        matches_loaded=380,
        matches_priced=363,
        period_start="2018-08-10",
        period_end="2019-05-12",
        full=_stop_summary(stop_on_breach=False, bets=bets),
        stopped=_stop_summary(stop_on_breach=True, bets=bets),
        bootstrap=bootstrap
        or BootstrapSummary(
            result=BootstrapResult(
                fraction_profitable=0.61,
                mean_roi=0.05,
                ci_5=-0.10,
                ci_95=0.20,
                n_bets=bets,
            ),
            n_resamples=10000,
            seed=42,
            note=None,
        ),
        calibration=calibration
        or CalibrationSummary(
            brier=0.2043,
            log_loss=0.5104,
            n_predictions=363,
            excluded=2,
            bins=(
                ReliabilityBin(0.0, 0.1, 0.05, 0.04, 40),
                ReliabilityBin(0.1, 0.2, 0.15, 0.16, 55),
            ),
            notes=(),
        ),
        odds_sources=OddsSourceSummary(
            run_id="epl_mw_naive_stamped",
            sources=(
                SourceStats("pinnacle_close", bets, 30.0 * bets, -3.0 * bets, -0.1),
            ),
            matches_skipped=17,
        ),
        notes=notes,
    )


def test_render_contains_every_section_a_reader_is_promised():
    text = render_markdown(_report_data())

    for section in (
        "## Run",
        "## Parameters as configured",
        "## Bankroll — both stop modes side by side",
        "## Bootstrap significance",
        "## Calibration (all predictions)",
        "## Odds sources",
        "## Notes",
    ):
        assert section in text, f"missing section: {section}"
    assert "Config hash: sha256:" in text
    assert "`stop_on_breach=False` (play on)" in text
    assert "`stop_on_breach=True` (stop on breach)" in text
    assert "epl_mw_naive_stamped_stop" in text
    assert "Profitable resamples: 61.0%" in text
    assert "Brier score: 0.2043" in text
    assert "| pinnacle_close |" in text


def test_render_labels_a_small_sample_and_silences_a_large_one():
    small = render_markdown(_report_data(bets=SMALL_SAMPLE_BETS - 1))
    large = render_markdown(_report_data(bets=SMALL_SAMPLE_BETS))

    assert "too small a sample" in small
    assert str(SMALL_SAMPLE_BETS - 1) in small
    assert "too small a sample" not in large


def test_render_states_the_control_invariant_for_pure_market():
    text = render_markdown(_report_data(model_type="pure_market", bets=0))

    assert "Invariant 4" in text
    assert "placed 0 bets" in text
    assert "No bets were placed" in text


def test_render_explains_a_bootstrap_that_did_not_run():
    summary = BootstrapSummary(
        result=None,
        n_resamples=10000,
        seed=42,
        note="no bets were placed, so there is no sequence to resample",
    )

    text = render_markdown(_report_data(bootstrap=summary, bets=0))

    assert "Not run: no bets were placed" in text
    assert "90% CI" not in text


def test_render_reports_a_refused_log_loss_as_a_note():
    calibration = CalibrationSummary(
        brier=0.9,
        log_loss=None,
        n_predictions=10,
        excluded=0,
        bins=(),
        notes=("log loss refused: bad row(s): [0]",),
    )

    text = render_markdown(_report_data(calibration=calibration))

    assert "Brier score: 0.9000" in text
    assert "log loss refused: bad row(s): [0]" in text
    assert "Not scored" not in text  # Brier still stands; only log loss refused


def test_render_reports_run_level_notes():
    text = render_markdown(
        _report_data(notes=("3 match(es) had no price under odds policy.",))
    )

    assert "- 3 match(es) had no price under odds policy." in text


def test_config_hash_is_short_and_deterministic():
    first = config_hash('{"a": 1, "b": 2}')
    second = config_hash('{"a": 1, "b": 2}')
    other = config_hash('{"a": 1, "b": 3}')

    assert len(first) == 12
    assert first == second
    assert first != other
