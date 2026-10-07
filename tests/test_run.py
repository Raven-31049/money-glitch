"""Tests for run.py: the one-command pipeline, end to end on fixture data.

The contract these prove, against real files and a real database:

* invariant 4: the pure-market control places 0 bets in BOTH stop modes, and
  the tripwire raises rather than reporting a bet as an edge;
* invariant 2: both stop modes exist, are stored under their own run ids, and
  the stopped ledger is a prefix of the full one;
* the odds policy's skips are rows in the skipped table, never predictions;
* the report document renders for every outcome — bets, no bets, no folds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bet_engine import db
from bet_engine.config import VariantConfig
from bet_engine.run import main, run

FIXTURES = Path(__file__).parent / "fixtures"


def _config(name: str, model_type: str, **overrides) -> VariantConfig:
    base: dict = {
        "name": name,
        "market": "match_winner",
        "league": "E0",
        "seasons": ["1819", "1920", "2021"],
        "odds_policy": "pinnacle_only",
        "model": {"type": model_type, "params": {}},
        "staking": {
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "max_stake_frac": 0.05,
        },
        "bankroll": {"starting": 1000.0, "max_drawdown": 0.3},
        "backtest": {"min_train_days": 1, "refit_every_days": 1},
        "bootstrap": {"n": 200, "seed": 42},
    }
    base.update(overrides)
    return VariantConfig(**base)


def _run(config: VariantConfig, tmp_path: Path, data_dir: Path = FIXTURES):
    return run(
        config,
        data_dir=data_dir,
        db_path=tmp_path / "engine.sqlite3",
        reports_dir=tmp_path / "reports",
    )


def _report_text(result) -> str:
    return result.report_path.read_text(encoding="utf-8")


# --- invariant 4: the control ------------------------------------------------

def test_control_places_no_bets_in_either_stop_mode(tmp_path):
    result = _run(_config("control_fx", "pure_market", staking={
        "ev_threshold": 0.0, "kelly_fraction": 1.0, "max_stake_frac": 0.05,
    }), tmp_path)

    assert result.sim_full.bets.empty
    assert result.sim_stopped.bets.empty
    assert result.report.full.bets == 0
    assert result.report.stopped.bets == 0
    assert result.report.bootstrap.result is None
    assert "Invariant 4" in _report_text(result)
    # It still scored every settled prediction — a control that refuses bets
    # is not a control that refuses to be measured.
    assert result.report.calibration.n_predictions == 9
    assert result.report.calibration.brier is not None


def test_tripwire_raises_when_the_control_somehow_selects_a_bet(
    tmp_path, monkeypatch
):
    """The invariant is an exception in the pipeline, not a report footnote."""
    import bet_engine.run as run_module

    def _sneak_one_through(candidates, threshold):
        return candidates.head(1)

    monkeypatch.setattr(run_module, "select_bets", _sneak_one_through)

    with pytest.raises(RuntimeError, match="invariant 4 violated"):
        _run(_config("control_fx", "pure_market"), tmp_path)


# --- the baseline actually bets ---------------------------------------------

def test_naive_places_bets_and_persists_both_ledgers(tmp_path):
    result = _run(_config("naive_fx", "naive_frequency"), tmp_path)

    assert len(result.sim_full.bets) >= 1

    conn = db.connect(tmp_path / "engine.sqlite3")
    try:
        full_run = db.read_run(conn, result.run_id)
        stopped_run = db.read_run(conn, result.stopped_run_id)
        assert full_run is not None and full_run.stop_on_breach is False
        assert stopped_run is not None and stopped_run.stop_on_breach is True
        assert stopped_run.config["derived_from"] == result.run_id

        full_ledger = db.read_bets(conn, result.run_id)
        stopped_ledger = db.read_bets(conn, result.stopped_run_id)
        assert len(full_ledger) == len(result.sim_full.bets)
        # No breach on fixtures: identical runs. In general the stopped
        # sequence must be a prefix of the full one (invariant 2).
        assert [b["seq_no"] for b in stopped_ledger] == [
            b["seq_no"] for b in full_ledger[: len(stopped_ledger)]
        ]

        predictions = db.read_predictions(conn, result.run_id)
        assert len(predictions) == 3 * result.report.calibration.n_predictions
        assert {p["market"] for p in predictions} == {"match_winner"}
        assert all(p["run_id"] == result.run_id for p in predictions)
    finally:
        conn.close()


def test_naive_report_states_both_modes_and_the_eval_sections(tmp_path):
    result = _run(_config("naive_fx", "naive_frequency"), tmp_path)
    text = _report_text(result)

    assert "## Bankroll — both stop modes side by side" in text
    assert "stop_on_breach=False" in text
    assert "stop_on_breach=True" in text
    assert "## Bootstrap significance" in text
    assert "## Calibration (all predictions)" in text
    assert "## Odds sources" in text
    assert "## Notes" in text
    assert result.run_id in text
    assert result.stopped_run_id in text
    assert "Config hash: sha256:" in text
    # Small samples are labelled, never sold as an edge (AGENTS.md).
    assert "too small a sample" in text


# --- the odds policy's skips -------------------------------------------------

def test_unpriced_matches_are_skipped_not_predicted(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.read_csv(FIXTURES / "E0_1920.csv")
    unpriced = {column: float("nan") for column in frame.columns}
    unpriced.update(
        {
            "Div": "E0",
            "Date": "20/08/2019",
            "HomeTeam": "Test Home",
            "AwayTeam": "Test Away",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "B365H": 1.9,
            "B365D": 3.5,
            "B365A": 4.0,
        }
    )
    pd.concat([frame, pd.DataFrame([unpriced])], ignore_index=True).to_csv(
        data_dir / "E0_1920.csv", index=False
    )

    result = _run(
        _config("skip_fx", "pure_market", seasons=["1920"]), tmp_path, data_dir
    )

    assert result.report.matches_loaded == 5
    assert result.report.matches_priced == 4
    assert result.report.odds_sources.matches_skipped == 1

    conn = db.connect(tmp_path / "engine.sqlite3")
    try:
        skipped = db.read_skipped(conn, result.run_id)
        predicted_ids = {p["match_id"] for p in db.read_predictions(
            conn, result.run_id
        )}
    finally:
        conn.close()
    assert len(skipped) == 1
    assert "pinnacle_only" in skipped[0]["reason"]
    assert skipped[0]["match_id"] not in predicted_ids
    assert any("no price under odds policy" in note for note in result.report.notes)


# --- degenerate inputs still produce a report -------------------------------

def test_no_folds_still_produces_a_report(tmp_path):
    result = _run(
        _config(
            "nofold_fx",
            "naive_frequency",
            backtest={"min_train_days": 100_000, "refit_every_days": 1},
        ),
        tmp_path,
    )

    assert result.sim_full.bets.empty
    assert result.report.calibration.brier is None
    assert "no predictions" in result.report.calibration.notes[0]
    assert result.report.bootstrap.result is None
    assert result.report.period_start is None
    text = _report_text(result)
    assert "n/a (no predictions)" in text
    assert "Not scored:" in text


# --- the CLI -----------------------------------------------------------------

def test_cli_runs_a_variant_from_a_yaml_file(tmp_path):
    config = _config("cli_fx", "pure_market", staking={
        "ev_threshold": 0.0, "kelly_fraction": 1.0, "max_stake_frac": 0.05,
    })
    config_path = tmp_path / "variant.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
    )
    reports = tmp_path / "cli_reports"

    exit_code = main(
        [
            str(config_path),
            "--db", str(tmp_path / "cli.sqlite3"),
            "--reports", str(reports),
            "--data-dir", str(FIXTURES),
        ]
    )

    assert exit_code == 0
    written = list(reports.glob("cli_fx_*.md"))
    assert len(written) == 1
    assert "cli_fx" in written[0].read_text(encoding="utf-8")
    assert (tmp_path / "cli.sqlite3").exists()
