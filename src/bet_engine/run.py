"""The one command: config -> data -> walk-forward -> bets -> sim -> eval -> db -> report.

``uv run python -m bet_engine.run configs/<variant>.yaml`` executes a variant
end to end and writes two things: rows in SQLite (provenance a reader can
re-query) and ``reports/<variant>_<stamp>.md`` (a document a human reads).

WHY the stages are ordered the way they are:

* **The odds policy splits the data FIRST.** A row the policy cannot price
  never reaches the model, so it is recorded as *skipped* (with the policy
  that skipped it) rather than predicted against a missing price. The report
  accounts for every match: priced + skipped = loaded.
* **EV, Kelly and settlement all use the raw odds actually paid.** A bet is
  worth placing only if ``model_prob * raw_odds - 1`` clears the threshold
  (IMPLEMENTATION_NOTES.md §5): the money is won or lost at the price the book offered, so the
  edge that decides a bet must be measured at that same price. De-vigged odds
  are used only to produce ``market_prob`` — the control's prediction and the
  report's market column — never for EV or bet selection.
* **``market_prob`` and the control's prediction are the SAME computation.**
  Both call :func:`bet_engine.models.pure_market.devig_long` on the same price
  columns, so they are bit-identical. Measured against the raw odds paid, the
  control's edge is ``market_prob * raw_odds - 1 < 0`` wherever the book prices
  above fair, which ``select_bets``' strict ``>`` refuses. ``pure_market``
  selecting ANY bet therefore means an EV/Kelly bug — the run raises rather
  than reporting it as an edge (invariant 4).
* **Both stop modes simulate the SAME selection frame.** Stop-on-breach
  changes how many bets get *placed*, never which bets are considered or what
  they size at, so the two curves differ only in the breach (invariant 2).
* **The bootstrap reads the play-on ledger** — the full, untruncated sequence
  (invariant 3); the stopped ledger is by construction truncated, and
  ``bootstrap_pnl`` refuses it. Calibration scores every *settled* prediction;
  unsettled matches are excluded and counted, never scored as losses.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

from .backtest import Simulation, kelly_stake, select_bets, simulate, walk_forward
from .config import VariantConfig, load_variant
from .data.ingest import load_league
from .db import connect, write_bets, write_predictions, write_run, write_skipped
from .eval import (
    BootstrapSummary,
    CalibrationSummary,
    ReportData,
    bootstrap_pnl,
    brier_score,
    config_hash,
    log_loss,
    odds_source_summary,
    reliability_table,
    render_markdown,
    summarise_simulation,
)
from .markets import Market, attach_odds, get_market
from .markets.base import valid_price
from .models import create_model
from .models.pure_market import devig_long

#: Repo-level defaults resolved from the source tree, so the command works the
#: same from any working directory (mirrors data.ingest's default data dir).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _REPO_ROOT / "data" / "processed" / "bet_engine.sqlite3"
_DEFAULT_REPORTS = _REPO_ROOT / "reports"


@dataclass(frozen=True)
class RunResult:
    """What one execution produced, for callers and tests to inspect."""

    run_id: str
    stopped_run_id: str
    report_path: Path
    report: ReportData
    sim_full: Simulation
    sim_stopped: Simulation


def run(
    config: VariantConfig,
    *,
    config_path: str = "<in-memory>",
    data_dir: Path | None = None,
    db_path: Path | None = None,
    reports_dir: Path | None = None,
) -> RunResult:
    """Execute one variant end to end and return what it produced."""
    market = get_market(config.market)
    stamp_now = datetime.now(timezone.utc)
    stamp = stamp_now.strftime("%Y%m%dT%H%M%S%f")

    raw = load_league(config.league, config.seasons, data_dir)
    priced = attach_odds(raw, market, policy=config.odds_policy)
    mask = _priceable_mask(priced, market.outcomes)
    matches_loaded = int(priced["match_id"].nunique())
    matches_priced = int(priced.loc[mask, "match_id"].nunique())
    skipped = priced.loc[~mask]
    winners = _winners(priced, market)

    predictions = walk_forward(
        priced.loc[mask].reset_index(drop=True),
        lambda: create_model(config.model.type, config.model.params, market.outcomes),
        config.backtest.min_train_days,
        config.backtest.refit_every_days,
    )
    full = _assemble(
        predictions,
        priced,
        market_long(priced, market),
        winners,
        market.name,
        market.outcomes,
    )

    # Invariant 4, enforced where the bug would actually manifest: before any
    # money logic runs, not as a note in a report nobody reads. Selection runs
    # on the raw odds actually paid (IMPLEMENTATION_NOTES.md §5); market_prob is carried only
    # for the control's identity check and for reporting.
    candidates = full.loc[full["settled"]].copy()
    candidates["raw_odds"] = candidates["odds"]
    selected = select_bets(candidates, config.staking.ev_threshold)
    if config.model.type == "pure_market" and len(selected):
        raise RuntimeError(
            f"invariant 4 violated: pure-market control {config.name!r} "
            f"selected {len(selected)} bet(s) at ev_threshold="
            f"{config.staking.ev_threshold}; the de-vigged market cannot "
            f"beat its own fair price — this is an EV/Kelly bug, not an edge"
        )
    bets, zero_staked = _stake(selected, config)
    bets = bets.sort_values(["date", "match_id"]).reset_index(drop=True)

    sim_full = simulate(
        bets, config.bankroll.starting, config.bankroll.max_drawdown, False
    )
    sim_stopped = simulate(
        bets, config.bankroll.starting, config.bankroll.max_drawdown, True
    )
    calibration = _calibrate(full, winners, market.outcomes)
    bootstrap = _bootstrap(sim_full, config)

    run_id = f"{config.name}_{stamp}"
    stopped_run_id = f"{run_id}_stop"
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    db_file = db_path or _DEFAULT_DB
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_file)
    try:
        _write_db(
            conn,
            run_id=run_id,
            stopped_run_id=stopped_run_id,
            config=config,
            config_json=config_json,
            predictions=full,
            bets_full=sim_full.bets,
            bets_stopped=sim_stopped.bets,
            skipped=skipped,
            market=market.name,
            skipped_reason=(
                f"no price under odds policy {config.odds_policy!r} "
                f"(incomplete or missing Pinnacle vector)"
            ),
        )
        sources = odds_source_summary(conn, run_id)
    finally:
        conn.close()

    report_path = (reports_dir or _DEFAULT_REPORTS) / f"{config.name}_{stamp}.md"
    report = ReportData(
        config=config,
        config_path=config_path,
        config_hash=config_hash(config_json),
        code_version=_code_version(),
        created_at=stamp_now.replace(microsecond=0).isoformat(),
        run_id=run_id,
        stopped_run_id=stopped_run_id,
        report_path=str(report_path),
        matches_loaded=matches_loaded,
        matches_priced=matches_priced,
        period_start=_day(full, "min"),
        period_end=_day(full, "max"),
        full=summarise_simulation(sim_full, stop_on_breach=False),
        stopped=summarise_simulation(sim_stopped, stop_on_breach=True),
        bootstrap=bootstrap,
        calibration=calibration,
        odds_sources=sources,
        notes=_run_notes(
            skipped_count=int(len(skipped)),
            config=config,
            unsettled=_unsettled_count(full),
            zero_staked=zero_staked,
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(report), encoding="utf-8")
    return RunResult(
        run_id=run_id,
        stopped_run_id=stopped_run_id,
        report_path=report_path,
        report=report,
        sim_full=sim_full,
        sim_stopped=sim_stopped,
    )


def market_long(priced: pd.DataFrame, market: Market) -> pd.DataFrame:
    """The run's ``market_prob`` column: de-vigged prices, long form.

    Delegated to ``models.pure_market.devig_long`` — the exact function the
    control's ``predict`` calls — so the two are bit-identical by
    construction (see the module docstring; invariant 4 depends on it).
    """
    return devig_long(priced, market.outcomes, value_name="market_prob")


def _priceable_mask(priced: pd.DataFrame, outcomes: Sequence[str]) -> pd.Series:
    """Rows whose policy-resolved source delivered a complete valid vector."""
    columns = [f"odds_{outcome}" for outcome in outcomes]
    valid = priced[columns].map(valid_price).all(axis=1)
    return priced["odds_source"].notna() & valid


def _winners(priced: pd.DataFrame, market: Market) -> dict[str, str | None]:
    """Match id -> winning outcome (None when the match has no result yet).

    Settlement goes through the market object, never a hard-coded column
    read: which code maps to which label is market knowledge (invariant 8).
    """
    unique = priced.drop_duplicates("match_id")
    return {
        str(record["match_id"]): market.settle(record)
        for record in unique.to_dict("records")
    }


def _assemble(
    predictions: pd.DataFrame,
    priced: pd.DataFrame,
    market_probs: pd.DataFrame,
    winners: dict[str, str | None],
    market_name: str,
    outcomes: Sequence[str],
) -> pd.DataFrame:
    """Walk-forward output + market probabilities + raw prices + settlement.

    Every join is on identity columns with ``validate=``: a merge that would
    fan out (two odds rows for one outcome) raises here instead of silently
    duplicating bets downstream. ``won`` is only ever True where a result
    exists; ``settled`` carries existence itself so an unsettled row is
    excluded from betting/scoring rather than scored as a loss.
    """
    frame = predictions.merge(
        market_probs, on=["match_id", "outcome"], how="left", validate="many_to_one"
    )
    odds_long = priced.melt(
        id_vars=["match_id"],
        value_vars=[f"odds_{outcome}" for outcome in outcomes],
        var_name="column",
        value_name="odds",
    )
    odds_long["outcome"] = odds_long["column"].str.removeprefix("odds_")
    frame = frame.merge(
        odds_long[["match_id", "outcome", "odds"]],
        on=["match_id", "outcome"],
        how="left",
        validate="many_to_one",
    )
    sources = priced[["match_id", "odds_source"]].drop_duplicates("match_id")
    frame = frame.merge(sources, on="match_id", how="left", validate="many_to_one")
    frame["market"] = market_name
    settled_label = frame["match_id"].map(winners)
    frame["settled"] = settled_label.notna()
    frame["won"] = (frame["outcome"] == settled_label) & frame["settled"]
    return frame


def _stake(
    selected: pd.DataFrame, config: VariantConfig
) -> tuple[pd.DataFrame, int]:
    """Fractional-Kelly stakes of the day's OPENING balance, as fractions.

    The stake is stored as a fraction of the configured starting bankroll so
    the same frame prices identically under both stop modes — ``simulate``
    applies each day's actual opening balance (bankroll.py). Kelly runs on
    ``raw_odds``: the money is won or lost at the book's price, not at the
    de-vigged fair one.
    """
    starting = config.bankroll.starting
    stakes = [
        kelly_stake(
            float(prob),
            float(raw_odds),
            config.staking.kelly_fraction,
            starting,
            config.staking.max_stake_frac,
        )
        for prob, raw_odds in zip(selected["model_prob"], selected["raw_odds"])
    ]
    frame = selected.copy()
    frame["stake_frac"] = pd.Series(
        [stake / starting for stake in stakes], index=frame.index, dtype="float64"
    )
    zero_staked = int((frame["stake_frac"] <= 0.0).sum())
    frame = frame.loc[frame["stake_frac"] > 0.0]
    return frame, zero_staked


def _bootstrap(sim_full: Simulation, config: VariantConfig) -> BootstrapSummary:
    """Significance on the FULL play-on sequence, or why there is none."""
    if sim_full.bets.empty:
        return BootstrapSummary(
            result=None,
            n_resamples=config.bootstrap.n,
            seed=config.bootstrap.seed,
            note="no bets were placed, so there is no sequence to resample",
        )
    result = bootstrap_pnl(
        sim_full.bets, config.bootstrap.n, seed=config.bootstrap.seed
    )
    return BootstrapSummary(
        result=result,
        n_resamples=config.bootstrap.n,
        seed=config.bootstrap.seed,
        note=None,
    )


def _calibrate(
    full: pd.DataFrame,
    winners: dict[str, str | None],
    outcomes: Sequence[str],
) -> CalibrationSummary:
    """Brier, log loss and reliability bins over ALL predictions.

    One row per match (wide, reindexed to the market's outcome order), scored
    only where a result exists; unsettled matches are *excluded and counted*,
    never scored as losses. A log-loss refusal (a model that gave the settled
    outcome probability 0) is reported as a note, not raised: the rest of the
    report — Brier, reliability, bets — still stands.
    """
    if full.empty:
        return CalibrationSummary(
            brier=None,
            log_loss=None,
            n_predictions=0,
            excluded=0,
            bins=(),
            notes=("no predictions to score (no fold met the training requirement)",),
        )
    wide = full.pivot(
        index="match_id", columns="outcome", values="model_prob"
    ).reindex(columns=list(outcomes))
    settled_ids = [match_id for match_id in wide.index if winners.get(match_id)]
    excluded = len(wide) - len(settled_ids)
    if not settled_ids:
        return CalibrationSummary(
            brier=None,
            log_loss=None,
            n_predictions=0,
            excluded=excluded,
            bins=(),
            notes=("no settled matches among the predictions",),
        )
    scored = wide.loc[settled_ids]
    actual = pd.Series(
        [winners[match_id] for match_id in settled_ids],
        index=scored.index,
        dtype="object",
    )
    notes: list[str] = []
    try:
        loss: float | None = log_loss(scored, actual)
    except ValueError as exc:
        loss = None
        notes.append(f"log loss refused: {exc}")
    return CalibrationSummary(
        brier=brier_score(scored, actual),
        log_loss=loss,
        n_predictions=len(settled_ids),
        excluded=excluded,
        bins=reliability_table(scored, actual),
        notes=tuple(notes),
    )


def _write_db(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stopped_run_id: str,
    config: VariantConfig,
    config_json: str,
    predictions: pd.DataFrame,
    bets_full: pd.DataFrame,
    bets_stopped: pd.DataFrame,
    skipped: pd.DataFrame,
    market: str,
    skipped_reason: str,
) -> None:
    """Persist both runs and their rows.

    Two rows in ``runs`` because the ledgers are genuinely two claims (one
    stopped, one did not) and each must carry its own ``stop_on_breach`` flag;
    the stopped run's config records ``derived_from`` so a reader can pair
    them without guessing from timestamps.
    """
    write_run(
        conn,
        run_id,
        variant_name=config.name,
        market=market,
        config=config_json,
        stop_on_breach=False,
    )
    write_run(
        conn,
        stopped_run_id,
        variant_name=config.name,
        market=market,
        config=json.dumps(
            {**json.loads(config_json), "derived_from": run_id}, sort_keys=True
        ),
        stop_on_breach=True,
    )
    if len(predictions):
        rows = predictions[
            [
                "match_id",
                "date",
                "league",
                "outcome",
                "model_prob",
                "market_prob",
                "odds",
                "odds_source",
            ]
        ].copy()
        rows["run_id"] = run_id
        rows["market"] = market
        write_predictions(conn, rows.to_dict("records"))
    if len(skipped):
        rows = skipped[["match_id", "date", "league"]].copy()
        rows["market"] = market
        rows["reason"] = skipped_reason
        rows["run_id"] = run_id
        write_skipped(conn, rows.to_dict("records"))
    _write_bets(conn, bets_full, run_id)
    _write_bets(conn, bets_stopped, stopped_run_id)


def _write_bets(conn, bets: pd.DataFrame, run_id: str) -> None:
    if not len(bets):
        return
    rows = bets.copy()
    rows["run_id"] = run_id
    write_bets(conn, rows.to_dict("records"))


def _run_notes(
    *,
    skipped_count: int,
    config: VariantConfig,
    unsettled: int,
    zero_staked: int,
) -> tuple[str, ...]:
    """Run-level facts a reader needs that no table carries on its own."""
    notes: list[str] = []
    if skipped_count:
        notes.append(
            f"{skipped_count} match(es) had no price under odds policy "
            f"{config.odds_policy!r} and were skipped: no prediction, no bet "
            f"(they are rows in the skipped table)."
        )
    if unsettled:
        notes.append(
            f"{unsettled} predicted match(es) had no settlement at run time; "
            f"excluded from betting and scoring."
        )
    if zero_staked:
        notes.append(
            f"{zero_staked} selection(s) were not placed: fractional Kelly "
            f"stakes nothing at the raw price — the fair-odds edge cleared "
            f"the threshold but the book's vig did not."
        )
    return tuple(notes)


def _day(frame: pd.DataFrame, which: str) -> str | None:
    if frame.empty:
        return None
    value = frame["date"].min() if which == "min" else frame["date"].max()
    return str(pd.Timestamp(value).date())


def _unsettled_count(full: pd.DataFrame) -> int:
    """Predicted matches with no result yet (match-level, not row-level)."""
    if full.empty:
        return 0
    return int(full.loc[~full["settled"], "match_id"].nunique())


def _code_version() -> str:
    try:
        return importlib_metadata.version("bet-engine")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: load one variant YAML and run it end to end."""
    parser = argparse.ArgumentParser(
        prog="python -m bet_engine.run",
        description="Run one backtest variant end to end.",
        epilog="example: uv run python -m bet_engine.run configs/epl_mw_naive.yaml",
    )
    parser.add_argument("config", type=Path, help="path to a variant YAML file")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database path (default: data/processed/bet_engine.sqlite3)",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=None,
        help="directory for markdown reports (default: reports/)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="raw CSV cache directory (default: data/raw/)",
    )
    args = parser.parse_args(argv)

    config = load_variant(args.config)
    result = run(
        config,
        config_path=str(args.config),
        data_dir=args.data_dir,
        db_path=args.db,
        reports_dir=args.reports,
    )
    print(f"variant: {config.name} ({config.model.type}, {config.market})")
    print(f"run id:  {result.run_id}")
    print(f"stopped: {result.stopped_run_id}")
    print(
        f"bets:    {result.report.full.bets} (play on) / "
        f"{result.report.stopped.bets} (stop on breach)"
    )
    print(f"report:  {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
