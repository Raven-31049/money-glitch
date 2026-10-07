"""Report assembly: odds sources, both stop modes, bootstrap, calibration.

The report states bet count, period, model and code version, and config hash,
and never issues an ROI conclusion from fewer than ~100 bets without saying the
sample is too small (AGENTS.md).

Two layers, deliberately separated:

* *summary* helpers (:func:`summarise_simulation`, and the dataclasses) turn
  engine outputs into plain typed numbers — no formatting, no markdown, so a
  test can assert the numbers directly;
* :func:`render_markdown` turns those into the ``reports/<variant>_<ts>.md``
  document: both stop modes side by side (invariant 2), the bootstrap run on
  the full sequence, and the calibration table over all predictions.

The bootstrap summary carries the :class:`~bet_engine.eval.bootstrap.BootstrapResult`
only when the run had bets to resample — an empty run reports *why* there is
no interval instead of inventing one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .calibration import ReliabilityBin

if TYPE_CHECKING:  # bootstrap imports this module's constant; keep it lazy
    from ..backtest.bankroll import Simulation
    from ..config import VariantConfig
    from .bootstrap import BootstrapResult

#: Bet counts below this are reported with an explicit small-sample warning.
#: A ROI from a handful of bets says nothing about an edge, so the number is
#: printed *and* labelled (AGENTS.md).
SMALL_SAMPLE_BETS = 100

#: Shown for rows whose source was never recorded — a pre-migration row, or a
#: writer that passed an empty string. Unattributed bets are reported, not
#: hidden: a bet that vanished from the breakdown would quietly change the
#: denominator under every ROI in it.
UNKNOWN_SOURCE = "(unknown)"


@dataclass(frozen=True)
class SourceStats:
    """Bet outcomes for one odds source within one run."""

    odds_source: str
    bets: int
    staked: float
    pnl: float
    #: None when nothing was staked: no exposure means no return *on*
    #: exposure, and printing 0.0% there would read as a break-even result.
    roi: float | None


@dataclass(frozen=True)
class OddsSourceSummary:
    """Which book priced a run's bets, and what happened to them."""

    run_id: str
    sources: tuple[SourceStats, ...]
    matches_skipped: int

    @property
    def total_bets(self) -> int:
        return sum(source.bets for source in self.sources)


@dataclass(frozen=True)
class StopModeSummary:
    """One simulated stop mode's whole curve, reduced to report numbers."""

    stop_on_breach: bool
    label: str
    bets: int
    staked: float
    pnl: float
    #: None when nothing was staked — no exposure, no return on exposure.
    roi: float | None
    starting: float
    final_bankroll: float
    peak: float
    max_drawdown: float
    breach_seq_no: int | None
    breach_date: str | None
    stopped: bool


@dataclass(frozen=True)
class BootstrapSummary:
    """The significance section: result, or the honest reason there is none."""

    result: BootstrapResult | None
    n_resamples: int
    seed: int
    note: str | None


@dataclass(frozen=True)
class CalibrationSummary:
    """Scores over ALL predictions, plus the reliability bins."""

    brier: float | None
    log_loss: float | None
    n_predictions: int
    excluded: int
    bins: tuple[ReliabilityBin, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReportData:
    """Everything :func:`render_markdown` needs, assembled by the runner.

    The config travels as the typed object itself rather than as copied
    fields: the report must print *the values actually used* (IMPLEMENTATION_NOTES.md §8),
    and one reference cannot drift from what the run executed.
    """

    config: VariantConfig
    config_path: str
    config_hash: str
    code_version: str
    created_at: str
    run_id: str
    stopped_run_id: str
    report_path: str
    matches_loaded: int
    matches_priced: int
    period_start: str | None
    period_end: str | None
    full: StopModeSummary
    stopped: StopModeSummary
    bootstrap: BootstrapSummary
    calibration: CalibrationSummary
    odds_sources: OddsSourceSummary
    notes: tuple[str, ...]


def summarise_simulation(
    sim: Simulation, *, stop_on_breach: bool
) -> StopModeSummary:
    """Reduce one :class:`Simulation` to the numbers a report row shows.

    The mode flag is an argument, not inferred from ``stopped``: a run that
    played on without ever breaching is still a ``stop_on_breach=True``
    *configuration*, and the report labels what was configured next to what
    actually happened. ROI is P&L over *staked* (money ROI), not over bets —
    a run that sized 0.5% and one that sized 5% have the same win rate and
    completely different meanings for a bankroll, and only the staked
    denominator captures that. Nothing staked gives None, never 0.0%.
    """
    label = "stop on breach" if stop_on_breach else "play on"
    bets = sim.bets
    staked = float(bets["stake"].sum()) if len(bets) else 0.0
    pnl = float(bets["pnl"].sum()) if len(bets) else 0.0
    breach_date = None
    if sim.breach_date is not None:
        breach_date = str(sim.breach_date.date())
    return StopModeSummary(
        stop_on_breach=stop_on_breach,
        label=label,
        bets=int(len(bets)),
        staked=staked,
        pnl=pnl,
        roi=(pnl / staked) if staked > 0.0 else None,
        starting=float(sim.starting),
        final_bankroll=float(sim.final_bankroll),
        peak=float(sim.peak),
        max_drawdown=float(sim.worst_drawdown),
        breach_seq_no=sim.breach_seq_no,
        breach_date=breach_date,
        stopped=bool(sim.stopped),
    )


def odds_source_summary(
    conn: sqlite3.Connection, run_id: str
) -> OddsSourceSummary:
    """Bets, staking and ROI per odds source for ``run_id``, plus skips.

    Derived from the ``bets`` and ``skipped`` tables at read time rather than
    maintained as a counter on ``runs``: a number that lives in two places can
    disagree, and this way the breakdown *is* the data. Empty string and NULL
    sources are folded together, because both mean "we do not know which book
    this came from" and reporting them separately would imply a distinction
    nobody recorded.
    """
    rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(odds_source, ''), ?) AS odds_source,
               COUNT(*)                      AS bets,
               COALESCE(SUM(stake), 0.0)     AS staked,
               COALESCE(SUM(pnl), 0.0)       AS pnl
          FROM bets
         WHERE run_id = ?
         GROUP BY COALESCE(NULLIF(odds_source, ''), ?)
         ORDER BY odds_source
        """,
        (UNKNOWN_SOURCE, run_id, UNKNOWN_SOURCE),
    ).fetchall()
    skipped = conn.execute(
        "SELECT COUNT(*) FROM skipped WHERE run_id = ?", (run_id,)
    ).fetchone()[0]

    sources = tuple(
        SourceStats(
            odds_source=row["odds_source"],
            bets=row["bets"],
            staked=row["staked"],
            pnl=row["pnl"],
            roi=(row["pnl"] / row["staked"]) if row["staked"] > 0 else None,
        )
        for row in rows
    )
    return OddsSourceSummary(
        run_id=run_id, sources=sources, matches_skipped=skipped
    )


def format_odds_source_summary(summary: OddsSourceSummary) -> str:
    """The summary as a plain-text table with skips and a size warning.

    Bet count and ROI share a line on purpose: a ROI without the sample it
    came from is the easiest way to over-read a backtest, and the warning is
    printed whenever the run placed fewer than ``SMALL_SAMPLE_BETS`` bets.
    """
    header = f"{'odds source':<18}{'bets':>6}{'staked':>12}{'pnl':>10}{'roi':>10}"
    lines = [header, "-" * len(header)]
    if summary.sources:
        for stat in summary.sources:
            roi = "n/a" if stat.roi is None else f"{stat.roi * 100:+.1f}%"
            lines.append(
                f"{stat.odds_source:<18}"
                f"{stat.bets:>6}"
                f"{stat.staked:>12.2f}"
                f"{stat.pnl:>10.2f}"
                f"{roi:>10}"
            )
    else:
        lines.append("(no bets placed)")
    lines.append(f"matches skipped by odds policy: {summary.matches_skipped}")
    total = summary.total_bets
    if 0 < total < SMALL_SAMPLE_BETS:
        lines.append(
            f"WARNING: {total} bets is too small a sample for an ROI "
            f"conclusion (threshold {SMALL_SAMPLE_BETS})."
        )
    return "\n".join(lines)


# --- markdown rendering ------------------------------------------------------


def render_markdown(report: ReportData) -> str:
    """The full report document for ``report`` as markdown text."""
    config = report.config
    lines: list[str] = [
        f"# {config.name} — {config.market} / {config.league} backtest report",
        "",
        "## Run",
        "",
        f"- Config: `{report.config_path}`",
        f"- Config hash: sha256:{report.config_hash}",
        f"- Code version: bet_engine {report.code_version}",
        f"- Model: `{config.model.type}`"
        + (f" {json.dumps(config.model.params)}" if config.model.params else ""),
        f"- Odds policy: `{config.odds_policy}`",
        f"- Seasons: {', '.join(config.seasons)}",
        f"- Period: {_period(report)}",
        f"- Matches: {report.matches_loaded} loaded, "
        f"{report.matches_priced} priced, "
        f"{report.odds_sources.matches_skipped} skipped by the odds policy",
        f"- Matches scored: {report.calibration.n_predictions}",
        f"- Run id (play-on ledger): `{report.run_id}`",
        f"- Run id (stopped ledger): `{report.stopped_run_id}`",
        f"- Created: {report.created_at}",
        "",
        "## Parameters as configured",
        "",
        f"- Staking: ev_threshold={config.staking.ev_threshold}, "
        f"kelly_fraction={config.staking.kelly_fraction}, "
        f"max_stake_frac={config.staking.max_stake_frac}",
        f"- Bankroll: starting={_money(config.bankroll.starting)}, "
        f"max_drawdown={config.bankroll.max_drawdown}",
        f"- Walk-forward: min_train_days={config.backtest.min_train_days}, "
        f"refit_every_days={config.backtest.refit_every_days} (calendar days; "
        f"training strictly earlier than each prediction day)",
        f"- Bootstrap: n={report.bootstrap.n_resamples}, "
        f"seed={report.bootstrap.seed}",
        "",
        "## Bankroll — both stop modes side by side",
        "",
        *(_stop_mode_table(report)),
        "",
        _headline(report),
        "",
        "## Bootstrap significance",
        "",
        *(_bootstrap_lines(report)),
        "",
        "## Calibration (all predictions)",
        "",
        *(_calibration_lines(report)),
        "",
        "## Odds sources",
        "",
        *(_odds_source_lines(report)),
        "",
        "## Notes",
        "",
        *(_notes(report)),
        "",
    ]
    return "\n".join(lines)


def _period(report: ReportData) -> str:
    if report.period_start is None or report.period_end is None:
        return "n/a (no predictions)"
    return f"{report.period_start} .. {report.period_end}"


def _stop_mode_table(report: ReportData) -> list[str]:
    """The side-by-side block: every row is a claim both modes make (or not).

    Invariant 2: 'we stopped' and 'we kept going' are different claims, so
    they share rows rather than living in two paragraphs a reader would have
    to hold in their head.
    """
    full, stopped = report.full, report.stopped
    rows = [
        ("Metric", f"`stop_on_breach=False` ({full.label})",
         f"`stop_on_breach=True` ({stopped.label})"),
        ("Bets placed", str(full.bets), str(stopped.bets)),
        ("Staked", _money(full.staked), _money(stopped.staked)),
        ("P&L", _money(full.pnl), _money(stopped.pnl)),
        ("ROI (P&L / staked)", _pct(full.roi), _pct(stopped.roi)),
        ("Starting bankroll", _money(full.starting), _money(stopped.starting)),
        ("Final bankroll", _money(full.final_bankroll),
         _money(stopped.final_bankroll)),
        ("Peak bankroll", _money(full.peak), _money(stopped.peak)),
        ("Max drawdown (from peak)", _pct_magnitude(full.max_drawdown),
         _pct_magnitude(stopped.max_drawdown)),
        ("First breach", _breach(full), _breach(stopped)),
        ("Halted early", "no" if not full.stopped else "yes",
         "no" if not stopped.stopped else "yes"),
    ]
    header = f"| {rows[0][0]} | {rows[0][1]} | {rows[0][2]} |"
    divider = "| --- | ---: | ---: |"
    body = [f"| {name} | {play} | {stop} |" for name, play, stop in rows[1:]]
    return [header, divider, *body]


def _breach(summary: StopModeSummary) -> str:
    if summary.breach_seq_no is None:
        return "none"
    text = f"bet {summary.breach_seq_no} ({summary.breach_date})"
    return text + (", halted after it" if summary.stopped else ", played on")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def _pct_magnitude(value: float | None) -> str:
    """A drop is a size, not a signed return: 13.7% down, never "+13.7%"."""
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _headline(report: ReportData) -> str:
    """Bet count, ROI and max drawdown in one sentence each mode's reader
    can quote without parsing a table — with the sample-size rule applied."""
    parts = []
    for summary in (report.full, report.stopped):
        roi = _pct(summary.roi) if summary.roi is not None else "n/a"
        parts.append(
            f"{summary.label}: {summary.bets} bets, ROI {roi}, "
            f"max drawdown {_pct_magnitude(summary.max_drawdown)}"
        )
    total = report.full.bets
    warning = ""
    if 0 < total < SMALL_SAMPLE_BETS:
        warning = (
            f"\n\n**WARNING: {total} bets is too small a sample for an ROI "
            f"conclusion** (under the {SMALL_SAMPLE_BETS}-bet threshold) — "
            f"report it as a small sample, never as an edge."
        )
    return "; ".join(parts) + warning


def _bootstrap_lines(report: ReportData) -> list[str]:
    bootstrap = report.bootstrap
    lines = [
        "Resampling unit: **single bet** (stated per docs/IMPLEMENTATION_NOTES.md §5); "
        "draws are with replacement over the FULL, untruncated bet sequence "
        "(invariant 3).",
        "",
        f"- Resamples: {bootstrap.n_resamples}, seed: {bootstrap.seed}",
    ]
    if bootstrap.result is None:
        lines.append(f"- Not run: {bootstrap.note}")
        return lines
    result = bootstrap.result
    lines += [
        f"- Bets resampled (n_bets): {result.n_bets}",
        f"- Profitable resamples: {result.fraction_profitable * 100:.1f}%",
        f"- Mean ROI: {_pct(result.mean_roi)}",
        f"- 90% CI (5th–95th percentile): [{_pct(result.ci_5)}, "
        f"{_pct(result.ci_95)}]",
    ]
    return lines


def _calibration_lines(report: ReportData) -> list[str]:
    calibration = report.calibration
    if calibration.brier is None:
        reason = calibration.notes[0] if calibration.notes else "no predictions."
        return [f"Not scored: {reason}"]
    lines = [
        f"- Matches scored: {calibration.n_predictions}"
        + (
            f" ({calibration.excluded} excluded: no result to settle)"
            if calibration.excluded
            else ""
        ),
        f"- Brier score: {calibration.brier:.4f} (0 = perfect; "
        f"mean squared error over the full outcome vector)",
    ]
    if calibration.log_loss is not None:
        lines.append(f"- Log loss: {calibration.log_loss:.4f}")
    elif not calibration.notes:
        lines.append("- Log loss: refused")
    lines += [f"- Note: {note}" for note in calibration.notes]
    if calibration.bins:
        lines += [
            "",
            "Reliability table (every predicted probability, 0/1 indicator):",
            "",
            "| Bin | Count | Mean predicted | Actual frequency |",
            "| --- | ---: | ---: | ---: |",
        ]
        for b in calibration.bins:
            mean = "—" if b.count == 0 else f"{b.predicted_mean:.3f}"
            freq = "—" if b.count == 0 else f"{b.actual_frequency:.3f}"
            lines.append(
                f"| [{b.low:.2f}, {b.high:.2f}) | {b.count} | {mean} | {freq} |"
            )
    return lines


def _odds_source_lines(report: ReportData) -> list[str]:
    summary = report.odds_sources
    lines = [
        "| Odds source | Bets | Staked | P&L | ROI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if summary.sources:
        for stat in summary.sources:
            lines.append(
                f"| {stat.odds_source} | {stat.bets} | {_money(stat.staked)} | "
                f"{_money(stat.pnl)} | {_pct(stat.roi)} |"
            )
    else:
        lines.append("| (no bets placed) | 0 | 0.00 | 0.00 | n/a |")
    lines.append(
        f"\nMatches skipped by the odds policy: {summary.matches_skipped}"
    )
    return lines


def _notes(report: ReportData) -> list[str]:
    notes = list(report.notes)
    if report.config.model.type == "pure_market":
        notes.append(
            f"Invariant 4 (pure-market control): placed "
            f"{report.full.bets} bets — expected 0; any bet would be an "
            f"EV/Kelly bug, not an edge."
        )
    total = report.full.bets
    if 0 < total < SMALL_SAMPLE_BETS:
        notes.append(
            f"Sample too small for an ROI conclusion: {total} bets is under "
            f"the {SMALL_SAMPLE_BETS}-bet threshold. The numbers above are "
            f"reported, not interpreted."
        )
    if total == 0:
        notes.append("No bets were placed, so there is no ROI to report.")
    if not notes:
        notes.append("None.")
    return [f"- {note}" for note in notes]


def config_hash(config_json: str) -> str:
    """Short, stable hash of the exact JSON a run stored as its config.

    Hashing the stored serialisation (sorted keys, one canonical spacing)
    rather than a live object means two runs of the same YAML — on any
    machine, at any time — carry the same hash, and a reader can recompute
    it from the database row alone.
    """
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    return digest[:12]


__all__ = [
    "SMALL_SAMPLE_BETS",
    "UNKNOWN_SOURCE",
    "BootstrapSummary",
    "CalibrationSummary",
    "OddsSourceSummary",
    "ReportData",
    "SourceStats",
    "StopModeSummary",
    "config_hash",
    "format_odds_source_summary",
    "odds_source_summary",
    "render_markdown",
    "summarise_simulation",
]
