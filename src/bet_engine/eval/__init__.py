"""Evaluation: significance, calibration, and report assembly."""

from .bootstrap import BootstrapResult, PnlSource, bootstrap_pnl
from .calibration import (
    Actual,
    Predicted,
    ReliabilityBin,
    brier_score,
    log_loss,
    reliability_table,
)
from .report import (
    SMALL_SAMPLE_BETS,
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

__all__ = [
    "SMALL_SAMPLE_BETS",
    "Actual",
    "BootstrapResult",
    "BootstrapSummary",
    "CalibrationSummary",
    "OddsSourceSummary",
    "Predicted",
    "PnlSource",
    "ReliabilityBin",
    "ReportData",
    "SourceStats",
    "StopModeSummary",
    "bootstrap_pnl",
    "brier_score",
    "config_hash",
    "format_odds_source_summary",
    "log_loss",
    "odds_source_summary",
    "reliability_table",
    "render_markdown",
    "summarise_simulation",
]
