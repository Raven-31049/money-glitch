"""Evaluation: significance, calibration, and report assembly."""

from .bootstrap import BootstrapResult, PnlSource, bootstrap_pnl
from .calibration import (
    Actual,
    LogisticCalibration,
    Predicted,
    ReliabilityBin,
    brier_score,
    calibration_slope,
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
from .uncertainty import interval_width, marginal_percentiles, percentile_probabilities

__all__ = [
    "SMALL_SAMPLE_BETS",
    "Actual",
    "BootstrapResult",
    "BootstrapSummary",
    "CalibrationSummary",
    "LogisticCalibration",
    "OddsSourceSummary",
    "Predicted",
    "PnlSource",
    "ReliabilityBin",
    "ReportData",
    "SourceStats",
    "StopModeSummary",
    "bootstrap_pnl",
    "brier_score",
    "calibration_slope",
    "config_hash",
    "format_odds_source_summary",
    "interval_width",
    "log_loss",
    "marginal_percentiles",
    "odds_source_summary",
    "percentile_probabilities",
    "reliability_table",
    "render_markdown",
    "summarise_simulation",
]
