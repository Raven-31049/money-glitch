"""Significance of a P&L sequence: bootstrap confidence intervals.

Invariant 3 is enforced here, not merely documented: the bootstrap always runs
on the FULL, untruncated bet sequence. A ledger that a stop cut short is
refused rather than quietly resampled, because an interval over a truncated
curve would describe a different claim than the played-on curve the report
shows alongside it (invariant 2). The guard is what keeps "the bootstrap ran
on everything" true by construction instead of by convention.

Resampling unit is the single bet, drawn with replacement — the unit is stated
here because reports must state it (docs/PLAN.md §5). Draws are produced in
vectorised numpy batches: no Python-level loop over resamples, and batches are
sized so a large ``n`` never materialises one giant ``(n, n_bets)`` array.

``seed`` is a required argument on purpose: a significance number nobody can
reproduce is not evidence, so every caller has to say which stream it drew
from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.bankroll import Simulation
from .report import SMALL_SAMPLE_BETS

#: What ``bootstrap_pnl`` accepts: per-bet P&Ls as a 1-D sequence, a ledger
#: frame carrying ``pnl`` (and, for money-on-money ROI, ``stake``), or the
#: Simulation that finished such a run. Only the frame forms can carry the
#: truncation flag — a bare column of numbers cannot say where it came from.
PnlSource = Simulation | pd.DataFrame | pd.Series | np.ndarray | Sequence[float]

#: Resample draws are produced in batches of at most this many elements, so
#: ``n=10000`` over a 10000-bet sequence costs ~16 MB per batch instead of
#: ~800 MB as one array. Batching bounds memory only — every resample is still
#: pure numpy, with no per-resample Python work.
_BATCH_ELEMENTS = 2_000_000

#: What the truncated-ledger guard raises; named in full because invariant 3
#: is the reason, and a caller hitting it needs to know which run to pass.
_TRUNCATED_MESSAGE = (
    "pnl is a ledger truncated by a stop; the bootstrap must run on the full, "
    "untruncated bet sequence (invariant 3) — pass the stop_on_breach=False "
    "ledger for this run"
)


@dataclass(frozen=True)
class BootstrapResult:
    """The resampled distribution of profit, summarised.

    ``fraction_profitable`` is the share of resamples whose total P&L is
    strictly positive. ``mean_roi`` and the ``ci_5``/``ci_95`` percentiles
    describe the resampled ROI distribution — total P&L over total staked,
    where a bare P&L array is taken as unit stakes (one bet, one unit).
    ``n_bets`` is the length of the sequence the draws came from, so the
    interval always travels with the sample it describes.
    """

    fraction_profitable: float
    mean_roi: float
    ci_5: float
    ci_95: float
    n_bets: int


def bootstrap_pnl(
    pnl: PnlSource,
    n: int = 10000,
    *,
    seed: int | None,
    stakes: np.ndarray | Sequence[float] | None = None,
) -> BootstrapResult:
    """Resample ``pnl`` with replacement and summarise the profit distribution.

    Each of the ``n`` resamples draws ``n_bets`` bets with replacement, so the
    distribution answers "if this sequence of bets had fallen differently, how
    often would it still be profitable, and where would its ROI sit?" — a
    significance estimate that needs no assumption about independent,
    identically distributed stakes.

    ``pnl`` may be a bare 1-D array of per-bet P&Ls, a ledger DataFrame (a
    ``pnl`` column; its ``stake`` column becomes the ROI denominator unless
    ``stakes`` is given), or the :class:`~bet_engine.backtest.Simulation`
    holding one. A ledger a stop truncated — ``Simulation.stopped``, or the
    ledger's ``attrs["truncated_by_stop"]`` — raises ValueError: resampling it
    would put a confidence interval on a curve the run never actually finished
    (invariant 3). A bare column such as ``sim.bets["pnl"]`` cannot carry that
    flag, so pass the ledger itself.

    Fewer than ``SMALL_SAMPLE_BETS`` bets prints a warning: the interval still
    computes, but a small sample cannot carry a significance claim (AGENTS.md).

    ``seed`` is required (keyword-only) so every reported interval names the
    stream it was drawn from.
    """
    n = _resample_count(n)
    _check_seed(seed)

    values, staked = _sequence(pnl, stakes)
    n_bets = int(values.size)
    if n_bets < SMALL_SAMPLE_BETS:
        print(
            f"WARNING: bootstrap over {n_bets} bets is under the "
            f"{SMALL_SAMPLE_BETS}-bet threshold; this interval is a "
            f"small-sample estimate and cannot support a significance claim "
            f"on its own."
        )

    rng = np.random.default_rng(seed)
    totals, exposure = _draws(values, staked, n, rng)
    # A resample that drew only zero-stake bets staked nothing and won
    # nothing (pnl comes from stake), so 0.0 is its exact ROI, not a fudge.
    rois = np.divide(totals, exposure, out=np.zeros(n), where=exposure > 0.0)
    return BootstrapResult(
        fraction_profitable=float(np.count_nonzero(totals > 0.0) / n),
        mean_roi=float(rois.mean()),
        ci_5=float(np.percentile(rois, 5.0)),
        ci_95=float(np.percentile(rois, 95.0)),
        n_bets=n_bets,
    )


def _sequence(pnl: PnlSource, stakes: object | None) -> tuple[np.ndarray, np.ndarray]:
    """P&L and per-bet stake arrays from whichever form the caller passed.

    The truncation check lives here so every accepted form — Simulation,
    ledger, bare array — passes through exactly one gate before any resampling
    starts.
    """
    frame: pd.DataFrame | None
    if isinstance(pnl, Simulation):
        frame = pnl.bets
        if pnl.stopped:
            raise ValueError(_TRUNCATED_MESSAGE)
    elif isinstance(pnl, pd.DataFrame):
        frame = pnl
    else:
        frame = None

    if frame is not None:
        if frame.attrs.get("truncated_by_stop"):
            raise ValueError(_TRUNCATED_MESSAGE)
        if "pnl" not in frame.columns:
            raise ValueError(
                f"pnl ledger is missing its 'pnl' column; have {list(frame.columns)}"
            )
        values = _floats(frame["pnl"], "pnl")
        if stakes is None and "stake" in frame.columns:
            stakes = frame["stake"]
    else:
        values = _floats(pnl, "pnl")

    if stakes is None:
        return values, np.ones_like(values)

    staked = _floats(stakes, "stakes")
    if staked.size != values.size:
        raise ValueError(
            f"stakes must have one value per bet; got {staked.size} "
            f"stake(s) for {values.size} bet(s)"
        )
    if (staked < 0.0).any():
        raise ValueError(
            f"stakes must be >= 0 on every bet; "
            f"bad index/row(s): {np.flatnonzero(staked < 0.0)[:10].tolist()}"
        )
    if staked.sum() <= 0.0:
        raise ValueError(
            "stakes must include a positive amount; ROI is undefined on zero exposure"
        )
    return values, staked


def _floats(values: object, name: str) -> np.ndarray:
    """A finite 1-D float array, or a ValueError naming ``name``."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric values") from None
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one value per bet (1-D); got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one bet; got 0 values")
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite on every bet; "
            f"bad index/row(s): {np.flatnonzero(~finite)[:10].tolist()}"
        )
    return array


def _draws(
    values: np.ndarray,
    staked: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` resample totals and exposures, drawn in vectorised batches."""
    n_bets = values.size
    batch = max(1, _BATCH_ELEMENTS // n_bets)
    totals = np.empty(n, dtype=float)
    exposure = np.empty(n, dtype=float)
    for start in range(0, n, batch):
        count = min(batch, n - start)
        picks = rng.integers(0, n_bets, size=(count, n_bets))
        totals[start : start + count] = values[picks].sum(axis=1)
        exposure[start : start + count] = staked[picks].sum(axis=1)
    return totals, exposure


def _resample_count(n: object) -> int:
    """Validate ``n``, keeping bools out of an integer's place."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return int(n)


def _check_seed(seed: object) -> None:
    """Seeds are ints or None; ``np.random.default_rng`` would take more."""
    if seed is None:
        return
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")


__all__ = ["BootstrapResult", "PnlSource", "bootstrap_pnl"]
