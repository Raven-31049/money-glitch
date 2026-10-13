"""Discrete count distributions and their convolution (cards, Phase 1 step 3).

WHY this module exists: the cards model predicts each *side's* own cards, and
the market settles on the *match total* = home + away. For a Poisson side the
total is again Poisson (sums of independent Poissons are Poisson), but for the
negative-binomial model the total is not a named distribution — it is the
convolution of two per-side negative-binomial pmfs. That convolution is the one
piece of arithmetic step 3 adds, so it lives here, computed once and shared by
the model and the report.

WHY vectorised: AGENTS.md rule 6 forbids per-cell Python loops in probability
grids. Each pmf grid is built by broadcasting ``(n_matches, 1)`` against
``(1, n_counts)``, and the row-wise convolution is a single matrix product
against a constant 0/1 selector, so no Python loop touches a grid cell.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

#: Highest per-side card count kept in a pmf grid. The convolution of two such
#: grids covers totals 0 .. 2*SIDE_MAX_COUNT - 2. Real per-side card counts top
#: out in the low teens, so 48 truncates nothing a match has ever produced while
#: keeping the grids (and the convolution) small.
SIDE_MAX_COUNT: int = 48

#: Cache of the convolution selector matrix, keyed by grid length. It is a pure
#: function of the length, so building it once per run is enough.
_SELECTORS: dict[int, np.ndarray] = {}


def side_pmf(mu: np.ndarray, alpha: float | None | np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-side card pmf on ``counts`` for each row of ``mu``.

    ``mu`` is an array of per-match expected own cards. ``alpha`` is the
    negative-binomial dispersion (variance = mu + alpha * mu**2), either one
    scalar for a fitted model or an array with one value per match; ``None`` or
    a non-positive value means the Poisson limit, which is what the recruit
    models use and what a negative-binomial fit collapses to when the data
    carries no overdispersion.

    Returns an ``(n_matches, n_counts)`` array; every row sums to ~1.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    counts = np.asarray(counts)
    if alpha is None:
        return poisson.pmf(counts[None, :], mu[:, None])

    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 0:
        value = float(alpha)
        if not np.isfinite(value) or value <= 0.0:
            return poisson.pmf(counts[None, :], mu[:, None])
        return nbinom.pmf(counts[None, :], 1.0 / value, (1.0 / (1.0 + value * mu))[:, None])

    alpha = np.broadcast_to(alpha, mu.shape)
    out = poisson.pmf(counts[None, :], mu[:, None])
    positive = np.isfinite(alpha) & (alpha > 0.0)
    if positive.any():
        a = alpha[positive]
        out[positive] = nbinom.pmf(
            counts[None, :],
            (1.0 / a)[:, None],
            (1.0 / (1.0 + a * mu[positive]))[:, None],
        )
    return out


def _selector(length: int) -> np.ndarray:
    """The ``(length**2, 2*length-1)`` 0/1 matrix that sums row products on
    anti-diagonals — i.e. convolution as one matrix product, no Python loop."""
    matrix = _SELECTORS.get(length)
    if matrix is None:
        index = np.arange(length)
        diagonal = index[:, None] + index[None, :]
        matrix = np.zeros((length * length, 2 * length - 1), dtype=float)
        matrix[np.arange(length * length), diagonal.ravel()] = 1.0
        _SELECTORS[length] = matrix
    return matrix


def convolve_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise convolution of two ``(n_matches, length)`` pmf grids.

    ``result[i, t] = sum_j left[i, j] * right[i, t-j]`` — the distribution of
    the sum of the two sides for match ``i``. Implemented as an outer product
    and one matrix product so the whole batch is a single numpy call.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"convolve_rows needs two equal-shaped grids, got {left.shape} and "
            f"{right.shape}"
        )
    if left.ndim != 2:
        raise ValueError(f"convolve_rows needs 2-D grids, got shape {left.shape}")
    n, length = left.shape
    outer = left[:, :, None] * right[:, None, :]
    return outer.reshape(n, length * length) @ _selector(length)


#: The three settlement states a cards over/under market can read off a total
#: pmf, in the order :func:`over_under_push` returns them: over, under, and the
#: push (exact hit on a whole-number line). A half-line always has a push mass
#: of exactly 0, so over + under == 1 there; a whole line splits the mass three
#: ways and the bet on a push is void (markets.base.PUSH).
OVER_UNDER_PUSH: tuple[str, ...] = ("over", "under", "push")


def over_under_push(
    pmf: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row ``(over, under, push)`` from a match-total pmf and per-row lines.

    ``pmf`` is ``(n_matches, n_counts)`` on the integer counts
    ``0, 1, ..., n_counts - 1`` and ``lines`` is one threshold per row. ``over``
    is P(total > line), ``under`` is P(total < line), and ``push`` is
    P(total == line) — nonzero only when the line is a whole number, where the
    bet is void. The three always sum to 1 by construction (they partition the
    pmf), so the caller never has to renormalise (invariant 5 concerns
    percentiles, but this is exact even so). Vectorised: one broadcast compare
    per side, no Python loop over rows or cells (invariant 6).
    """
    grid = np.asarray(pmf, dtype=float)
    thresholds = np.asarray(lines, dtype=float)
    if grid.ndim != 2:
        raise ValueError(
            f"pmf must be 2-D (n_matches, n_counts), got shape {grid.shape}"
        )
    if thresholds.shape != (grid.shape[0],):
        raise ValueError(
            f"lines must have one value per row ({grid.shape[0]}), got "
            f"shape {thresholds.shape}"
        )
    counts = np.arange(grid.shape[1])
    greater = counts[None, :] > thresholds[:, None]
    smaller = counts[None, :] < thresholds[:, None]
    over = (grid * greater).sum(axis=1)
    under = (grid * smaller).sum(axis=1)
    push = (grid * ~(greater | smaller)).sum(axis=1)
    return over, under, push


def total_pmf(
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    alpha: float | None | np.ndarray,
    *,
    side_max: int = SIDE_MAX_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of home+away cards: ``(counts, pmf)``.

    ``counts`` is ``0 .. 2*side_max - 2`` and ``pmf`` has shape
    ``(n_matches, len(counts))``, each row summing to ~1. The per-side grids
    are convolved, so this is exact for the Poisson limit and captures the
    two-side convolution for the negative binomial.
    """
    counts = np.arange(2 * side_max - 1)
    side_counts = np.arange(side_max)
    home = side_pmf(mu_home, alpha, side_counts)
    away = side_pmf(mu_away, alpha, side_counts)
    return counts, convolve_rows(home, away)


__all__ = [
    "OVER_UNDER_PUSH",
    "SIDE_MAX_COUNT",
    "convolve_rows",
    "over_under_push",
    "side_pmf",
    "total_pmf",
]
