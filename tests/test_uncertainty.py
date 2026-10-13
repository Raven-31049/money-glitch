"""Tests for eval/uncertainty.py — percentiles are marginals, never renormed.

The load-bearing one is :func:`test_percentiles_are_not_renormalised`: it builds
draws whose per-outcome 10th percentiles deliberately sum to well under 1, so an
implementation that renormalised each percentile to sum to 1 across outcomes
would return something else and fail. That is AGENTS.md rule 5 / invariant 5,
and it must not be possible to "fix" it by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from bet_engine.eval.uncertainty import interval_width, marginal_percentiles


def _bimodal_draws(draws: int = 4000, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Half the draws nearly certain over, half nearly certain under.

    Per-outcome marginals are then symmetric but the 10th percentiles of *both*
    outcomes are tiny (an uncertain match looks small for over and for under),
    which is exactly the shape a renormalising implementation would erase.
    """
    samples = np.empty((draws, 1, 1, 2), dtype=float)
    half = draws // 2
    samples[:half, 0, 0, :] = [high, low]
    samples[half:, 0, 0, :] = [low, high]
    return samples


def test_percentiles_are_not_renormalised():
    samples = _bimodal_draws()

    result = marginal_percentiles(samples, [10, 50, 90])

    low = result[0, 0, 0]
    # Each outcome's own 10th percentile, unnormalised: they sum to ~0.02, not 1.
    assert low[0] == pytest.approx(0.01, abs=0.01)
    assert low[1] == pytest.approx(0.01, abs=0.01)
    assert low.sum() == pytest.approx(0.02, abs=0.02)
    assert low.sum() != pytest.approx(1.0, abs=0.5)


def test_percentiles_match_independent_numpy_per_outcome():
    rng = np.random.default_rng(0)
    samples = rng.dirichlet([2.0, 5.0], size=(500, 7)).reshape(500, 7, 1, 2)

    result = marginal_percentiles(samples, [5, 50, 95])

    assert result.shape == (3, 7, 1, 2)
    for position, percent in enumerate([5, 50, 95]):
        for outcome in range(2):
            expected = np.percentile(samples[:, :, 0, outcome], percent, axis=0)
            assert np.allclose(result[position, :, 0, outcome], expected)


def test_interval_width_is_the_high_minus_low_marginal():
    samples = _bimodal_draws()

    width = interval_width(samples, low=10.0, high=90.0)

    assert width.shape == (1, 1, 2)
    assert width[0, 0, 0] == pytest.approx(0.98, abs=0.05)


def test_percentiles_are_computed_from_the_draw_axis():
    """Axis 0 is draws: growing it must not change a point's percentile shape."""
    samples = np.stack([np.full((4, 1, 2), 0.3), np.full((4, 1, 2), 0.7)], axis=0)

    result = marginal_percentiles(samples, [50])

    assert result.shape == (1, 4, 1, 2)
    assert np.allclose(result[0], 0.5)


def test_rejects_a_bad_shape_and_bad_percentiles():
    with pytest.raises(ValueError, match="draw axis"):
        marginal_percentiles(np.zeros((5,)), [50])

    with pytest.raises(ValueError, match="at least one percentile"):
        marginal_percentiles(np.zeros((5, 2)), [])

    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        marginal_percentiles(np.zeros((5, 2)), [101])
