"""Guards for the step-3c diagnostic's per-line reliability rebuild.

An earlier revision of ``scripts/phase1_step3c_diagnostics.py`` read the P(over)
reliability table for every line off the one walk-forward run at line 4.5, so
lines 3.5/4.5/5.5 showed byte-identical predictions — impossible, since a higher
line is strictly harder to clear. These tests pin the two guards that make that
impossible to reintroduce, plus the signature that let it happen:

* :func:`assert_reconstruction` — the per-side means captured from the run must
  rebuild the run's *own* line exactly, and must reject another line's
  probabilities claiming to be it;
* :func:`assert_line_ordering` — P(over) must fall as the line rises, on every
  match;
* :func:`over_reliability` — takes that line's probabilities and outcomes, so it
  cannot silently reuse one line's numbers for another.

The script is loaded by path because ``scripts/`` is not an importable package.
Importing it defines the functions but does not run ``main`` (the
``if __name__ == "__main__"`` guard).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase1_step3c_diagnostics.py"
)


def _load_diagnostics():
    spec = importlib.util.spec_from_file_location(
        "phase1_step3c_diagnostics", _SCRIPT
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load_diagnostics()

#: Two matches with different expected totals, both Poisson (alpha_ effective 0).
SIDES = {"m1": (1.6, 1.4, 0.0), "m2": (0.8, 0.9, 0.0)}
IDS = np.array(["m1", "m2"])


def _long_form(wide: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """The walk-forward's long shape from a wide over/under frame."""
    return pd.DataFrame(
        {
            "match_id": np.repeat(ids, 2),
            "outcome": np.tile(np.array(["over", "under"], dtype=object), len(ids)),
            "model_prob": wide[["over", "under"]].to_numpy(dtype=float).ravel(),
        }
    )


def test_probabilities_fall_as_the_line_rises():
    over35 = diag.probabilities_for_line(SIDES, IDS, 3.5)["over"].to_numpy(dtype=float)
    over45 = diag.probabilities_for_line(SIDES, IDS, 4.5)["over"].to_numpy(dtype=float)
    over55 = diag.probabilities_for_line(SIDES, IDS, 5.5)["over"].to_numpy(dtype=float)

    # The three lines are genuinely different, and ordered, per match.
    assert np.all(over35 > over45)
    assert np.all(over45 > over55)
    diag.assert_line_ordering(SIDES, IDS, diag.LINES)


def test_line_ordering_guard_rejects_an_inversion(monkeypatch):
    # Rebuilding from independent Poissons is always ordered, so to check the
    # guard itself fires, hand it probabilities that are not.
    def fake(sides, ids, line):
        value = {3.5: 0.5, 4.5: 0.6, 5.5: 0.4}[line]  # 4.5 above 3.5: inverted
        return pd.DataFrame(
            {
                "over": np.full(len(ids), value, dtype=float),
                "under": np.full(len(ids), 1.0 - value, dtype=float),
            },
            index=pd.Index(ids, name="match_id"),
        )

    monkeypatch.setattr(diag, "probabilities_for_line", fake)
    with pytest.raises(AssertionError, match="must exceed"):
        diag.assert_line_ordering(SIDES, IDS, diag.LINES)


def test_reconstruction_accepts_the_runs_own_line_and_rejects_another():
    matching = diag.probabilities_for_line(SIDES, IDS, diag.RUN_LINE)
    prediction = _long_form(matching, IDS)

    # The captured sides rebuild the run's own line exactly: accepted.
    diag.assert_reconstruction(prediction, SIDES, IDS, diag.RUN_LINE)

    # A prediction built from a *different* line must be rejected — this is the
    # exact bug (one line's probabilities claiming to be another's).
    mismatched = diag.probabilities_for_line(SIDES, IDS, 3.5)
    prediction_wrong = _long_form(mismatched, IDS)
    with pytest.raises(AssertionError, match="do not match the walk-forward"):
        diag.assert_reconstruction(prediction_wrong, SIDES, IDS, diag.RUN_LINE)


def test_over_reliability_reports_the_arrays_it_is_given():
    # One probability per decile bin, so each populated bin holds exactly its
    # own value: the table's `predicted` is the array passed in, nothing else.
    p = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    y = np.zeros_like(p)
    rows = diag.over_reliability(p, y)
    populated = [row for row in rows if row["n"]]

    assert len(populated) == 10
    assert all(row["n"] == 1 for row in populated)
    assert [row["predicted"] for row in populated] == pytest.approx(list(p))
    assert [row["actual"] for row in populated] == pytest.approx([0.0] * 10)
