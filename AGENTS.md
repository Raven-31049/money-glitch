# AGENTS.md — Project Rules

Betting analytics research system. Full roadmap: docs/PLAN.md (read it before any task).
We are building phase by phase. Never build ahead of the current phase.

## Tech
- Python 3.11+, managed with uv. Package lives in src/bet_engine/. Tests in tests/ (pytest).
- pandas, numpy, scipy, statsmodels, pydantic, pyyaml. SQLite for storage.
- Run tests with: uv run pytest

## Non-negotiable rules (from lessons learned — do not violate)
1. Walk-forward backtests batch by CALENDAR DAY. Training data must be strictly earlier
   than the prediction day. An assertion enforces this at runtime and must never be removed.
2. Every backtest supports stop_on_breach=True and stop_on_breach=False. Reports show both.
3. Bootstrap significance always runs on the FULL, untruncated bet sequence.
4. Every market has a "pure market" control variant (de-vigged odds, zero model weight).
   It must place ~0 bets. If it doesn't, there is a bug in the EV/Kelly code.
5. Uncertainty percentiles per outcome are NOT renormalized to sum to 1. Ever.
6. Probability grids and sampling must be vectorized numpy, never per-cell Python loops.
7. One model per league. Never pool leagues into one fit unless the task explicitly says so.
8. The validation engine (backtest, bankroll, bootstrap, calibration, control) is
   market-agnostic. Market-specific logic lives only in markets/ and models/.

## Working style
- Never edit or delete a test just to make it pass. If a test seems wrong, stop and explain why.
- Small modules, type hints, docstrings that explain WHY, not just what.
- After each task: run the full test suite and report results honestly, including failures.
- Never report an ROI conclusion from fewer than ~100 bets without stating the sample is too small.
