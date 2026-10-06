"""Block bootstrap over calendar days for significance testing.

Bets within a day share conditions, so resampling whole days is the unit that
keeps the resampled series realistic. The bootstrap always runs on the full,
untruncated bet sequence even when the reported backtest halted on a breach
(invariants 3 and 2).
"""

# TODO: day-block resampling, confidence intervals, p-values on the full
# sequence.
