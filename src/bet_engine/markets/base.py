"""Contract every market must fulfil: outcomes, settlement, tick size, de-vig.

The validation engine depends only on this contract and never on a concrete
market, which is what keeps backtest, bankroll, bootstrap, calibration and
control market-agnostic (invariant 8).
"""

# TODO: Market protocol/ABC with outcome labels, settlement, tick size, and
# de-vig wiring.
