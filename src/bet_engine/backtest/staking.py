"""Staking: edge threshold, fractional Kelly, hard per-bet caps.

The pure market control (de-vigged odds, zero model weight) must place ~0 bets,
so this module's threshold behaviour is the tripwire for EV/Kelly bugs
(invariant 4).
"""

# TODO: edge filter, fractional Kelly with cap, per-bet stake ceiling.
