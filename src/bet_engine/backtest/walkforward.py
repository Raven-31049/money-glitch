"""Calendar-day walk-forward fold engine.

Training data must be strictly earlier than the prediction day, enforced at
runtime by an assertion that must never be removed or weakened (invariant 1).
The hard boundary is a calendar day in the league's timezone, so folds cannot
quietly absorb same-day information through a timestamp rounding quirk.
"""

# TODO: fold iteration over calendar days plus the temporal assertion
# `max(train.timestamp) < fold.day_start`.
