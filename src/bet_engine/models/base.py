"""Contract for models: probability output plus uncertainty percentiles.

One model per league, never pooled (invariant 7), and every prediction carries
model and code version because an unversioned prediction cannot be reproduced.
"""

# TODO: Model protocol with predict() -> PredictionBatch, carrying provenance
# fields and per-outcome uncertainty percentiles that are never renormalized
# (invariant 5).
