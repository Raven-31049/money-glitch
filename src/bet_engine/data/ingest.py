"""Pull raw odds/fixtures/results and land them in normalized form.

Source timestamps are preserved through ingestion so downstream staleness
filters can make an honest call instead of trusting a reconstructed clock.
"""

# TODO: provider client, raw file landing, normalization into the events /
# markets / odds_snapshots / results tables.
