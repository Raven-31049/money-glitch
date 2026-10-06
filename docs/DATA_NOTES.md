# DATA_NOTES.md — Known Data Caveats

Running log of known data-quality caveats for the football-data.co.uk source.
Keep this as the single place caveats live. Entries are appended oldest to
newest; each carries the date it was checked so we know when to re-check. Do
not delete or rewrite an old entry — add a new one when the state changes.

## E0 2526 — Pinnacle odds missing for the second half of the season

- **Checked:** 2026-10-06
- Pinnacle opening (`PSH`/`PSD`/`PSA`) and closing (`PSCH`/`PSCD`/`PSCA`)
  prices are missing for **170 of 380** rows in the E0 2025-26 file.
- All gaps fall in **Jan–May 2026**; Aug–Dec 2025 is fully covered.
- Confirmed present in the source file itself: a fresh download is
  md5-identical to our cached copy (2026-10-06), so this is not cache staleness.
- `B365H`/`B365D`/`B365A` are complete (0% missing).

Implication: do not use Pinnacle as the reference book for de-vig / EV on
2025-26 post-Christmas matches. Re-check the source file before that data is
used, and again when 2526 analyses are started.

Also observed on 2026-10-06: the download URL responds with a 302 redirect;
the followed GET returns the same bytes as the cache, so nothing to act on now,
but a future refactor should follow redirects explicitly.