Development Plan v2 — Multi-Market, Multi-League Betting Analytics System

This supersedes the original roadmap. It reflects real lessons learned from a full build-and-test cycle on match-winner/Premier League, not theory. Read the "Lessons Learned" section first — it explains WHY this plan is shaped the way it is.

Lessons Learned (carry these forward, don't relearn them)
Walk-forward backtesting must batch by calendar day, not by match index. Multiple matches share a day; splitting train/predict by index lets same-day matches leak into their own training set. Always assert that training data is strictly from earlier days than what's being predicted — this caught a real bug once already.
A drawdown-triggered stop makes results path-dependent and can flatter a losing strategy. Always test both stop_on_breach=True (the real risk-managed version) AND stop_on_breach=False (full-sequence, to see true underlying performance) separately. A strategy that looks profitable only because it got lucky early and stopped at the right moment is not a working strategy.
Always run a bootstrap significance check on the FULL, untruncated bet sequence — resample actual bet P&Ls with replacement, many times, and see what fraction of resamples are profitable. Under ~15-20% is evidence of real negative edge, not noise. Running bootstrap on a cherry-picked/truncated sequence gives a meaningless number.
Always include a "pure market" control variant (bet using only the de-vigged bookmaker odds, zero weight on your own model). It should place ~0 bets by construction. If it doesn't, your EV/Kelly code has a bug. If it does (correctly placing ~0), it confirms your pipeline code is sound and any losses from your actual model are a property of the model, not a bug in the betting logic.
Point-estimate probabilities can be systematically overconfident, and Kelly staking amplifies that overconfidence (stakes most on exactly the picks the model is most wrong about — the "optimizer's curse"). Uncertainty-aware estimation is necessary, but:
Bootstrap-based uncertainty (refitting the model many times) is computationally impractical at season scale — don't attempt it.
Analytical uncertainty (sampling from the fitted model's own parameter covariance, one fit only) is fast IF the downstream probability-grid computation is vectorized (numpy array ops, not a Python loop calling scipy per-cell — this was a 300x+ speedup when fixed).
Taking percentiles of each outcome (home/draw/away) INDEPENDENTLY does not and should not sum to 1 — this is mathematically correct, not a bug, and is caused by the exponential link function producing skewed (not symmetric) output distributions. Do not "fix" this by renormalizing — that silently destroys the conservatism you're trying to add. A renormalization bug here previously caused a result that looked like "massive edge" and was actually inverted/wrong.
A very conservative percentile (e.g. 20th) can collapse bet volume to near-zero, making the result statistically meaningless either way. There's a real trade-off between "conservative enough to filter bad bets" and "not so conservative that sample size collapses" — this needs tuning per-market, and tuning it on the same data you'll evaluate on risks overfitting the threshold itself. Validate any chosen percentile on a league/season NOT used to pick it.
Match-winner is plausibly the wrong market to focus on. It's the most liquid, most-modeled, most efficiently-priced market in football betting. A full validated test (2 leagues, 5 combined seasons, multiple uncertainty configurations) found no demonstrated edge and reasonable evidence of negative edge. This is a real result, not a failure of the process — but it means the project's energy should shift toward markets with structurally less scrutiny.
Different markets need different models, sharing the same validation engine. Dixon-Coles is specifically a paired-team goals model — it doesn't generalize to corners or cards. The backtest/calibration/bootstrap/control-variant harness is market-agnostic and should be reused untouched; only the prediction model and target variable change per market.
Don't fit one model across multiple leagues' data pooled together unless explicitly testing whether that matters — team strength parameters are only meaningful relative to the other teams they were fit alongside. Default to one independent model per league, pooling only at the final bet-selection stage across leagues/markets.
Phase 0 — Rebuild the Proven Foundation

Goal: Recreate the validation engine, informed by lessons 1-4 from the start this time (don't rediscover these bugs again).

Build:

Config system (per-market, per-league variant configs)
Database schema: generalize predictions/bets tables to carry a market field (not just implicitly "match winner")
Walk-forward backtest runner, day-batched, with the no-lookahead assertion built in from day one
stop_on_breach as a parameter from day one (both modes supported from the start)
Bootstrap significance checker, built to run on full/untruncated sequences
Control-variant pattern, generalized so any new market model can be checked against a "zero-skill" baseline

Gate checklist:

 No-lookahead assertion present and tested against a deliberately-broken case
 Both stop_on_breach modes implemented and produce different results on a toy case
 Bootstrap checker runs on a synthetic known-edge and known-no-edge sequence, correctly distinguishing them
 Control-variant pattern works for at least one market before building a second
Phase 1 — Cards Market (first new market, team + referee level)

Goal: Test the "overlooked secondary market" thesis properly, on a market with a plausible, documented source of edge (referee tendency).

Build:

Data ingestion extended to pull HY/AY/HR/AR (cards) and Referee columns — already present in the existing football-data.co.uk files, just unused so far
A team+referee card-rate model: simplest reasonable version is a Poisson regression, predicting each team's expected cards per match as a function of team discipline rate and referee's historical card rate (NOT Dixon-Coles — that's goals-specific)
Analytical uncertainty estimation for this model (reuse the covariance-sampling pattern from the match-winner work, adapted to this model's parameters)
Cards betting markets typically quoted as over/under a threshold (e.g. "over 4.5 cards") — EV/Kelly logic needs adapting from 3-outcome (home/draw/away) to 2-outcome (over/under) markets

Testing process:

Full gauntlet from lessons 1-4: walk-forward, both stop_on_breach modes, bootstrap on full sequence, control variant
Specifically check: does referee identity meaningfully improve calibration over a team-only model? (Compare a team-only version vs. team+referee version as two sub-variants — this tests whether "referee tendency" is real signal or a red herring)

Gate checklist:

 Calibration check passes (predicted over/under frequencies match actual)
 Control variant (pure market) places ~0 bets, confirming pipeline correctness
 Bootstrap on full sequence run and reported honestly, whatever it shows
 Team+referee version compared against team-only version specifically
Phase 2 — Corners Market

Goal: Second secondary market, different data characteristics than cards (more tactical/random, less referee-driven, more pace-of-match-driven).

Build:

Same pattern as Phase 1: team-level Poisson-style model using HC/AC columns
Likely predictors: team's historical corner rate for/against, possibly pace proxies if available (shots data, also already present in the files: HS/AS/HST/AST)

Gate checklist: same shape as Phase 1's — calibration, control variant, full bootstrap, honest reporting regardless of outcome.

Phase 3 — Multi-League Expansion (per market)

Goal: Once a market (cards or corners) shows a real signal on one league, test whether it generalizes — this is also where sample-size problems from narrow uncertainty percentiles get solved by volume rather than by picking a flattering cutoff.

Build:

Independent model fit per league (lesson 8) — not one pooled fit
Cross-league pooling ONLY at the bet-selection stage: gather every league's candidate bets for a given market, apply one EV/uncertainty threshold across the combined pool
Explicitly validate any uncertainty percentile choice on a league NOT used to pick it (lesson 5's overfitting risk) — e.g. tune on Premier League + Eredivisie, validate on La Liga/Serie A/Ligue 1 before trusting the result

Gate checklist:

 Per-league models checked individually before pooling (no silently broken league dragging down/inflating the pooled result)
 Chosen uncertainty percentile validated on a held-out league, not just the tuning leagues
 Pooled bet volume large enough for bootstrap to be meaningful (learned the hard way: single or low-double-digit bet counts make every stat meaningless)
Phase 4 — Revisit Match-Winner (multi-market, multi-league pooling)

Goal: Not re-litigating whether match-winner alone has edge (it doesn't, we tested that) — instead, let match-winner compete as just one more candidate market in the same pooled bet-selection pipeline as cards/corners. If a market-agnostic pool occasionally still finds a genuine match-winner opportunity amid the noise, this architecture should surface it without us having to bet match-winner every week regardless of whether there's anything good there.

Build:

Wire the existing (now better-understood) Dixon-Coles pipeline into the same pooled candidate-selection system as cards/corners
No separate gate needed — this reuses Phase 0's engine and Phase 3's pooling pattern, just adding a third market type to the pool
Phase 5 — Population/Survival Layer (variants, across markets and leagues)

Goal: The prop-firm-style variant framework from the original plan — but now applied across a genuinely multi-market, multi-league candidate pool, which is what it was always meant to select between.

Build:

Variant configs now vary by: EV threshold, Kelly fraction, uncertainty percentile, AND which market(s)/leagues a variant is allowed to draw from
Promotion/demotion exactly as originally speced: profit target + drawdown limit + minimum sample size
Phase 6 onward — Live paper trading, real money, in-play, player stats

Unchanged in substance from the original roadmap's Phase 4 onward, EXCEPT:

Player-level stats (shots, individual cards, assists) are explicitly deferred here, not folded into Phase 1-2, because football-data.co.uk has no player-level columns — this needs a different, likely paid, data source (API-Football/Understat-style), and player models need to handle rotation/suspension/form in ways team-level models don't. Scope this as its own research phase once team-level multi-market is proven, not before.
In-play/live adaptation remains its own separate track (the latency/data-cost argument from earlier still holds, unchanged).
What NOT to do, explicitly (hard-won from this build cycle)
Don't fit uncertainty via refitting/bootstrapping the model repeatedly — use the analytical (covariance-sampling) approach from day one.
Don't trust an ROI number from fewer than ~100 bets, regardless of how good it looks.
Don't pool multiple leagues into one model fit without explicitly testing whether it matters first.
Don't tune and validate an uncertainty threshold on the same data.
Don't conclude "no edge" or "found edge" from a stop_on_breach=True run alone — always check the full-sequence version too.