# Reverse-engineering the tester

The tester is everything that produces and scores the hidden test cases: the scoring code, the data generator or market simulator, and the judge with its checker and limits. Organisers write it once and use it for both the data you get and the data you are scored on. Whoever models it best usually wins: the user reports that the top teams at the HRT Algo Arena all worked this way. Build the strategy from it.

## Contents
1. Ground rules
2. The scorer (both modes)
3. The generator (prediction and trading contests)
4. The judge (algorithm contests)
5. Turning the model of the tester into strategy
6. Worked example: HRT Algo Arena, Nov 2026
7. Worked example: a Masyu solving contest

## 1. Ground rules

- **Organiser code is evidence; read it like source, not documentation.** Comments say what the organisers meant. Code says what the tester does.
- **Re-implement, then match.** Every fact you rely on (the score formula, the label definition) should be reproduced numerically on known data, not just read.
- **Facts and hypotheses stay separate** in the master state. A hypothesis about the generator becomes a fact only after a test that could have refuted it.
- **Stay inside the rules.** Use what every competitor gets: data, starter code, docs, your own scores. Don't exploit platform bugs, try to extract hidden labels, or probe the scorer beyond normal use. If a probe might be against the rules, ask first.

## 2. The scorer

| Question | Why it matters |
|---|---|
| Exact formula, including std convention (ddof), annualisation and any epsilon | Decides the optimal transformation of forecasts into submissions |
| Treatment of NaN, inf, missing rows, constant predictions | A sanitiser that maps non-finite to 0 is safe; a crash or a rejection costs an upload |
| Is the score invariant to scale, shift or sign? | Scale-invariant scores make sizing *relative*; a sign error flips the score |
| What is the id column, and in what order? | `headline_id` meant session id here: a misread zeroes the score |
| Is the label adjusted (demeaned, market-adjusted, clipped)? | Demeaned labels make constant predictions score 0 and remove the need for a bias term |
| Tie-breaks, best vs latest upload, public/private split | Decide the upload plan (`leaderboard-strategy.md`) |

Then derive the optimal submission from the score. For a Sharpe of `position × return` across sessions, that is `mu / (k·var + mu²)`. For accuracy-like scores it is a threshold rule. For rank scores, only the ordering matters.

## 3. The generator (prediction and trading)

Work through `synthetic-markets.md` §1 (label, templates, event study, entity structure, arrivals, volatility). In addition:
- **Train vs hidden.** Compare every input distribution between the labelled and hidden sets (counts, volatility, event mix, id ranges). Same generator → in-sample cross-validation is representative and fitting on all labelled data is right. Different → find out how they differ before modelling.
- **Generator fingerprints.** Rounding (prices to 4 d.p.), fixed starting values (price = 1), fixed lengths, round-number parameters and discrete template lists all reveal how the simulator was written. Structures are usually simple: fixed kernels, small integer counts, uniform or Gaussian draws.
- **Hidden variables.** If responses vary more than one fixed effect explains, look for a latent variable the generator draws per session (which entity is the asset, a regime, a volatility level). Test with split-half consistency: does it repeat within a session?
- **What the generator ignores.** Proving that something does *not* matter (the numbers inside headlines here) removes whole model families from consideration. That is as valuable as finding what does.

## 4. The judge (algorithm contests)

| Question | How to find out | Strategic consequence |
|---|---|---|
| How are tests generated: random, hand-made, from a public puzzle generator, adversarial? | Statement, samples, past editions, the organiser's other contests | Hand-made or deduction-solvable puzzles reward propagation rules; random or adversarial ones reward robust search and restarts |
| Size distribution; how many tests at the maximum size | Statement limits, samples, past editions | Optimise the worst case of the *actual* distribution |
| Unique solutions? Any valid answer accepted, or a reference answer? | Statement, checker description, sample outputs | Decides whether the solver may stop at the first solution |
| Output format strictness (whitespace, line endings, ordering) | Statement, samples, a trivial early submission if feedback is per test | A format mismatch scores 0 on everything |
| Time measurement: per test or total, compile and startup counted, judge machine speed | Statement, an early timing probe on a sample if allowed | Sets the safety margin and the language choice |
| Feedback per submission: per-test verdicts, score only, nothing | Rules, first submission | Per-test feedback allows targeted fixes; none means the internal test set must carry the load |

## 5. From model to strategy

Write a short table in the master state: **property of the tester → how it was established → what it implies**. Every later decision should cite a row. The rows drive:
- **What to predict or solve:** the quantity the scorer rewards, computed from the mechanism the generator uses.
- **How to size or order the work:** from the score formula and the test distribution.
- **Where the headroom is:** the latent variable or test class the current solution handles worst.
- **What the internal tests must cover:** the generator's actual distribution, plus its extremes.

## 6. Worked example: HRT Algo Arena, Nov 2026

| Property of the tester | How it was established | Strategy it produced |
|---|---|---|
| Score = `sqrt(252)·mean/std` (pandas ddof=1, +1e-12) of `pred × return`. The local helper maps non-finite predictions to 0, but the upload page rejects them | Read `sessions.py` and the upload page's checks; re-implemented | Scale-free; size as `mu/(k·var + mu²)`; never emit NaN (the validator script checks it) |
| Label = simple return close49 → close59, minus the cross-session mean | Recomputed from bars, exact to 1e-19 | Constant predictions score 0; no bias term needed |
| `headline_id` column = session number | Read `write_submission` | Correct upload format first time |
| 80 companies, 50 event templates; numbers and slot words carry no effect | Template parsing, then tests | Categorical event model; no NLP |
| Each event moves the price at a constant rate for 10 bars (half weight at the ends) | Event study | Forecast = drift still owed by events from bars 40–49 |
| 5 sectors × 16 companies; each session has 3 main-sector + 1 random company; one hidden target company | Co-occurrence clustering; EM; split-half test | Weight events by the posterior that they concern the target; ignore off-sector |
| Event effects proportional to session volatility; ranges measure it well | Quintile test; Garman–Klass vs close-to-close | Range-based variance in the sizing |
| Hidden set from the same generator (ids 11000+, bars 0–49 only) | Starter code; distribution comparison | In-sample cross-validation is representative; fit on all labelled data |
| Best upload counts | Result page (found late) | Should have shaped the upload plan from the start |

The remaining gap to the leaders was in the hidden-target inference: our posterior needed an ad-hoc temperature, a sign that our model of the generator's exposure structure was still wrong. A better model of the tester was the next step, not more features.

## 7. Worked example: a Masyu solving contest (hypothetical)

| Property of the tester | How to establish it | Strategy it produces |
|---|---|---|
| Puzzles come from a generator that thins clues to a unique solution | Past editions; sample density; uniqueness check on samples | Deduction rules plus probing; prepare for sparse 40×40 grids |
| The checker accepts any valid loop | Statement | Stop at the first loop that passes self-verification |
| Time is per puzzle and includes process startup | Statement; timing a trivial submission if allowed | C++; keep startup trivial; a time guard below the limit |
| Output format: exact grid with fixed characters | Samples | Isolated, tested I/O layer; checker that parses our own output |
| Most tests are small, a few are maximum size | Past editions | Spend tuning time on the maximum-size tail |
