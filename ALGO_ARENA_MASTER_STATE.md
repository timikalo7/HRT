# ALGO_ARENA_MASTER_STATE

Last updated: 2026-11-24 (session 1 — research and baseline, before any official materials were provided)

Evidence labels used throughout:
**[OFFICIAL]** organiser's own files · **[MEASURED]** computed by us on the data · **[PUBLIC-CLAIM]** stated by a third party, not verified · **[HYPOTHESIS]** our inference, untested · **[UNKNOWN]**

---

## Session 2 update (2026-11-24, live competition window)

**Confirmed.** The user supplied the official notebook, challenge page text and data. The five parquet files and `sessions.py` are byte-identical (SHA-256) to the public edition, and the page text matches section B. Hypothesis C.1 is now fact. The challenge page as pasted contains no restriction on outside help.
**Deadline.** One hour from the start; started about 18:29 UTC, so uploads must be in by about 19:25. Current leaderboard leader: 3.609. 10 uploads.
**User decisions.** Build on the public ~3.6 solution and improve it; the strategy must be explainable (see `STRATEGY.md`).

**New findings [MEASURED]**
- Sectors: exactly 5 sectors of 16 companies, from co-occurrence (the suffix-based map in one public solution misplaces several names). Every session = 3 companies from one sector + 1 drawn from anywhere (83.5% other-sector, 16.5% same-sector).
- Hidden target: EM on full sessions gives role weights own / same-sector peer / other-sector = 1 / 0.25–0.28 / −0.02. The target is weakly identified (mean max posterior 0.65 even with 100 bars).
- At temperature 1 the bar-49 target posterior is over-confident and costs Sharpe; flattening it (temperature 4) is better than both trusting it and ignoring it.

**Experiments (5-fold CV, identical folds, seeds 0 and 1 unless noted)**

| ID | Change | Result | Decision |
|---|---|---|---|
| E2 | Latent-target forecast, position = mu / sigma (seed 0) | 3.297 | superseded |
| E3 | Posterior temperature 1 / 2 / 4 / prior-only, sizing mu/(10 sigma^2 + mu^2) (seed 0) | 3.392 / 3.433 / 3.442 / 3.414 | keep temperature 4 |
| E4 | Sizing mu/sigma^k for k = 0, 0.5, 1, 1.5, 2 vs mu/(10 sigma^2 + mu^2), temperature 4 (seed 0) | 3.328, 3.370, 3.374, 3.334, 3.250 vs 3.442 | keep the Sharpe-optimal form |
| E5 | Variance from Garman–Klass + gaps instead of close-to-close sd | 3.442 -> 3.493 / 3.492; flat for scale 12–25 | keep, scale 15 |
| E6 | Public `improved_all` re-run on our folds | 3.483 / 3.463; regenerated out-of-sample file equals the published one (max diff 7e-13) | used as blend component |
| E7 | Public forecast re-sized as p/(v + p^2) | 3.495 / 3.477 alone, no gain inside the blend | rejected (no added value) |
| E8 | 50/50 blend of public model and latent-target model, each at unit sd | **3.546 / 3.537**; vs public alone +0.063 / +0.074, bootstrap sd 0.027, P(>0) 0.992 / 0.998; weights 0.4–0.6 flat | **final candidate** |

**Status.** `arena/latent.py` (model B), `arena/blend.py`, `STRATEGY.md`. Upload files are written to `submissions/` (git-ignored): `predictions_c2_blend.csv` (final candidate), `predictions_c1_latent.csv` (our model alone), `predictions_public_improved_all_rerun.csv` (public model re-run; not to be uploaded as our own).
**Checks run by us.** Predictions unchanged when bars 50–99 of test sessions are overwritten with noise (max change 0.0); sector map identical when built from first-half headlines only; all files pass the upload format checks; no position exceeds 3.1 sd.
**Independent audit (fresh-context agent), returned 19:00 UTC — verdict SAFE TO UPLOAD for `predictions_c1_latent.csv` (sha256 97cd47dd...).**
- Critical: none. Empirical leak tests all gave identical predictions (max diff 0): block rebuilt from raw parquet with only bars 0-49; bars 50-99, late headlines and labels of test sessions scrambled then refit; training labels shuffled (`fit` never reads labels); sector map built from the training fold only.
- Major 1: the brief quoted a stale score. Resolved: the current code scores 3.493 / 3.492 / 3.493 (seeds 0-2), confirmed independently by the auditor.
- Major 2: sampling noise. Bootstrap sd of the CV Sharpe 0.146 (5-95%: 3.26-3.72). Accepted; nothing to fix.
- Minor, accepted: `predict()` relies on the caller passing a first-half block (3.7% of predictions change on an untruncated block; never triggered because `cross_validate` and the out-of-sample files are truncated); hyperparameters tuned on the same CV but the curves are flat (temperature 2-8: 3.485-3.493; variance scale 12-25: 3.486-3.493), selection bias at most ~0.05; slot overflow for a fifth company would be a bug but the maximum is 4; 21% of model B's predictions are exactly zero (sessions with nothing pending) — the blend has none; `mu` is over-stated ~17% (slope 0.834), which only touches the small mu^2 term.
- Not covered by the audit: `arena/blend.py`, the blend file and the public re-run file. Our own checks: the blend file equals 0.5/0.5 of the audited file and the public re-run at unit scale (max diff ~1e-15) and passes the format checks; the public re-run equals that repo's published file (max diff 7e-13), which its author reports scored about 3.6.
**Rapid arXiv q-fin search (fresh-context agent).** Sentiment literature: not applicable to templated headlines (consistent with the measured 2.6 vs 3.0+). One idea tested — use the posterior second moment of the owed drift in the sizing denominator (after Spears, Zohren, Roberts, arXiv:2007.15982): model B 3.493 -> 3.497, blend 3.546 -> 3.548 / 3.537 -> 3.539. Inside noise; **rejected** so the audited file stays unchanged. Blend-weight optimisation (Kakushadze, arXiv:1405.4716) already covered by E8: weights 0.4-0.6 are flat.
**Independent advisor (fresh-context agent, not `/advisor`).** Launched 18:41 UTC; had not returned when the candidate was handed over.

### Later in the window (19:03-19:10 UTC) — leaderboard leaders reported at 3.8

| ID | Change | Result (seed 0 / seed 1) | Decision |
|---|---|---|---|
| E9 | Combine the two models at forecast level with one sizing rule instead of blending positions | 3.547 / 3.538 vs position blend 3.546 / 3.537 | no gain, rejected |
| E10 | Second-stage least squares on [owed drift, headline count, unexplained first-half move, live main-sector headlines] | model B **3.493 -> 3.589 / 3.592**. Unexplained move alone gives 3.561 / 3.568; adding all eight candidate features is slightly worse (3.580 / 3.587) | **kept** — now in `arena/latent.py` |
| E11 | Blend weight of the public model against the upgraded model B | 0 / 0.1 / 0.25 / 0.5 -> 3.589, 3.591, 3.589, 3.571 (seed 0) | public weight cut to 0.15 |
| E12 | Second public model (`model_3564.py`, default config) re-run on our folds | 3.455 alone (seed 0 only); three-way 0.15 public / 0.15 m3564 / 0.70 ours = **3.601** vs 3.589 without it | **kept as `predictions_c5_blend3.csv`**; evidence on one fold assignment only |

**Current best candidate: `submissions/predictions_c5_blend3.csv` (3.60 cross-validated).** Leak test on the upgraded model: predictions unchanged (max diff 0.0) after overwriting bars 50-99 of test sessions. Format valid. The independent audit covered the earlier version of model B; the second stage added since then has only our own leak test.
Cross-validation read about 0.13 below the leaderboard for the public model (3.47 vs a reported ~3.6); if that offset holds, 3.60 maps to roughly 3.7. That is an extrapolation from one data point, not a measurement.
Upload files in `submissions/`: c1 (model B, first version, 3.49), c2 (50/50 with public, 3.54), c3 (model B with second stage, 3.59), c4 (0.25 public + 0.75 c3, 3.59), **c5 (three-way, 3.60)**, plus the two public re-runs (blend components only).

---

## A. Competition identity

| Field | Value | Evidence |
|---|---|---|
| Name | "Algo Arena" / "AlgoArena"; organiser's code header reads "Hudson River Trading -- Synthetic Market Prediction Challenge" | [OFFICIAL] `sessions.py` header, as redistributed in two public repos |
| Operating mode | **B — Quantitative trading**, specifically a *return-prediction + position-sizing* task (not market making, not live execution) | [OFFICIAL] starter notebook |
| Organiser / sponsor | Hudson River Trading | [OFFICIAL] file header |
| Platform | HRT's "arcade" challenge page; CSV upload; live leaderboard | [OFFICIAL] starter notebook |
| Host society / university for *our* event | — | [UNKNOWN] |
| Deadline | — | [UNKNOWN] |
| Submission | `predictions.csv`, columns `headline_id,pred` (`headline_id` = session number), one row per out-of-sample session, **10 uploads** | [OFFICIAL] starter notebook |

**Update 2026-11-24, later in session 1.** The user supplied a challenge link on `arcadegames.hudson-trading.com` ("Arcade"). It could not be read from this session: without a personal join link the page shows only "No join link detected. Open the personal link your recruiter sent you." [MEASURED]. The challenge page text, its rules, deadline and upload allowance are therefore still unseen. What the site's public client code does establish is in B.1 and I.1.

> **The central open question.** Everything below describes the edition of the challenge whose starter kit is public on GitHub (active since at least 2026-10-20). It is a **[HYPOTHESIS]**, not a fact, that the event we are entering is this same challenge with this same data. It must be confirmed against the official materials before any of it is relied on. HRT has run at least one variant with a different target (see D.1, ETH Datathon).

## B. Specification of the public edition

All [OFFICIAL] from the organiser's starter notebook and `sessions.py` unless marked.

- **Setting.** Each *session* is one synthetic stock trading for 100 bars. Fictional news headlines about fictional companies are published along the way.
- **Inputs at decision time.** Bars 0–49 (`open`, `high`, `low`, `close`; prices start at 1, rounded to 4 d.p.) and the headlines published in bars 0–49 with their bar index.
- **Decision.** At the close of bar 49 take a position equal to the prediction; hold 10 bars to the close of bar 59.
- **Target.** `returns` = simple return `close[59]/close[49] − 1`, minus the mean of that quantity across sessions ("market-adjusted"). [MEASURED] exact to 1e-19 in-sample; the in-sample mean removed is +0.00066.
- **Score.** `Sharpe = sqrt(252) · mean(pnl) / std(pnl)`, `pnl = pred · return`, one session = one "day". Sample std (pandas default).
- **Data.** 10,000 in-sample sessions (ids 1000–10999) with all 100 bars, all headlines and the label; 10,000 out-of-sample sessions (ids 11000–20999) with bars 0–49 only. [MEASURED]
- **No costs, no position limits, no turnover penalty, no leverage rule.** Only the Sharpe of `pred · return` matters, so the *scale* of predictions is irrelevant and the *relative size* across sessions is everything.
- **Interface.** Either a function `row -> float` run through `sessions.get_oos_predictions`, or any DataFrame of predictions passed to `sessions.write_submission`. Non-finite predictions are written as 0.
- **Libraries.** The notebook targets Google Colab; `sessions.py` needs `attrs`, `rich`, `tqdm`, `numpy`, `pandas`. Since only a CSV is uploaded, any offline tooling is usable. [HYPOTHESIS — no rule seen restricting tools]

### B.1 What the Arcade site's public client code shows [MEASURED from the served JavaScript, not from a challenge page]

- Upload check: header must be exactly `headline_id,pred`; two columns; every `pred` finite; no repeated `headline_id`; at least one row; file up to 2 MB. Our baseline file (10,000 rows, ~0.3 MB) passes all of these.
- CSV challenges are "Scored against hidden labels and ranked by Sharpe ratio", with an "N of M submissions used" counter and a list of earlier uploads with their metrics. The allowance M is set per challenge.
- Events can be team events (shared notebook, invite codes, name shown on the leaderboard) and can have timed rounds with a "Rules for this round" panel.
- Access is by a personal link from a recruiter, and the site has a built-in assistant described as: it "explains Python, pandas, numpy, errors and statistics. It cannot see your notebook or the data, and it does not help find patterns in the data. Your questions and its answers are saved and shown to the hiring team."

## C. Assumptions and unknowns

**Confirmed facts** — section B, for the public edition only.

**Hypotheses**
1. Our event is the same challenge, same data, same 49→59 target.
2. The leaderboard score is computed on all 10,000 out-of-sample sessions, and the final ranking uses that same score (no hidden private split).
3. Using out-of-sample *inputs* (first-half bars and headlines, no labels) to fit model parameters is permitted. The self-described winner did this.

**Unknowns that could change the strategy**
- Deadline and time budget.
- Whether ranking is by best upload, last upload, or something else — this changes how the 10 uploads should be spent.
- Individual or team; whether code must be submitted or explained; whether there is an interview/presentation stage.
- **Rules on outside material.** Two complete solutions and the organiser's data are public. Whether consulting them is allowed is a rules question only the official materials can answer. See I.1.

## D. Research

### D.1 Historical and comparable events

| Source | What it is | Reliability |
|---|---|---|
| `github.com/suspiciously-haunted/algo-arena` (single commit 2026-11-07; benchmark log dated 2026-10-20) | Describes itself as "Winning entry for the Hudson River Trading Algo Arena challenge"; reports "about 3.6" on the out-of-sample leaderboard. Contains organiser's notebook, `sessions.py`, all five data files, code, three prediction CSVs. | Participant repo. "Winning" is **self-described**; no official leaderboard found to verify rank. Code and data read directly. |
| `github.com/ABCPeter123/Quant-Comp-2026` → `algoarena/` (last commit 2026-11-22) | Second participant. Byte-identical data and `sessions.py` (SHA-256 match). Reports an official 2.564 for a TF-IDF/sentiment model and **3.564** for a different latent-company model (`model_3564.py`). Calls AlgoArena "the separate existing HRT project" next to a Waterloo Quant Club competition. | Participant repo. Scores are self-reported ("supplied by the user, not reproduced locally"). |
| `github.com/chr13b/synthetic-market-sharpe` (2026-06) | "2026 ETH Datathon" synthetic-market task: same 100-bar/headline format but hold **49→99**, score `mean/std·16`, returns **not** market-adjusted (a long bias dominated), public/private test split. | Participant repo. Link to HRT **not stated** in its README — treat as a comparable event, not a predecessor. |
| Web search for an official Algo Arena page, past leaderboards or a society announcement | Nothing found. | — |

Not relevant despite surfacing in search: Hack the Burgh VI trading bot (Optiver challenge), a Kaggle competition also named "Algo Arena" (unrelated tabular task), Qminers Algo Trade Arena.

### D.2 What the public solutions do

| Area | Self-described winner (`improved_all`, ~3.6) | Second entry (`model_3564`, 3.564) |
|---|---|---|
| Signal | Drift still owed at bar 49 by headlines from bars 40–49 | Same |
| Maths | Per event type, one drift rate; 11-bar kernel `[½,1×9,½]`; OLS on [owed drift, "surprise" of recent headlines, unexplained first-half move] | Latent variable: one of the session's companies *is* the stock. Posterior over which, from the likelihood of first-half returns; role coefficients own / same-sector peer / other ≈ 2.5 / 0.6 / 0.3 fitted by EM |
| Company structure | 5 industries found by spectral clustering of co-occurrence; keep only main-industry headlines | Sector read from the company-name suffix |
| Extra features | Headline count, repeated-company news, conflicting news, same-industry company count | Deviation of first-half move from model |
| Sizing | Divide by sqrt of a predicted variance (linear model on Garman–Klass and gap variances); clip at 2 sd | Divide by Garman–Klass vol; clip at 2%/98% quantiles |
| Semi-supervised | Drift rates estimated on in-sample **plus out-of-sample first halves** | No |
| Validation | 5-fold × 3 seeds, block splits, headline dropout, noise, bootstrap | Inner out-of-fold features for the second stage |
| Reported not to help | Gradient boosting, past price moves, headline words, company identity dummies, nearest-path matching | TF-IDF / sentiment (2.56 on the leaderboard) |

**Transferable principles** (hold for any edition of this format): reverse-engineer the generator rather than fit a black box; text matters only as a *category label*; the edge is in knowing *which* headlines move *this* stock; size by predicted risk.
**Edition-specific** (re-derive, never assume): kernel length, event vocabulary, sector map, role coefficients, horizon.

### D.3 What we verified ourselves [MEASURED]

| # | Check | Result |
|---|---|---|
| V1 | Data shape and label definition | As in section B. 4 companies per session (9,577 of 10,000 show all four within 100 bars). 80 companies. |
| V2 | Event taxonomy | 302 raw templates collapse to **50 event types** once numbers and slot fillers are masked. |
| V3 | Response kernel (event study, mean bar return by lag −3…+15) | Pooled over strong types, ×1e-4 per bar: lag 0 = 1.64, lags 1–9 = 2.9–3.1, lag 10 = 1.59, lags ≥11 ≈ 0.1 = background. **Half weight at lags 0 and 10, flat between, nothing outside.** No pre-announcement drift. |
| V4 | Sector outsiders | In sessions with 3 same-sector companies + 1 outsider (62% of sessions), the outsider's headlines have response **0.06** vs **1.30** for the majority (pooled amplitude = 1). |
| V5 | Latent company identity | Per (session, company) response estimated on alternate headlines: split-half correlation **0.18** (n=12,734 pairs, ≥4 headlines), 0 expected if all companies mattered equally. Response is heterogeneous *within* the majority sector. |
| V6 | Score noise | Bar-return sd 0.0027; 10-bar target sd 0.0093. Bootstrap 95% half-width of a 10,000-session Sharpe ≈ **0.28** (s.e. ≈ 0.15). [PUBLIC-CLAIM, from the winner's logged bootstrap; consistent with theory] |

## E. Architecture and decisions

**Objective, derived from the score.** With conditional mean μᵢ and conditional variance σᵢ² of session *i*'s return, the Sharpe of Σ pᵢrᵢ is maximised (to first order) by

    pᵢ ∝ μᵢ / (σᵢ² + μᵢ²)

So the task splits cleanly into (1) the best estimate of μᵢ and (2) an estimate of the conditional second moment. Position is *not* proportional to the forecast: large forecasts are damped and noisy sessions are down-weighted. This is the principled version of the clipping and vol scaling both public entries found empirically.

**Intended architecture — NOT yet committed to; pending confirmation of the competition and an independent challenge (section H).**
1. Generative model of a session: latent target company, role-dependent headline amplitudes, 11-bar kernel, heteroskedastic noise.
2. Posterior over the latent company from first-half returns and sector structure → μᵢ as a posterior-weighted owed drift.
3. Variance model from OHLC range estimators, posterior uncertainty about the latent company, and expected second-half news.
4. Position from the formula above.

**Rejected so far**
- Sentiment / TF-IDF / language models on headline text: headlines are templated; text carries nothing beyond the event type (public leaderboard 2.56 for that route vs 3.04 for one structural feature).
- Black-box learners on price paths: deprioritised, not disproven. V3 shows headlines have no effect before publication or after lag 10; that past price moves, boosting and path matching add nothing is a [PUBLIC-CLAIM] we have not tested.

**Language/tooling.** Python with numpy/pandas/scipy; 8 s for a 15-fit cross-validation. No reason to consider anything else.

## F. Implementation status

| Component | Status |
|---|---|
| `arena/core.py` — loading, headline parsing, exact score, leak-safe CV harness, upload writer | Done, run on the public data |
| `arena/baseline.py` — pending-drift model (one feature) | Done — **current working version and fallback** |
| Latent-company model, variance model, sizing | Not started |
| Data | **Not in this repo.** Put the organiser's five parquet files in `data/` (git-ignored) or set `ARENA_DATA`. |

Leak guard: the CV harness hands `predict` a block that physically contains only bars 0–49, headlines ≤ bar 49 and no labels. Self-reviewed; not yet independently audited.

## G. Experiments and results

| ID | Hypothesis | Configuration | Result | Decision |
|---|---|---|---|---|
| E1 | The drift still owed by bars-40–49 headlines, with one amplitude per event type, is most of the available signal | `arena.baseline`, 5-fold CV × 3 seeds, public data, fit on full sessions of the training folds only | Sharpe **3.039** (seeds 3.050 / 3.043 / 3.025); corr 0.21; individual 2,000-session folds range 2.40–3.78; fitted β = 1.013 on all data (forecast is calibrated) | Keep as floor and fallback |

Reference points, not ours: starter keyword idea 0.57, base drift model 3.04, best logged CV 3.50 [PUBLIC-CLAIM]; leaderboard ~3.6 and 3.564 [PUBLIC-CLAIM].

**Reading results correctly.** A single 10,000-session score has s.e. ≈ 0.15. Differences between models must be judged on *paired* per-session PnL on identical folds, never by comparing two headline Sharpes — and 3.56 vs 3.6 on the leaderboard is not a real difference.

## H. Advisor feedback

No `/advisor` command or tool exists in this session, so none has been used. Substitute, clearly labelled as such: an independent fresh-context agent given the specification and data but **not** our conclusions or the public solutions, asked to reverse-engineer the generator and propose a strategy. Not yet run — scheduled for immediately after the competition identity is confirmed (consultation point 1).

## I. Auditor feedback and open risks

No independent audit has been run yet. Open risks:

1. **Integrity / rules.** A public repo contains a ready-made upload file scoring ~3.6. Uploading someone else's predictions is not something this project will do. Whether *reading* public solutions is within the rules is unconfirmed; all code in this repo is our own, but the structural ideas in D.2 were seen there first, and D.3 records which of them we then established independently from the data.
   **Raised 2026-11-24:** the platform is a recruiting tool (personal recruiter links, a "hiring team" that reviews assistant conversations) and its own assistant is deliberately barred from helping "find patterns in the data" (B.1). That is evidence about the organiser's intent, not a rule we have read. **No further modelling until the "Rules" text for this challenge has been seen.** If it forbids outside help with the modelling, this project stops at explanation and review of the user's own work.
2. **Wrong-competition risk.** Section C, hypothesis 1.
3. **Leaderboard overfitting.** 10 uploads against a score with s.e. 0.15; picking the max of several near-identical models buys noise, not skill, if the final ranking uses different data.
4. **Vocabulary drift.** The slot lists in `arena/core.py` are from this edition. Unknown fillers degrade gracefully (types split rather than break); check that `Data.summary()` reports 50 event types on the official files.

## J. Remaining work

| Priority | Task | Expected benefit | Effort | Blocked on |
|---|---|---|---|---|
| 1 | Confirm the competition, rules, deadline against official materials | Decides whether everything here applies | Minutes | Official materials |
| 2 | Independent advisor consultation (section H) | Catches a wrong generative story before building on it | ~1 run | 1 |
| 3 | Latent-company / sector-aware exposure | Largest known step: the winner's log shows 3.04 → 3.37 CV from industry filtering alone [PUBLIC-CLAIM] | Half a day | 1–2 |
| 4 | Variance model + `μ/(σ²+μ²)` sizing | Small: the winner's log shows +0.02 to +0.04 from vol scaling [PUBLIC-CLAIM]; the exact formula is untested | Hours | 3 |
| 5 | Unify both public ideas in one likelihood; estimate ceiling with an oracle that knows the true company | Unknown — the open question is how close 3.6 is to the ceiling | 1 day | 3 |
| 6 | Semi-supervised amplitudes from out-of-sample first halves | Small; rules check needed | Hours | 1 |
| 7 | Upload plan for the 10 submissions | Avoids wasting uploads on noise | Minutes | Ranking rule |

**Next action:** obtain the official starter notebook / challenge page and compare with section B.
