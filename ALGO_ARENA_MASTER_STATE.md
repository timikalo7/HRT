# ALGO_ARENA_MASTER_STATE

_Last updated: 2026-11-24 — exploration + CV model done; two candidate CSVs written, **nothing uploaded**._

## A. Competition identity
- Name: Algo Arena (HRT) — "Predict how a simulated stock moves after its news" (`sessions.py` header: "Hudson River Trading -- Synthetic Market Prediction Challenge")
- Mode: **B — Quantitative trading** (cross-sectional return prediction on synthetic data)
- Organiser / sponsor: not stated beyond the HRT header in `sessions.py`
- Deadline: **UNKNOWN (asked user)**
- Platform: Arcade (upload CSV), Colab notebook `algoarena.ipynb`; runs locally
- Submission: CSV `headline_id,pred`, one row per OOS session (ids 11000..), **10 uploads total**; best-vs-last counting **UNKNOWN (asked user)**

## B. Confirmed specification (problem statement, algoarena.ipynb, sessions.py)
- Session = one synthetic stock, 100 bars, fictional headlines/companies. Prices start at 1.
- Visible: bars 0-49 OHLC + headlines in bars 0-49. Trade at close 49, hold to close 59.
- Target `returns` = close59/close49 - 1 minus cross-sectional mean (**verified exactly on IS**, simple not log).
- IS 10,000 sessions full 100 bars + returns; OOS 10,000 sessions bars 0-49 only.
- Score = sqrt(252) * mean(pred*ret) / std(pred*ret). Scale-invariant globally; non-finite pred -> 0.
- `write_submission(prediction_df=...)` accepts a vectorised DataFrame (index `session`, column `pred`) and validates ids.
- Data: Drive folder `1mYOtE9R-XtdcJ4cPVpQYAxiPLD1Dm4vX` -> local `data/` (sizes verified).

## C. Assumptions and unknowns
- Resolved: H1 (headline text carries signal: yes, via 50 stems), H2 (post-headline drift: yes, 10-bar constant drift), H3 (price features: only weak mean reversion + short-term continuation), H4 (timing: only headlines at bars 40-49 matter), H5 (companies recur across sessions and IS/OOS: yes, but company identity carries no signal).
- Unknown: deadline; public/private leaderboard split; best-vs-last upload; whether OOS demeaning uses OOS mean (assumed; centring makes this irrelevant).

## D. Research
- Historical Algo Arena / HRT competitions: **not researched** (deprioritised — the generator structure was recovered directly from data; historical strategies are unlikely to transfer to this synthetic set-up).

## E. Architecture and decisions (current)
- Python 3.12 venv, numpy/pandas/scipy; no ML libraries needed so far.
- **Model** (`scripts/pipeline.py`, `scripts/make_submission.py`):
  1. Headline -> stem: mask numbers, $ amounts, and filler vocabularies (products, regions, counterparties, titles, events), strip "Group"/"Co" company suffix -> 50 stems.
  2. Session vol: Parkinson (high-low) estimator on bars 0-49, calibrated to close-to-close scale; vs = sigma / 0.0026.
  3. Stem rates: ridge (lambda=400) least-squares deconvolution of vol-normalised bar returns on 11-bar drift kernels (half-full-half), all IS bars + OOS first-half bars.
  4. Features (normalised): g = sum over first-half headlines of rate x remaining units in 50..59 (w(b) = b - 39.5 for b>=40); n_first (headline count); cum_noise (de-drifted cumulative noise bars 0-49); [optional] noise5, n_live.
  5. OLS of y/vs on features (IS); pred = yhat / vs (since drift and noise both scale with sigma); **centred** over OOS.
- Rejected: template-level rates (fillers carry no info), lag-mean rate estimator (overlap bias), per-event Bayesian update (gain ~0), vol power 2, sentiment persistence, company effects.

## F. Implementation status
- Working: `scripts/model.py`, `scripts/pipeline.py`, `scripts/make_submission.py [n_cum|all|g_only]`.
- Candidate files (git-excluded): `predictions_n_cum.csv` (primary), `predictions_all.csv` (alternative). Both 10,000 rows, finite, centred, written via `sessions.write_submission`.
- Fallback: `make_submission.py g_only` (pure news drift).
- Research scripts: smoke_check, explore_headlines, event_study, event_study2, lag_profile, cv_baseline, experiments{,2..6}, cv_pipeline, cv_linear, oos_check.
- Git: nothing committed. `data/`, `scratch_*.pkl`, `predictions*.csv` in `.git/info/exclude`. `requirements.txt` pinned (modified, uncommitted).

## G. Verified data facts
- returns std 0.93%, excess kurtosis 0.43 (not fat-tailed).
- 80 fictional companies (40 sector words x 2, some with "Group"/"Co" suffix), ~4 per session, all 80 in both IS and OOS; ~17.4 headlines/session; arrival rate falls from 0.20/bar (early) to 0.105/bar (bars 90-99).
- 50 stems in three clusters: negative (-2.5..-3.9 bps/bar), neutral (-0.7..+0.6), positive (+1.6..+4.0). Rate SE ~0.14 bps for large stems.
- **Drift kernel**: a headline at bar b adds constant drift at lags 0..10 (half weight at 0 and 10). Nothing before, nothing after.
- Drift **scales with session vol** (slope 0.74 -> 1.31 across vol quintiles, pooled 1.01).
- Per-event drift multiplier varies a lot (implied sd ~0.85x of mean for non-neutral) but **nothing observable explains it** (company chi2 100/79, sector 49/39, amount 1.7/3, timing 12/8, mention count ns).
- No generic momentum on quiet bars (autocorr ~0). De-drifted cumulative noise mean-reverts weakly (corr -0.04 with next 10 bars). n_first vs n_future(50-59) corr -0.09.
- **OOS transfer** (oos_check.py): IS rates explain OOS first half with slope 1.012 +- 0.012 (IS 0.996), identical lag profile, identical feature distributions, 0 unseen stems/templates/companies.

## G2. Experiments (5-fold OOF on IS, 2-3 seeds; z = bootstrap z of Sharpe difference)
| # | Config | OOF Sharpe | Notes |
|---|---|---|---|
| E0 | notebook keyword ("secures" - "decline") | 0.57 | reference |
| E1 | template lag-mean rates x w(b), raw | 2.48-2.59 | |
| E2 | E1 + shrink n/(n+50) | 2.61-2.63 | |
| E2c | E2 centred | 2.85-2.87 | centring = pure variance reduction |
| E5 | deconvolved template rates, ridge 400, centred | 2.96-2.98 | paired t +2.5 vs E2c |
| E6 | **stem** deconv ridge 400 | 3.05 | t +2.7..+3.5 vs E5; fillers irrelevant |
| E7 | + vol-normalised rates, pred = g/sigma (Parkinson) | 3.08 | z ~+1.0 (small: sigma CV only 0.2); p=2 hurts (-2.0) |
| E8 | + per-event Bayesian update (c=0.8) | 3.08 | no gain; c=1.5 hurts |
| E9 | + n_first + cum_noise (OLS, OOF) | **3.13-3.14** | z +2.2..+2.5 vs E7 — **primary (n_cum)** |
| E10 | + noise5 + n_live | 3.15-3.17 | z vs E9 ~+0.3-0.5 (noise) — alternative (all) |
- Full-fit in-sample Sharpe: n_cum 3.18, all 3.22 (vs OOF 3.14 / 3.16 -> little overfit).
- Expected OOS score ~3.1 +- 0.16 (1 SE), if the OOS generator matches (supported by oos_check).

## G3. Session 2 (2026-11-24 ~19:00-19:17 UTC): beat-3.8 push + Timmy's branch
- Exposure test (`scripts/exposure_test.py`): within-session same-company Cov(m1,m2) = +0.756 (se 0.04), different-company -0.291 (se 0.02), Var(m) 1.53 -> exposures differ by company within a session and sum to ~constant.
- Our Gaussian exposure posterior (`scripts/exposure.py`, `cv_exposure.py`): 3.14 -> **3.26** (+n+cum; z +2.2..+3.9, seed 0).
- Teammates' branches (read-only worktrees in `..\HRT_branches\`): **zen = Timmy's (top-two)**, CV 3.59 (c3) / 3.60 (c5 three-way blend). Its structure explains our exposure result: 5 sectors x 16 companies; session = 3 same-sector + 1 any; one same-sector company is the stock (weight 1), peers ~0.25, odd-sector ~0. Built on public ~3.6 GitHub solutions. tender = CV 3.40; origin/main (Rohit) = CV 3.24.
- Tested our one missing finding on Timmy's model (`scripts/zen_volscale.py`, identical folds, seed 0): mu x relative range-vol -> **3.5911 vs 3.5889** (fold range 3.14-3.93 vs 3.11-3.95). **No meaningful gain.** Files written anyway: `predictions_zen_c3_vs.csv`, `predictions_zen_c5_vs.csv` (corr 0.9956 with c5) — not recommended over c5.

## H. Advisor feedback
- Consultation 1 (after spec extraction): all accepted (Drive connected; raw sessions.py; ask before download; centre; OOF only; first checks).
- Consultation 2 (after baseline, OOF 2.61):
  1. Centre preds — **done, +0.24**.
  2. Overlap bias in lag-mean rates -> deconvolution — **done, +0.1 (t +2.5)**; residual n-headline effect partly remained -> explained by n_first/n_future correlation, added as feature.
  3. Pool templates by stem — **done, +0.08 (t +3)**.
  4. OOS first-half transfer check + coverage + use OOS events in rate fit — **done; transfer confirmed; OOS events included in final fit**.
  5. Paired/bootstrap comparisons instead of eyeballing — **done**.
  6. Realised drift vs momentum; drift vs vol — **done**: no generic momentum; drift ∝ vol; per-event update useless.
  7. Housekeeping + fallback CSV — **done**.

- Consultation 3 (before handing over candidates): **advisor tool unavailable** — not used. Replaced by the internal adversarial check below.

## I. Auditor feedback
- **Internal adversarial review (NOT an independent auditor; advisor unavailable), 2026-11-24:**
  - Leakage: features use bars <= 49 and headlines <= 49 only (`features`, `sigma` slice [:, :50]); IS second half used only to fit rates (allowed). OK.
  - End-to-end pseudo-OOS (`scripts/pseudo_oos.py`): held-out fold truncated to first half, its events added to the rate fit as in the real OOS, coefficients from the train fold, centred on test. Pooled Sharpe: g_only 3.08/3.09, **n_cum 3.15/3.15**, all 3.17/3.18 (2 seeds) — matches OOF, so the submission path has no hidden leak or misalignment. Per-fold (2k sessions) 2.6-3.8 shows single-score noise.
  - ID alignment: preds built in sorted-session order; `write_submission` validates the id set and reorders. OK.
  - Outstanding: no truly independent audit yet; OOS demeaning convention unverifiable (centring makes it irrelevant).

## J. Remaining work
1. User: deadline; best-vs-last upload rule; decide first upload (recommend `predictions_n_cum.csv`).
2. Independent audit of the pipeline (leakage, ID alignment, centring).
3. Possible small gains (each likely <= +0.05): joint deconvolution of event multipliers, better sigma (combine cc+range), nonlinear combination (LightGBM on normalised features) — low expected value given SE 0.16.
4. Upload strategy: first upload = primary; use the result only as a sanity check (SE 0.16), not for tuning.
