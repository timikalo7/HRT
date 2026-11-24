# ALGO_ARENA_MASTER_STATE

_Last updated: 2026-11-24 — problem statement received; v0/v1 built and CV'd; submission files ready; independent audit running._

## A. Competition identity
| Field | Value | Status |
|---|---|---|
| Competition name | Algo Arena — "Hudson River Trading – Synthetic Market Prediction Challenge" (header of sessions.py) | Confirmed (starter code) |
| Operating mode | **B — quantitative trading** (return prediction scored by Sharpe), with an NLP flavour | Confirmed |
| Organiser | HRT ("Algo" team), hosted on Arcade | Confirmed (problem page) |
| Sponsor | Hudson River Trading | Confirmed (sessions.py header) |
| Deadline | — | Unknown |
| Platform | "arcade" challenge page; live OOS leaderboard | Confirmed (notebook) |
| Submission requirements | `predictions.csv` with columns `headline_id,pred` (headline_id = session number), all 10,000 OOS sessions, written by `sessions.write_submission`. **Max 10 uploads.** | Confirmed (notebook + sessions.py) |

## B. Confirmed specification
Sources: `notebooks/algoarena.ipynb`, `data/sessions.py` (from the user's Drive).
- Each session = one synthetic stock, 100 bars; prices start at 1. Fictional headlines about fictional companies.
- Visible per session: OHLC for bars 0–49, headlines (and their bar) published in bars 0–49.
- Target: market-adjusted return close(49)→close(59) (mean across sessions subtracted; INS mean ≈ 0, std ≈ 0.0093, range ±0.042).
- Data: 10,000 INS sessions (with returns, plus full bars 0–99 and all headlines for research); 10,000 OOS sessions (bars/headlines 0–49 only).
- Interface: `f(row: pd.Series) -> float`; row has open/high/low/close (len-50 arrays), headlines (list[str]), headline_bars (list[int]). Non-finite → 0; exceptions abort.
- **Score: OOS Sharpe = sqrt(252) · mean(pred·r) / std(pred·r)** over sessions. Scale-invariant in pred; zeros allowed (a zero contributes 0 to mean but still counts in N and the std).
- INS data: ~9.9 headlines per session before bar 49 (1–25); 174k headlines total, 34k unique strings. Headlines mention several different companies per session.

## C. Assumptions and unknowns
**Confirmed facts:** see §B.

**Hypotheses (untested):**
- H1: Only headlines about the session's *own* company move its price; other companies' headlines are distractors. (Several companies appear per session.)
- H2: Headline effects are delayed/decaying, so recent headlines (near bar 49) matter more for bars 50–59.
- H3: Price path features (momentum/reversal, volatility) carry signal independent of news.
- H4: Because score is Sharpe of pred·r, sizing by 1/expected variance (e.g. 1/realised vol) can matter as much as the direction forecast.

**Hypothesis outcomes:**
- H1 REJECTED (EXP-01/05): headlines about all 4 companies move the stock equally; per-company slope split-half corr −0.07.
- H2 CONFIRMED (EXP-01): response is a 10-bar box (≈½ weight lag 0, ≈0.1/bar lags 1–9, ½ lag 10, 0 after). Only headlines in bars 40–49 matter for 50–59.
- H3 WEAK (EXP-02): 3–5 bar momentum corr ≈ +0.03 with residual; 50-bar reversal ≈ −0.03.
- H4 WEAK (EXP-04): 1/vol sizing adds ≈ +0.03–0.05 Sharpe; vol spread across sessions is small.

**Unresolved questions:**
- Q1: Deadline (not on the problem page).
- Q2: Does the leaderboard use the best or the last upload? Public/private split of OOS?
- Q3: Are external data / LLMs allowed? (Not needed so far: templates are a closed set of 428.)

## D. Research
- Data structure (measured): 80 fictional companies, 4 per session (95.8%), 428 distinct headline templates once the company name is stripped (numbers kept). Every OOS template and company appears in INS. ~9.9 headlines before bar 49 per session; ~1.95 in bars 40–49.
- Template impact is highly reliable: split-half corr of per-template 10-bar mean move = 0.91 (~370 events per template).
- Target = simple return C59/C49 − 1 minus the cross-sectional mean (verified to 1e-19).
- Historical research on past editions: not yet done. The challenge is a self-contained synthetic simulator, so outside strategies transfer poorly; given the size of the data-driven edge, priority stays on empirical modelling. Will look if time allows (low expected value).

## E. Architecture and decisions
- **Core: rank-1 impulse-response model** (`research/irf.py`): bar_logret = Σ_h a[template]·k[lag], fitted by ALS on all 100 bars of training sessions (≈1M bar observations rather than 10k targets). Forecast = Σ_h a_h·Σ_{u=50..59} k[u−t_h].
- **v1 second stage** (`research/model.py`): linear on [IRF forecast, 3-bar momentum, 50-bar log return]; demeaned; divided by Parkinson vol (bars 0–49).
- Language: Python/NumPy/SciPy (matches harness; whole-array code, CV in ~1 min).
- Rejected: per-company effects, template-specific response lengths (no reliable signal, EXP-05); "own company only" filtering (H1 rejected); 1/variance sizing (weaker than 1/vol, EXP-04).

## F. Implementation status
- `notebooks/algoarena.ipynb` (starter), `data/sessions.py` (harness), data parquet files in `data/` (gitignored; re-download from Drive folder 1mYOtE9R-XtdcJ4cPVpQYAxiPLD1Dm4vX via the Drive connector — drive.google.com is blocked by the container proxy so gdown fails).
- Harness verified locally: INS (10000×7) and OOS (10000×6) load.
- `research/run.py cv|submit v0|v1`: one code path for CV and submission.
- **Working version: v1** → `submissions/v1.csv` (validated by `sessions.write_submission`).
- **Fallback: v0** → `submissions/v0.csv`; commit df0f1de (tag push blocked by proxy).
- Uploads used: 0 / 10.

## G. Experiments and results
All CV = 5-fold by session on INS; Sharpe = mean fold Sharpe unless stated.

| # | Hypothesis | Change | Baseline | Result | Decision |
|---|---|---|---|---|---|
| 00 | Starter "stocks go up" | pred = 1 | — | ≈ 0 (per notebook) | reference |
| 01 | Headline templates drive returns with a delayed response | IRF model | 0 | **2.97 ± 0.07** (3 seeds), corr 0.20 | keep (v0) |
| 02 | Residual has price signal | residual analysis | v0 | mom3 +0.031, rev50 −0.033, vol↔|res| 0.19 | feed into v1 |
| 03 | Per-instance impact noise is large | 10-bar move vs a | — | slope 1.00; residual var rises only ~5% with |a| | small; ignore |
| 04 | Sizing + price feats | OOF linear + 1/vol | 2.97 | 3.07 (Parkinson 1/vol) | keep |
| 05 | Company / duration heterogeneity | split-half reliability | — | corr −0.07 / 0.02–0.04 | reject |
| 06 | v1 end-to-end | `run.py cv v1` | v0 2.97 | **3.08 ± 0.06**, better than v0 on all 3 seeds | **current best** |

In-sample fit Sharpe vs CV: v0 3.09 vs 2.97; v1 3.20 vs 3.08 → small overfit gap.

## H. Advisor feedback
_No consultations yet._ Next: advisor-style challenge of the model ceiling and alternative approaches. Note: no dedicated `/advisor` tool is exposed in this session; independent review will be done with separate review agents (clearly labelled) unless a real advisor becomes available.

## I. Auditor feedback
- Audit #1 (independent agent, labelled internal review, not /advisor): running on v0/v1.

## J. Remaining work
| Priority | Task | Benefit | Effort | Deadline risk |
|---|---|---|---|---|
| 1 | Respond to audit #1 findings | Correctness | Low | — |
| 2 | Upload v1 once (1/10) to get an OOS anchor | Checks CV ↔ OOS agreement | Trivial | — |
| 3 | Shrink template impacts using numeric structure (e.g. $ size, %) | Small (+≈0.05?) | Med | low |
| 4 | Estimate the achievable ceiling (oracle a, k) to see how much headroom is left | Decides whether to keep researching | Low | — |
| 5 | Nonlinear second stage (GBM on residual) | Uncertain | Med | low |

**Next action:** audit response, then upload v1.
