# ALGO_ARENA_MASTER_STATE

_Last updated: 2026-11-24 — starter notebook + data received; awaiting problem-statement screenshot._

## A. Competition identity
| Field | Value | Status |
|---|---|---|
| Competition name | Algo Arena — "Hudson River Trading – Synthetic Market Prediction Challenge" (header of sessions.py) | Confirmed (starter code) |
| Operating mode | **B — quantitative trading** (return prediction scored by Sharpe), with an NLP flavour | Confirmed |
| Organiser | — | Unknown |
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

**Unresolved questions:**
- Q1: Deadline.
- Q2: Any rules on the screenshot/problem page not in the notebook (team size, allowed external data/LLMs, tie-breaks, whether final score = best or last upload).
- Q3: Is the leaderboard OOS set the full 10k, or a public/private split?

## D. Research
_Not started — research is scoped to the event once identified, to avoid conflating similarly named events._

## E. Architecture and decisions
_None yet._

## F. Implementation status
- `notebooks/algoarena.ipynb` (starter), `data/sessions.py` (harness), data parquet files in `data/` (gitignored; re-download from Drive folder 1mYOtE9R-XtdcJ4cPVpQYAxiPLD1Dm4vX via the Drive connector — drive.google.com is blocked by the container proxy so gdown fails).
- Harness verified locally: INS (10000×7) and OOS (10000×6) load.
- Working version: none. Fallback: none.

## G. Experiments and results
| # | Hypothesis | Change | Baseline | Result | Decision |
|---|---|---|---|---|---|

## H. Advisor feedback
_No consultations yet._ Note: no dedicated `/advisor` tool is exposed in this session; independent review will be done with separate review agents (clearly labelled) unless a real advisor becomes available.

## I. Auditor feedback
_No audits yet._

## J. Remaining work
| Priority | Task | Benefit | Effort | Deadline risk |
|---|---|---|---|---|
| 1 | Receive problem-statement screenshot (deadline, extra rules) | Completes spec | — | — |
| 2 | Build fast vectorised backtester + K-fold CV on INS (avoid the slow per-row apply for research) | Measurement | Low | — |
| 3 | First advisor checkpoint on the spec | Catch misreadings early | Low | — |
| 4 | Tier-1 baseline + evaluation harness | Guarantees a valid submission | Low–Med | — |

**Next action:** read screenshot, then the full initial analysis + baseline.
