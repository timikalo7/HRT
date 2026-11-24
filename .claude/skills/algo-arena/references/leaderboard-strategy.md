# Leaderboard and upload strategy

Read this before the first upload of any competition with a limited number of scored submissions.

## Contents
1. Questions to settle before uploading
2. The anchor upload
3. Spending the remaining uploads
4. Leaderboard forensics (what scores of known blends reveal)
5. The ledger
6. Reporting a file to the user

## 1. Questions to settle before uploading

| Question | Why it changes the plan |
|---|---|
| Does the board rank the **best** or the **latest** upload? | Best-of: every unused upload is a wasted option, so plan to use them all, and later uploads can be riskier. Latest-of: the final upload must be the safest file, and experiments go early. |
| Is the board computed on all hidden data, or a public subset with a private final ranking? | Public/private split: tuning to the public board overfits, so trust cross-validation. Board = final set: leaderboard-informed re-weighting is legitimate. |
| How many uploads, and is there a rate limit or cooldown? | Decides the schedule. |
| Is the score deterministic for a given file? | If yes, re-uploading the same file is pure waste. |

If the materials don't say, ask once, and meanwhile act as if it were latest-of with a private split (the safe assumption).

## 2. The anchor upload

Upload the first file that passes the gate early, within the first third of the window. It is worth more than any later upload:
- It **calibrates** the cross-validation → leaderboard offset. Here it was +0.06 to +0.12; it can be negative.
- It puts a score on the board in case later work breaks.
- It gives the user something concrete.

## 3. Spending the remaining uploads

- Rank candidates by **expected leaderboard score**: cross-validated Sharpe plus the calibrated offset. Prefer candidates that differ structurally (a new feature family, model or component); they carry information.
- Re-weightings of highly correlated files (corr > 0.99) move the score by about one leaderboard standard error of the *difference*: here roughly 0.01–0.02. They are lottery tickets, worth it only with best-of scoring and spare uploads.
- Keep one upload in reserve for the last minutes when a genuinely better model may land, unless scoring is best-of and the clock is nearly out.
- Never upload a file that has not passed the gate (format, leak test, reproducible cross-validation).

## 4. Leaderboard forensics

Each upload of a known linear blend of components gives one exact equation about the components' quality on the hidden set. With out-of-fold PnL streams `P` (sessions × components, each component scaled to unit sd) you have the covariance `C = cov(P)`. A blend with weights `w` scores

    Sharpe(w) = sqrt(ann) · (w · m) / sqrt(w' C w)

where `m` is the components' unknown mean PnL on the hidden set. Each leaderboard score therefore fixes `w · m`. With a few scores, solve for `m` using the in-sample `m0` as a prior (Gaussian update with prior covariance `C/n + tau² m0 m0'`), then:
- read off each component's implied hidden-set Sharpe (which components over- or under-perform out of sample);
- find the best non-negative blend under the updated `m`, and its predicted score.

`scripts/sharpe_tools.py::lb_forensics` implements this. In this competition it showed the ceiling of re-blending was 3.70, which correctly predicted that further blends would not reach 3.75. Use it to stop wasting uploads, not to chase noise. With a public/private split, use it only diagnostically.

## 5. The ledger

`scripts/ledger.py` keeps `submissions/LEDGER.csv`. Add a row when a file is created, with its SHA-256, composition, cross-validated Sharpe, expected leaderboard score and gate result. Record the leaderboard score **against the hash** the moment it arrives. A teammate session once fitted blends to a score attributed to the wrong file, and two uploads were wasted.

Keep the best confirmed file at a fixed path, `submissions/CURRENT_BEST.csv`, so it can be sent instantly when the user asks for it.

## 6. Reporting a file to the user

Lead with one line, then details:

> **Upload `c9_blend3_pos.csv`** — expected 3.70 ± 0.02 on the leaderboard (CV 3.61); gate passed (format, leak test 0.0, CV reproduced). P(beats current best 3.699) ≈ 40%. P(≥ 3.75) < 5%.

When the user sets a bar ("only send ≥ 3.75"), give P(clears the bar) for every candidate. If none clears it, say so plainly, and offer below-bar files only as optional extra tickets, and only under best-of scoring.
