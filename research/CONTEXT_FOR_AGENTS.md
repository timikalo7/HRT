# Context for research agents — HRT Algo Arena

Read this first. It is the shared factual base; your own task prompt says what to do with it.

## 1. The task

Simulated data. Each session = one synthetic stock over 100 bars, with templated fictional news headlines. At prediction time you may use only bars 0–49 (open/high/low/close; price starts at 1; 4 decimals) and headlines with `bar_ix <= 49`. Predict the market-adjusted return `close[59]/close[49] − 1 − mean`. A trader takes position = prediction; score = `sqrt(252) * mean(pnl) / std(pnl)`, `pnl = position * return`, over 10,000 hidden out-of-sample sessions. Only relative size across sessions matters. No costs.

Data (`/home/user/HRT/data/`, read-only): 10,000 in-sample sessions (ids 1000–10999) with all 100 bars, all headlines and labels; 10,000 out-of-sample sessions (ids 11000–20999) with bars 0–49 only.

## 2. What is established about the generator (measured on in-sample data)

1. **Templates.** 80 companies; 50 event types once numbers and slot words (segment, region, counterparty, conference, executive title) are masked. `arena/core.py` parses each headline into `company` and `ev`.
2. **Kernel.** A headline moves the price at a constant per-bar rate over lags 0..10 (half weight at lags 0 and 10, full weight at lags 1–9), nothing before or after. Each event type has its own amplitude (about 15 strongly positive, 15 strongly negative, 20 near zero).
3. **Sectors.** Exactly 5 sectors of 16 companies (recovered from co-occurrence; `arena.latent.all_sectors`). Every session has 3 companies from one "main" sector plus 1 drawn from anywhere (other-sector 83.5%, same-sector 16.5%). Other-sector headlines have no effect (weight −0.02 ± noise).
4. **Hidden target.** Within the main sector, responses are heterogeneous: an EM with "exactly one company is the stock (weight 1), same-sector peers weight w_p" gives w_p ≈ 0.25–0.28. The target is weakly identified: mean max posterior 0.65–0.67 even using all 100 bars. At temperature 1 the bar-49 posterior is over-confident and costs Sharpe; temperature 4 is used (flat between 2 and 8).
5. **Noise.** Per-bar return sd about 0.0027, varying by session (1%–99%: 0.0017–0.0037). 10-bar label sd 0.0093. Garman–Klass range variance of bars 0–49 is a much better per-session volatility estimate than close-to-close sd.
6. **Reversal.** The part of the first-half move not explained by headlines predicts the label with a negative slope (about −0.014 to −0.025 per unit of log move). Worth +0.07 Sharpe in cross-validation.
7. **Arrivals.** About 17.4 headlines per session; arrival rate falls from ~0.20/bar at the start to ~0.10/bar at the end; more early headlines → fewer later ones.

## 3. Current model and scores

`arena/latent.py` — `LatentTarget`: EM for amplitudes and peer fraction on full training sessions; at bar 49 a posterior over which seen main-sector company is the target (or "not seen yet"); forecast feature `mu` = posterior-weighted drift still owed by headlines from bars 40–49; second-stage least squares on `[mu, headline count, unexplained first-half move, live main-sector headlines, 1]`; position = `f / (15 * range_variance + f^2)` (Sharpe-optimal sizing).

| Model | 5-fold CV Sharpe (seeds 0 / 1) | Leaderboard |
|---|---|---|
| One feature (pending drift, no company information) | 3.04 | — |
| `LatentTarget` without second stage | 3.49 | implied ≈ 3.59 |
| `LatentTarget` (the baseline in `research/harness.py`) | 3.589 / 3.592 | implied ≈ 3.65 |
| + continuous-exposure feature + positive part of mu (`research/final_variant.py`) | 3.611 / 3.613 | not uploaded |
| Public `improved_all` regression model (other people's code) | 3.483 / 3.463 | implied ≈ 3.63 |
| Best uploaded blend (70% a variant of ours + 15% + 15% two public models) | 3.61 | **3.699** |

Leaders on the board are at about **3.8**. A re-blend of existing files cannot beat 3.699 (leaderboard forensics, `research/lb_forensics.py`: best achievable ≈ 3.70). One 10,000-session Sharpe has a standard error of about 0.15, but two models that are 99% correlated differ by a standard error of only about 0.01–0.02, so always compare on identical sessions.

Caution: cross-validation gains from the second stage were +0.10 but the leaderboard-implied gain was only about +0.06. Treat small cross-validated gains (< 0.02) as noise.

## 4. Tested and closed — do not repeat

No effect or no gain: sentiment / TF-IDF / word features; numbers inside headlines; slot words (segment, region, counterparty) as main effects, as company or sector attributes, or as a session-level "relevance" attribute; event type as a marker of the target; company-specific volatility, drift or strength; pair-specific peer effects (weak at most); gradient boosting on the baseline features; recent price momentum (1/5/10 bars); "surprise" of recent headlines; splitting the unexplained move by window; fitted variance models (≤ +0.006); semi-supervised EM using out-of-sample first halves (+0.001 to +0.006); forecast-level vs position-level blending; about 30 extra second-stage features (all within ±0.01, except positive part of mu and a ridge exposure feature at +0.01 each).

## 5. Open evidence that the current model is wrong in a specific way

- **Exposure structure.** For main-sector headline pairs in the same session with non-overlapping response windows, the product of their normalised responses averages **+0.80 (se 0.05) for the same company** and **−0.23 (se 0.03) for different companies**. "Exactly one target at weight 1, peers at a fixed 0.25" predicts +0.80 and about **−0.37**. The same-company variance fits; the cross-company anticorrelation is only about 60% of what an exactly-one-target model implies. This points to a per-session, per-company random exposure component (peer weights that vary by session, or exposures that are not sum-constrained). A model with the right exposure distribution should give a calibrated posterior without the temperature fudge.
- **Over-confidence at temperature 1** says the likelihood is misspecified: exposure structure (above), noise that is not i.i.d. Gaussian (the reversal in §2.6 suggests mean-reverting noise), or both.
- **Diagnostic only, not a usable model:** computing the target posterior from all 100 bars (which includes the label window) gives Sharpe 4.2. It shows the headroom is in knowing the exposures, not in more correction features.

## 6. How to evaluate (mandatory)

Use `research/harness.py` (import with `sys.path.insert(0, '/home/user/HRT')`):

- `run(make_model, tag, outdir)` — any model with `fit(train_block)` / `predict(test_block) -> positions`. Cross-validates seeds 0 and 1 on the project folds, prints Sharpe against the cached baseline with a paired bootstrap, runs the look-ahead test, fits on all in-sample sessions and writes out-of-sample positions. Saves `<tag>_oof_seed0.npy`, `<tag>_oof_seed1.npy`, `<tag>_oos.npy`, `<tag>.json`. About 3 minutes.
- `run_variant(VariantCls, tag, outdir)` — for subclasses of `arena.latent.LatentTarget` that only override `_features` and/or `predict` (see `research/final_variant.py` for the pattern). Reuses cached EM fits; about 30 s.
- `get_base(seed, fold)` / `get_base()` — the cached fitted baseline for a training fold / for all in-sample sessions (attributes `amp`, `peer`, `types`, `mu0`, `beta`, `sectors`). `data()` returns `(Data, sectors)`; `folds(n, seed)`; `baseline_oof(seed)`; `paired(new, base, returns)`.

Rules:
- Test blocks are built only by `Block.take(idx, first_half=True)`; never read bars after 49, later headlines or labels at prediction time. Every parameter is fitted on training folds only. Using out-of-sample *inputs* (first-half bars and headlines, no labels) for unsupervised fitting is allowed but must be stated.
- The look-ahead test must return exactly 0.0.
- An improvement counts only if it is positive on **both** seeds and at least +0.02 on each. Report per-seed numbers; do not report the best of several tries without saying how many you tried.
- Do not edit anything under `/home/user/HRT`. Work only in your own directory under `/tmp/claude-0/-home-user-HRT/1b646c27-9e64-5f9e-b4f8-43257d3086ff/scratchpad/wf/`.
- The machine has 4 cores shared with other agents: set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before importing numpy. Python 3.13 with numpy, pandas, pyarrow, scipy, scikit-learn. Do not search the web.
- Save results to disk after each step so partial work survives.
