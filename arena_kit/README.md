# Algo Arena starter kit (HRT x CompSoc x QuantSoc, 24 Nov 2026)

Built around the format of the previous HRT Algo Arena. Tonight's task may differ, so check the rules first.

## Setup (do before 18:00)
```
pip install pandas numpy scikit-learn pyarrow
python make_fake.py
python baseline.py --data fake_data
```
Seeing a CV table and `submission.csv` means the kit works.

## First 10 minutes with the real data
1. Put the files in `data/`. The loader matches the file names `*ins_bars*`, `*ins_headlines*`, `*ins_returns*`, `*oos_bars*` and `*oos_headlines*`. Parquet and CSV both work.
2. Check the column names against `CFG` at the top of `kit.py`. Fix `session`, `bar`, `headline`, `target`, `sub_id`, `n_seen` and `horizon` as needed.
3. Check that `kit.score` matches the official metric. The previous metric was `sharpe = mean(pred*ret)/std(pred*ret)*sqrt(252)`.
4. Run `python baseline.py --data data`. It runs CV on every model, refits the best one and writes `submission.csv`.

## Real-data results (24 Nov 2026)
Best model: `python model2.py --submit "+seen"` (CV Sharpe 3.20 ± 0.27), which writes `../predictions.csv`.
- Each headline type pushes the price at a fixed rate (about ±3 bps per bar) for 10 bars, starting the bar after it appears.
- The numbers inside headlines ($ amounts, %) do not change the effect, and every company in a session counts the same.
- Helped: coarse types (`type_short`), a 9-bar window, vol sizing to the power 1.25, and estimating rates from all visible paths, including oos bars 0-49.
- Did not help: a fitted lag profile, residual-vol or variance sizing, or snapping types to shared rates.
- Ceiling: headlines after bar 49 hold the rest of the signal (with perfect foresight the score is about 4.1), and they are unpredictable from past news.
- `explore.py` and `explore2.py` hold the research and `tune.py` the CV grid.

## Files
| file | what |
|---|---|
| `kit.py` | load, `score`, `report`, `cv(fit_predict, y)`, `write_submission`, `clip_sd`, `vol_size` |
| `features.py` | price features; headline parsing (company split, number masking into types); `fit_rates`; `drift_features` (owed / done / unexplained); `target_type_scores`; `session_text` |
| `baseline.py` | model ladder with CV and submission. Flags: `--type type_short`, `--lmax`, `--alpha`, `--clip`, `--power`, `--only`, `--model` |
| `make_fake.py` | synthetic data with a planted headline signal |

## Adding a model
In `baseline.py`, add an entry to `M`: `M["mine"] = lambda tr, te: <preds for te>`. Here `tr` and `te` are row indices; in-sample rows come first, then out-of-sample rows. Fit using only `tr`. Use `fit_lin(X, tr, te, sized=True)` for ridge with clipping and vol sizing.

## What won last time (claimed Sharpe about 3.6)
- Each headline type moves the price at a fixed rate for about 10 bars. Mask numbers and names to get the type, then fit the per-type rate from full in-sample paths.
- Features: drift still owed into the target window, plus the move the headlines don't explain.
- Fit a linear model, then clip at 2 SD.
- Use only headlines from the session's main industry.
- Size positions by 1/sqrt(predicted vol).
- These did not help: gradient boosting, momentum, word features, company identity.

## Time plan (about 2 hours of work)
1. 0–15 min: load the data, run the baseline, make the first submission.
2. 15–60 min: find the main signal.
3. 60–90 min: features, then sizing and clipping.
4. 90–110 min: tune on CV, not the leaderboard, because uploads are limited.
5. Last 10 min: final submission.
