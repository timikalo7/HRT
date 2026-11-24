# HRT Algo Arena — read this first

Any Claude session picking this repo up: read `ALGO_ARENA_MASTER_STATE.md` (decisions, experiments, open risks) and `STRATEGY.md` (plain-language explanation of the approach) before doing anything. The operating procedure the owner wants followed is the `algo-arena` skill in `.claude/skills/algo-arena/SKILL.md`.

## The task in two lines

Simulated sessions: one stock, 100 bars, templated fictional headlines. Given bars 0–49 and their headlines, predict the return from the close of bar 49 to the close of bar 59; score is `sqrt(252) * mean / std` of `pred * return` over 10,000 hidden sessions. Upload is a CSV `headline_id,pred`; **10 uploads only**.

## Where things are

| Path | What |
|---|---|
| `arena/core.py` | Loader, headline parser (company, event type), exact score, leak-safe cross-validation, upload writer |
| `arena/baseline.py` | One-feature fallback model (3.04 cross-validated) |
| `arena/latent.py` | Hidden-target model (3.49 cross-validated) |
| `arena/blend.py` | Blend upload files at unit scale |
| `research/` | Scripts behind the numbers in the master state |
| `submissions/` | Candidate upload files (committed so another session can upload them) |
| `data/` | **Not committed.** Put the five organiser parquet files here (`ins_bars`, `ins_headlines`, `ins_returns`, `oos_bars`, `oos_headlines`) or set `ARENA_DATA` |

## Commands

```bash
pip install -r requirements.txt
python -m arena.baseline cv            # ~10 s
python -m arena.latent cv 0 1          # ~1 min; expect about 3.49
python -m arena.latent write submissions/predictions_c1_latent.csv
python -m arena.blend submissions/predictions_c2_blend.csv 0.5:submissions/predictions_public_improved_all_rerun.csv 0.5:submissions/predictions_c1_latent.csv
```

`submissions/predictions_public_improved_all_rerun.csv` is a re-run of the `improved_all` model from the public repo `github.com/suspiciously-haunted/algo-arena` (clone it next to this repo and see `research/pub_run.py`). It is a blend component, not our own work.

## Rules of the road

- Never let a model see bars after 49, headlines after bar 49, or labels at prediction time. `cross_validate` enforces this by construction; keep it that way.
- Compare models on identical folds (`np.random.default_rng(seed).permutation(n) % 5`). One 10,000-session Sharpe has a standard error of about 0.15, so small differences between two separate numbers mean nothing.
- Do not upload anything that has not been through the checks listed in the master state. Uploads are scarce.
- Keep `ALGO_ARENA_MASTER_STATE.md` current: every experiment, kept or rejected.
