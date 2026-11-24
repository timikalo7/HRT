"""5-fold CV of the news-drift baseline (template rates estimated on train folds only)."""
import numpy as np
import pandas as pd

from model import bar_returns, folds, load, news_feature, sharpe, template_rates

d = load("ins")
R = bar_returns(d)
y = d.returns
S = len(d.sessions)

# reference: notebook keyword predictor
kw = np.zeros(S)
first = d.h[d.h["bar_ix"] <= 49]
np.add.at(kw, first["si"].to_numpy(),
          first["headline"].str.contains("secures").astype(float).to_numpy()
          - first["headline"].str.contains("decline").astype(float).to_numpy())
print(f"notebook keyword predictor (no fitting): Sharpe {sharpe(kw, y):.3f}")

for shrink in (0.0, 50.0, 200.0):
    for seed in (0, 1, 2):
        fid = folds(S, 5, seed)
        oof = np.zeros(S)
        for k in range(5):
            tr = fid != k
            rates = template_rates(d, R, tr, shrink=shrink)
            f = news_feature(d, rates)
            oof[~tr] = f[~tr]
        per_fold = [sharpe(oof[fid == k], y[fid == k]) for k in range(5)]
        print(f"shrink={shrink:5.0f} seed={seed}: OOF Sharpe {sharpe(oof, y):.3f}  corr {np.corrcoef(oof, y)[0,1]:.4f}"
              f"  folds {np.round(per_fold, 2)}  frac_nonzero {(oof != 0).mean():.3f}")
