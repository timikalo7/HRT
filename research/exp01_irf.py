"""EXP-01: impulse-response headline model, 5-fold CV on INS sessions."""
import numpy as np, pandas as pd
from lib import load, sharpe, folds
import irf

piv, hl, sess, ret = load("ins")
close = piv["close"]
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}
hl["tid"] = hl.tmpl.map(tm)
n = len(sess)
pred = np.zeros(n)
for f, te in enumerate(folds(n)):
    tr = np.setdiff1d(np.arange(n), te)
    a, k = irf.fit(close, hl, tr, len(tm))
    pred[te] = irf.forecast(hl, te, a, k)
    print(f"fold {f}: sharpe {sharpe(pred[te]-pred[te].mean(), ret[te]):.2f}")
p = pred - pred.mean()
print("CV sharpe", round(sharpe(p, ret), 3), "corr", round(np.corrcoef(p, ret)[0, 1], 4))
print("k profile (first 25 lags):", np.round(k[:25], 3))
np.save("/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/pred_irf.npy", pred)
