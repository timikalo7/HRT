"""EXP-02: what explains the residual of the IRF forecast?"""
import numpy as np, pandas as pd
from lib import load, sharpe
piv, hl, sess, ret = load("ins")
S='/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/'
pred = np.load(S+'pred_irf.npy'); p = pred - pred.mean()
beta = np.polyfit(p, ret, 1)[0]; res = ret - beta*p
print("beta", beta)
C = piv["close"]; lc = np.log(np.c_[np.ones(len(C)), C]); br = np.diff(lc, axis=1)  # br[:,u]
def c(x, y=res): return round(np.corrcoef(x, y)[0,1], 4)
for w in [1,3,5,10,20,50]:
    print(f"past {w}-bar ret vs resid", c(lc[:,50]-lc[:,50-w]), " vs ret", c(lc[:,50]-lc[:,50-w], ret))
# realized drift since recent headlines vs expectation
vol = br[:, :50].std(axis=1)
print("vol vs |resid|", c(vol, np.abs(res)))
nrec = np.zeros(len(C)); np.add.at(nrec, hl[(hl.bar_ix>=40)&(hl.bar_ix<=49)].si.values, 1)
print("n headlines in 40-49 vs |resid|", c(nrec, np.abs(res)), "mean n", nrec.mean())
# intrabar: close-open, high-low features
o,hh,ll = piv["open"],piv["high"],piv["low"]
print("last bar (c-o) vs resid", c(C[:,49]-o[:,49]))
print("gap open49-close48", c(o[:,49]-C[:,48]))
