"""EXP-03: does realised drift since a recent headline reveal its true (instance-level) impact?"""
import numpy as np, pandas as pd
from lib import load, folds
import irf
piv, hl, sess, ret = load("ins")
C = piv["close"]; lc = np.log(np.c_[np.ones(len(C)), C])   # lc[:, u+1] = log C[u]
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"] = hl.tmpl.map(tm)
n=len(sess); tr=np.arange(n)[np.arange(n)%2==0]; te=np.arange(n)[np.arange(n)%2==1]
a,k = irf.fit(C, hl, tr, len(tm))
h = hl[hl.si.isin(te)].copy()
h["a"] = a[h.tid]
# per-headline: realised move over the 10 bars after it, vs a  (needs full bars, research only)
t = h.bar_ix.values; s=h.si.values
ok = t <= 89
real = lc[s[ok], t[ok]+11] - lc[s[ok], t[ok]]   # log C[t+10] - log C[t-1]
print("instance: corr(real move, a)", np.corrcoef(real, h.a.values[ok])[0,1].round(3),
      "slope", np.polyfit(h.a.values[ok], real, 1)[0].round(3))
# residual move variance vs. pure bar-noise variance (does instance noise exist?)
resid = real - h.a.values[ok]
br = np.diff(lc,axis=1); bv = br.var(axis=1)[s[ok]]
print("var(resid)/(11*bar var) =", (resid.var()/ (11*bv.mean())).round(3), " (>1 => extra per-headline noise or overlaps)")
# by |a| bucket: if multiplicative noise, resid var grows with |a|
q = pd.qcut(np.abs(h.a.values[ok]), 5, labels=False)
print("resid var by |a| quintile", [round(resid[q==i].var()*1e6,2) for i in range(5)])
