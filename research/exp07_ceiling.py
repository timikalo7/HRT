"""EXP-07: headroom. Compare forecast from headlines <=49 with an oracle that also knows headlines in 50-59."""
import numpy as np
from lib import load, sharpe
import irf
piv, hl, sess, ret = load("ins")
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"] = hl.tmpl.map(tm)
n=len(sess); idx=np.arange(n)
a,k = irf.fit(piv["close"], hl, idx, len(tm))
f49 = irf.forecast(hl, idx, a, k, start=49)
# contribution of headlines published in bars 50..59 to the 50..59 return
h2 = hl[(hl.bar_ix>=50)&(hl.bar_ix<=59)]
kc = np.concatenate([[0.0], np.cumsum(np.concatenate([k, np.zeros(200)]))])
w = kc[59 - h2.bar_ix.values + 1]  # lags 0..59-t
fut = np.zeros(n); np.add.at(fut, h2.si.values, a[h2.tid.values]*w)
for name, p in [("known (<=49)", f49), ("oracle (+50-59 headlines)", f49+fut)]:
    p=p-p.mean(); print(f"{name}: sharpe {sharpe(p,ret):.2f}  R2 {np.corrcoef(p,ret)[0,1]**2:.3f}")
print("var shares of ret: known", round(np.var(f49)/np.var(ret),3), " future-headlines", round(np.var(fut)/np.var(ret),3))
