"""EXP-08: are future headlines (bars 50-59) predictable from past ones (story arcs)?"""
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import Ridge
from lib import load, sharpe, folds
import irf
piv, hl, sess, ret = load("ins")
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"] = hl.tmpl.map(tm); T=len(tm)
n=len(sess); idx=np.arange(n)
a,k = irf.fit(piv["close"], hl, idx, T)
kc = np.concatenate([[0.0], np.cumsum(np.concatenate([k, np.zeros(200)]))])
h2 = hl[(hl.bar_ix>=50)&(hl.bar_ix<=59)]
fut = np.zeros(n); np.add.at(fut, h2.si.values, a[h2.tid.values]*kc[59-h2.bar_ix.values+1])
f49 = irf.forecast(hl, idx, a, k)
past = hl[hl.bar_ix<=49]
def X_of(wfun):
    return sp.csr_matrix((wfun(past.bar_ix.values), (past.si.values, past.tid.values)), shape=(n,T))
# same-company arcs: template pair features would be huge; start with session-level template counts
for name, wf in [("counts", lambda t: np.ones_like(t, float)), ("recent(>=30)", lambda t: (t>=30).astype(float))]:
    X = X_of(wf); pf = np.zeros(n)
    for te in folds(n,5,0):
        tr=np.setdiff1d(idx,te); m=Ridge(alpha=30).fit(X[tr], fut[tr]); pf[te]=m.predict(X[te])
    print(f"{name}: corr(pred_fut, fut) {np.corrcoef(pf,fut)[0,1]:.3f}  corr with ret-resid {np.corrcoef(pf, ret-f49)[0,1]:.3f}",
          f" sharpe f49 {sharpe(f49-f49.mean(),ret):.3f} -> f49+pf {sharpe(f49+pf-(f49+pf).mean(),ret):.3f}")
# company-level arcs: next headline template of same company given its previous template
hs = hl.sort_values(["si","co","bar_ix"]); prev = hs.groupby(["si","co"]).tid.shift(1)
pair = hs.assign(prev=prev).dropna(subset=["prev"]); pair["a_next"]=a[pair.tid.values]
g = pair.groupby("prev").a_next.agg(["mean","count"]); g2 = pair.groupby(["prev", pair.si%2]).a_next.mean().unstack()
print("same-company: E[a_next | prev template] split-half corr", round(g2.corr().iloc[0,1],3), " std of means", round(g["mean"].std(),5), " overall std a", round(a.std(),5))
