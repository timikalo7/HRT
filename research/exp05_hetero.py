"""EXP-05: heterogeneity in impact — by company, by template duration. Split-half reliability."""
import numpy as np, pandas as pd
from lib import load
import irf
piv, hl, sess, ret = load("ins")
C = piv["close"]; lc = np.log(np.c_[np.ones(len(C)), C])
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"] = hl.tmpl.map(tm)
n=len(sess); A_=np.arange(n)[np.arange(n)%2==0]
a,k = irf.fit(C, hl, A_, len(tm))
h = hl.copy(); h["a"]=a[h.tid]; h=h[h.bar_ix<=84]
s=h.si.values; t=h.bar_ix.values
h["m10"] = lc[s,t+11]-lc[s,t]; h["m5"]=lc[s,t+6]-lc[s,t]; h["m15"]=lc[s,t+16]-lc[s,t]
h["half"]=h.si%4  # use 2 disjoint halves of the held-out half
B=h[h.si%2==1].copy(); B["g"]=(B.si//2)%2
def slope(d): return np.sum(d.a*d.m10)/np.sum(d.a**2)
cs = B.groupby(["co","g"]).apply(slope).unstack()
print("company slope: mean", cs.mean().round(3).tolist(), " std", cs.std().round(3).tolist(), " split-half corr", round(cs.corr().iloc[0,1],3))
# duration: per template ratio m15/m10 and m5/m10 reliability
B["big"]=B.a.abs()>B.a.abs().quantile(.5)
d=B[B.big].groupby(["tid","g"]).apply(lambda x: np.sum(x.a*x.m5)/np.sum(x.a*x.m10)).unstack()
print("template m5/m10 ratio mean", d.mean().round(3).tolist(), "split-half corr", round(d.corr().iloc[0,1],3))
d=B[B.big].groupby(["tid","g"]).apply(lambda x: np.sum(x.a*x.m15)/np.sum(x.a*x.m10)).unstack()
print("template m15/m10 ratio mean", d.mean().round(3).tolist(), "split-half corr", round(d.corr().iloc[0,1],3))
# template a reliability between disjoint halves (using simple 10-bar move means)
r = h.groupby(["tid", h.si%2]).m10.mean().unstack(); print("template mean-move split-half corr", round(r.corr().iloc[0,1],3), "n per template", int(len(h)/len(tm)))
