"""EXP-09: explainable 3-class model (bullish / neutral / bearish) vs full per-template model. CV."""
import numpy as np
from lib import load, sharpe, folds
import irf
piv, hl, sess, ret = load("ins")
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"]=hl.tmpl.map(tm); n=len(sess)
P={k:np.zeros(n) for k in ["full","3class","box3class"]}
for te in folds(n,5,0):
    tr=np.setdiff1d(np.arange(n),te)
    a,k=irf.fit(piv["close"],hl,tr,len(tm))
    P["full"][te]=irf.forecast(hl,te,a,k)
    cls=np.where(a>10e-4,1,np.where(a<-10e-4,-1,0))
    a3=np.array([a[cls==c].mean() if c else 0 for c in cls])
    P["3class"][te]=irf.forecast(hl,te,a3,k)
    box=np.r_[0.5,np.ones(9),0.5,np.zeros(len(k)-11)]/10   # hand rule: linear drift over 10 bars
    P["box3class"][te]=irf.forecast(hl,te,a3,box)
for name,p in P.items():
    p=p-p.mean(); print(f"{name:10s} CV sharpe {sharpe(p,ret):.3f}  hit rate (sign right, pred!=0) {np.mean(np.sign(p[p!=0])==np.sign(ret[p!=0])):.3f}")
f=P["full"]; big=np.abs(f-f.mean())>np.quantile(np.abs(f-f.mean()),.8)
print("hit rate on top-20% strongest forecasts", round(np.mean(np.sign(f[big]-f.mean())==np.sign(ret[big])),3))
print("class thresholds ±10bp; class means bp:", round(a[cls==1].mean()*1e4,1), round(a[cls==-1].mean()*1e4,1), "counts", (cls==1).sum(), (cls==0).sum(), (cls==-1).sum())
