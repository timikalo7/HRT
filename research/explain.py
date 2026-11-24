"""Produce the explanation material: impact table, response shape, a worked example."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lib import load
import irf
piv, hl, sess, ret = load("ins")
tm = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}; inv={i:t for t,i in tm.items()}
hl["tid"]=hl.tmpl.map(tm); n=len(sess)
a,k = irf.fit(piv["close"], hl, np.arange(n), len(tm))
imp = pd.Series(a*1e4, index=[inv[i] for i in range(len(a))]).sort_values()  # bp total move per headline
cnt = hl.tmpl.value_counts()
print("TOP 8 bullish (bp total move over 10 bars):"); print(imp.tail(8)[::-1].round(1).to_string())
print("\nTOP 8 bearish:"); print(imp.head(8).round(1).to_string())
print("\nshare of templates |impact|<5bp:", round((imp.abs()<5).mean(),2), " median |impact|", round(imp.abs().median(),1))
# event study figure
C=piv["close"]; lc=np.log(np.c_[np.ones(n),C])
h=hl[(hl.bar_ix>=5)&(hl.bar_ix<=79)].copy(); h["a"]=a[h.tid]
q=h.a.quantile([.2,.8]).values
lags=np.arange(-5,21)
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
for lab,m,col in [("bullish headlines (top 20%)",h.a>q[1],"tab:green"),("bearish headlines (bottom 20%)",h.a<q[0],"tab:red"),("neutral (middle)", (h.a>=q[0])&(h.a<=q[1]),"gray")]:
    g=h[m]; s=g.si.values; t=g.bar_ix.values
    path=[np.mean(lc[s,t+1+L]-lc[s,t]) for L in lags]   # log C[t+L] - log C[t-1]
    ax[0].plot(lags, np.array(path)*1e4, label=lab, color=col)
ax[0].axvline(0,ls=":",c="k"); ax[0].axvline(10,ls=":",c="k"); ax[0].axhline(0,c="k",lw=.5)
ax[0].set_xlabel("bars since headline"); ax[0].set_ylabel("avg cumulative move (bp)")
ax[0].set_title("Price drifts for exactly 10 bars after a headline, then stops"); ax[0].legend(fontsize=8)
top=pd.concat([imp.head(6),imp.tail(6)])
ax[1].barh([t[:55] for t in top.index], top.values, color=["tab:red"]*6+["tab:green"]*6)
ax[1].set_xlabel("total move per headline (bp)"); ax[1].set_title("Learned impact of headline types (6 worst / 6 best)")
ax[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig("../docs/explain.png", dpi=140)
imp.to_csv("../docs/headline_impacts.csv", header=["impact_bp"])
# worked example: a session with a strong prediction
f=irf.forecast(hl, np.arange(n), a, k); i=int(np.argmax(np.abs(f)*(np.abs(f)<np.quantile(np.abs(f),.995))))
ex=hl[(hl.si==i)&(hl.bar_ix>=40)&(hl.bar_ix<=49)]
print("\nWORKED EXAMPLE session", sess[i], "forecast bp", round(f[i]*1e4,1), "actual bp", round(ret[i]*1e4,1))
for _,r in ex.iterrows():
    rem=max(0,min(10,r.bar_ix+10-49)); print(f"  bar {r.bar_ix}: {r.headline[:70]}  impact {a[r.tid]*1e4:+.0f}bp, ~{rem-0.5 if rem else 0}/10 still to come")
