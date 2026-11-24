"""Quick research on the real data: impulse response of headline types, and which companies matter."""
import numpy as np
import pandas as pd

import features as F
import kit

D = kit.load("data")
ins = D["ins"]
lp = np.log(kit.close_of(ins["full"], ins["cols"]))
r = np.diff(lp, axis=1)                      # r[:, t] = return of bar t+1
mu = r.mean()
H = F.prep_heads(ins["heads"], ins["ids"])   # all headlines, full session

print("headlines per session:", H.groupby("sess_idx").size().describe().round(1).to_dict())
nc = H.groupby("sess_idx")["company"].nunique()
print("companies per session:", nc.describe().round(1).to_dict())
top = H.groupby(["sess_idx", "company"]).size().groupby("sess_idx").max() / H.groupby("sess_idx").size()
print("share of headlines from the session's most frequent company:", top.describe().round(2).to_dict())

# impulse response per type (lags 0..20), excess per-bar log return
T = lp.shape[1]
rows = []
s, t = H["sess_idx"].to_numpy(), H["t"].to_numpy()
for L in range(0, 21):
    ok = t + L < T - 1
    rows.append(pd.DataFrame({"type": H["type_full"].to_numpy()[ok], "L": L,
                              "r": r[s[ok], t[ok] + L] - mu}))
d = pd.concat(rows)
g = d.groupby(["type", "L"])["r"].mean().unstack()
cnt = H["type_full"].value_counts()
g = g.loc[cnt.index[:25]]
print("\nmean excess return (bps) by lag after headline, top 25 types:")
print((g * 1e4).round(1).iloc[:, :16].to_string())
print("\nnumber of distinct types:", H["type_full"].nunique())

# does the main company's headline matter more than others?
H["is_top"] = H.groupby("sess_idx")["company"].transform(lambda c: c == c.value_counts().index[0])
for flag in (True, False):
    sub = d[d.index.isin([])] if False else None
rates = g.iloc[:, 1:11].mean(1)
H["rate"] = H["type_full"].map(rates)
for flag in (True, False):
    h = H[H["is_top"] == flag].dropna(subset=["rate"])
    s2, t2 = h["sess_idx"].to_numpy(), h["t"].to_numpy()
    ok = t2 + 10 < T - 1
    fwd = lp[s2[ok], t2[ok] + 10] - lp[s2[ok], t2[ok]] - 10 * mu
    print(f"top-company={flag}: n={ok.sum()}, corr(rate, fwd10)={np.corrcoef(h['rate'].to_numpy()[ok], fwd)[0,1]:.3f}")
