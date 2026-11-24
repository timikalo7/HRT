"""Does the number in a headline scale its price impact?"""
import re

import numpy as np
import pandas as pd

import features as F
import kit

D = kit.load("data")
ins = D["ins"]
lp = np.log(kit.close_of(ins["full"], ins["cols"]))
mu = np.diff(lp, axis=1).mean()
T = lp.shape[1]
H = F.prep_heads(ins["heads"], ins["ids"])
s, t = H["sess_idx"].to_numpy(), H["t"].to_numpy()
ok = t + 10 < T
H = H[ok].copy()
s, t = s[ok], t[ok]
H["fwd"] = lp[s, t + 10] - lp[s, t] - 10 * mu


def num(h):
    m = re.search(r"\$\s?(\d[\d,.]*)\s?([KMB])?", h)
    if m:
        v = float(m.group(1).replace(",", ""))
        v *= {"K": 1e-3, "M": 1, "B": 1e3}.get((m.group(2) or "M").upper(), 1)
        return v
    m = re.search(r"(\d[\d,.]*)\s?%", h)
    return float(m.group(1).replace(",", "")) if m else np.nan


H["val"] = H["headline"].map(num)
print("headlines with a number:", H["val"].notna().mean().round(3))
for typ, g in H.dropna(subset=["val"]).groupby("type_full"):
    if len(g) < 1500:
        continue
    q = pd.qcut(g["val"].rank(method="first"), 4, labels=False)
    means = (g.groupby(q)["fwd"].mean() * 1e4).round(1).tolist()
    print(f"{typ[:60]:<60} n={len(g):5d} vals {g['val'].min():g}..{g['val'].max():g}  fwd10 bps by value quartile {means}")

# rare types: how many headlines fall in types with < 200 occurrences?
vc = H["type_full"].value_counts()
print("\nshare of headlines in types with <200 obs:", round(vc[vc < 200].sum() / vc.sum(), 3), "types:", (vc < 200).sum())
print("\nrare-type examples:\n", vc[vc < 200].head(15).to_string())
