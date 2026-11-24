"""What explains per-event drift magnitude? drift_e = rate_stem * sigma_s/sigma_bar * m_e.

For each event: realised rate x_e = mean over lags 1..9 of (R - E_others), where E is the expected
drift path from all headlines (vol-scaled). Expected e_e = rate * sigma_s/sigma_bar.
Per group g: slope_g = sum(e*x)/sum(e^2) with SE; spread of slopes vs SEs says if the group matters.
"""
import copy
import re

import numpy as np
import pandas as pd

from model import UNIT, bar_returns, load, template_rates_deconv

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
d = load("ins")
R = bar_returns(d)
S, NB = R.shape
ds = copy.copy(d)
ds.h = d.h.assign(templ=d.h["stem"])
rates = template_rates_deconv(ds, R, np.ones(S, bool), lam=400)
sig = R[:, :50].std(axis=1)
vs = sig / sig.mean()

h = d.h.copy()
h["rate"] = h["stem"].map(rates)
h["e"] = h["rate"] * vs[h["si"].to_numpy()]
si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
E = np.zeros_like(R)
for L in range(11):
    t = b + L
    ok = t < NB
    np.add.at(E, (si[ok], t[ok]), h["e"].to_numpy()[ok] * UNIT[L])
keep = b + 9 < NB
h = h[keep].copy()
si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
own = h["e"].to_numpy()
x = np.zeros(len(h))
for L in range(1, 10):
    x += R[si, b + L] - E[si, b + L] + own  # remove others' expected drift, keep own
h["x"] = x / 9
h = h[np.abs(h["rate"]) > 1e-4]  # non-neutral only
print("non-neutral events:", len(h), " pooled slope:",
      round(float((h.e * h.x).sum() / (h.e ** 2).sum()), 3))

# candidate grouping variables
tok = h["headline"].str.split()
h["company3"] = np.where(tok.str[2].isin(["Group", "Co"]), tok.str[:3].str.join(" "), tok.str[:2].str.join(" "))
h["suffix"] = np.where(tok.str[2].isin(["Group", "Co"]), tok.str[2], "none")
h["sector"] = tok.str[1]
h["bar_bucket"] = pd.cut(h["bar_ix"], [-1, 9, 19, 29, 39, 49, 59, 69, 79, 89])
def amt(s):
    m = re.search(r"\$([\d.]+)([MB])", s)
    if m:
        return float(m.group(1)) * (1000 if m.group(2) == "B" else 1)
    m = re.search(r"(\d+(?:\.\d+)?)%", s)
    return float(m.group(1)) if m else np.nan
h["amount"] = h["headline"].map(amt)
h["amt_q"] = h.groupby("stem")["amount"].transform(lambda a: pd.qcut(a.rank(method="first"), 4, labels=False)
                                                    if a.notna().all() else np.nan)
mention = h.groupby(["si", "company"]).size().rename("n_mentions")
h = h.join(mention, on=["si", "company"])
h["sign"] = np.sign(h["rate"])


def group_test(col):
    g = h.groupby(col, observed=True)
    rows = []
    for k, sub in g:
        ee, xx = sub["e"].to_numpy(), sub["x"].to_numpy()
        s = (ee * xx).sum() / (ee ** 2).sum()
        res = xx - s * ee
        se = np.sqrt(res.var() / (ee ** 2).sum())
        rows.append((k, len(sub), s, se))
    t = pd.DataFrame(rows, columns=[col, "n", "slope", "se"]).set_index(col)
    w = 1 / t["se"] ** 2
    mu = (t["slope"] * w).sum() / w.sum()
    chi2 = ((t["slope"] - mu) ** 2 * w).sum()
    print(f"\n{col}: groups {len(t)}  chi2 {chi2:.1f} on {len(t)-1} df  (slope sd {t['slope'].std():.3f},"
          f" mean se {t['se'].mean():.3f})")
    return t


for col in ("sign", "suffix", "bar_bucket", "amt_q", "n_mentions"):
    print(group_test(col).round(3).to_string())
for col in ("sector", "company3", "stem"):
    t = group_test(col)
    print(t.sort_values("slope").round(3).iloc[[0, 1, 2, -3, -2, -1]].to_string())
