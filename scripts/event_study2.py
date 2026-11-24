"""Drift profile by template, company role (fixed), and size dependence."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

h = pd.read_pickle(ROOT / "scratch_h2.pkl")
g = pd.read_pickle(ROOT / "scratch_templ.pkl")

cols = ["n", "r0", "oc", "d1", "d10", "t_r0", "t_d10"]
num = g[g["n"] > 3000].sort_values("d10")
print("Numeric templates (bps):")
out = num[cols].copy()
for c in ["r0", "oc", "d1", "d10"]:
    out[c] = (out[c] * 1e4).round(2)
print(out.round(2))

# drift profile per bar after the headline, for positive vs negative numeric templates
pos = num[num["d10"] > 0].index
neg = num[num["d10"] < 0].index
for name, idx in (("pos", pos), ("neg", neg)):
    sub = h[h["templ"].isin(idx)]
    print(name, "mean r0/d1/d2/d5/d10 (bps):",
          [round(sub[c].mean() * 1e4, 2) for c in ["r0", "d1", "d2", "d5", "d10"]])

# Company role, ranked per company (not per row), using first-half mentions.
first = h[h["bar_ix"] <= 49]
co = first.groupby(["session", "company"]).size().rename("n_first").reset_index()
co["rank"] = co.groupby("session")["n_first"].rank(ascending=False, method="first")
h = h.merge(co[["session", "company", "rank", "n_first"]], on=["session", "company"], how="left",
            suffixes=("_old", ""))
h["sgn"] = np.sign(h["templ"].map(g["d10"]))
num_rows = h[h["templ"].isin(num.index)]
print("\nSigned 10-bar drift (bps) on numeric templates by company mention-rank (first half):")
print(num_rows.assign(sd=num_rows["d10"] * num_rows["sgn"] * 1e4)
      .groupby("rank")["sd"].agg(["mean", "count"]).round(2))

# Size dependence: does $X or N% scale the drift?
def amount(s):
    m = re.search(r"\$([\d.]+)([MB])", s)
    if m:
        return float(m.group(1)) * (1000 if m.group(2) == "B" else 1)
    m = re.search(r"(\d+(?:\.\d+)?)%", s)
    return float(m.group(1)) if m else np.nan

num_rows = num_rows.assign(amt=num_rows["headline"].map(amount))
print("\nDrift (bps) by amount, per numeric template:")
for t in num.index:
    s = num_rows[num_rows["templ"] == t]
    tab = s.groupby("amt")["d10"].agg(["mean", "count"])
    tab["mean"] = (tab["mean"] * 1e4).round(1)
    print(f"\n{t}\n", tab.T.to_string())
