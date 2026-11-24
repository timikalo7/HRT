"""Event study on full in-sample sessions (bars 0-99).

For each headline at bar b: same-bar reaction r0 = C[b]/C[b-1]-1 (b=0: C/O-1), next-bar r1, and
drift C[b+k]/C[b]-1. Compare by template and by the company's role in the session.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

h = pd.read_pickle(ROOT / "scratch_h.pkl")
bars = pd.read_parquet(ROOT / "data" / "ins_bars.parquet")
C = bars.pivot(index="session", columns="bar_ix", values="close")
O = bars.pivot(index="session", columns="bar_ix", values="open")
sess_index = {s: i for i, s in enumerate(C.index)}
Cm, Om = C.to_numpy(), O.to_numpy()

si = h["session"].map(sess_index).to_numpy()
b = h["bar_ix"].to_numpy()
prev = np.where(b > 0, Cm[si, np.maximum(b - 1, 0)], Om[si, b])
h["r0"] = Cm[si, b] / prev - 1
# open-to-close of the headline bar (isolates intrabar reaction)
h["oc"] = Cm[si, b] / Om[si, b] - 1
for k in (1, 2, 5, 10):
    ok = b + k <= 99
    v = np.full(len(h), np.nan)
    v[ok] = Cm[si[ok], b[ok] + k] / Cm[si[ok], b[ok]] - 1
    h[f"d{k}"] = v

# per-bar return vol for scale
bar_ret = Cm[:, 1:] / Cm[:, :-1] - 1
print("typical |bar return|:", np.nanmean(np.abs(bar_ret)), " bar std:", np.nanstd(bar_ret))

# company role: mention count over the FIRST HALF only (what we can see OOS)
first = h[h["bar_ix"] <= 49]
cnt = first.groupby(["session", "company"]).size().rename("n_first")
h = h.join(cnt, on=["session", "company"])
h["n_first"] = h["n_first"].fillna(0)
h["rank_first"] = h.groupby("session")["n_first"].rank(ascending=False, method="min")
# order of first appearance in session
firstbar = h.groupby(["session", "company"])["bar_ix"].min().rename("first_bar")
h = h.join(firstbar, on=["session", "company"])
h["appear_order"] = h.groupby("session")["first_bar"].rank(method="dense")

# Template-level: which templates move the price (abs and signed)
g = h.groupby("templ").agg(n=("r0", "size"), r0=("r0", "mean"), oc=("oc", "mean"),
                           d1=("d1", "mean"), d10=("d10", "mean"), absr0=("r0", lambda x: x.abs().mean()))
g["t_r0"] = g["r0"] / (h.groupby("templ")["r0"].std() / np.sqrt(g["n"]))
g["t_d10"] = g["d10"] / (h.groupby("templ")["d10"].std() / np.sqrt(g["n"]))
print("\nTop templates by |t(r0)| (n>=300):")
print(g[g["n"] >= 300].reindex(g[g["n"] >= 300]["t_r0"].abs().sort_values(ascending=False).index).head(40).round(5))

# Is the reaction concentrated on one company per session? Use sentiment proxy = template mean r0.
h["sent"] = h["templ"].map(g["r0"])
h["sgn"] = np.sign(h["sent"])
print("\nSigned reaction (r0 * sign(template mean)) by mention rank in first half:")
print(h.assign(sr=h["r0"] * h["sgn"]).groupby("rank_first")["sr"].agg(["mean", "count"]))
print("\nSigned reaction by appearance order:")
print(h.assign(sr=h["r0"] * h["sgn"]).groupby("appear_order")["sr"].agg(["mean", "count"]))

h.to_pickle(ROOT / "scratch_h2.pkl")
g.to_pickle(ROOT / "scratch_templ.pkl")
