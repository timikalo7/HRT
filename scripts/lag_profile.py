"""Per-bar signed return at lag L after a headline (L = 0..60), numeric templates only.

Sign = template direction. Answers: how long does the drift last, and what's its shape?
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
h = pd.read_pickle(ROOT / "scratch_h2.pkl")
g = pd.read_pickle(ROOT / "scratch_templ.pkl")
bars = pd.read_parquet(ROOT / "data" / "ins_bars.parquet")
C = bars.pivot(index="session", columns="bar_ix", values="close").to_numpy()
O = bars.pivot(index="session", columns="bar_ix", values="open").to_numpy()
sess = np.sort(bars["session"].unique())
si = np.searchsorted(sess, h["session"].to_numpy())
b = h["bar_ix"].to_numpy()

# per-bar return R[s, t] = C[t]/C[t-1]-1, R[s,0] = C0/O0-1
R = np.empty_like(C)
R[:, 0] = C[:, 0] / O[:, 0] - 1
R[:, 1:] = C[:, 1:] / C[:, :-1] - 1

sgn = np.sign(h["templ"].map(g["d10"]).to_numpy())
numeric = h["templ"].isin(g.index[g["n"] > 3000]).to_numpy()

rows = []
for L in range(-5, 61):
    t = b + L
    ok = (t >= 0) & (t <= 99)
    for name, mask in (("numeric", numeric), ("qualitative", ~numeric)):
        m = ok & mask
        v = R[si[m], t[m]] * sgn[m]
        rows.append((name, L, v.mean() * 1e4, v.std() / np.sqrt(m.sum()) * 1e4, m.sum()))
df = pd.DataFrame(rows, columns=["kind", "lag", "bps", "se", "n"])
piv = df.pivot(index="lag", columns="kind", values="bps").round(2)
piv["se_num"] = df[df.kind == "numeric"].set_index("lag")["se"].round(2)
pd.set_option("display.max_rows", 100)
print(piv.to_string())
