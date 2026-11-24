"""Is per-event drift strength shared within (session, company)?

For each event e (full IS sessions, window fully inside 100 bars):
  x_e = sum_L UNIT[L] * (Rn - E_all)[s, b+L] / U2          (excess rate, normalised units)
  model: true rate = rate_e * m_e  =>  x_e ~ rate_e * (m_e - 1) + noise
For pairs (e1, e2) in the same session with non-overlapping windows:
  Cov(m1, m2) ~ sum(x1 x2 r1 r2) / sum(r1^2 r2^2)
split by same company vs different company. Also Var(m) via the split-window trick.
"""
import copy

import numpy as np
import pandas as pd

from model import UNIT, bar_returns, load, template_rates_deconv
from pipeline import SIG_REF, sigma

d = load("ins")
R = bar_returns(d)
S, NB = R.shape
vs = sigma(d, "park") / SIG_REF
Rn = R / vs[:, None]
ds = copy.copy(d)
ds.h = d.h.assign(templ=d.h["stem"])
rates = template_rates_deconv(ds, Rn, np.ones(S, bool), lam=400)

h = d.h.copy()
tok = h["headline"].str.split()
h["co"] = np.where(tok.str[2].isin(["Group", "Co"]), tok.str[:3].str.join(" "), tok.str[:2].str.join(" "))
h["rate"] = h["stem"].map(rates).to_numpy()
si, b, r = h["si"].to_numpy(), h["bar_ix"].to_numpy(), h["rate"].to_numpy()
E = np.zeros_like(Rn)
for L in range(11):
    t = b + L
    ok = t < NB
    np.add.at(E, (si[ok], t[ok]), r[ok] * UNIT[L])
X = Rn - E
X -= X.mean()
U2 = (UNIT[:11] ** 2).sum()
full = b + 10 < NB
x = np.zeros(len(h))
for L in range(11):
    t = np.minimum(b + L, NB - 1)
    x += UNIT[L] * X[si, t]
h["x"] = x / U2
ev = h[full].reset_index(drop=True)
print("events with full window:", len(ev), "  companies/session:", ev.groupby("si")["co"].nunique().mean().round(2))

# pairs within session
ev["eid"] = np.arange(len(ev))
p = ev[["si", "eid", "co", "bar_ix", "x", "rate"]].merge(
    ev[["si", "eid", "co", "bar_ix", "x", "rate"]], on="si", suffixes=("1", "2"))
p = p[(p.eid1 < p.eid2) & ((p.bar_ix1 - p.bar_ix2).abs() > 10)]
p["same"] = p.co1 == p.co2
rr = p.rate1 * p.rate2
for name, m in (("same company", p.same), ("different company", ~p.same)):
    q = p[m]
    w = (q.rate1 * q.rate2)
    cov = (q.x1 * q.x2 * w).sum() / (w ** 2).sum()
    # SE via session-cluster bootstrap-lite: split sessions into 20 groups
    grp = q.si % 20
    parts = [((q.x1 * q.x2 * w)[grp == g].sum() / (w[grp == g] ** 2).sum()) for g in range(20)]
    print(f"{name:18s} pairs {len(q):8d}  Cov(m1,m2) = {cov:+.3f}  (se ~{np.std(parts)/np.sqrt(20):.3f})")

# Var(m) from early/late split of the same event window (lags 0-4 vs 5-10), non-overlap noise
xe = np.zeros(len(h)); xl = np.zeros(len(h))
for L in range(11):
    t = np.minimum(b + L, NB - 1)
    if L <= 4:
        xe += UNIT[L] * X[si, t]
    else:
        xl += UNIT[L] * X[si, t]
ue, ul = (UNIT[:5] ** 2).sum(), (UNIT[5:11] ** 2).sum()
xe, xl = (xe / ue)[full], (xl / ul)[full]
w = ev["rate"].to_numpy() ** 2
print(f"Var(m) (same event, split window) = {(xe * xl * w).sum() / (w ** 2).sum():+.3f}")
