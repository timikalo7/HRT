"""Advisor item 4: does the IS model transfer to the OOS first half? (no uploads spent)"""
import numpy as np
import pandas as pd

from model import UNIT, bar_returns, load
from pipeline import SIG_REF, features, fit, sigma

ins, oos = load("ins"), load("oos")
print("stems IS:", ins.h["stem"].nunique(), " OOS:", oos.h["stem"].nunique(),
      " OOS stems unseen in IS:", sorted(set(oos.h["stem"]) - set(ins.h["stem"])))
print("templates OOS unseen in IS:", len(set(oos.h["templ"]) - set(ins.h["templ"])))
print("companies OOS unseen:", len(set(oos.h["company"]) - set(ins.h["company"])))

rates = fit(ins, np.ones(len(ins.sessions), bool), sig_kind="park")


def lag_slope(d, label):
    vs = sigma(d, "park") / SIG_REF
    Rn = bar_returns(d)[:, :50] / vs[:, None]
    h = d.h[d.h["bar_ix"] <= 49]
    si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
    r = h["stem"].map(rates).fillna(0).to_numpy()
    E = np.zeros_like(Rn)
    for L in range(11):
        t = b + L
        ok = t <= 49
        np.add.at(E, (si[ok], t[ok]), r[ok] * UNIT[L])
    e, x = E.ravel() - E.mean(), Rn.ravel() - Rn.mean()
    slope = np.dot(e, x) / np.dot(e, e)
    se = np.sqrt(((x - slope * e) ** 2).mean() / np.dot(e, e))
    # signed per-lag profile
    sg = np.sign(r)
    prof = []
    for L in range(-2, 13):
        t = b + L
        ok = (t >= 0) & (t <= 49)
        prof.append((Rn[si[ok], t[ok]] * sg[ok]).mean() * 1e4)
    print(f"{label}: first-half slope of R on expected drift = {slope:.3f} +- {se:.3f}")
    print(f"   signed bps per lag -2..12: {np.round(prof, 2)}")


lag_slope(ins, "IS ")
lag_slope(oos, "OOS")

# distribution shift of features / basic stats
Fi, vi = features(ins, rates, "park")
Fo, vo = features(oos, rates, "park")
cmp = pd.DataFrame({"IS mean": Fi.mean(), "OOS mean": Fo.mean(), "IS sd": Fi.std(), "OOS sd": Fo.std()})
cmp.loc["vs"] = [vi.mean(), vo.mean(), vi.std(), vo.std()]
print(cmp.round(5).to_string())
print("headlines/session first half  IS", round((ins.h.bar_ix <= 49).sum() / len(ins.sessions), 3),
      " OOS", round(len(oos.h) / len(oos.sessions), 3), " OOS max bar", oos.h["bar_ix"].max())
