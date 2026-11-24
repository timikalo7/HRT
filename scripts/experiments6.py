"""Is anything beyond visible-event drift predictable?

(a) future headlines (bars 50-59): count and sentiment vs first-half count/sentiment; company bias
(b) noise mean reversion: de-drifted cumulative noise vs future noise; price level
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
ds = copy.copy(d)
ds.h = d.h.assign(templ=d.h["stem"])
rates = template_rates_deconv(ds, R / vs[:, None], np.ones(S, bool), lam=400)
h = d.h.copy()
h["rate"] = h["stem"].map(rates)
si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()

first = h[h.bar_ix <= 49]
fut = h[(h.bar_ix >= 50) & (h.bar_ix <= 59)]
n1 = np.bincount(first.si, minlength=S)
nf = np.bincount(fut.si, minlength=S)
s1 = np.bincount(first.si, weights=first.rate, minlength=S)
sf = np.bincount(fut.si, weights=fut.rate, minlength=S)
print("(a) corr(n_first, n_future 50-59):", round(np.corrcoef(n1, nf)[0, 1], 4))
print("    corr(sum rate first, sum rate future):", round(np.corrcoef(s1, sf)[0, 1], 4))
print("    corr(mean rate first, sum rate future):", round(np.corrcoef(s1 / np.maximum(n1, 1), sf)[0, 1], 4))
# headlines arrival per bar
print("    mean headlines per bar by decile of bar:", np.round(np.bincount(b // 10, minlength=10) / S / 10, 3))

# company bias: mean rate of a company's headlines in one half of sessions vs the other
h["half"] = h["si"] % 2
cm = h.groupby(["company", "half"])["rate"].mean().unstack()
print("    company mean-rate split-half corr:", round(cm[0].corr(cm[1]), 4),
      " (company sd", round(cm.mean(1).std() * 1e4, 3), "bps)")
# within session: does the same company's future news follow its earlier news?
fc = first.groupby(["si", "company"])["rate"].mean().rename("r_first")
fu = fut.groupby(["si", "company"])["rate"].mean().rename("r_fut")
j = pd.concat([fc, fu], axis=1).dropna()
print("    within-session company sentiment persistence corr:", round(j.r_first.corr(j.r_fut), 4), "n", len(j))

# (b) noise: subtract expected event drift (all events, vol-scaled)
E = np.zeros_like(R)
for L in range(11):
    t = b + L
    ok = t < NB
    np.add.at(E, (si[ok], t[ok]), h["rate"].to_numpy()[ok] * UNIT[L])
N = R / vs[:, None] - E           # normalised noise
N -= N.mean()
past = {w: N[:, 50 - w:50].sum(1) for w in (5, 10, 20, 50)}
futn = N[:, 50:60].sum(1)
for w, p in past.items():
    print(f"(b) corr(past {w:2d}-bar noise, future 10-bar noise): {np.corrcoef(p, futn)[0,1]:+.4f}")
logp = np.log(d.C[:, 49])
print("    corr(log C49, future noise):", round(np.corrcoef(logp, futn)[0, 1], 4))
print("    corr(log C49 - cum expected drift, future noise):",
      round(np.corrcoef(logp - (E[:, :50] * vs[:, None]).sum(1), futn)[0, 1], 4))
# generic: at many anchor points t, past-20 vs next-10 noise
cs = np.concatenate([np.zeros((S, 1)), N.cumsum(1)], axis=1)
rows = []
for t in range(20, 90):
    rows.append(np.corrcoef(cs[:, t] - cs[:, t - 20], cs[:, t + 10] - cs[:, t])[0, 1])
print("    mean corr(past20, next10) over anchors 20..89:", round(float(np.mean(rows)), 4))
rows = [np.corrcoef(cs[:, t] - cs[:, 0], cs[:, t + 10] - cs[:, t])[0, 1] for t in range(20, 90)]
print("    mean corr(cum since start, next10) over anchors:", round(float(np.mean(rows)), 4))
