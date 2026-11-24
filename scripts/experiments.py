"""Candidate improvements over the news-drift baseline, all out-of-fold.

A. vol scaling        pred = f / sigma^p
B. early sentiment    sum of rates for headlines at bars < 40 (persistent session sentiment?)
C. realised drift     for events at 40..49, observed excess return since the event
D. price features     past returns over several windows
"""
import numpy as np

from model import UNIT, bar_returns, folds, load, news_feature, sharpe, template_rates

d = load("ins")
R = bar_returns(d)
y = d.returns
S = len(d.sessions)
fid = folds(S, 5, 0)

# ---- out-of-fold template rates -> features that depend on rates
f_news = np.zeros(S)
f_early = np.zeros(S)
f_real = np.zeros(S)        # realised excess drift on live events, weighted by remaining units
rates_oof = {}
h1 = d.h[d.h["bar_ix"] <= 49]
si, b = h1["si"].to_numpy(), h1["bar_ix"].to_numpy()
for k in range(5):
    tr = fid != k
    rates = template_rates(d, R, tr, shrink=50)
    fn = news_feature(d, rates)
    f_news[~tr] = fn[~tr]
    r = h1["templ"].map(rates).fillna(0).to_numpy()
    fe = np.zeros(S)
    m = b < 40
    np.add.at(fe, si[m], r[m])
    f_early[~tr] = fe[~tr]
    # realised: for live events (b>=40) observed sum R[b..49] minus expected rate*units(0..49-b)
    live = b >= 40
    obs = np.array([R[s, bb:50].sum() for s, bb in zip(si[live], b[live])])
    exp_units = np.array([UNIT[0:50 - bb].sum() for bb in b[live]])
    excess_per_bar = (obs - r[live] * exp_units) / (50 - b[live])
    rem_units = np.array([UNIT[50 - bb:60 - bb].sum() for bb in b[live]])
    fr = np.zeros(S)
    np.add.at(fr, si[live], excess_per_bar * rem_units)
    f_real[~tr] = fr[~tr]

print(f"base news OOF Sharpe {sharpe(f_news, y):.3f}")

# ---- A. vol scaling
sig = R[:, :50].std(axis=1)
print(f"session bar-vol: mean {sig.mean()*1e4:.1f}bps  cv {sig.std()/sig.mean():.3f}  "
      f"min {sig.min()*1e4:.1f}  max {sig.max()*1e4:.1f}")
# does first-half vol predict |y| ?
print("corr(sig, |y|):", round(np.corrcoef(sig, np.abs(y))[0, 1], 4))
for p in (0.5, 1, 2):
    print(f"A vol scaling p={p}: Sharpe {sharpe(f_news / sig**p, y):.3f}")

# ---- residual correlations of candidate features
beta = np.dot(f_news, y) / np.dot(f_news, f_news)
res = y - beta * f_news
def rc(name, x):
    print(f"{name:28s} corr(y) {np.corrcoef(x, y)[0,1]:+.4f}  corr(resid) {np.corrcoef(x, res)[0,1]:+.4f}")

rc("B early sentiment", f_early)
rc("C realised excess drift", f_real)
C = d.C
for w in (1, 2, 5, 10, 20, 49):
    rc(f"D past return {w} bars", C[:, 49] / C[:, 49 - w] - 1)
rc("D range (H-L)/C mean", ((d.H - d.L) / d.C)[:, :50].mean(axis=1))
rc("D n headlines first half", np.bincount(si, minlength=S).astype(float))
