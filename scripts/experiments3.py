"""Stem pooling (advisor item 3): do region/product/counterparty/title fillers matter?

Compare (all deconvolved, centred, OOF):
  T400   template rates, ridge 400 towards 0          (current best)
  S      stem rates only (fillers ignored)
  TS     template rates, ridge towards stem rate (lam 400 / 1600)
"""
import copy

import numpy as np

from model import bar_returns, folds, load, news_feature, sharpe, template_rates_deconv

d = load("ins")
R = bar_returns(d)
y = d.returns
S = len(d.sessions)
print("templates:", d.h["templ"].nunique(), " stems:", d.h["stem"].nunique())
print(d.h.groupby("stem").size().sort_values(ascending=False).head(60).to_string())

ds = copy.copy(d)
ds.h = d.h.assign(templ=d.h["stem"])
stem_of = d.h.drop_duplicates("templ").set_index("templ")["stem"]


def paired(a, b, name):
    pa, pb = a / a.std(), b / b.std()
    diff = (pa - pb) * y
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    print(f"   {name}: Sharpe {sharpe(a, y):.3f} vs {sharpe(b, y):.3f}  t(diff)={t:+.2f}")


for seed in (0, 1):
    fid = folds(S, 5, seed)
    oof = {c: np.zeros(S) for c in ("T400", "S0", "S400", "TS400", "TS1600", "TS6400")}
    for k in range(5):
        tr = fid != k
        oof["T400"][~tr] = news_feature(d, template_rates_deconv(d, R, tr, lam=400))[~tr]
        for lam in (0, 400):
            sr = template_rates_deconv(ds, R, tr, lam=lam)
            oof[f"S{lam}"][~tr] = news_feature(ds, sr)[~tr]
        sr = template_rates_deconv(ds, R, tr, lam=400)
        prior = stem_of.map(sr)
        for lam in (400, 1600, 6400):
            oof[f"TS{lam}"][~tr] = news_feature(d, template_rates_deconv(d, R, tr, lam=lam, prior=prior))[~tr]
    print(f"\nseed {seed}")
    base = oof["T400"] - oof["T400"].mean()
    for c, f in oof.items():
        paired(f - f.mean(), base, f"{c:7s} vs T400")
