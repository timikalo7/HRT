"""Centering + deconvolved template rates, with paired comparisons (advisor items 1, 2, 5)."""
import numpy as np

from model import (bar_returns, folds, load, news_feature, sharpe, template_rates,
                   template_rates_deconv)

d = load("ins")
R = bar_returns(d)
y = d.returns
S = len(d.sessions)
h1 = d.h[d.h["bar_ix"] <= 49]
nhead = np.bincount(h1["si"].to_numpy(), minlength=S).astype(float)


def paired(a, b, name):
    """t-stat of per-session pnl difference with preds scaled to unit std."""
    pa, pb = a / a.std(), b / b.std()
    diff = pa * y - pb * y
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    print(f"   paired {name}: Sharpe {sharpe(a, y):.3f} vs {sharpe(b, y):.3f}  t(diff)={t:+.2f}")


configs = {"lagmean_k50": None, "deconv_l0": 0.0, "deconv_l100": 100.0, "deconv_l400": 400.0,
           "deconv_l1600": 1600.0}
results = {}
for seed in (0, 1):
    fid = folds(S, 5, seed)
    oof = {c: np.zeros(S) for c in configs}
    for k in range(5):
        tr = fid != k
        for c, lam in configs.items():
            rates = (template_rates(d, R, tr, shrink=50) if lam is None
                     else template_rates_deconv(d, R, tr, lam=lam))
            oof[c][~tr] = news_feature(d, rates)[~tr]
    print(f"\nseed {seed}")
    for c, f in oof.items():
        fc = f - f.mean()
        beta = np.dot(fc, y) / np.dot(fc, fc)
        res = y - beta * fc
        print(f"{c:14s} raw {sharpe(f, y):.3f}  centred {sharpe(fc, y):.3f}  mean(f)={f.mean()*1e4:.2f}bps"
              f"  corr(nhead,resid) {np.corrcoef(nhead, res)[0,1]:+.4f}")
    base = oof["lagmean_k50"] - oof["lagmean_k50"].mean()
    paired(oof["lagmean_k50"] - oof["lagmean_k50"].mean(), oof["lagmean_k50"], "centred vs raw (lagmean)")
    for c in list(configs)[1:]:
        paired(oof[c] - oof[c].mean(), base, f"{c} vs lagmean (both centred)")
    results[seed] = oof

np.save("../scratch_oof_exp2.npy", np.array([results[0][c] for c in configs]))
