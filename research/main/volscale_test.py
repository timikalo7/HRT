"""Does a headline's effect scale with the session's volatility? And long-lag / non-additive effects."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import HALF
from arena.baseline import fit_amplitudes
from arena.latent import range_variance
from research.harness import data
pd.set_option('display.width', 200)
d, sectors = data(); ins = d.ins; n = len(ins)
lr = ins.logret(); y = lr - lr.mean()
amp = fit_amplitudes(ins)
h = ins.headlines.copy(); h['a'] = h.ev.map(amp); h['sec'] = h.company.map(sectors)
main = h.groupby('si').sec.agg(lambda x: x.value_counts().index[0]); hm = h[h.sec == h.si.map(main)].copy()
# session volatility from ALL 100 bars' ranges (research only) and from the first half
O, H, L, C = (np.log(x) for x in (ins.open, ins.high, ins.low, ins.close))
prev = np.concatenate([np.zeros((n, 1)), C[:, :-1]], 1)
gk = (0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2 + (O - prev) ** 2)
vol_full = np.sqrt(gk.mean(1)); vol_half = np.sqrt(gk[:, :50].mean(1))
print('session bar-sd (range based): quantiles 5/25/50/75/95 %:', np.quantile(vol_full, [.05, .25, .5, .75, .95]).round(5))
s, t = hm.si.to_numpy(), hm.bar_ix.to_numpy(); ok = (t + 10 <= 99) & (hm.a.abs().to_numpy() > 1.2e-4); hs = hm[ok].copy(); s, t = s[ok], t[ok]
hs['resp'] = sum(y[s, t + k] for k in range(1, 10))          # 9 full-weight bars
hs['a9'] = 9 * hs.a
hs['vol'] = vol_full[s]; hs['volh'] = vol_half[s]
print('--- (1) response slope by session-volatility quintile (slope of 9-bar response on pooled amplitude) ---')
hs['q'] = pd.qcut(hs.vol, 5, labels=False)
rows = []
for q, g in hs.groupby('q'):
    b = (g.a9 * g.resp).sum() / (g.a9 ** 2).sum(); se = np.sqrt(((g.a9 * (g.resp - b * g.a9)) ** 2).sum()) / (g.a9 ** 2).sum()
    rows.append((q, g.vol.mean(), b, se, b / g.vol.mean() * hs.vol.mean()))
print(pd.DataFrame(rows, columns=['vol quintile', 'mean bar sd', 'slope', 'se', 'slope / relative vol']).round(4).to_string(index=False))
# regression: resp = b0*a9 + b1*a9*(vol/mean - 1)
rv = hs.vol / hs.vol.mean() - 1; X = np.column_stack([hs.a9, hs.a9 * rv]); beta, *_ = np.linalg.lstsq(X, hs.resp.to_numpy(), rcond=None)
res = hs.resp.to_numpy() - X @ beta; cov = np.linalg.inv(X.T @ X) @ (X.T * res ** 2) @ X @ np.linalg.inv(X.T @ X)
print('regression resp = b0*a + b1*a*(vol/mean-1):  b0 = %.3f (se %.3f)   b1 = %.3f (se %.3f)   [b1/b0 = 1 means effect proportional to volatility, 0 means fixed]' % (beta[0], np.sqrt(cov[0, 0]), beta[1], np.sqrt(cov[1, 1])))
rvh = hs.volh / hs.volh.mean() - 1; X = np.column_stack([hs.a9, hs.a9 * rvh]); beta, *_ = np.linalg.lstsq(X, hs.resp.to_numpy(), rcond=None)
print('same with first-half volatility only: b0 = %.3f  b1 = %.3f  ratio %.2f' % (beta[0], beta[1], beta[1] / beta[0]))
print('--- (2) long-lag effects: mean signed return per bar at lags 11..60 after a strong main-sector headline (x1e-5), pooled amplitude sign ---')
s2, t2 = hm.si.to_numpy(), hm.bar_ix.to_numpy(); sg = np.sign(hm.a.to_numpy()); strong = hm.a.abs().to_numpy() > 1.2e-4
for lo, hi in [(1, 9), (11, 20), (21, 30), (31, 40), (41, 60)]:
    vals = []
    for k in range(lo, hi + 1):
        m = strong & (t2 + k <= 99); vals.append(sg[m] * y[s2[m], t2[m] + k])
    v = np.concatenate(vals); print(f'   lags {lo:2d}-{hi:2d}: mean {v.mean() * 1e5:+7.2f}  se {v.std() / np.sqrt(len(v)) * 1e5:.2f}  (n={len(v)})')
print('--- (3) additivity: realised bar return vs predicted total drift from all active main-sector headlines (pooled amplitudes) ---')
K = np.array([0.5] + [1.0] * 9 + [0.5]); pred = np.zeros((n, 100)); cnt = np.zeros((n, 100))
a_all = hm.a.to_numpy()
for lag, w in enumerate(K):
    m = t2 + lag <= 99; np.add.at(pred, (s2[m], t2[m] + lag), a_all[m] * w); np.add.at(cnt, (s2[m], t2[m] + lag), (np.abs(a_all[m]) > 1.2e-4) * w)
pf, yf, cf = pred.ravel(), y.ravel(), cnt.ravel()
bins = np.quantile(pf[np.abs(pf) > 1e-5], [0, .02, .1, .3, .5, .7, .9, .98, 1]); idx = np.digitize(pf, bins)
tab = pd.DataFrame({'bin': idx, 'pred': pf, 'y': yf})[np.abs(pf) > 1e-5].groupby('bin').agg(n=('y', 'size'), pred=('pred', 'mean'), realised=('y', 'mean'))
tab['ratio'] = tab.realised / tab.pred; print((tab.assign(pred=tab.pred * 1e4, realised=tab.realised * 1e4)).round(3).to_string())
for c_lo, c_hi in [(0.4, 1.01), (1.01, 2.01), (2.01, 9)]:
    m = (cf > c_lo) & (cf <= c_hi) & (np.abs(pf) > 1e-5); b = (pf[m] * yf[m]).sum() / (pf[m] ** 2).sum()
    print(f'   bars with {c_lo:.1f} < active strong headlines <= {c_hi:.1f}: slope realised/predicted {b:.3f} (n={m.sum()})')
