"""Do the numbers and slot fillers inside a headline change its effect? (in-sample, main-sector headlines only)"""
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load
from arena.latent import all_sectors
pd.set_option('display.width', 220)
d = load(); ins = d.ins; sec = all_sectors(d)
h = ins.headlines.copy(); lr = ins.logret(); lr = lr - lr.mean()
s, t = h.si.to_numpy(), h.bar_ix.to_numpy(); ok = t + 9 <= 99
h = h[ok].copy(); s, t = s[ok], t[ok]
h['y'] = sum(lr[s, t + k] for k in range(1, 10))
h['sec'] = h.company.map(sec)
main = h.groupby('si').sec.agg(lambda x: x.value_counts().index[0]); h = h[h.sec == h.si.map(main)]
def amount(x):
    m = re.search(r'\$([\d.]+)([MB])', x); return float(m.group(1)) * (1000 if m.group(2) == 'B' else 1) if m else np.nan
def pct(x):
    m = re.search(r'(\d+(?:\.\d+)?)%', x); return float(m.group(1)) if m else np.nan
h['amt'] = h.headline.map(amount); h['pct'] = h.headline.map(pct)
for ev, g in h[h.amt.notna()].groupby('ev'):
    print(f'{ev[:50]:50s} n={len(g):6d} distinct amounts={g.amt.nunique():3d} corr(log amt, y)={np.corrcoef(np.log(g.amt), g.y)[0,1]:+.3f}  mean y by amount tercile x1e4:', (g.groupby(pd.qcut(g.amt, 3, duplicates="drop"), observed=True).y.mean() * 1e4).round(1).tolist())
for ev, g in h[h.pct.notna()].groupby('ev'):
    print(f'{ev[:50]:50s} n={len(g):6d} distinct pcts={g.pct.nunique():3d} corr(pct, y)={np.corrcoef(g.pct, g.y)[0,1]:+.3f}  mean y by pct tercile x1e4:', (g.groupby(pd.qcut(g.pct, 3, duplicates="drop"), observed=True).y.mean() * 1e4).round(1).tolist())
# slot fillers: does the filler change the effect within an event type? F-test style: variance of filler means vs expected
rows = []
for ev, g in h.groupby('ev'):
    rest = g.headline.str.replace(r'\$[\d.]+[MB]', '$X', regex=True).str.replace(r'\d+(?:\.\d+)?%', 'N%', regex=True)
    rest = [x[len(c) + 1:] for x, c in zip(rest, g.company)]
    gm = pd.Series(g.y.to_numpy()).groupby(rest)
    if gm.ngroups < 2: continue
    means, ns = gm.mean(), gm.size(); se2 = g.y.var() / ns
    chi = (((means - g.y.mean()) ** 2) / se2).sum()
    rows.append((ev[:50], gm.ngroups, len(g), chi, gm.ngroups - 1))
t = pd.DataFrame(rows, columns=['ev', 'fillers', 'n', 'chi2', 'dof'])
print('slot fillers: total chi2 %.1f on %d dof (expect ~dof if fillers do not matter)' % (t.chi2.sum(), t.dof.sum()))
print(t.sort_values('chi2', ascending=False).head(6).round(1).to_string(index=False))
