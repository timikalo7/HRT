"""Sector structure from co-occurrence, and session composition."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
d = load()
allh = pd.concat([d.ins.headlines.assign(src='ins', key=lambda x: x.si), d.oos.headlines.assign(src='oos', key=lambda x: x.si + 100000)], ignore_index=True)
cos = sorted(allh.company.unique()); ci = {c: i for i, c in enumerate(cos)}
# sessions with all 4 companies visible (in-sample, 100 bars) for clean composition
pres = pd.crosstab(allh.key, allh.company).gt(0).astype(float)
occ = pres.values.T @ pres.values; np.fill_diagonal(occ, 0)
p = pres.values.mean(0); lift = occ / (np.outer(p, p) * len(pres))
print('lift distribution quantiles:', np.quantile(lift[np.triu_indices(80, 1)], [0, .1, .25, .5, .75, .9, .95, .99, 1]).round(2))
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
dist = 1 / (lift + 0.05); np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), 'average')
for k in (4, 5, 6, 7):
    lab = fcluster(Z, k, 'maxclust')
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(80, 1)
    print(f'k={k}: sizes {np.bincount(lab)[1:].tolist()}  mean lift within {lift[iu][same[iu]].mean():.2f}  between {lift[iu][~same[iu]].mean():.3f}')
lab = fcluster(Z, 5, 'maxclust')
sec = pd.Series(lab, index=cos)
for k in range(1, 6):
    members = sec[sec == k].index.tolist()
    print(k, len(members), sorted(set(m.split()[1] for m in members)))
# within vs between lift histogram to see how clean it is
same = lab[:, None] == lab[None, :]; iu = np.triu_indices(80, 1)
print('within-lift quantiles', np.quantile(lift[iu][same[iu]], [0, .05, .5, .95, 1]).round(2), ' between-lift quantiles', np.quantile(lift[iu][~same[iu]], [0, .05, .5, .95, 1]).round(3))
# session composition with the co-occurrence sectors, full in-sample sessions
hi = d.ins.headlines.copy(); hi['sec'] = hi.company.map(sec)
first = hi.drop_duplicates(['si', 'company'])
pat = first.groupby('si').sec.agg(lambda s: tuple(sorted(s.value_counts().tolist(), reverse=True)))
print('composition (100 bars, co-occurrence sectors):', pat.value_counts().to_dict())
nco = first.groupby('si').size()
print('companies per session:', nco.value_counts().sort_index().to_dict())
pickle.dump(sec.to_dict(), open('sector_map.pkl', 'wb'))
# headline share: per session, headline counts per company sorted desc (sessions with 4 companies)
cnt = hi.groupby(['si', 'company']).size().rename('n').reset_index()
cnt['rank'] = cnt.groupby('si').n.rank(ascending=False, method='first')
print('mean headline count by within-session count rank:', cnt.groupby('rank').n.mean().round(2).to_dict())
print('headlines per session by composition:', hi.groupby('si').size().groupby(pat).mean().round(2).to_dict())
