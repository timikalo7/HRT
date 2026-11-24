"""Are slot fillers (segment, region, counterparty...) attributes that decide relevance?
(1) Is the filler tied to the company or the sector? (2) Within a session, do headlines sharing a filler value have more similar price responses?"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, _SLOTS, split_company
from arena.baseline import fit_amplitudes
from arena.latent import all_sectors
from scipy.stats import chi2_contingency
pd.set_option('display.width', 220)
d = load(); ins = d.ins; sectors = all_sectors(d)
h = ins.headlines.copy()
rest = [split_company(s)[1] for s in h.headline]
for tag, lst in _SLOTS.items():
    h[tag] = [next((f for f in lst if f in r), None) for r in rest]
h['sec'] = h.company.map(sectors)
allh = pd.concat([h, d.oos.headlines.assign(sec=lambda x: x.company.map(sectors))], ignore_index=True)
print('--- (1) filler vs company / sector: chi-square tests of independence (in-sample headlines) ---')
for tag in _SLOTS:
    g = h[h[tag].notna()]
    c1 = chi2_contingency(pd.crosstab(g.company, g[tag])); c2 = chi2_contingency(pd.crosstab(g.sec, g[tag]))
    print(f'{tag:5s} n={len(g):6d}  company x filler: chi2 {c1[0]:8.1f} on {c1[2]:4d} dof (p={c1[1]:.3f}) | sector x filler: chi2 {c2[0]:7.1f} on {c2[2]:3d} dof (p={c2[1]:.3f})')
# within a session: is the filler value shared across headlines more than chance? (session-level attribute)
print('--- (1b) within-session repetition of filler values (a session attribute would make values repeat) ---')
for tag in _SLOTS:
    g = h[h[tag].notna()]
    k = g.groupby('si')[tag].agg(['size', 'nunique']); k = k[k['size'] >= 2]
    nvals = len(_SLOTS[tag]); exp_unique = nvals * (1 - (1 - 1 / nvals) ** k['size'])
    print(f'{tag:5s} sessions {len(k):5d}  mean distinct values per session {k["nunique"].mean():.3f} vs {exp_unique.mean():.3f} expected if drawn uniformly at random')
print('--- (2) response similarity of headline pairs within a session ---')
amp = fit_amplitudes(ins); lr = ins.logret(); lr = lr - lr.mean()
h['a'] = h.ev.map(amp); main = h.groupby('si').sec.agg(lambda x: x.value_counts().index[0]); hm = h[(h.sec == h.si.map(main))].copy()
s, t = hm.si.to_numpy(), hm.bar_ix.to_numpy(); ok = (t + 9 <= 99) & (hm.a.abs().to_numpy() > 1.2e-4); hm = hm[ok]; s, t = s[ok], t[ok]
hm['b'] = sum(lr[s, t + k] for k in range(1, 10)) / 9.0 / hm.a.to_numpy()
hm['z'] = hm.b - hm.b.mean()
rows = []
for si, g in hm.groupby('si'):
    if len(g) < 2: continue
    v = g[['bar_ix', 'company', 'z', 'SEG', 'REG', 'CP', 'ev']].to_numpy()
    for i, j in itertools.combinations(range(len(v)), 2):
        if abs(v[i][0] - v[j][0]) < 10: continue                      # non-overlapping response windows only
        rows.append((v[i][1] == v[j][1], v[i][3] is not None and v[i][3] == v[j][3], v[i][3] is not None and v[j][3] is not None,
                     v[i][4] is not None and v[i][4] == v[j][4], v[i][4] is not None and v[j][4] is not None,
                     v[i][5] is not None and v[i][5] == v[j][5], v[i][5] is not None and v[j][5] is not None, v[i][6] == v[j][6], v[i][2] * v[j][2]))
P = pd.DataFrame(rows, columns=['same_co', 'same_seg', 'both_seg', 'same_reg', 'both_reg', 'same_cp', 'both_cp', 'same_ev', 'zz'])
def line(name, m):
    x = P.zz[m]; print(f'{name:55s} pairs {len(x):7d}  mean product {x.mean():+.3f}  (se {x.std() / np.sqrt(len(x)):.3f})')
line('same company', P.same_co); line('different company', ~P.same_co)
for tag in ('seg', 'reg', 'cp'):
    both = P[f'both_{tag}']
    line(f'diff company, both have {tag.upper()}, SAME {tag.upper()} value', ~P.same_co & both & P[f'same_{tag}'])
    line(f'diff company, both have {tag.upper()}, different value', ~P.same_co & both & ~P[f'same_{tag}'])
    line(f'same company, both have {tag.upper()}, SAME {tag.upper()} value', P.same_co & both & P[f'same_{tag}'])
    line(f'same company, both have {tag.upper()}, different value', P.same_co & both & ~P[f'same_{tag}'])
line('diff company, same event type', ~P.same_co & P.same_ev); line('same company, same event type', P.same_co & P.same_ev)
