"""Do headline effects depend on WHICH company it is about, or on which other company is present?
(company-specific strength; pair-specific peer effects such as competitors vs partners)"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load
from arena.baseline import fit_amplitudes
from arena.latent import all_sectors
d = load(); ins = d.ins; sectors = all_sectors(d)
amp = fit_amplitudes(ins); lr = ins.logret(); lr = lr - lr.mean()
h = ins.headlines.copy(); h['a'] = h.ev.map(amp); h['sec'] = h.company.map(sectors)
main = h.groupby('si').sec.agg(lambda x: x.value_counts().index[0]); h = h[h.sec == h.si.map(main)]
s, t = h.si.to_numpy(), h.bar_ix.to_numpy(); ok = (t + 9 <= 99) & (h.a.abs().to_numpy() > 1.2e-4)
h = h[ok]; s, t = s[ok], t[ok]
h['y'] = sum(lr[s, t + k] for k in range(1, 10)) / 9.0
h['b'] = h.y / h.a                                   # per-headline response relative to the pooled amplitude
h['w'] = h.a ** 2
def chi(groups):
    g = h.groupby(groups).apply(lambda x: pd.Series({'b': np.average(x.b, weights=x.w), 'n': len(x), 'se': np.sqrt((x.w ** 2 * (x.b - np.average(x.b, weights=x.w)) ** 2).sum()) / x.w.sum()}), include_groups=False)
    g = g[g.n >= 20]; mu = np.average(g.b, weights=1 / g.se ** 2)
    return g, ((g.b - mu) ** 2 / g.se ** 2).sum(), len(g) - 1, mu
g, c, dof, mu = chi('company')
print('by company the headline is about: chi2 %.0f on %d dof | mean response %.3f | range of company means %.2f..%.2f' % (c, dof, mu, g.b.min(), g.b.max()))
# pairs: headline about company i, with company j also present in the session's main sector
pres = ins.headlines.assign(sec=ins.headlines.company.map(sectors)); pres = pres[pres.sec == pres.si.map(main)].drop_duplicates(['si', 'company'])[['si', 'company']].rename(columns={'company': 'other'})
hp = h.merge(pres, on='si'); hp = hp[hp.company != hp.other]
h = hp
g, c, dof, mu = chi(['company', 'other'])
print('by (company, other company present): chi2 %.0f on %d dof | sd of pair means %.3f vs typical se %.3f' % (c, dof, g.b.std(), g.se.median()))
print('most extreme pairs:'); print(g.assign(z=(g.b - mu) / g.se).sort_values('z').round(2).iloc[[0, 1, 2, -3, -2, -1]].to_string())
