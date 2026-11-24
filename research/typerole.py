"""Diagnostics on full in-sample sessions: (1) does the event type say whether a headline is about the target? (2) how much would knowing the target be worth?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe, HALF
from arena.latent import LatentTarget, all_sectors, _Prep, _posterior, range_variance, VAR_SCALE
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
d = load(); ins = d.ins; r = ins.returns; n = len(ins); sectors = all_sectors(d)
m = LatentTarget(sectors); m.fit(ins)
prep = _Prep(ins, m.types, sectors)
lr = ins.logret(); y = lr - m.mu0; sig = y.std(1)
X = prep.paths(m.amp, 100)
for temp in (1.0,):
    p, _ = _posterior(y, sig, X, prep, m.peer, False, temp)
p4 = p[:, :4]
print('full-session posterior: mean max %.3f, share > 0.9: %.3f' % (p4.max(1).mean(), (p4.max(1) > 0.9).mean()))
# (1) per event type: average posterior that the headline's company is the target (main-sector headlines only)
h = ins.headlines
ph = p4[prep.hs, prep.hj]; main_h = prep.main[prep.hs, prep.hj]
base = 1.0 / np.maximum(prep.main.sum(1), 1)[prep.hs]
t = pd.DataFrame({'ev': h['ev'].to_numpy(), 'p_target': ph, 'prior': base, 'main': main_h, 'amp': m.amp[np.maximum(prep.he, 0)]})
g = t[t.main].groupby('ev').agg(n=('p_target', 'size'), p_target=('p_target', 'mean'), prior=('prior', 'mean'), amp=('amp', 'first'))
g['lift'] = g.p_target / g.prior
print(g.sort_values('lift', ascending=False).round(3).head(14).to_string()); print('...'); print(g.sort_values('lift').round(3).head(8).to_string())
print('share of each type that is about a main-sector company (odd-sector share = rest), rare types:')
print(t.groupby('ev').main.agg(['mean', 'size']).query('size < 1000').round(3).to_string())
# (2) value of knowing the target: oracle forecast using the full-session posterior (NOT a valid model, diagnostic only)
pend = prep.pending(m.amp)
peer_pend = np.einsum('scj,sj->sc', prep.peer.astype(float), pend)
var = range_variance(ins.take(np.arange(n), first_half=True))
for name, q in [('full-session posterior (soft)', p4), ('full-session MAP (hard)', np.eye(4)[p4.argmax(1)])]:
    mu = (q * (pend + m.peer * peer_pend)).sum(1)
    print(f'ORACLE {name}: Sharpe {sharpe(mu / (VAR_SCALE * var + mu ** 2), r):.3f}   (first-half posterior model scores ~3.49 before corrections)')
