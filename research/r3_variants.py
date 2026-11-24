import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe, HALF
from arena.latent import LatentTarget, all_sectors, _Prep, _posterior
d = load(); ins = d.ins; sectors = all_sectors(d); n = len(ins)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
folds = np.random.default_rng(seed).permutation(n) % 5
out = {}
def put(k, te, v):
    out.setdefault(k, np.zeros(n))[te] = v
for f in range(5):
    tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
    m = LatentTarget(sectors); m.fit(ins.take(tr, first_half=False))
    test = ins.take(te, first_half=True)
    prep = _Prep(test, m.types, sectors)
    lr = test.logret()[:, :HALF + 1]; y = lr - m.mu0; sig = np.maximum(y.std(1), 1e-6)
    X = prep.paths(m.amp, HALF + 1); pend = prep.pending(m.amp)
    peer_pend = np.einsum('scj,sj->sc', prep.peer.astype(float), pend)
    for temp in (1.0, 2.0, 4.0, 1e9):
        p, _ = _posterior(y, sig, X, prep, m.peer, True, temp)
        mu = (p[:, :4] * (pend + m.peer * peer_pend)).sum(1) + p[:, 4] * m.peer * (prep.main * pend).sum(1)
        put(('mu', temp), te, mu)
    put('sig', te, sig)
    put('main_equal', te, (prep.main * pend).sum(1))
    print('fold', f, 'peer', round(m.peer, 3), flush=True)
np.save(f'oof_seed{seed}.npy', out, allow_pickle=True)
r = ins.returns; sig = out['sig']
rows = []
for k, v in out.items():
    if k == 'sig': continue
    for pw in (0, 0.5, 1, 1.5, 2):
        rows.append((str(k), f'/sig^{pw}', sharpe(v / sig ** pw, r)))
    s2 = 10 * sig ** 2
    rows.append((str(k), 'mu/(10s2+mu2)', sharpe(v / (s2 + v ** 2), r)))
print(pd.DataFrame(rows, columns=['signal', 'sizing', 'sharpe']).pivot(index='signal', columns='sizing', values='sharpe').round(3).to_string())
