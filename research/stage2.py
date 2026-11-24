"""Latent-target forecast plus a few correction features in a second-stage linear fit."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe, HALF
from arena.latent import LatentTarget, all_sectors, _Prep, _posterior, range_variance, VAR_SCALE

def feats(m, block):
    prep = _Prep(block, m.types, m.sectors)
    lr = block.logret()[:, :HALF + 1]; y = lr - m.mu0; sig = np.maximum(y.std(1), 1e-6)
    X = prep.paths(m.amp, HALF + 1)
    p, fit = _posterior(y, sig, X, prep, m.peer, True, m.temp)
    pend = prep.pending(m.amp)
    peer_pend = np.einsum('scj,sj->sc', prep.peer.astype(float), pend)
    mu = (p[:, :4] * (pend + m.peer * peer_pend)).sum(1) + p[:, 4] * m.peer * (prep.main * pend).sum(1)
    explained = np.einsum('sc,sct->s', p, fit)
    unexpl = y.sum(1) - explained
    n = len(block)
    cnt = np.bincount(prep.hs, minlength=n).astype(float)
    main_h = prep.main[prep.hs, prep.hj]
    cnt_main = np.bincount(prep.hs, weights=main_h.astype(float), minlength=n)
    live = np.bincount(prep.hs, weights=(main_h & (prep.hb >= 40)).astype(float), minlength=n)
    late = np.bincount(prep.hs, weights=(prep.hb >= 40).astype(float), minlength=n)
    pmax = p[:, :4].max(1)
    return dict(mu=mu, cnt=cnt, cnt_main=cnt_main, live=live, late=late, unexpl=unexpl, move=y.sum(1), pmax_mu=mu * pmax, var=range_variance(block))

if __name__ == '__main__':
    d = load(); ins = d.ins; r = ins.returns; n = len(ins); sectors = all_sectors(d)
    seed = int(sys.argv[1]); folds = np.random.default_rng(seed).permutation(n) % 5
    SETS = {'mu only': ['mu'], '+cnt': ['mu', 'cnt'], '+unexpl': ['mu', 'unexpl'], '+cnt+unexpl': ['mu', 'cnt', 'unexpl'],
            '+cnt+unexpl+live': ['mu', 'cnt', 'unexpl', 'live'], 'all': ['mu', 'cnt', 'cnt_main', 'live', 'late', 'unexpl', 'move', 'pmax_mu']}
    F = {k: np.zeros(n) for k in SETS}; VAR = np.zeros(n)
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LatentTarget(sectors); m.fit(ins.take(tr, first_half=False))
        ftr = feats(m, ins.take(tr, first_half=True)); fte = feats(m, ins.take(te, first_half=True))
        VAR[te] = fte['var']
        for k, cols in SETS.items():
            A = np.column_stack([ftr[c] for c in cols] + [np.ones(len(tr))]); B = np.column_stack([fte[c] for c in cols] + [np.ones(len(te))])
            beta = np.linalg.lstsq(A, r[tr], rcond=None)[0]
            F[k][te] = B @ beta
            if f == 0 and k == 'all': print('beta(all):', dict(zip(cols, beta.round(5))), flush=True)
    np.savez(f'stage2_seed{seed}.npz', VAR=VAR, **{k.replace('+', 'p').replace(' ', '_'): v for k, v in F.items()})
    a = np.load(f'pub_oof_improved_all_seed{seed}.npy'); z = lambda x: x / x.std()
    for k, f in F.items():
        f = f - f.mean()
        pos = f / (VAR_SCALE * VAR + f ** 2)
        print(f'seed {seed}  {k:18s} alone {sharpe(pos, r):.3f}   blend with public {sharpe(0.5 * z(a) + 0.5 * z(pos), r):.3f}', flush=True)
