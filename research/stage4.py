"""Alternative target inference: continuous per-company exposures (ridge / Gaussian posterior) as extra second-stage features."""
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
    n = len(block)
    out = dict(mu=mu, cnt=np.bincount(prep.hs, minlength=n).astype(float), unexpl=y.sum(1) - np.einsum('sc,sct->s', p, fit),
               live=np.bincount(prep.hs, weights=(prep.main[prep.hs, prep.hj] & (prep.hb >= 40)).astype(float), minlength=n), var=range_variance(block))
    # continuous exposures: beta_j ~ N(b0, tau2) for main-sector companies, 0 for the odd one
    Xm = X * prep.main[:, :, None]
    gv = np.sqrt(out['var'])                                   # range-based bar sd, less noisy than std(y)
    b0 = (1 + 2 * m.peer) / 3
    for tau2 in (0.05, 0.15, 0.5):
        for nm, s in (('', sig), ('g', gv * np.median(sig / gv))):
            A = np.einsum('sjt,skt->sjk', Xm, Xm) / s[:, None, None] ** 2 + np.eye(4)[None] / tau2
            b = np.einsum('sjt,st->sj', Xm, y) / s[:, None] ** 2 + b0 / tau2
            beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
            out[f'ridge{nm}{tau2}'] = (prep.main * beta * pend).sum(1)
            out[f'uridge{nm}{tau2}'] = y.sum(1) - np.einsum('sj,sjt->s', beta, Xm)
    return out

if __name__ == '__main__':
    d = load(); ins = d.ins; r = ins.returns; n = len(ins); sectors = all_sectors(d)
    seed = int(sys.argv[1]); folds = np.random.default_rng(seed).permutation(n) % 5
    base = ['mu', 'cnt', 'unexpl', 'live']
    SETS = {'base4': base}
    for t in (0.05, 0.15, 0.5):
        for g in ('', 'g'):
            SETS[f'+ridge{g}{t}'] = base + [f'ridge{g}{t}']
            SETS[f'ridge{g}{t} only'] = [f'ridge{g}{t}', 'cnt', f'uridge{g}{t}', 'live']
    SETS['+ridge0.15+uridge'] = base + ['ridge0.15', 'uridge0.15']
    F = {k: np.zeros(n) for k in SETS}; VAR = np.zeros(n)
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LatentTarget(sectors); m.fit(ins.take(tr, first_half=False))
        ftr = feats(m, ins.take(tr, first_half=True)); fte = feats(m, ins.take(te, first_half=True)); VAR[te] = fte['var']
        for k, cols in SETS.items():
            A = np.column_stack([ftr[c] for c in cols] + [np.ones(len(tr))]); B = np.column_stack([fte[c] for c in cols] + [np.ones(len(te))])
            F[k][te] = B @ np.linalg.lstsq(A, r[tr], rcond=None)[0]
    np.savez(f'stage4_seed{seed}.npz', VAR=VAR, keys=np.array(list(F)), vals=np.column_stack(list(F.values())))
    for k, f in F.items():
        f = f - f.mean(); print(f'seed {seed}  {k:22s} {sharpe(f / (VAR_SCALE * VAR + f ** 2), r):.3f}', flush=True)
