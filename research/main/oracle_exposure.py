"""DIAGNOSTIC ONLY (uses bars after 49): how much would a sharper estimate of each company's exposure be worth?
Exposures / target posterior are estimated from bars 0-49 plus 60-99 (the label window 50-59 is excluded), then used
to weight the drift still owed at bar 49. Everything else (second stage, sizing) is the real pipeline."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import numpy as np
sys.path.insert(0, '/home/user/HRT')
from arena.core import sharpe, HALF
from arena.latent import LatentTarget, _Prep, _posterior, range_variance, VAR_SCALE
from research.harness import data, folds, get_base, baseline_oof

d, sectors = data(); ins = d.ins; r = ins.returns; n = len(ins)

def feats(base, block_full):
    first = block_full.take(np.arange(len(block_full)), first_half=True)
    Z = base._features(first)                                   # [mu, count, unexplained, live, 1] from bars 0-49 only
    var = range_variance(first)
    prep = _Prep(block_full, base.types, base.sectors)          # company slots/sectors from the whole session
    lr = block_full.logret(); y = lr - base.mu0
    X = prep.paths(base.amp, 100); pend = prep.pending(base.amp)
    out = {'Z': Z, 'var': var}
    for name, mask in [('0-49', np.r_[np.ones(50), np.zeros(50)]), ('0-49+60-99', np.r_[np.ones(50), np.zeros(10), np.ones(40)]), ('60-99 only', np.r_[np.zeros(60), np.ones(40)])]:
        m = mask.astype(bool); ym = y[:, m]; Xm = X[:, :, m]; sig = np.maximum(ym.std(1), 1e-6)
        for temp in (1.0, 4.0):
            p, _ = _posterior(ym, sig, Xm, prep, base.peer, True, temp)
            peer_pend = np.einsum('scj,sj->sc', prep.peer.astype(float), pend)
            out[f'post[{name}] T={temp:g}'] = (p[:, :4] * (pend + base.peer * peer_pend)).sum(1) + p[:, 4] * base.peer * (prep.main * pend).sum(1)
        Xmain = Xm * prep.main[:, :, None]; b0 = (1 + 2 * base.peer) / 3
        for tau2 in (0.15, 0.5, 2.0):
            A = np.einsum('sjt,skt->sjk', Xmain, Xmain) / sig[:, None, None] ** 2 + np.eye(4)[None] / tau2
            b = np.einsum('sjt,st->sj', Xmain, ym) / sig[:, None] ** 2 + b0 / tau2
            beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
            out[f'ridge[{name}] tau2={tau2:g}'] = (prep.main * beta * pend).sum(1)
    return out

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
f = folds(n, seed); P = {}
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    base = get_base(seed, k)
    Ftr = feats(base, ins.take(tr, first_half=False)); Fte = feats(base, ins.take(te, first_half=False))
    for key in Ftr:
        if key in ('Z', 'var'): continue
        for mode in ('replace mu', 'add'):
            cols = (lambda F: np.column_stack([F[key], F['Z'][:, 1:]])) if mode == 'replace mu' else (lambda F: np.column_stack([F[key], F['Z']]))
            beta = np.linalg.lstsq(cols(Ftr), r[tr], rcond=None)[0]; fc = cols(Fte) @ beta
            P.setdefault((key, mode), np.zeros(n))[te] = fc / (VAR_SCALE * Fte['var'] + fc ** 2)
print(f'seed {seed}: baseline {sharpe(baseline_oof(seed), r):.3f}')
for (key, mode), p in P.items():
    print(f'  {key:28s} {mode:10s} {sharpe(p, r):.3f}')
