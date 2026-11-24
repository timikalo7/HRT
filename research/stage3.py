"""More second-stage features on top of the current four, on identical folds."""
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
    own = (p[:, :4] * pend).sum(1); peer = (p[:, :4] * m.peer * peer_pend).sum(1) + p[:, 4] * m.peer * (prep.main * pend).sum(1)
    mu = own + peer
    resid = y - np.einsum('sc,sct->st', p, fit)        # unexplained bar returns
    n = len(block)
    cnt = np.bincount(prep.hs, minlength=n).astype(float)
    main_h = prep.main[prep.hs, prep.hj]
    live = np.bincount(prep.hs, weights=(main_h & (prep.hb >= 40)).astype(float), minlength=n)
    main_pend = (prep.main * pend).sum(1)              # owed drift if every main-sector company counted fully
    top = p[:, :4].argmax(1); top_pend = pend[np.arange(n), top]
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return dict(mu=mu, cnt=cnt, unexpl=resid.sum(1), live=live, u_0_29=resid[:, :30].sum(1), u_30_39=resid[:, 30:40].sum(1), u_40_49=resid[:, 40:].sum(1),
                u_45_49=resid[:, 45:].sum(1), own=own, peer=peer, main_pend=main_pend, top_pend=top_pend, ent_mu=ent * mu,
                pos=np.maximum(mu, 0), cnt_late=np.bincount(prep.hs, weights=(prep.hb >= 25).astype(float), minlength=n), var=range_variance(block))

if __name__ == '__main__':
    d = load(); ins = d.ins; r = ins.returns; n = len(ins); sectors = all_sectors(d)
    seed = int(sys.argv[1]); folds = np.random.default_rng(seed).permutation(n) % 5
    base = ['mu', 'cnt', 'unexpl', 'live']
    SETS = {'base4': base, 'split unexpl': ['mu', 'cnt', 'live', 'u_0_29', 'u_30_39', 'u_40_49'], '+u_45_49': base + ['u_45_49'], '+main_pend': base + ['main_pend'],
            'own/peer split': ['own', 'peer', 'cnt', 'unexpl', 'live'], '+top_pend': base + ['top_pend'], '+ent_mu': base + ['ent_mu'], '+pos': base + ['pos'], '+cnt_late': base + ['cnt_late'],
            'split+main_pend': ['mu', 'cnt', 'live', 'u_0_29', 'u_30_39', 'u_40_49', 'main_pend'], 'kitchen': ['mu', 'cnt', 'live', 'u_0_29', 'u_30_39', 'u_40_49', 'main_pend', 'top_pend', 'ent_mu', 'pos', 'cnt_late']}
    F = {k: np.zeros(n) for k in SETS}; VAR = np.zeros(n)
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LatentTarget(sectors); m.fit(ins.take(tr, first_half=False))
        ftr = feats(m, ins.take(tr, first_half=True)); fte = feats(m, ins.take(te, first_half=True)); VAR[te] = fte['var']
        for k, cols in SETS.items():
            A = np.column_stack([ftr[c] for c in cols] + [np.ones(len(tr))]); B = np.column_stack([fte[c] for c in cols] + [np.ones(len(te))])
            F[k][te] = B @ np.linalg.lstsq(A, r[tr], rcond=None)[0]
    np.savez(f'stage3_seed{seed}.npz', VAR=VAR, **{f'k{i}': v for i, v in enumerate(F.values())})
    for k, f in F.items():
        f = f - f.mean(); print(f'seed {seed}  {k:16s} {sharpe(f / (VAR_SCALE * VAR + f ** 2), r):.3f}', flush=True)
