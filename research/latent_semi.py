"""Semi-supervised version of Tim's LatentTarget (timikalo7/HRT, branch claude/zen-mendel-953rlv).

The EM that fits event amplitudes and the peer fraction also uses UNLABELLED first halves
(bars 0..49 and their headlines; no returns): out-of-sample sessions and, in CV, the test
fold's own first halves. The second stage is still fitted on labelled training sessions only.

    python research/latent_semi.py cv              # paired CV vs LatentTarget, seeds 0 and 1
    python research/latent_semi.py write OUT.csv   # fit on in-sample + oos first halves, write OOS preds

Result (5-fold, identical folds, seeds 0 / 1): LatentTarget 3.589 / 3.592; semi-supervised
3.590 / 3.598 (extra weight 1) and 3.591 / 3.597 (weight 0.5). Peer fraction 0.25 -> 0.23.
Gain +0.001 to +0.006: inside noise, so not used for an upload.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.environ.get("TIM_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from arena.baseline import fit_amplitudes  # noqa: E402
from arena.core import KERNEL, Block, load, sharpe, write_submission  # noqa: E402
from arena.latent import LatentTarget, _posterior, _Prep, all_sectors  # noqa: E402


def concat_blocks(a: Block, b: Block) -> Block:
    """Stack two first-half blocks (same bar count)."""
    h2 = b.headlines.copy()
    h2["si"] = h2["si"] + len(a)
    import pandas as pd
    return Block(np.concatenate([a.sessions, b.sessions]), *(np.concatenate([getattr(a, c), getattr(b, c)])
                 for c in ("open", "high", "low", "close")), pd.concat([a.headlines, h2], ignore_index=True), None)


class LatentSemi(LatentTarget):
    name = "latent_semi"

    def __init__(self, sectors, extra: Block | None = None, extra_weight: float = 1.0, **kw):
        super().__init__(sectors, **kw)
        self.extra, self.extra_weight = extra, extra_weight

    def _parts(self, blk: Block, amp):
        prep = _Prep(blk, self.types, self.sectors)
        lr = blk.logret()
        return prep, lr.shape[1], lr

    def fit(self, train: Block) -> None:
        amp0 = fit_amplitudes(train)
        self.types = sorted(amp0)
        amp = np.array([amp0[e] for e in self.types])
        blocks = [(train, 1.0)] + ([(self.extra, self.extra_weight)] if self.extra is not None else [])
        parts = []
        mu0 = float(train.logret().mean())
        for blk, wt in blocks:
            prep = _Prep(blk, self.types, self.sectors)
            lr = blk.logret()
            y = lr - mu0
            parts.append((prep, lr.shape[1], y, y.std(1), wt))
        self.mu0 = mu0
        peer = 0.3
        for _ in range(self.em_iters):
            G, g, As, bs = np.zeros((2, 2)), np.zeros(2), [], []
            cache = []
            for prep, nbars, y, sig, wt in parts:
                X = prep.paths(amp, nbars)
                p, _ = _posterior(y, sig, X, prep, peer, with_unseen=(nbars < 60))
                if nbars < 60:   # unlabelled first half: the target may not have appeared yet
                    p = p[:, :4] / np.maximum(p[:, :4].sum(1, keepdims=True), 1e-12)
                else:
                    p = p[:, :4]
                peer_sum = np.einsum("scj,sjt->sct", prep.peer.astype(float), X)
                w = wt * p / sig[:, None] ** 2
                G += np.array([[np.einsum("sc,sct,sct->", w, a, b) for b in (X, peer_sum)] for a in (X, peer_sum)])
                g += np.array([np.einsum("sc,sct,st->", w, a, y) for a in (X, peer_sum)])
                cache.append((prep, nbars, y, sig, wt, p))
            own, pr = np.linalg.solve(G, g)
            peer = pr / own
            off = 0
            for prep, nbars, y, sig, wt, p in cache:
                role = p + peer * np.einsum("sc,scj->sj", p, prep.peer.astype(float))
                rh = role[prep.hs, prep.hj]
                rows, cols, vals = [], [], []
                sw = np.sqrt(wt)
                for lag, wk in enumerate(KERNEL):
                    ok = (prep.hb + lag < nbars) & (prep.he >= 0)
                    rows.append(off + prep.hs[ok] * nbars + prep.hb[ok] + lag)
                    cols.append(prep.he[ok])
                    vals.append(sw * rh[ok] * wk / sig[prep.hs[ok]])
                As.append((np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)))
                bs.append(sw * (y / sig[:, None]).reshape(-1))
                off += prep.n * nbars
            r_, c_, v_ = (np.concatenate(z) for z in zip(*As))
            A = sp.csr_matrix((v_, (r_, c_)), shape=(off, len(self.types)))
            amp = lsqr(A, np.concatenate(bs), atol=1e-12, btol=1e-12)[0]
        self.amp, self.peer = amp, float(peer)
        Z = self._features(train.take(np.arange(len(train)), first_half=True))
        self.beta = np.linalg.lstsq(Z, train.returns, rcond=None)[0]


if __name__ == "__main__":
    D = load()
    ins, oos = D.ins, D.oos
    r, n = ins.returns, len(ins)
    sectors = all_sectors(D)
    if sys.argv[1] == "write":
        m = LatentSemi(sectors, extra=oos)
        m.fit(ins)
        p = m.predict(oos)
        write_submission(oos.sessions, p, sys.argv[2])
        print(f"wrote {sys.argv[2]} (peer {m.peer:.3f})")
        sys.exit()
    for seed in (0, 1):
        folds = np.random.default_rng(seed).permutation(n) % 5
        base, semi, semi_half = np.zeros(n), np.zeros(n), np.zeros(n)
        for f in range(5):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            trb, teb = ins.take(tr, first_half=False), ins.take(te, first_half=True)
            extra = concat_blocks(teb, oos)
            m0 = LatentTarget(sectors); m0.fit(trb); base[te] = m0.predict(teb)
            m1 = LatentSemi(sectors, extra=extra); m1.fit(trb); semi[te] = m1.predict(teb)
            m2 = LatentSemi(sectors, extra=extra, extra_weight=0.5); m2.fit(trb); semi_half[te] = m2.predict(teb)
            print(f"  seed {seed} fold {f}: peer base {m0.peer:.3f} semi {m1.peer:.3f}", flush=True)
        rng = np.random.default_rng(7)
        for nm, p in (("semi w=1", semi), ("semi w=0.5", semi_half)):
            d = []
            for _ in range(500):
                i = rng.integers(0, n, n)
                d.append(sharpe(p[i], r[i]) - sharpe(base[i], r[i]))
            d = np.array(d)
            print(f"seed {seed}: Tim {sharpe(base, r):.3f}  {nm} {sharpe(p, r):.3f}  "
                  f"diff {sharpe(p, r) - sharpe(base, r):+.3f}  P(diff>0) {(d > 0).mean():.2f}", flush=True)
