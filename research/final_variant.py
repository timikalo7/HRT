"""Latent model plus two extra second-stage features that were each +0.01 on both fold assignments:
continuous per-company exposures (Gaussian posterior from the first-half price reaction) and the positive part of mu."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, write_submission, sharpe, cross_validate, HALF
from arena.latent import LatentTarget, all_sectors, _Prep, _posterior, range_variance

TAU2 = 0.5

class Plus(LatentTarget):
    def _features(self, test):
        prep = _Prep(test, self.types, self.sectors)
        lr = test.logret()[:, : HALF + 1]; y = lr - self.mu0; sig = np.maximum(y.std(1), 1e-6)
        X = prep.paths(self.amp, HALF + 1)
        p, fit = _posterior(y, sig, X, prep, self.peer, with_unseen=True, temp=self.temp)
        pend = prep.pending(self.amp)
        peer_pend = np.einsum("scj,sj->sc", prep.peer.astype(float), pend)
        mu = (p[:, :4] * (pend + self.peer * peer_pend)).sum(1) + p[:, 4] * self.peer * (prep.main * pend).sum(1)
        n = len(test)
        count = np.bincount(prep.hs, minlength=n).astype(float)
        unexplained = y.sum(1) - np.einsum("sc,sct->s", p, fit)
        live = np.bincount(prep.hs, weights=(prep.main[prep.hs, prep.hj] & (prep.hb >= HALF - 9)).astype(float), minlength=n)
        # continuous exposures: beta_j ~ N(b0, TAU2) for main-sector companies; noise scale from the bar ranges
        Xm = X * prep.main[:, :, None]
        gv = np.sqrt(range_variance(test)); s = gv * 1.11          # 1.11 = typical ratio of close-to-close sd to the range-based sd
        b0 = (1 + 2 * self.peer) / 3
        A = np.einsum("sjt,skt->sjk", Xm, Xm) / s[:, None, None] ** 2 + np.eye(4)[None] / TAU2
        b = np.einsum("sjt,st->sj", Xm, y) / s[:, None] ** 2 + b0 / TAU2
        beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        ridge = (prep.main * beta * pend).sum(1)
        return np.column_stack([mu, count, unexplained, live, ridge, np.maximum(mu, 0.0), np.ones(n)])

if __name__ == "__main__":
    d = load(); ins = d.ins; sectors = all_sectors(d); n = len(ins)
    if sys.argv[1] == "cv":
        res = cross_validate(lambda: Plus(sectors), ins, seeds=(int(sys.argv[2]),))
        print(res.round(3).to_string(index=False))
    else:
        rng = np.random.default_rng(0); idx = np.sort(rng.choice(n, 2000, replace=False)); tr = np.setdiff1d(np.arange(n), idx)
        m = Plus(sectors); m.fit(ins.take(tr, first_half=False)); p1 = m.predict(ins.take(idx, first_half=True))
        cb = ins.take(idx, first_half=False)
        for x in (cb.open, cb.high, cb.low, cb.close): x[:, HALF + 1:] = rng.uniform(0.5, 2.0, x[:, HALF + 1:].shape)
        cb.headlines = cb.headlines[cb.headlines.bar_ix <= HALF].reset_index(drop=True); cb.returns = None
        print("leak test max |change|:", float(np.abs(p1 - m.predict(cb)).max()), "| held-out Sharpe (2000): %.3f" % sharpe(p1, ins.returns[idx]))
        m = Plus(sectors); m.fit(ins)
        write_submission(d.oos.sessions, m.predict(d.oos), sys.argv[2]); print("beta:", m.beta.round(5))
