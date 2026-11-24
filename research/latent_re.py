"""Tim's LatentTarget with a random-effects likelihood for "which company is the stock".

Tim's posterior assumes every headline moves the price by exactly its type's amplitude, so it is
over-confident and needs temperature 4. Here each headline's strength is a_h * (1 + d_h) with
d_h ~ N(0, tau^2) (measured spread is large). The marginal likelihood of the first-half returns
under candidate c is Gaussian with covariance sig^2 I + U U' (U = tau * role * a_h * kernel_h),
evaluated exactly per session with the Woodbury identity. Optionally the posterior mean of each
headline's multiplier (revealed by how the price has reacted so far) rescales its pending drift.

    python research/latent_re.py cv                 # paired CV vs LatentTarget, seeds 0 and 1
    python research/latent_re.py write OUT.csv      # tau 0.85, temperature 1.5, write OOS preds

Result (5-fold, identical folds, seeds 0 / 1): LatentTarget 3.589 / 3.592; random effects
tau 0.85 temp 1.5 -> 3.616 / 3.620 (+0.027 / +0.028, P>0 0.96); temp 2 -> +0.023 / +0.024 (P>0 0.98);
tau 0.6 temp 2 -> +0.022 / +0.023. Rescaling each headline's pending drift by its posterior
multiplier ("+mult") is worse (-0.09), so it is off.
Upload files built from it: submissions/IMPROVED_c6_with_RE_core.csv (c6 + 0.7 x (RE - c3), unit scale)
and submissions/IMPROVED_c5_structure_RE_core.csv (0.70 RE + 0.15 public + 0.15 m3564).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TIM_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arena.baseline import pending_weight  # noqa: E402
from arena.core import HALF, KERNEL, load, sharpe  # noqa: E402
from arena.latent import MAIN_PER_SESSION, LatentTarget, _Prep, all_sectors, range_variance  # noqa: E402

T = HALF + 1


def _pad(prep: _Prep):
    """Headline rows grouped per session into (S, Hmax) slots."""
    order = np.lexsort((prep.hb, prep.hs))
    hs = prep.hs[order]
    pos = np.arange(len(hs)) - np.searchsorted(hs, hs)        # rank within session
    keep = prep.hb[order] <= HALF
    order, hs, pos = order[keep], hs[keep], np.arange(keep.sum()) - np.searchsorted(hs[keep], hs[keep])
    H = int(pos.max()) + 1 if len(pos) else 1
    return order, hs, pos, H


class LatentRE(LatentTarget):
    name = "latent_re"

    def __init__(self, sectors, *, tau=0.85, temp_re=1.0, mult=False, sig_mode="cc", **kw):
        super().__init__(sectors, **kw)
        self.tau, self.temp_re, self.mult, self.sig_mode = tau, temp_re, mult, sig_mode

    def _re_posterior(self, y, sig, prep: _Prep):
        n = prep.n
        order, hs, pos, H = _pad(prep)
        a = np.where(prep.he >= 0, self.amp[np.maximum(prep.he, 0)], 0.0)[order]
        hb, hj = prep.hb[order], prep.hj[order]
        K = np.zeros((n, H, T))
        for lag, w in enumerate(KERNEL):
            b = hb + lag
            ok = b < T
            K[hs[ok], pos[ok], b[ok]] = w
        A = np.zeros((n, H))
        A[hs, pos] = a
        J = np.full((n, H), -1)
        J[hs, pos] = hj
        PW = np.zeros((n, H))
        PW[hs, pos] = pending_weight(hb)
        valid = J >= 0
        ll = np.full((n, 5), -np.inf)
        owed = np.zeros((n, 5))
        fits = np.zeros((n, 5, T))
        s2 = sig ** 2
        for c in range(5):
            if c < 4:
                is_c = (J == c)
                is_peer = np.take_along_axis(prep.peer[:, c, :], np.maximum(J, 0), 1) & valid
                role = is_c * 1.0 + is_peer * self.peer
            else:
                role = np.take_along_axis(prep.main, np.maximum(J, 0), 1) & valid
                role = role * self.peer
            ra = role * A                                         # (n, H) expected strength
            mean = np.einsum("sh,sht->st", ra, K)
            U = self.tau * ra[:, :, None] * K                     # (n, H, T)
            r = y - mean
            b = np.einsum("sht,st->sh", U, r)
            M = np.einsum("sht,sgt->shg", U, U) + s2[:, None, None] * np.eye(H)[None]
            Minv_b = np.linalg.solve(M, b[..., None])[..., 0]
            quad = ((r * r).sum(1) - (b * Minv_b).sum(1)) / s2
            _, logdet = np.linalg.slogdet(M)
            ll[:, c] = -0.5 * (quad + logdet)
            m_hat = 1.0 + (self.tau * Minv_b if self.mult else 0.0)
            owed[:, c] = (ra * m_hat * PW).sum(1)
            fits[:, c] = mean + (np.einsum("sh,sht->st", self.tau * Minv_b * ra * self.tau, K) if self.mult else 0.0)
        # same prior as Tim's model
        total = np.maximum(prep.n_main, MAIN_PER_SESSION)
        prior_seen = 1.0 / total
        prior_unseen = np.maximum(MAIN_PER_SESSION - prep.n_main, 0) / total
        prior = np.concatenate([np.where(prep.main, prior_seen[:, None], 0.0), prior_unseen[:, None]], 1)
        prior[prior.sum(1) == 0, 4] = 1.0
        with np.errstate(divide="ignore"):
            lp = ll / self.temp_re + np.log(prior)
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        p /= p.sum(1, keepdims=True)
        return p, owed, fits

    def _features(self, test):
        prep = _Prep(test, self.types, self.sectors)
        lr = test.logret()[:, :T]
        y = lr - self.mu0
        sig = np.maximum(y.std(1), 1e-6) if self.sig_mode == "cc" else np.sqrt(range_variance(test))
        p, owed, fits = self._re_posterior(y, sig, prep)
        mu = (p * owed).sum(1)
        n = len(test)
        count = np.bincount(prep.hs, minlength=n).astype(float)
        unexplained = y.sum(1) - np.einsum("sc,sct->s", p, fits)
        main_h = prep.main[prep.hs, prep.hj]
        live = np.bincount(prep.hs, weights=(main_h & (prep.hb >= HALF - 9)).astype(float), minlength=n)
        return np.column_stack([mu, count, unexplained, live, np.ones(n)])


if __name__ == "__main__":
    D = load()
    ins = D.ins
    r, n = ins.returns, len(ins)
    sectors = all_sectors(D)
    variants = {
        "Tim base (temp 4)": lambda: LatentTarget(sectors),
        "RE tau0.85 temp1": lambda: LatentRE(sectors, tau=0.85),
        "RE tau0.85 temp2": lambda: LatentRE(sectors, tau=0.85, temp_re=2.0),
        "RE tau0.85 temp1 +mult": lambda: LatentRE(sectors, tau=0.85, mult=True),
        "RE tau0.85 temp1 rangevol": lambda: LatentRE(sectors, tau=0.85, sig_mode="range"),
        "RE tau1.2 temp1": lambda: LatentRE(sectors, tau=1.2),
        "RE tau0.6 temp2": lambda: LatentRE(sectors, tau=0.6, temp_re=2.0),
        "RE tau0.85 temp1.5": lambda: LatentRE(sectors, tau=0.85, temp_re=1.5),
        "RE tau0.85 temp3": lambda: LatentRE(sectors, tau=0.85, temp_re=3.0),
        "RE tau1.2 temp2": lambda: LatentRE(sectors, tau=1.2, temp_re=2.0),
    }
    if sys.argv[1] == "write":   # python latent_re.py write OUT.csv
        from arena.core import write_submission
        m = LatentRE(sectors, tau=0.85, temp_re=1.5)
        m.fit(ins)
        write_submission(D.oos.sessions, m.predict(D.oos), sys.argv[2])
        print(f"wrote {sys.argv[2]} (peer {m.peer:.3f})")
        sys.exit()
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(variants)
    seeds = (0,) if "quick" in sys.argv[1:2] else (0, 1)
    for seed in seeds:
        folds = np.random.default_rng(seed).permutation(n) % 5
        oof = {k: np.zeros(n) for k in only}
        for f in range(5):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            trb, teb = ins.take(tr, first_half=False), ins.take(te, first_half=True)
            base = None
            for k in only:
                m = variants[k]()
                if base is not None and isinstance(m, LatentRE):
                    # reuse the EM fit (identical) -- only the posterior differs
                    m.types, m.amp, m.peer, m.mu0 = base.types, base.amp, base.peer, base.mu0
                    Z = m._features(trb.take(np.arange(len(trb)), first_half=True))
                    m.beta = np.linalg.lstsq(Z, trb.returns, rcond=None)[0]
                else:
                    m.fit(trb)
                    if base is None:
                        base = m
                oof[k][te] = m.predict(teb)
            print(f"  seed {seed} fold {f} done", flush=True)
        b = oof[only[0]]
        rng = np.random.default_rng(7)
        idx = [rng.integers(0, n, n) for _ in range(300)]
        for k in only:
            d = np.array([sharpe(oof[k][i], r[i]) - sharpe(b[i], r[i]) for i in idx])
            print(f"seed {seed}: {k:28s} {sharpe(oof[k], r):.3f}  diff {sharpe(oof[k], r) - sharpe(b, r):+.3f}  "
                  f"P(>0) {(d > 0).mean():.2f}", flush=True)
