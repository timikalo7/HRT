"""Latent-target model.

Verified structure of a session: three companies from one sector plus one drawn from anywhere.
One of the main-sector companies is the stock itself; its headlines move the price at full
amplitude, same-sector peers at a fraction, and an other-sector company not at all.

Training (full sessions) fits the event amplitudes and the peer fraction by EM over the hidden
target. Prediction uses bars 0-49 only: a posterior over which seen company is the target (or
that it has not appeared yet), then the posterior-weighted drift that first-half headlines
still owe during bars 50-59. A second-stage linear fit adds three corrections (headline count,
the first-half move the headlines do not explain, live main-sector headlines), and the position
is forecast / (variance + forecast^2).

  python -m arena.latent cv [seeds]          # cross-validated Sharpe
  python -m arena.latent write preds.csv     # fit on all in-sample, write the upload file
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import squareform

from arena.baseline import fit_amplitudes, pending_weight
from arena.core import HALF, KERNEL, Block, cross_validate, load, write_submission

N_SECTORS = 5
MAIN_PER_SESSION = 3
# Holding-window variance = VAR_SCALE x per-bar range variance. Cross-validated Sharpe is flat for
# values between about 12 and 25 (10 bars, times ~1.24 because the range estimate reads low).
VAR_SCALE = 15.0


def sector_map(headlines: pd.DataFrame, key: np.ndarray) -> dict[str, int]:
    """Company -> sector, from which companies share sessions. Uses headline text only, no prices
    or labels. `key` identifies the session of each headline row."""
    pres = pd.crosstab(key, headlines["company"].to_numpy()).gt(0).astype(float)
    occ = pres.to_numpy().T @ pres.to_numpy()
    np.fill_diagonal(occ, 0)
    p = pres.to_numpy().mean(0)
    lift = occ / (np.outer(p, p) * len(pres))
    dist = 1.0 / (lift + 0.05)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), N_SECTORS, "maxclust")
    return dict(zip(pres.columns, labels.tolist()))


def range_variance(block: Block) -> np.ndarray:
    """Per-bar return variance of each session from bars 0..HALF: Garman-Klass plus opening gaps.
    Uses the high-low range, so it is a much less noisy estimate than close-to-close moves."""
    o, h, l, c = (np.log(x[:, : HALF + 1]) for x in (block.open, block.high, block.low, block.close))
    prev = np.concatenate([np.zeros((len(c), 1)), c[:, :-1]], 1)
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return np.maximum((gk + (o - prev) ** 2).mean(1), 1e-10)


class _Prep:
    """Padded company slots per session (order of first appearance) and headline index arrays."""

    def __init__(self, block: Block, types: list[str], sectors: dict[str, int]):
        h = block.headlines
        self.n = n = len(block)
        si = h["si"].to_numpy()
        comp = pd.factorize(h["company"])[0]
        sec_h = h["company"].map(sectors).fillna(0).astype(int).to_numpy()
        self.sec = np.full((n, 4), -1)
        nslots = np.zeros(n, int)
        slot: dict[tuple[int, int], int] = {}
        hj = np.empty(len(h), int)
        for i in range(len(h)):
            k = (si[i], comp[i])
            j = slot.get(k)
            if j is None:
                j = min(nslots[si[i]], 3)
                slot[k] = j
                nslots[si[i]] += 1
                self.sec[si[i], j] = sec_h[i]
            hj[i] = j
        tid = {e: i for i, e in enumerate(types)}
        self.hs, self.hj, self.hb = si, hj, h["bar_ix"].to_numpy()
        self.he = h["ev"].map(tid).fillna(-1).astype(int).to_numpy()
        seen = self.sec >= 0
        # main sector = plurality among seen companies; ties keep every tied sector
        counts = np.stack([((self.sec == s) & seen).sum(1) for s in range(1, N_SECTORS + 1)], 1)
        top = counts.max(1, keepdims=True)
        is_main_sector = (counts == top) & (top > 0)
        self.main = seen & np.take_along_axis(
            np.concatenate([np.zeros((n, 1), bool), is_main_sector], 1), np.maximum(self.sec, 0), 1)
        same = (self.sec[:, :, None] == self.sec[:, None, :]) & seen[:, :, None] & seen[:, None, :]
        self.peer = same & ~np.eye(4, dtype=bool)[None]
        self.ambiguous = is_main_sector.sum(1) > 1
        self.n_main = np.where(self.ambiguous, 1, self.main.sum(1))

    def paths(self, amp: np.ndarray, nbars: int) -> np.ndarray:
        X = np.zeros((self.n, 4, nbars))
        a = np.where(self.he >= 0, amp[np.maximum(self.he, 0)], 0.0)
        for lag, w in enumerate(KERNEL):
            ok = self.hb + lag < nbars
            np.add.at(X, (self.hs[ok], self.hj[ok], self.hb[ok] + lag), a[ok] * w)
        return X

    def pending(self, amp: np.ndarray) -> np.ndarray:
        """Drift each company slot still owes during the holding window, at full amplitude."""
        a = np.where(self.he >= 0, amp[np.maximum(self.he, 0)], 0.0) * pending_weight(self.hb)
        out = np.zeros((self.n, 4))
        np.add.at(out, (self.hs, self.hj), np.where(self.hb <= HALF, a, 0.0))
        return out


def _posterior(y, sig, X, prep: _Prep, peer_coef: float, with_unseen: bool, temp: float = 1.0):
    """P(target = slot c | bars), plus P(target not seen yet) as a fifth column."""
    peer_sum = np.einsum("scj,sjt->sct", prep.peer.astype(float), X)
    fit = X + peer_coef * peer_sum  # (n, 4, T): candidate c is the target
    main_sum = np.einsum("sj,sjt->st", prep.main.astype(float), X)
    fit = np.concatenate([fit, peer_coef * main_sum[:, None, :]], 1)  # candidate 4: unseen target
    ll = -0.5 * ((y[:, None, :] - fit) ** 2).sum(2) / sig[:, None] ** 2 / temp
    total = np.maximum(prep.n_main, MAIN_PER_SESSION)
    prior_seen = 1.0 / total
    prior_unseen = np.where(with_unseen, np.maximum(MAIN_PER_SESSION - prep.n_main, 0) / total, 0.0)
    prior = np.concatenate([np.where(prep.main, prior_seen[:, None], 0.0), prior_unseen[:, None]], 1)
    prior[prior.sum(1) == 0, 4] = 1.0  # no headlines at all
    with np.errstate(divide="ignore"):
        ll = ll + np.log(prior)
    ll -= ll.max(1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(1, keepdims=True), fit


class LatentTarget:
    name = "latent_target"

    def __init__(self, sectors: dict[str, int], *, em_iters: int = 8, temp: float = 4.0):
        # temp > 1 flattens the target posterior: at temp 1 it is over-confident and costs Sharpe.
        self.sectors, self.em_iters, self.temp = sectors, em_iters, temp

    def fit(self, train: Block) -> None:
        amp0 = fit_amplitudes(train)
        self.types = sorted(amp0)
        amp = np.array([amp0[e] for e in self.types])
        prep = _Prep(train, self.types, self.sectors)
        lr = train.logret()
        nbars = lr.shape[1]
        self.mu0 = float(lr.mean())
        y = lr - self.mu0
        sig = y.std(1)
        peer = 0.3
        for _ in range(self.em_iters):
            X = prep.paths(amp, nbars)
            p, _ = _posterior(y, sig, X, prep, peer, with_unseen=False)
            p = p[:, :4]
            # peer fraction: weighted least squares on [own path, peer path]
            peer_sum = np.einsum("scj,sjt->sct", prep.peer.astype(float), X)
            w = p / sig[:, None] ** 2
            G = np.array([[np.einsum("sc,sct,sct->", w, a, b) for b in (X, peer_sum)] for a in (X, peer_sum)])
            g = np.array([np.einsum("sc,sct,st->", w, a, y) for a in (X, peer_sum)])
            own, pr = np.linalg.solve(G, g)
            peer = pr / own
            # amplitudes: least squares with each headline's posterior-mean role weight
            role = p + peer * np.einsum("sc,scj->sj", p, prep.peer.astype(float))
            rh = role[prep.hs, prep.hj]
            rows, cols, vals = [], [], []
            for lag, wk in enumerate(KERNEL):
                ok = (prep.hb + lag < nbars) & (prep.he >= 0)
                rows.append(prep.hs[ok] * nbars + prep.hb[ok] + lag)
                cols.append(prep.he[ok])
                vals.append(rh[ok] * wk / sig[prep.hs[ok]])
            A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(prep.n * nbars, len(self.types)))
            amp = lsqr(A, (y / sig[:, None]).reshape(-1), atol=1e-12, btol=1e-12)[0]
        self.amp, self.peer = amp, float(peer)
        # Second stage: label on [owed drift, headline count, unexplained first-half move,
        # live main-sector headlines], using first-half features of the training sessions.
        Z = self._features(train.take(np.arange(len(train)), first_half=True))
        self.beta = np.linalg.lstsq(Z, train.returns, rcond=None)[0]

    def _features(self, test: Block) -> np.ndarray:
        """Second-stage inputs from bars 0..HALF and their headlines only."""
        prep = _Prep(test, self.types, self.sectors)
        lr = test.logret()[:, : HALF + 1]
        y = lr - self.mu0
        sig = np.maximum(y.std(1), 1e-6)
        X = prep.paths(self.amp, HALF + 1)
        p, fit = _posterior(y, sig, X, prep, self.peer, with_unseen=True, temp=self.temp)
        pend = prep.pending(self.amp)
        peer_pend = np.einsum("scj,sj->sc", prep.peer.astype(float), pend)
        owed_if_target = pend + self.peer * peer_pend
        owed_if_unseen = self.peer * (prep.main * pend).sum(1)
        mu = (p[:, :4] * owed_if_target).sum(1) + p[:, 4] * owed_if_unseen
        n = len(test)
        count = np.bincount(prep.hs, minlength=n).astype(float)
        unexplained = y.sum(1) - np.einsum("sc,sct->s", p, fit)
        main_h = prep.main[prep.hs, prep.hj]
        live = np.bincount(prep.hs, weights=(main_h & (prep.hb >= HALF - 9)).astype(float), minlength=n)
        return np.column_stack([mu, count, unexplained, live, np.ones(n)])

    def forecast(self, test: Block) -> tuple[np.ndarray, np.ndarray]:
        """Forecast of the 10-bar return and the session's per-bar variance, from bars 0..HALF only."""
        return self._features(test) @ self.beta, range_variance(test)

    def predict(self, test: Block) -> np.ndarray:
        """Sharpe-optimal position: forecast / conditional second moment of the 10-bar return."""
        mu, var = self.forecast(test)
        return mu / (VAR_SCALE * var + mu ** 2)


def all_sectors(data) -> dict[str, int]:
    h = pd.concat([data.ins.headlines, data.oos.headlines], ignore_index=True)
    key = np.concatenate([data.ins.headlines["si"].to_numpy(), data.oos.headlines["si"].to_numpy() + len(data.ins)])
    return sector_map(h, key)


def main(argv: list[str]) -> None:
    data = load()
    print(data.summary())
    sectors = all_sectors(data)
    print("sector sizes:", np.bincount(list(sectors.values()))[1:].tolist())
    if argv[:1] == ["write"]:
        model = LatentTarget(sectors)
        model.fit(data.ins)
        pred = model.predict(data.oos)
        path = write_submission(data.oos.sessions, pred, argv[1])
        print(f"wrote {len(pred)} predictions to {path} (peer fraction {model.peer:.3f})")
    else:
        seeds = tuple(int(s) for s in argv[1:]) or (0,)
        res = cross_validate(lambda: LatentTarget(sectors), data.ins, seeds=seeds)
        print(res.round(3).to_string(index=False))
        print(f"mean 5-fold Sharpe over {len(res)} seed(s): {res['sharpe'].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
