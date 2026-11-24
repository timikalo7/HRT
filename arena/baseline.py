"""Tier-1 fallback model: pending headline drift.

Each headline type has one amplitude, fitted by least squares on per-bar returns using the
verified 11-bar response kernel. The forecast for bars 50-59 is the drift that first-half
headlines (bars 40-49) still owe, scaled by one regression coefficient.

Deliberately ignores which company a headline is about, the price path, and risk sizing --
it is the simplest model that uses the verified structure, kept as the floor to beat.

  python -m arena.baseline cv                 # cross-validated Sharpe on in-sample data
  python -m arena.baseline write preds.csv    # fit on all in-sample, write the upload file
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from arena.core import HALF, HORIZON, KERNEL, Block, cross_validate, load, write_submission


def fit_amplitudes(block: Block) -> dict[str, float]:
    """Per-bar drift of each event type, from every bar of the block (overlaps handled jointly)."""
    lr = block.logret()
    n, nbars = lr.shape
    types = sorted(block.headlines["ev"].unique())
    col = {e: i for i, e in enumerate(types)}
    si = block.headlines["si"].to_numpy()
    bar = block.headlines["bar_ix"].to_numpy()
    ev = block.headlines["ev"].map(col).to_numpy()
    rows, cols, vals = [], [], []
    for j, w in enumerate(KERNEL):
        ok = bar + j < nbars
        rows.append(si[ok] * nbars + bar[ok] + j)
        cols.append(ev[ok])
        vals.append(np.full(ok.sum(), w))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * nbars, len(types)))
    y = (lr - lr.mean()).reshape(-1)
    return dict(zip(types, lsqr(X, y, atol=1e-12, btol=1e-12)[0]))


def pending_weight(bar: np.ndarray) -> np.ndarray:
    """Kernel weight of a headline at `bar` that falls inside the holding window."""
    lags = np.arange(len(KERNEL))
    target = bar[:, None] + lags[None, :]
    inside = (target > HALF) & (target <= HALF + HORIZON)
    return (inside * KERNEL[None, :]).sum(1)


class PendingDrift:
    name = "pending_drift"

    def feature(self, block: Block) -> np.ndarray:
        h = block.headlines[block.headlines["bar_ix"] <= HALF]
        owed = h["ev"].map(self.amp).fillna(0.0).to_numpy() * pending_weight(h["bar_ix"].to_numpy())
        return np.bincount(h["si"].to_numpy(), weights=owed, minlength=len(block))

    def fit(self, train: Block) -> None:
        self.amp = fit_amplitudes(train)
        m = self.feature(train)
        self.centre = m.mean()
        self.beta = float(np.dot(m - self.centre, train.returns) / np.dot(m - self.centre, m - self.centre))

    def predict(self, test: Block) -> np.ndarray:
        return self.beta * (self.feature(test) - self.centre)


def main(argv: list[str]) -> None:
    data = load()
    print(data.summary())
    if argv[:1] == ["write"]:
        model = PendingDrift()
        model.fit(data.ins)
        path = write_submission(data.oos.sessions, model.predict(data.oos), argv[1])
        print(f"wrote {len(data.oos)} predictions to {path} (beta={model.beta:.3f})")
    else:
        res = cross_validate(PendingDrift, data.ins)
        print(res.round(3).to_string(index=False))
        print(f"mean 5-fold Sharpe over {len(res)} seeds: {res['sharpe'].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
