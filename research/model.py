"""Production model. Same code path for CV and for the OOS submission (no train/serve skew).

v0: headline impulse-response forecast only.
v1: v0 + short-term momentum + 50-bar reversal (linear), sized by 1/Parkinson-vol.
Only bars 0-49 and headlines with bar <= 49 are used at prediction time.
"""
from __future__ import annotations

import numpy as np

import irf
from lib import HALF


def price_feats(piv):
    C, H, Lo = piv["close"][:, :HALF + 1], piv["high"][:, :HALF + 1], piv["low"][:, :HALF + 1]
    lc = np.log(np.c_[np.ones(len(C)), C])  # lc[:, u+1] = log C[u]; C[-1] = 1 (prices start at 1)
    mom3 = lc[:, HALF + 1] - lc[:, HALF - 2]
    rev50 = lc[:, HALF + 1]
    vol = np.sqrt((np.log(H / Lo) ** 2).mean(axis=1) / (4 * np.log(2)))
    return mom3, rev50, vol


class Model:
    def __init__(self, version: str = "v1"):
        self.version = version

    def fit(self, piv, hl, idx, ret, tmap):
        self.tmap = tmap
        hl = hl.assign(tid=hl.tmpl.map(tmap))
        self.a, self.k = irf.fit(piv["close"], hl, idx, len(tmap))
        if self.version == "v0":
            return self
        # second stage fitted on in-sample IRF forecasts (a,k have ~430+60 params on 1M bar obs,
        # so in-sample overfit of the forecast is negligible; CV below checks this end-to-end)
        f = irf.forecast(hl, idx, self.a, self.k)
        mom3, rev50, vol = (x[idx] for x in price_feats(piv))
        X = np.c_[np.ones(len(idx)), f, mom3, rev50]
        self.b = np.linalg.lstsq(X, ret[idx], rcond=None)[0]
        self.vol_med = np.median(vol)
        return self

    def predict(self, piv, hl, idx):
        hl = hl.assign(tid=hl.tmpl.map(self.tmap))
        f = irf.forecast(hl, idx, self.a, self.k)
        if self.version == "v0":
            return f - f.mean()
        mom3, rev50, vol = (x[idx] for x in price_feats(piv))
        mu = np.c_[np.ones(len(idx)), f, mom3, rev50] @ self.b
        mu = mu - mu.mean()  # target is market-adjusted: a common offset only adds noise
        return mu / (vol / self.vol_med)
