"""Full prediction pipeline: fit on in-sample (optionally + OOS first half), predict first halves.

Model (from event studies):
  per-bar return R[s,t] = vs_s * sum_e r_e * UNIT[t - b_e] + noise,  vs_s = sigma_s / SIG_REF
  r_e = rate[stem_e] (+ per-event deviation, inferred from the observed path if het_c > 0)
Prediction for the 49->59 window: E[y] ~ vs * g, Var[y] ~ vs^2  =>  pred = g / vs, centred.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from model import UNIT, Data, bar_returns, template_rates_deconv

SIG_REF = 0.0026
HALF = 49
# remaining drift units inside bars 50..59 for a headline at bar b (0..49)
W = np.array([UNIT[max(0, 50 - b):60 - b].sum() for b in range(HALF + 1)])


def sigma(d: Data, kind: str = "cc") -> np.ndarray:
    R = bar_returns(d)[:, :HALF + 1]
    cc = R.std(axis=1)
    if kind == "cc":
        return cc
    lr = np.log(d.H[:, :HALF + 1] / d.L[:, :HALF + 1])
    park = np.sqrt((lr ** 2).mean(axis=1) / (4 * np.log(2)))
    if kind == "park":
        return park * (cc.mean() / park.mean())  # calibrate scale to close-to-close
    if kind == "mix":
        return 0.5 * cc + 0.5 * park * (cc.mean() / park.mean())
    raise ValueError(kind)


def _stem_view(d: Data, first_half_only: bool = False) -> Data:
    ds = copy.copy(d)
    h = d.h.assign(templ=d.h["stem"])
    ds.h = h[h["bar_ix"] <= HALF] if first_half_only else h
    return ds


def fit(ins: Data, train_mask: np.ndarray, lam: float = 400.0, sig_kind: str = "cc",
        extra: list[Data] = ()) -> pd.Series:
    """Stem rates in normalised units (per-bar return at sigma = SIG_REF)."""
    R = bar_returns(ins) / (sigma(ins, sig_kind) / SIG_REF)[:, None]
    ex = []
    for e in extra:  # first-half-only data (e.g. OOS): bars 0..49 and headlines <= 49
        Re = bar_returns(e)[:, :HALF + 1] / (sigma(e, sig_kind) / SIG_REF)[:, None]
        ex.append((_stem_view(e, first_half_only=True), Re))
    return template_rates_deconv(_stem_view(ins), R, train_mask, lam=lam, extra=ex)


def features(d: Data, rates: pd.Series, sig_kind: str = "park") -> tuple[pd.DataFrame, np.ndarray]:
    """Normalised features (units: per-bar return at SIG_REF) and vs. Uses bars/headlines <= 49 only."""
    S = len(d.sessions)
    vs = sigma(d, sig_kind) / SIG_REF
    h = d.h[d.h["bar_ix"] <= HALF]
    si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
    r = h["stem"].map(rates).fillna(0.0).to_numpy()
    g = np.zeros(S)
    np.add.at(g, si, r * W[b])
    Rn = bar_returns(d)[:, :HALF + 1] / vs[:, None]
    Ex = np.zeros_like(Rn)
    for L in range(11):
        t = b + L
        ok = t <= HALF
        np.add.at(Ex, (si[ok], t[ok]), r[ok] * UNIT[L])
    N = Rn - Ex
    n1 = np.bincount(si, minlength=S).astype(float)
    f = pd.DataFrame({
        "g": g,
        "cum_noise": N.sum(1),
        "noise5": N[:, -5:].sum(1),
        "n_first": n1,
        "n_live": np.bincount(si[b >= 40], minlength=S).astype(float),
    })
    return f, vs


def predict(d: Data, rates: pd.Series, sig_kind: str = "cc", het_c: float = 0.0, het_c0: float = 0.0,
            vol_power: float = 1.0) -> np.ndarray:
    S = len(d.sessions)
    vs = sigma(d, sig_kind) / SIG_REF
    h = d.h[d.h["bar_ix"] <= HALF]
    si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
    r = h["stem"].map(rates).fillna(0.0).to_numpy()
    r_post = r.copy()
    if het_c > 0 or het_c0 > 0:
        Rn = bar_returns(d)[:, :HALF + 1] / vs[:, None]
        Ex = np.zeros_like(Rn)
        for L in range(11):
            t = b + L
            ok = t <= HALF
            np.add.at(Ex, (si[ok], t[ok]), r[ok] * UNIT[L])
        live = np.flatnonzero(b >= 40)
        for i in live:
            n = HALF - b[i] + 1
            u = UNIT[:n]
            ex = np.dot(u, Rn[si[i], b[i]:HALF + 1] - Ex[si[i], b[i]:HALF + 1]) / np.dot(u, u)
            tau2 = (het_c * abs(r[i]) + het_c0) ** 2
            k = tau2 / (tau2 + SIG_REF ** 2 / np.dot(u, u))
            r_post[i] = r[i] + k * ex
    g = np.zeros(S)
    np.add.at(g, si, r_post * W[b])
    return g / vs ** vol_power
