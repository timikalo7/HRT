"""Per-session linear-Gaussian posterior over event multipliers m_e (true rate = r_e * m_e).

Prior (from exposure_test.py on IS):  E[m]=1, Var(m)=v_self, Cov(m_e,m_e') = v_same (same company,
same session) or v_diff (different company). Observation (normalised bars 0..49):
    y0[t] = Rn[t] - sum_e r_e UNIT[t-b_e] - mu = sum_e r_e (m_e - 1) UNIT[t-b_e] + noise(sig2)
Posterior mean: delta = C A^T (A C A^T + sig2 I)^-1 y0,  m_hat = 1 + delta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import UNIT, Data, bar_returns
from pipeline import HALF, SIG_REF, W, sigma


def company_of(h: pd.DataFrame) -> pd.Series:
    tok = h["headline"].str.split()
    return pd.Series(np.where(tok.str[2].isin(["Group", "Co"]), tok.str[:3].str.join(" "),
                              tok.str[:2].str.join(" ")), index=h.index)


def posterior_multipliers(d: Data, rates: pd.Series, v_self: float, v_same: float, v_diff: float,
                          sig2: float, mu: float = 0.0, sig_kind: str = "park", max_e: int = 32,
                          return_all: bool = False):
    """Returns per-headline DataFrame (first half) with columns si, b, r, m_hat (and company)."""
    S = len(d.sessions)
    vs = sigma(d, sig_kind) / SIG_REF
    Rn = bar_returns(d)[:, :HALF + 1] / vs[:, None]
    h = d.h[d.h["bar_ix"] <= HALF].copy()
    h["co"] = company_of(h)
    h = h.sort_values(["si", "bar_ix"]).reset_index(drop=True)
    si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
    r = h["stem"].map(rates).fillna(0.0).to_numpy()
    h["r"] = r
    # residual after prior-mean drift
    E = np.zeros_like(Rn)
    for L in range(11):
        t = b + L
        ok = t <= HALF
        np.add.at(E, (si[ok], t[ok]), r[ok] * UNIT[L])
    Y = Rn - E - mu
    # slot index of each event within its session
    slot = h.groupby("si").cumcount().to_numpy()
    n_per = np.bincount(si, minlength=S)
    K = min(max_e, n_per.max())
    keep = slot < K
    T = HALF + 1
    A = np.zeros((S, T, K))
    for L in range(11):
        t = b + L
        ok = keep & (t <= HALF)
        A[si[ok], t[ok], slot[ok]] = r[ok] * UNIT[L]
    co_codes = pd.factorize(h["co"])[0]
    CO = -np.ones((S, K), int)
    CO[si[keep], slot[keep]] = co_codes[keep]
    valid = CO >= 0
    same = (CO[:, :, None] == CO[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    C = np.where(same, v_same, v_diff) * (valid[:, :, None] & valid[:, None, :])
    idx = np.arange(K)
    C[:, idx, idx] = np.where(valid, v_self, 1.0)
    M = A @ C @ A.transpose(0, 2, 1)
    M[:, np.arange(T), np.arange(T)] += sig2
    z = np.linalg.solve(M, Y[:, :, None])[:, :, 0]
    delta = (C @ A.transpose(0, 2, 1) @ z[:, :, None])[:, :, 0]
    m_hat = np.ones(len(h))
    m_hat[keep] = 1.0 + delta[si[keep], slot[keep]]
    h["m_hat"] = m_hat
    h["b"] = b
    return h


def g_posterior(d: Data, rates: pd.Series, **kw) -> np.ndarray:
    h = posterior_multipliers(d, rates, **kw)
    S = len(d.sessions)
    g = np.zeros(S)
    np.add.at(g, h["si"].to_numpy(), h["r"].to_numpy() * h["m_hat"].to_numpy() * W[h["b"].to_numpy()])
    return g
