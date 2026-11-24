"""Rank-1 impulse-response model of headline impact on bar returns.

    bar_logret[s, u] = sum_{headline h in s, t_h <= u} a[tmpl_h] * k[u - t_h] + noise

a = per-template impact, k = shared response profile over lags (normalised so sum(k[:L]) = 1).
Fitted by alternating least squares on ALL 100 bars of the training sessions (more data than
the 10-bar target alone). The forecast for 49->59 is sum_h a_h * sum_{u=50..59} k[u - t_h].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

L = 60  # max lag modelled


def _bar_returns(close: np.ndarray) -> np.ndarray:
    lc = np.log(np.concatenate([np.ones((close.shape[0], 1)), close], axis=1))
    r = np.diff(lc, axis=1)  # r[:, u] = log C[u] - log C[u-1], C[-1] = 1
    return r - r.mean(axis=0, keepdims=True)


def _events(hl, sess_idx, max_bar=99):
    """(row-session position, bar, template id) for headlines of the given sessions."""
    pos = {s: i for i, s in enumerate(sess_idx)}
    m = hl["si"].isin(pos) & (hl["bar_ix"] <= max_bar)
    h = hl[m]
    return h["si"].map(pos).to_numpy(), h["bar_ix"].to_numpy(), h["tid"].to_numpy()


def fit(close, hl, train_idx, n_tmpl, iters=8, ridge_a=50.0, ridge_k=1.0):
    """Fit (a, k) on sessions train_idx. hl must have column 'tid' (template id)."""
    R = _bar_returns(close[train_idx])
    n = len(train_idx)
    y = R.ravel()
    s, t, tid = _events(hl, train_idx)
    # expand each event over lags
    lag = np.arange(L)
    S = np.repeat(s, L); T = np.repeat(t, L) + np.tile(lag, len(s)); J = np.tile(lag, len(s))
    TID = np.repeat(tid, L)
    ok = T <= 99
    S, T, J, TID = S[ok], T[ok], J[ok], TID[ok]
    rows = S * 100 + T
    k = np.exp(-np.arange(L) / 8.0); k /= k.sum()
    a = np.zeros(n_tmpl)
    for _ in range(iters):
        A = sp.csr_matrix((k[J], (rows, TID)), shape=(n * 100, n_tmpl))
        a = np.linalg.solve((A.T @ A).toarray() + ridge_a * np.eye(n_tmpl), A.T @ y)
        K = sp.csr_matrix((a[TID], (rows, J)), shape=(n * 100, L))
        k = np.linalg.solve((K.T @ K).toarray() + ridge_k * np.eye(L) * 1e-6, K.T @ y)
        sc = k.sum(); k /= sc; a *= sc
    return a, k


def forecast(hl, sess_idx, a, k, start=49, end=59):
    """sum_h a_h * sum_{u=start+1..end} k[u - t_h] for headlines with t_h <= start."""
    s, t, tid = _events(hl, sess_idx, max_bar=start)
    kc = np.concatenate([[0.0], np.cumsum(np.concatenate([k, np.zeros(200)]))])
    w = kc[end - t + 1] - kc[start - t + 1]  # sum k[start+1-t .. end-t]
    out = np.zeros(len(sess_idx))
    np.add.at(out, s, a[tid] * w)
    return out
