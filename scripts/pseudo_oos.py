"""End-to-end audit of make_submission.py logic: held-out IS fold treated exactly like OOS.

Test fold is truncated to bars 0-49 / headlines <= 49 (its 2nd half and returns are invisible),
its events are added to the rate fit as `extra`, coefficients fit on the train fold, preds centred
on the test fold, then scored against the hidden returns.
"""
import copy

import numpy as np

from model import Data, folds, load, sharpe
from pipeline import features, fit

SETS = {"g_only": ["g"], "n_cum": ["g", "n_first", "cum_noise"],
        "all": ["g", "n_first", "cum_noise", "noise5", "n_live"]}
d = load("ins")
y = d.returns
S = len(d.sessions)


def subset(d: Data, idx: np.ndarray, first_half: bool) -> Data:
    e = copy.copy(d)
    e.sessions = d.sessions[idx]
    nb = 50 if first_half else d.C.shape[1]
    e.O, e.H, e.L, e.C = (m[idx, :nb] for m in (d.O, d.H, d.L, d.C))
    h = d.h[np.isin(d.h["si"], idx)].copy()
    if first_half:
        h = h[h["bar_ix"] <= 49]
    remap = -np.ones(S, int)
    remap[idx] = np.arange(len(idx))
    h["si"] = remap[h["si"].to_numpy()]
    e.h = h
    e.returns = None if first_half else d.returns[idx]
    return e


for seed in (0, 1):
    fid = folds(S, 5, seed)
    res = {k: [] for k in SETS}
    pooled = {k: np.zeros(S) for k in SETS}
    for k in range(5):
        tr_idx, te_idx = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
        train, test = subset(d, tr_idx, False), subset(d, te_idx, True)
        rates = fit(train, np.ones(len(tr_idx), bool), sig_kind="park", extra=[test])
        Ft, vt = features(train, rates, "park")
        Fe, ve = features(test, rates, "park")
        for name, cols in SETS.items():
            Xt = np.column_stack([np.ones(len(Ft)), Ft[cols].to_numpy()])
            beta = np.linalg.lstsq(Xt, train.returns / vt, rcond=None)[0]
            p = np.column_stack([np.ones(len(Fe)), Fe[cols].to_numpy()]) @ beta / ve
            p -= p.mean()
            res[name].append(sharpe(p, y[te_idx]))
            pooled[name][te_idx] = p
    for name in SETS:
        print(f"seed {seed} {name:7s} pooled Sharpe {sharpe(pooled[name], y):.3f}  "
              f"per-fold {np.round(res[name], 2)}  mean {np.mean(res[name]):.3f}")
