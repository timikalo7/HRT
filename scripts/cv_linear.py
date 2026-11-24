"""OOF linear combination of normalised features; target y/vs, pred = yhat/vs (then centred)."""
import sys

import numpy as np

from model import folds, load, sharpe
from pipeline import features, fit

d = load("ins")
y = d.returns
S = len(d.sessions)
BOOT = np.random.default_rng(123).integers(0, S, size=(400, S))


def ztest(a, b):
    pa, pb = a * y, b * y
    A, B = pa[BOOT], pb[BOOT]
    da = A.mean(1) / A.std(1) - B.mean(1) / B.std(1)
    return (pa.mean() / pa.std() - pb.mean() / pb.std()) / (da.std() + 1e-18)


sets = {
    "g_only":   ["g"],
    "+n":       ["g", "n_first"],
    "+cum":     ["g", "cum_noise"],
    "+n+cum":   ["g", "n_first", "cum_noise"],
    "+all":     ["g", "n_first", "cum_noise", "noise5", "n_live"],
}
seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
for seed in seeds:
    fid = folds(S, 5, seed)
    oof = {k: np.zeros(S) for k in sets}
    coefs = {k: [] for k in sets}
    for k in range(5):
        tr = fid != k
        rates = fit(d, tr, sig_kind="park")
        F, vs = features(d, rates, "park")
        yn = y / vs
        for name, cols in sets.items():
            X = np.column_stack([np.ones(S), F[cols].to_numpy()])
            beta = np.linalg.lstsq(X[tr], yn[tr], rcond=None)[0]
            coefs[name].append(beta)
            oof[name][~tr] = (X[~tr] @ beta) / vs[~tr]
    print(f"\nseed {seed}")
    base = oof["g_only"] - oof["g_only"].mean()
    for name, f in oof.items():
        fc = f - f.mean()
        cb = np.mean(coefs[name], axis=0)
        print(f"  {name:8s} Sharpe {sharpe(fc, y):.3f}  z vs g_only {ztest(fc, base):+.2f}  "
              f"coef {np.array2string(cb[1:], precision=4)}")
