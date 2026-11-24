"""Variants of the drift model on CV. Patches features.fit_rates to quantise per-type rates.

    python tune.py            # print CV for each variant
"""
import sys

import numpy as np

import features as F

_orig = F.fit_rates
MODE = {"q": None}


def fit_rates_q(logp_full, heads, train_mask, type_col="type_full", lmax=10, shrink=20.0):
    r = _orig(logp_full, heads, train_mask, type_col, lmax, shrink)
    q = MODE["q"]
    if q is None:
        return r
    if q == "sign3":
        # snap to {-neg, 0, +pos} using the median of clearly signed types
        thr = MODE.get("thr", 1e-4)
        pos, neg = r[r > thr], r[r < -thr]
        out = r * 0.0
        out[r > thr] = pos.median()
        out[r < -thr] = neg.median()
        return out
    return r


F.fit_rates = fit_rates_q

if __name__ == "__main__":
    import baseline
    variants = [
        ("raw", None, {}),
        ("sign3 thr0.5bp", "sign3", {"thr": 0.5e-4}),
        ("sign3 thr1bp", "sign3", {"thr": 1e-4}),
        ("sign3 thr1.5bp", "sign3", {"thr": 1.5e-4}),
    ]
    base = ["baseline.py", "--data", "data", "--only", "drift_ridge_sized", "--seeds", "2",
            "--out", "_tmp.csv"] + sys.argv[1:]
    for name, q, kw in variants:
        MODE.clear()
        MODE.update({"q": q, **kw})
        print(f"--- {name}")
        sys.argv = base
        baseline.main()
