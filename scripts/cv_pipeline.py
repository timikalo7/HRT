"""OOF comparison of pipeline variants, paired against the stem-S400 baseline (no vol handling)."""
import sys

import numpy as np

from model import folds, load, sharpe
from pipeline import fit, predict

d = load("ins")
y = d.returns
S = len(d.sessions)


BOOT = np.random.default_rng(123).integers(0, S, size=(400, S))


def paired(a, b):
    """Bootstrap z-score of the Sharpe difference (paired resampling of sessions)."""
    pa, pb = a * y, b * y
    A, B = pa[BOOT], pb[BOOT]
    da = A.mean(1) / A.std(1) - B.mean(1) / B.std(1)
    point = pa.mean() / pa.std() - pb.mean() / pb.std()
    return point / (da.std() + 1e-18)


variants = {
    "base_noVol":     dict(fit=dict(sig_kind="cc"), pred=dict(sig_kind="cc", vol_power=0.0), fit_raw=True),
    "vol_cc_p1":      dict(fit=dict(sig_kind="cc"), pred=dict(sig_kind="cc", vol_power=1.0)),
    "vol_cc_p0":      dict(fit=dict(sig_kind="cc"), pred=dict(sig_kind="cc", vol_power=0.0)),
    "vol_cc_p2":      dict(fit=dict(sig_kind="cc"), pred=dict(sig_kind="cc", vol_power=2.0)),
    "vol_park_p1":    dict(fit=dict(sig_kind="park"), pred=dict(sig_kind="park", vol_power=1.0)),
    "vol_mix_p1":     dict(fit=dict(sig_kind="mix"), pred=dict(sig_kind="mix", vol_power=1.0)),
    "mix_het0.8":     dict(fit=dict(sig_kind="mix"), pred=dict(sig_kind="mix", het_c=0.8, het_c0=0.6e-4)),
    "mix_het1.5":     dict(fit=dict(sig_kind="mix"), pred=dict(sig_kind="mix", het_c=1.5, het_c0=1e-4)),
}
seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
for seed in seeds:
    fid = folds(S, 5, seed)
    oof = {k: np.zeros(S) for k in variants}
    for k in range(5):
        tr = fid != k
        cache = {}
        for name, v in variants.items():
            key = (v["fit"]["sig_kind"], v.get("fit_raw", False))
            if key not in cache:
                if v.get("fit_raw"):
                    from pipeline import SIG_REF  # raw = no normalisation: use constant sigma
                    import pipeline
                    orig = pipeline.sigma
                    pipeline.sigma = lambda dd, kind="cc": np.full(len(dd.sessions), SIG_REF)
                    cache[key] = fit(d, tr)
                    pipeline.sigma = orig
                else:
                    cache[key] = fit(d, tr, **v["fit"])
            oof[name][~tr] = predict(d, cache[key], **v["pred"])[~tr]
    print(f"\nseed {seed}")
    base = oof["base_noVol"] - oof["base_noVol"].mean()
    for name, f in oof.items():
        fc = f - f.mean()
        print(f"  {name:12s} centred Sharpe {sharpe(fc, y):.3f}  corr {np.corrcoef(fc, y)[0,1]:.4f}"
              f"  paired t vs base {paired(fc, base):+.2f}")
