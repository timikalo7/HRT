"""OOF evaluation of the per-session exposure posterior (g_post) vs the prior-mean drift (g)."""
import sys

import numpy as np

from exposure import g_posterior
from model import bar_returns, folds, load, sharpe
from pipeline import SIG_REF, features, fit, sigma

d = load("ins")
y = d.returns
S = len(d.sessions)
BOOT = np.random.default_rng(123).integers(0, S, size=(400, S))


def ztest(a, b):
    pa, pb = a * y, b * y
    A, B = pa[BOOT], pb[BOOT]
    da = A.mean(1) / A.std(1) - B.mean(1) / B.std(1)
    return (pa.mean() / pa.std() - pb.mean() / pb.std()) / (da.std() + 1e-18)


vs_all = sigma(d, "park") / SIG_REF
Rn = bar_returns(d)[:, :50] / vs_all[:, None]
priors = {
    "post_base": dict(v_self=1.53, v_same=0.756, v_diff=-0.252),
    "post_x0.5": dict(v_self=0.765, v_same=0.378, v_diff=-0.126),
    "post_x1.5": dict(v_self=2.3, v_same=1.134, v_diff=-0.378),
    "post_noco": dict(v_self=1.53, v_same=0.0, v_diff=0.0),   # event-level only (old idea)
}
seeds = [int(s) for s in sys.argv[1:]] or [0]
for seed in seeds:
    fid = folds(S, 5, seed)
    G = {k: np.zeros(S) for k in ["g"] + list(priors)}
    F_all = None
    oof = {}
    feats = {}
    for k in range(5):
        tr = fid != k
        rates = fit(d, tr, sig_kind="park")
        F, vs = features(d, rates, "park")
        # noise variance per normalised bar, from training sessions' first-half residuals
        resid = Rn[tr] - 0  # rough: total per-bar var minus nothing (drift var is tiny)
        sig2 = resid.var()
        feats[k] = (F, vs, tr)
        G["g"][~tr] = F["g"].to_numpy()[~tr]
        for name, pr in priors.items():
            gp = g_posterior(d, rates, sig2=sig2, **pr)
            G[name][~tr] = gp[~tr]
            feats[k] = feats[k] + ((name, gp),) if False else feats[k]
            F[name] = gp
        feats[k] = (F, vs, tr)
    print(f"seed {seed}: sig2 (normalised) = {sig2:.3e}  (sd {np.sqrt(sig2)*1e4:.1f} bps)")
    for name in ["g"] + list(priors):
        cols = [name, "n_first", "cum_noise"]
        p_raw = np.zeros(S)
        p_st = np.zeros(S)
        for k in range(5):
            F, vs, tr = feats[k]
            p_raw[~tr] = (F[name].to_numpy() / vs)[~tr]
            X = np.column_stack([np.ones(S), F[cols].to_numpy()])
            beta = np.linalg.lstsq(X[tr], y[tr] / vs[tr], rcond=None)[0]
            p_st[~tr] = (X @ beta / vs)[~tr]
        oof[name] = p_st - p_st.mean()
        print(f"  {name:10s} alone Sharpe {sharpe(p_raw - p_raw.mean(), y):.3f}   "
              f"+n+cum Sharpe {sharpe(oof[name], y):.3f}  z vs g {ztest(oof[name], oof['g']):+.2f}")
