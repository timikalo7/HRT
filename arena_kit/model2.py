"""Drift model v2: rates from all visible paths, fitted lag profile, residual-vol sizing.

    python model2.py                 # CV the variants
    python model2.py --submit NAME   # refit NAME on all in-sample and write ../predictions.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import kit
from kit import CFG

N, HZ = CFG["n_seen"], CFG["horizon"]

D = kit.load("data")
ins, oos = D["ins"], D["oos"]
S_ins = len(ins["ids"])
S = S_ins + len(oos["ids"])
y = ins["y"]
cols = ins["cols"]
T = ins["full"].shape[1]
lp = np.full((S, T), np.nan)
lp[:S_ins] = np.log(kit.close_of(ins["full"], cols))
lp[S_ins:, :N] = np.log(kit.close_of(oos["seen"], cols))
R = np.diff(lp, axis=1)                      # R[:, k] = return of bar k+1
seen = np.concatenate([ins["seen"], oos["seen"]])

hi = F.prep_heads(ins["heads"], ins["ids"])
ho = F.prep_heads(oos["heads_seen"], oos["ids"])
ho["sess_idx"] += S_ins
HA = pd.concat([hi, ho], ignore_index=True)  # all headlines incl. ins bars >= N (research only)
HS = HA[HA["t"] < N].reset_index(drop=True)  # what a model may see for prediction
oos_idx = np.arange(S_ins, S)


def fit_rates(train_full, type_col, lmax=10, shrink=20.0, use_seen=True, profile=True):
    """train_full: bool mask of sessions whose full path may be used.
    use_seen: also use bars < N of every other session (no targets involved)."""
    h = HA if use_seen else HA[train_full[HA["sess_idx"].to_numpy()]]
    s, t = h["sess_idx"].to_numpy(), h["t"].to_numpy()
    last = np.where(train_full[s], T - 1, N - 1)          # last usable bar index per headline
    mu = np.nanmean(R[train_full])
    recs = []
    for L in range(1, lmax + 1):
        ok = (t + L <= last) & (t + L >= 1)
        rr = R[s[ok], t[ok] + L - 1] - mu
        recs.append(pd.DataFrame({"type": h[type_col].to_numpy()[ok], "L": L, "r": rr}))
    d = pd.concat(recs)
    core = d[d["L"] <= min(8, lmax)]
    g = core.groupby("type")["r"].agg(["sum", "count"])
    rate = g["sum"] / (g["count"] + shrink)
    w = np.ones(lmax)
    if profile:
        d["k"] = d["type"].map(rate)
        num = (d["r"] * d["k"]).groupby(d["L"]).sum()
        den = (d["k"] ** 2).groupby(d["L"]).sum()
        w = (num / den).reindex(range(1, lmax + 1)).fillna(0).to_numpy()
    return rate, w


def drift_feats(rate, w, type_col):
    lmax = len(w)
    s, t = HS["sess_idx"].to_numpy(), HS["t"].to_numpy()
    k = HS[type_col].map(rate).fillna(0.0).to_numpy()
    cw = np.concatenate([[0], np.cumsum(w)])               # cw[j] = sum w[1..j]
    lo = np.clip(N - t, 1, lmax + 1)                       # first lag landing in bars N..N+HZ-1 (bar index t+L)
    hi_ = np.clip(N - 1 + HZ - t, 0, lmax)
    owed_w = np.where(hi_ >= lo, cw[hi_] - cw[lo - 1], 0.0)
    done_w = cw[np.clip(N - 1 - t, 0, lmax)]
    owed = np.bincount(s, k * owed_w, S)
    done = np.bincount(s, k * done_w, S)
    # expected per-bar drift path over seen bars, to get residual volatility
    drift_path = np.zeros((S, N - 1))
    for L in range(1, lmax + 1):
        b = t + L - 1                                      # index into R
        ok = b < N - 1
        np.add.at(drift_path, (s[ok], b[ok]), k[ok] * w[L - 1])
    lps = lp[:, :N]
    resid = np.diff(lps, axis=1) - drift_path
    X = pd.DataFrame({
        "owed": owed,
        "unexpl": (lps[:, -1] - lps[:, 0]) - done,
        "n_recent": np.bincount(s[t >= N - lmax], minlength=S).astype(float),
    })
    return X, resid.std(1), np.diff(lps, axis=1).std(1)


def make_model(type_col="type_short", lmax=10, use_seen=True, profile=True, power=1.25,
               resid_vol=True, alpha=10.0, clip=2.0, raw=False):
    def f(tr, te):
        m = np.zeros(S, bool)
        m[tr] = True
        rate, w = fit_rates(m, type_col, lmax, use_seen=use_seen, profile=profile)
        X, rv, v = drift_feats(rate, w, type_col)
        vol = rv if resid_vol else v
        if raw:
            p, pr = X["owed"].to_numpy()[te], X["owed"].to_numpy()[tr]
        else:
            Xn = X.to_numpy()
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xn[tr], y[tr])
            p, pr = mdl.predict(Xn[te]), mdl.predict(Xn[tr])
        p, pr = kit.vol_size(p, vol[te], power), kit.vol_size(pr, vol[tr], power)
        return kit.clip_sd(p, pr, clip)
    return f


VARIANTS = {
    "v1_like": make_model(use_seen=False, profile=False, resid_vol=False, lmax=9),
    "+seen": make_model(use_seen=True, profile=False, resid_vol=False, lmax=9),
    "+seen+profile": make_model(use_seen=True, profile=True, resid_vol=False),
    "+seen+profile+rvol": make_model(use_seen=True, profile=True, resid_vol=True),
    "all, power2": make_model(power=2.0),
    "all, power1.5": make_model(power=1.5),
    "all, type_full": make_model(type_col="type_full"),
    "all, raw owed": make_model(raw=True),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.submit:
        kit.cv(VARIANTS[a.submit], y, k=5, seeds=(0, 1, 2), name=a.submit)
        pred = VARIANTS[a.submit](np.arange(S_ins), oos_idx)
        kit.write_submission(oos["ids"], pred, "../predictions.csv", expected_ids=oos["ids"])
    else:
        for nm in (a.only.split(",") if a.only else VARIANTS):
            kit.cv(VARIANTS[nm], y, k=5, seeds=(0, 1), name=nm)
