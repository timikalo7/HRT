"""Stage 2: evaluate second-stage / variance variants on the cached per-fold features.
All coefficients are fit on the training fold only (Ftr, ytr); test rows only get predicted."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/user/HRT")
from arena.core import sharpe, load

OUT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["mu", "count", "unexpl", "live", "one", "s_raw", "s_dir", "mu_pos", "mu_neg", "conflict",
         "mu_own", "mu_peer", "mu_top", "ent_mu", "r1", "r5", "r10", "rv", "rv10", "ent"]
IX = {k: i for i, k in enumerate(NAMES)}
BASE = ["mu", "count", "unexpl", "live", "one"]

MEAN_VARIANTS = {
    "base": BASE,
    "a_s_raw": BASE + ["s_raw"],
    "a_s_dir": BASE + ["s_dir"],
    "a_both": BASE + ["s_raw", "s_dir"],
    "b_sign": BASE + ["mu_pos"],            # mu = mu_pos + mu_neg, so adding mu_pos splits by sign
    "b_sign_conf": BASE + ["mu_pos", "conflict"],
    "b_conf": BASE + ["conflict"],
    "c_own": BASE + ["mu_own"],             # mu = own + peer
    "c_top": BASE + ["mu_top"],
    "d_ent": BASE + ["ent_mu"],
    "e_r1": BASE + ["r1"],
    "e_r5": BASE + ["r5"],
    "e_r10": BASE + ["r10"],
    "e_all": BASE + ["r1", "r5", "r10"],
}


def var_design(F, f):
    return np.column_stack([F[:, IX["rv"]], F[:, IX["rv10"]], F[:, IX["count"]] * F[:, IX["rv"]],
                            F[:, IX["live"]] * F[:, IX["rv"]], F[:, IX["mu"]] ** 2])


def var_design_raw(F, f):
    return np.column_stack([F[:, IX["rv"]], F[:, IX["rv10"]], F[:, IX["count"]],
                            F[:, IX["live"]], F[:, IX["mu"]] ** 2])


def positions(Ftr, ytr, Fte, cols, vmode):
    c = [IX[k] for k in cols]
    beta = np.linalg.lstsq(Ftr[:, c], ytr, rcond=None)[0]
    ftr, fte = Ftr[:, c] @ beta, Fte[:, c] @ beta
    if vmode == "base":
        return fte / (15.0 * Fte[:, IX["rv"]] + fte ** 2)
    kind, design = vmode.split(":")
    D = var_design if design == "scaled" else var_design_raw
    Ztr, Zte = D(Ftr, ftr), D(Fte, fte)
    sc = Ztr.std(0) + 1e-300
    Ztr, Zte = Ztr / sc, Zte / sc
    if design == "raw":
        Ztr, Zte = np.column_stack([Ztr, np.ones(len(Ztr))]), np.column_stack([Zte, np.ones(len(Zte))])
    target = ytr ** 2 if kind == "m2" else (ytr - ftr) ** 2
    g = np.linalg.lstsq(Ztr, target, rcond=None)[0]
    floor = np.percentile(Ztr @ g, 5)          # floor from the training fold only
    v = np.maximum(Zte @ g, floor)
    return fte / (v + fte ** 2) if kind == "res" else fte / np.maximum(v, floor)


def run(variants, save=None):
    y = load().ins.returns
    res = {}
    oofs = {}
    for seed in (0, 1):
        oof = {k: np.zeros(len(y)) for k in variants}
        for f in range(5):
            z = np.load(f"{OUT}/feat_s{seed}_f{f}.npz")
            for k, (cols, vmode) in variants.items():
                oof[k][z["te"]] = positions(z["Ftr"], z["ytr"], z["Fte"], cols, vmode)
        for k in variants:
            res.setdefault(k, []).append(sharpe(oof[k], y))
            oofs[(k, seed)] = oof[k]
    b = res["base"]
    print(f"{'variant':28s} seed0   seed1   d0      d1")
    for k, v in res.items():
        print(f"{k:28s} {v[0]:.4f}  {v[1]:.4f}  {v[0]-b[0]:+.4f} {v[1]-b[1]:+.4f}")
    return res, oofs


if __name__ == "__main__":
    variants = {k: (c, "base") for k, c in MEAN_VARIANTS.items()}
    for vm in ("m2:scaled", "res:scaled", "m2:raw", "res:raw"):
        variants["var_" + vm] = (BASE, vm)
    res, oofs = run(variants)
    json.dump(res, open(f"{OUT}/stage2_results.json", "w"), indent=1)
