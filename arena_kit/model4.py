"""model3 + main-industry split: rates fitted on main-industry headlines; owed drift split main/off."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import model2 as M, kit
from model2 import S, S_ins, y, lp, N, oos_idx, oos

IND = {"tech":"Systems Microchips Computing Software Technologies Networks Devices Solutions".split(),
 "health":"Sciences Genomics Labs Diagnostics Biotech Pharmaceuticals Therapeutics Biopharma Biosciences".split(),
 "consumer":"Commerce Marketplace Retail Outlets Brands Stores Goods".split(),
 "fin":"Securities Capital Investments Holdings Asset Financial Trading Partners Advisors".split(),
 "energy":"Fuels Power Resources Grid Renewables Energy Industries".split()}
w2i = {w: k for k, v in IND.items() for w in v}
HA0 = M.HA.copy()
import os
_ci = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_industry.csv"))
_cmap = dict(zip(_ci["co"], _ci["cl"].astype(str)))  # industries learned from company co-occurrence
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from lib import company_of as _co
HA0["co2"] = HA0["headline"].map(_co)
HA0["ind"] = HA0["co2"].map(_cmap).fillna("?")
# main industry from companies visible in bars < N only (same info in ins and oos)
vis = HA0[HA0["t"] < N].drop_duplicates(["sess_idx", "company"])
cs = vis.groupby(["sess_idx", "ind"]).size().unstack(fill_value=0)
main = cs.idxmax(axis=1)
TIES = os.environ.get("TIES") == "1"
if TIES:  # every industry tied for the max counts as main
    ismax = cs.eq(cs.max(axis=1), axis=0).stack()
    HA0["inmain"] = HA0.set_index(["sess_idx", "ind"]).index.map(ismax).fillna(False).to_numpy().astype(bool)
else:
    HA0["inmain"] = (HA0["ind"] == HA0["sess_idx"].map(main)).to_numpy()
HA0["type_ind"] = HA0["type_short"] + "|" + HA0["ind"]
HA0["typef_ind"] = HA0["type_full"] + "|" + HA0["ind"]
HS0 = HA0[HA0["t"] < N].reset_index(drop=True)
lps = lp[:, :N]; mom3 = lps[:, -1] - lps[:, -4]; rev = lps[:, -1]

def feats(rate, w, tc, sub):
    M.HS = sub.reset_index(drop=True)
    X, rv, v = M.drift_feats(rate, w, tc)
    return X, v

def make(tc="type_short", lmax=9, power=1.25, clip=2.0, alpha=10.0, off=True, shrink=20.0, price=True):
    def f(tr, te):
        m = np.zeros(S, bool); m[tr] = True
        M.HA = HA0[HA0["inmain"]]
        rate, w = M.fit_rates(m, tc, lmax, shrink=shrink, use_seen=True, profile=False)
        Xm, v = feats(rate, w, tc, HS0[HS0["inmain"]])
        X = pd.DataFrame({"owed_main": Xm["owed"], "n_recent_main": Xm["n_recent"]})
        if price: X["mom3"] = mom3; X["rev"] = rev
        Xall, _ = feats(rate, w, tc, HS0)
        X["unexpl"] = Xall["unexpl"]  # total move minus explained (all headlines, main-industry rates)
        if off:
            Xo, _ = feats(rate, w, tc, HS0[~HS0["inmain"]]); X["owed_off"] = Xo["owed"]
        Xn = X.to_numpy()
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xn[tr], y[tr])
        p, pr = mdl.predict(Xn[te]), mdl.predict(Xn[tr])
        p, pr = kit.vol_size(p, v[te], power), kit.vol_size(pr, v[tr], power)
        return kit.clip_sd(p, pr, clip)
    return f

V = {"ind": make(), "nooff": make(off=False), "lmax10": make(lmax=10), "full": make(tc="type_full"), "shr5": make(shrink=5.0), "shr60": make(shrink=60.0), "noprice": make(price=False), "p1.5": make(power=1.5), "clip3": make(clip=3.0), "a1": make(alpha=1.0), "type_ind": make(tc="type_ind"), "typef_ind": make(tc="typef_ind"), "ind nooff": make(off=False), "ind full": make(tc="type_full"), "ind p1.5": make(power=1.5), "ind lmax10": make(lmax=10)}
if __name__ == "__main__":
    if sys.argv[1:2] == ["--submit"]:
        nm = sys.argv[2]; f = V[nm]
        kit.cv(f, y, k=5, seeds=(0, 1, 2), name=nm)
        kit.write_submission(oos["ids"], f(np.arange(S_ins), oos_idx), f"../submissions/kit4_{nm.replace(' ','_')}.csv", expected_ids=oos["ids"])
    else:
        for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else V):
            kit.cv(V[nm], y, k=5, seeds=(0, 1, 2), name=nm)
