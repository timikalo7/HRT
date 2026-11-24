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
HA0["ind"] = HA0["company"].map(lambda c: next((w2i[w] for w in str(c).split()[1:] if w in w2i), "?"))
# main industry from companies visible in bars < N only (same info in ins and oos)
vis = HA0[HA0["t"] < N].drop_duplicates(["sess_idx", "company"])
cs = vis.groupby(["sess_idx", "ind"]).size().unstack(fill_value=0)
main = cs.idxmax(axis=1)
HA0["inmain"] = (HA0["ind"] == HA0["sess_idx"].map(main)).to_numpy()
HS0 = HA0[HA0["t"] < N].reset_index(drop=True)
lps = lp[:, :N]; mom3 = lps[:, -1] - lps[:, -4]; rev = lps[:, -1]

def feats(rate, w, tc, sub):
    M.HS = sub.reset_index(drop=True)
    X, rv, v = M.drift_feats(rate, w, tc)
    return X, v

def make(tc="type_short", lmax=9, power=1.25, clip=2.0, alpha=10.0, off=True):
    def f(tr, te):
        m = np.zeros(S, bool); m[tr] = True
        M.HA = HA0[HA0["inmain"]]
        rate, w = M.fit_rates(m, tc, lmax, use_seen=True, profile=False)
        Xm, v = feats(rate, w, tc, HS0[HS0["inmain"]])
        X = pd.DataFrame({"owed_main": Xm["owed"], "n_recent_main": Xm["n_recent"], "mom3": mom3, "rev": rev})
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

V = {"ind": make(), "ind nooff": make(off=False), "ind full": make(tc="type_full"), "ind p1.5": make(power=1.5), "ind lmax10": make(lmax=10)}
if __name__ == "__main__":
    if sys.argv[1:2] == ["--submit"]:
        nm = sys.argv[2]; f = V[nm]
        kit.cv(f, y, k=5, seeds=(0, 1, 2), name=nm)
        kit.write_submission(oos["ids"], f(np.arange(S_ins), oos_idx), f"../submissions/kit4_{nm.replace(' ','_')}.csv", expected_ids=oos["ids"])
    else:
        for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else V):
            kit.cv(V[nm], y, k=5, seeds=(0, 1, 2), name=nm)
