"""model4 (learned industries) + per-fold IRF forecast from main-industry headlines as an extra feature."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import model4 as Q, model2 as M, kit, irf
from lib import load
from model2 import S, S_ins, y, N, oos_idx, oos

piv_i, hl_i, sess_i, _ = load("ins"); piv_o, hl_o, sess_o, _ = load("oos")
assert (np.asarray(sess_i) == np.asarray(M.ins["ids"])).all() and (np.asarray(sess_o) == np.asarray(oos["ids"])).all()
main_by_s = dict(zip(Q.HA0["sess_idx"], Q.HA0["sess_idx"].map(Q.main)))  # kit session index -> main industry
cm = Q._cmap
def tag(hl, off):
    hl = hl.copy(); hl["ind"] = hl.co.map(cm); hl["main"] = (hl.si + off).map(Q.main)
    return hl[hl.ind == hl.main]
hi_m, ho_m = tag(hl_i, 0), tag(hl_o, S_ins)
tm = {t: i for i, t in enumerate(sorted(pd.concat([hl_i, hl_o]).tmpl.unique()))}
hi_m["tid"] = hi_m.tmpl.map(tm); ho_m["tid"] = ho_m.tmpl.map(tm)
base = Q.make()

def make(only_irf=False, power=1.25, clip=2.0):
    def f(tr, te):
        tri = tr[tr < S_ins]
        a, k = irf.fit(piv_i["close"], hi_m, tri, len(tm))
        fc = np.zeros(S)
        fc[:S_ins] = irf.forecast(hi_m, np.arange(S_ins), a, k)
        fc[S_ins:] = irf.forecast(ho_m.assign(si=ho_m.si), np.arange(len(sess_o)), a, k)
        # rebuild model4 features
        m = np.zeros(S, bool); m[tr] = True
        M.HA = Q.HA0[Q.HA0["inmain"]]
        rate, w = M.fit_rates(m, "type_short", 9, use_seen=True, profile=False)
        Xm, v = Q.feats(rate, w, "type_short", Q.HS0[Q.HS0["inmain"]])
        Xall, _ = Q.feats(rate, w, "type_short", Q.HS0)
        X = pd.DataFrame({"irf": fc}) if only_irf else pd.DataFrame({"irf": fc, "owed_main": Xm["owed"], "n_recent_main": Xm["n_recent"], "unexpl": Xall["unexpl"]})
        X["mom3"] = Q.mom3; X["rev"] = Q.rev
        Xn = X.to_numpy()
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xn[tr], y[tr])
        p, pr = mdl.predict(Xn[te]), mdl.predict(Xn[tr])
        p, pr = kit.vol_size(p, v[te], power), kit.vol_size(pr, v[tr], power)
        return kit.clip_sd(p, pr, clip)
    return f

V = {"irf+kit": make(), "irf only": make(only_irf=True)}
if __name__ == "__main__":
    if sys.argv[1:2] == ["--submit"]:
        nm = sys.argv[2]; f = V[nm]
        kit.cv(f, y, k=5, seeds=(0, 1, 2), name=nm)
        kit.write_submission(oos["ids"], f(np.arange(S_ins), oos_idx), f"../submissions/kit6_{nm.replace(' ','_').replace('+','_')}.csv", expected_ids=oos["ids"])
    else:
        for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else V):
            kit.cv(V[nm], y, k=5, seeds=(0, 1, 2), name=nm)
