"""model2 '+seen' + extra price features + sizing grid."""
import sys, numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import model2 as M, kit
from model2 import S, S_ins, y, lp, N, oos_idx, oos

lps = lp[:, :N]
mom3 = lps[:, -1] - lps[:, -4]; rev = lps[:, -1]; mom5 = lps[:, -1] - lps[:, -6]

def make(type_col="type_short", lmax=9, power=1.25, clip=2.0, extra=True, alpha=10.0, shrink=20.0):
    cache = {}
    def f(tr, te):
        m = np.zeros(S, bool); m[tr] = True
        rate, w = M.fit_rates(m, type_col, lmax, shrink=shrink, use_seen=True, profile=False)
        X, rv, v = M.drift_feats(rate, w, type_col)
        if extra:
            X = X.assign(mom3=mom3, rev=rev)
        Xn = X.to_numpy()
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xn[tr], y[tr])
        p, pr = mdl.predict(Xn[te]), mdl.predict(Xn[tr])
        p, pr = kit.vol_size(p, v[te], power), kit.vol_size(pr, v[tr], power)
        return kit.clip_sd(p, pr, clip)
    return f

V = {
 "base": make(extra=False),
 "+price": make(),
 "+price p1.5": make(power=1.5),
 "+price p1.0": make(power=1.0),
 "+price clip1.5": make(clip=1.5),
 "+price clip2.5": make(clip=2.5),
 "+price lmax10": make(lmax=10),
 "+price full types": make(type_col="type_full"),
}
if __name__ == "__main__":
    if sys.argv[1:2] == ["--submit"]:
        nm = sys.argv[2]; f = V[nm]
        kit.cv(f, y, k=5, seeds=(0, 1, 2), name=nm)
        kit.write_submission(oos["ids"], f(np.arange(S_ins), oos_idx), f"../submissions/kit_{nm.replace(' ','_')}.csv", expected_ids=oos["ids"])
    else:
        for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else V):
            kit.cv(V[nm], y, k=5, seeds=(0, 1, 2), name=nm)
