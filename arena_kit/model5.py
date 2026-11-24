"""model4 + infer main industry from price reaction in bars < N (plus company-count prior)."""
import sys, numpy as np, pandas as pd
import model4 as Q, model2 as M, kit
from model2 import S, S_ins, y, lp, N, oos_idx, oos
H = Q.HA0
inds = sorted(H["ind"].unique()); ii = {k: i for i, k in enumerate(inds)}

def relabel(lam, tc="type_short", lmax=9):
    m = np.zeros(S, bool); m[:S_ins] = True  # rates from main-by-count headlines (no targets used)
    M.HA = H[H["inmain"]]; rate, _ = M.fit_rates(m, tc, lmax, use_seen=True, profile=False)
    h = H[H["t"] < N - 1]
    s, t = h["sess_idx"].to_numpy(), h["t"].to_numpy()
    k = h[tc].map(rate).fillna(0).to_numpy()
    end = np.minimum(t + lmax, N - 1); nb = end - t
    e = k * (lp[s, end] - lp[s, t]); q = k * k * nb
    sig2 = np.nanvar(np.diff(lp[:, :N], axis=1), axis=1)[s]
    b1, b0 = 1.3, 0.2
    ll = ((b1 - b0) * e - 0.5 * (b1 ** 2 - b0 ** 2) * q) / sig2
    j = h["ind"].map(ii).to_numpy()
    L = np.zeros((S, len(inds))); np.add.at(L, (s, j), ll)
    cnt = Q.cs.reindex(columns=inds, fill_value=0).reindex(range(S), fill_value=0).to_numpy()
    main = np.array(inds)[np.argmax(L + lam * cnt, axis=1)]
    Q.HA0 = H.assign(inmain=(H["ind"] == pd.Series(main)[H["sess_idx"]].to_numpy()).to_numpy())
    Q.HS0 = Q.HA0[Q.HA0["t"] < N].reset_index(drop=True)
    return main

if __name__ == "__main__":
    base = np.array(Q.cs.reindex(range(S), fill_value=0).idxmax(axis=1))
    for lam in [float(x) for x in sys.argv[1].split(",")]:
        mn = relabel(lam)
        print("lam", lam, "agree w/ count-main", round((mn == base).mean(), 3))
        f = Q.make()
        kit.cv(f, y, k=5, seeds=(0, 1, 2), name=f"infer lam={lam}")
        if len(sys.argv) > 2:
            kit.write_submission(oos["ids"], f(np.arange(S_ins), oos_idx), f"../submissions/kit5_lam{lam}.csv", expected_ids=oos["ids"])
