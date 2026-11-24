"""End-to-end: load -> CV a ladder of models -> refit the best on all in-sample -> write submission.

    python baseline.py --data path/to/data_dir
    python baseline.py --data data --only drift_ridge_sized      # skip CV of others
    python baseline.py --data data --model price_ridge          # force which model is submitted
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import kit
from kit import CFG


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--type", default="type_full", help="headline type column: type_full or type_short")
    ap.add_argument("--lmax", type=int, default=10, help="bars a headline keeps pushing the price")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--power", type=float, default=0.5, help="vol sizing: pred / vol**power")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma list of model names to CV")
    ap.add_argument("--model", default=None, help="model to submit (default: best CV)")
    a = ap.parse_args()
    t0 = time.time()

    D = kit.load(a.data)
    ins, oos = D["ins"], D.get("oos")
    n = CFG["n_seen"]
    S_ins = len(ins["ids"])
    y = ins["y"]

    # ---- combined universe: ins rows first, then oos rows
    seen = ins["seen"] if oos is None else np.concatenate([ins["seen"], oos["seen"]])
    S = len(seen)
    cols = ins["cols"]
    oos_idx = np.arange(S_ins, S)
    y_all = np.concatenate([y, np.full(S - S_ins, np.nan)])
    lp_seen = np.log(np.maximum(kit.close_of(seen, cols), 1e-12))
    T = ins["full"].shape[1]
    lp_full = np.full((S, T), np.nan)
    lp_full[:S_ins] = np.log(np.maximum(kit.close_of(ins["full"], cols), 1e-12))
    lp_full[S_ins:, :n] = lp_seen[S_ins:]

    P = F.price_features(seen, cols)
    vol = F.realized_vol(seen, cols)

    H = None
    if ins["heads_seen"] is not None:
        hi = F.prep_heads(ins["heads_seen"], ins["ids"])
        parts = [hi]
        if oos is not None and oos["heads_seen"] is not None:
            ho = F.prep_heads(oos["heads_seen"], oos["ids"])
            ho["sess_idx"] += S_ins
            parts.append(ho)
        H = pd.concat(parts, ignore_index=True)
        print(f"[heads] {len(H)} seen headlines, {H['type_full'].nunique()} full types, "
              f"{H['type_short'].nunique()} short types, {H['company'].nunique()} companies")
        print("[heads] most common types:\n" + H[a.type].value_counts().head(8).to_string())
    has_full = T >= n + CFG["horizon"]

    def mask_of(tr):
        m = np.zeros(S, bool)
        m[tr] = True
        return m

    def fit_lin(X, tr, te, sized=False):
        X = np.nan_to_num(np.asarray(X, float))
        m = ridge(a.alpha).fit(X[tr], y_all[tr])
        p, pr = m.predict(X[te]), m.predict(X[tr])
        if sized:
            p, pr = kit.vol_size(p, vol[te], a.power), kit.vol_size(pr, vol[tr], a.power)
        return kit.clip_sd(p, pr, a.clip)

    def drift_X(tr):
        rates = F.fit_rates(lp_full, H, mask_of(tr), a.type, a.lmax)
        return F.drift_features(H, S, rates, lp_seen, a.type, a.lmax)

    # ---- model ladder. each: f(train_idx, test_idx) -> preds
    M = {
        "constant_long": lambda tr, te: np.ones(len(te)),
        "momentum_5": lambda tr, te: P["mom_5"].to_numpy()[te],
        "reversal_5": lambda tr, te: -P["mom_5"].to_numpy()[te],
        "reversal_1": lambda tr, te: -P["mom_1"].to_numpy()[te],
        "price_ridge": lambda tr, te: fit_lin(P, tr, te),
        "price_ridge_sized": lambda tr, te: fit_lin(P, tr, te, sized=True),
    }
    if H is not None and has_full:
        M["drift_owed_raw"] = lambda tr, te: drift_X(tr)["d_owed"].to_numpy()[te]
        M["drift_ridge"] = lambda tr, te: fit_lin(drift_X(tr), tr, te)
        M["drift_ridge_sized"] = lambda tr, te: fit_lin(drift_X(tr), tr, te, sized=True)
        M["drift_price_sized"] = lambda tr, te: fit_lin(pd.concat([drift_X(tr), P], axis=1), tr, te, sized=True)
    if H is not None:
        M["type_target_ridge"] = lambda tr, te: fit_lin(
            pd.concat([F.target_type_scores(H, y_all, mask_of(tr)), P], axis=1), tr, te)
        txt = F.session_text(H, S, window=a.lmax)

        def tfidf(tr, te):
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True)
            Xtr = v.fit_transform(txt[tr])
            m = Ridge(alpha=a.alpha * 10).fit(Xtr, y_all[tr])
            p, pr = m.predict(v.transform(txt[te])), m.predict(Xtr)
            return kit.clip_sd(p, pr, a.clip)
        M["tfidf_recent"] = tfidf

    names = a.only.split(",") if a.only else list(M)
    res = {}
    for nm in names:
        res[nm] = kit.cv(M[nm], y, k=a.folds, seeds=tuple(range(a.seeds)), name=nm)
    tab = pd.DataFrame(res, index=["sharpe", "sd", "pooled"]).T.sort_values("sharpe", ascending=False)
    print("\n" + tab.round(3).to_string())
    tab.to_csv("cv_results.csv")

    if oos is None:
        print("[done] no oos data, nothing to submit")
        return
    pick = a.model or tab.index[0]
    print(f"\n[fit] refitting '{pick}' on all {S_ins} in-sample sessions")
    pred = M[pick](np.arange(S_ins), oos_idx)
    kit.write_submission(oos["ids"], pred, a.out, expected_ids=oos["ids"])
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
