import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys, time, json
import multiprocessing as mp
import numpy as np
import pandas as pd

OUT = "/tmp/claude-0/-home-user-HRT/1b646c27-9e64-5f9e-b4f8-43257d3086ff/scratchpad/m3564"
DATA = "/home/user/HRT/data"
sys.path.insert(0, "/home/user/abcpeter123/quant-comp-2026/algoarena")
import model_3564 as M

T_START = time.time()


def log(*a):
    print(f"[{time.time() - T_START:7.1f}s]", *a, flush=True)


SD = None
R = None
FOLDS = None
CFG = None


def predict(sd, idx, model):
    F = M.features_for(sd, idx, model["P"], temp=model["cfg"]["temp"])
    return np.array([M.stage2_predict(row, model["S"]) for row in F.to_dict("records")], float)


def task(name):
    t0 = time.time()
    if name == "full":
        model = M.train(SD, R, np.arange(len(R)), CFG)
        ob = pd.read_parquet(f"{DATA}/oos_bars.parquet")
        oh = pd.read_parquet(f"{DATA}/oos_headlines.parquet")
        sdo = M.prepare_full(ob, oh)
        sess = sdo["sessions"]
        assert len(sess) == 10000 and sess[0] == 11000 and sess[-1] == 20999, (len(sess), sess[0], sess[-1])
        assert sdo["open"].shape[1] == 50, sdo["open"].shape
        p = predict(sdo, np.arange(len(sess)), model)
        np.save(f"{OUT}/m3564_oos.npy", p)
        np.save(f"{OUT}/m3564_oos_sessions.npy", sess)
        unseen = sorted(set(sdo["etypes"]) - set(SD["etypes"]))
        info = dict(name=name, secs=time.time() - t0, n=len(p), nan=int(np.isnan(p).sum()),
                    std=float(np.nanstd(p)), coef=[float(x) for x in model["P"]["coef"]],
                    beta=[float(x) for x in model["S"]["beta"]], lo=model["S"]["lo"], hi=model["S"]["hi"],
                    n_unseen_etypes=len(unseen), max_oos_hl_bar=int(oh.bar_ix.max()), max_oos_bar=int(ob.bar_ix.max()))
        return info
    f = int(name)
    tr = np.where(FOLDS != f)[0]
    te = np.where(FOLDS == f)[0]
    model = M.train(SD, R, tr, CFG)
    p = predict(SD, te, model)
    np.save(f"{OUT}/fold{f}.npy", np.stack([te.astype(float), p]))
    info = dict(name=name, secs=time.time() - t0, n=len(p), nan=int(np.isnan(p).sum()),
                coef=[float(x) for x in model["P"]["coef"]], beta=[float(x) for x in model["S"]["beta"]])
    if f == 0:
        # leakage check: blank out everything after bar 49 for the test sessions and re-predict
        sdt = dict(SD)
        for k in ("open", "high", "low", "close", "lr"):
            a = SD[k].copy()
            a[:, M.T0 + 1:] = np.nan
            sdt[k] = a
        sdt["hl_full"] = {s: (b[b <= M.T0], e[b <= M.T0], c[b <= M.T0]) for s, (b, e, c) in SD["hl_full"].items()}
        sub = te[:600]
        p2 = predict(sdt, sub, model)
        info["leak_check_n"] = int(len(sub))
        info["leak_check_maxabsdiff"] = float(np.max(np.abs(p2 - p[:600])))
        info["leak_check_nan"] = int(np.isnan(p2).sum())
    return info


def main():
    global SD, R, FOLDS, CFG
    bars = pd.read_parquet(f"{DATA}/ins_bars.parquet")
    hl = pd.read_parquet(f"{DATA}/ins_headlines.parquet")
    rets = pd.read_parquet(f"{DATA}/ins_returns.parquet").set_index("session").returns
    log("loaded", bars.shape, hl.shape, rets.shape)
    SD = M.prepare_full(bars, hl)
    sess = SD["sessions"]
    assert len(sess) == 10000 and sess[0] == 1000 and sess[-1] == 10999
    assert SD["open"].shape == (10000, 100), SD["open"].shape
    R = rets.reindex(sess).to_numpy(float)
    assert np.isfinite(R).all()
    FOLDS = np.random.default_rng(0).permutation(10000) % 5
    log("prepared; etypes", len(SD["etypes"]), "nan bars", int(np.isnan(SD["close"]).sum()))

    t0 = time.time()
    M.fit_base(SD, np.arange(2000))
    t2000 = time.time() - t0
    est_default = 45 * t2000 * 1.25 + 30
    log(f"fit_base(2000) = {t2000:.2f}s; est wall default cfg = {est_default:.0f}s")
    if est_default > 480:
        CFG = {"n_inner": 3, "em_iters": 5}
    else:
        CFG = None
    log("cfg override:", CFG)
    meta = dict(t2000=t2000, est_default=est_default, cfg_override=CFG, tasks=[])

    names = ["0", "1", "2", "3", "4", "full"]
    tp = time.time()
    with mp.get_context("fork").Pool(3) as pool:
        for info in pool.imap_unordered(task, names, chunksize=1):
            log("done", json.dumps(info))
            meta["tasks"].append(info)
            json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    meta["pool_wall"] = time.time() - tp

    oof = np.full(10000, np.nan)
    for f in range(5):
        a = np.load(f"{OUT}/fold{f}.npy")
        oof[a[0].astype(int)] = a[1]
    assert np.isfinite(oof).all()
    np.save(f"{OUT}/m3564_oof_seed0.npy", oof)
    pnl = oof * R
    sh = float(np.sqrt(252) * pnl.mean() / pnl.std(ddof=1))
    per = []
    for f in range(5):
        q = pnl[FOLDS == f]
        per.append(float(np.sqrt(252) * q.mean() / q.std(ddof=1)))
    meta["oof_sharpe"] = sh
    meta["fold_sharpes"] = per
    meta["total_wall"] = time.time() - T_START
    oos = np.load(f"{OUT}/m3564_oos.npy")
    meta["oos_shape"] = list(oos.shape)
    meta["oos_finite"] = bool(np.isfinite(oos).all())
    meta["oof_std"] = float(oof.std())
    meta["oos_std"] = float(oos.std())
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    log("OOF sharpe", sh, "per fold", per)
    log("ALL DONE", json.dumps({k: v for k, v in meta.items() if k != "tasks"}))


if __name__ == "__main__":
    main()
