"""Core plumbing for Algo Arena: load data, score like the leaderboard, cross-validate, write submission.

Edit CFG first if tonight's column / file names differ.
"""
import glob
import os

import numpy as np
import pandas as pd

CFG = dict(
    session="session",      # session id column
    bar="bar_ix",           # bar index column
    headline="headline",    # headline text column
    target="returns",       # target column in ins_returns
    sub_id="headline_id",   # submission id column name (yes, the old challenge called it this)
    sub_pred="pred",        # submission prediction column name
    n_seen=50,              # bars visible in oos (0..n_seen-1)
    horizon=10,             # target = close[n_seen-1+horizon] / close[n_seen-1] - 1
    price_cols=("open", "high", "low", "close"),
)


# ---------------------------------------------------------------- loading
def _read(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find(data_dir, key):
    hits = sorted(glob.glob(os.path.join(data_dir, f"*{key}*")))
    return hits[0] if hits else None


def bar_array(bars, n=None):
    """Long bars DataFrame -> (ids, array of shape (S, T, C)). Gaps forward-filled."""
    s, b = CFG["session"], CFG["bar"]
    if n is not None:
        bars = bars[bars[b] < n]
    ids = np.sort(bars[s].unique())
    cols = [c for c in CFG["price_cols"] if c in bars.columns]
    mats = []
    for c in cols:
        m = bars.pivot(index=s, columns=b, values=c).reindex(ids)
        mats.append(m.ffill(axis=1).bfill(axis=1).to_numpy(float))
    return ids, np.stack(mats, -1), cols


def load(data_dir="data"):
    """Returns dict(ins=..., oos=...). Each part has:
    ids        sorted session ids
    seen       (S, n_seen, C) bars the model may use
    full       (S, T, C) all bars (ins only; equals seen for oos)
    cols       price column names in the last axis
    heads      headlines DataFrame (or None), only bars < n_seen kept in 'heads_seen'
    y          target aligned to ids (ins only)
    """
    out = {}
    for split in ("ins", "oos"):
        bp = _find(data_dir, f"{split}_bars")
        if bp is None:
            continue
        bars = _read(bp)
        ids, full, cols = bar_array(bars)
        part = dict(ids=ids, full=full, seen=full[:, : CFG["n_seen"]], cols=cols)
        hp = _find(data_dir, f"{split}_headlines")
        if hp:
            h = _read(hp)
            part["heads"] = h
            part["heads_seen"] = h[h[CFG["bar"]] < CFG["n_seen"]].copy()
        else:
            part["heads"] = part["heads_seen"] = None
        out[split] = part

    ins = out.get("ins")
    if ins is not None:
        rp = _find(data_dir, "ins_returns")
        if rp:
            r = _read(rp).set_index(CFG["session"])[CFG["target"]]
            ins["y"] = r.reindex(ins["ids"]).to_numpy(float)
        else:  # derive target from full bars, market-adjusted
            ins["y"] = target_from_bars(ins["full"], ins["cols"])
            print("[load] no ins_returns file: target derived from bars (check this matches the rules!)")
    for k, v in out.items():
        nh = 0 if v["heads"] is None else len(v["heads"])
        print(f"[load] {k}: {len(v['ids'])} sessions, bars {v['full'].shape[1]}, headlines {nh}")
    return out


def close_of(arr, cols):
    return arr[..., cols.index("close") if "close" in cols else -1]


def target_from_bars(full, cols, adjust=True):
    c = close_of(full, cols)
    a, b = CFG["n_seen"] - 1, CFG["n_seen"] - 1 + CFG["horizon"]
    y = c[:, b] / c[:, a] - 1
    return y - y.mean() if adjust else y


# ---------------------------------------------------------------- scoring
def score(pred, y):
    """Leaderboard metric of the previous Algo Arena: Sharpe of pnl = pred * y."""
    pred = np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=0.0, neginf=0.0)
    pnl = pred * np.asarray(y, float)
    return pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)


def report(pred, y):
    pred = np.nan_to_num(np.asarray(pred, float))
    pnl = pred * y
    return dict(
        sharpe=round(score(pred, y), 3),
        corr=round(float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else 0.0, 4),
        hit=round(float((np.sign(pred) == np.sign(y))[pred != 0].mean()) if (pred != 0).any() else 0.0, 4),
        nonzero=round(float((pred != 0).mean()), 3),
        total_pnl=float(pnl.sum()),
    )


def cv(fit_predict, y, k=5, seeds=(0, 1, 2), verbose=True, name=""):
    """fit_predict(train_idx, test_idx) -> predictions for test_idx.
    Returns (mean fold Sharpe, std fold Sharpe, pooled OOF Sharpe of seed 0)."""
    n = len(y)
    fold_scores, oof0 = [], None
    for si, seed in enumerate(seeds):
        perm = np.random.default_rng(seed).permutation(n)
        oof = np.zeros(n)
        for f in range(k):
            te = np.sort(perm[f::k])
            tr = np.setdiff1d(np.arange(n), te)
            oof[te] = fit_predict(tr, te)
            fold_scores.append(score(oof[te], y[te]))
        if si == 0:
            oof0 = oof
    m, s, pooled = float(np.mean(fold_scores)), float(np.std(fold_scores)), score(oof0, y)
    if verbose:
        print(f"[cv] {name:<28} sharpe {m:6.3f} +- {s:5.3f}   pooled {pooled:6.3f}")
    return m, s, pooled


# ---------------------------------------------------------------- submission
def write_submission(ids, pred, path="submission.csv", expected_ids=None):
    ids = np.asarray(ids)
    pred = np.asarray(pred, float)
    assert len(ids) == len(pred), "ids / pred length mismatch"
    assert len(np.unique(ids)) == len(ids), "duplicate ids"
    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"[submit] {bad.sum()} non-finite preds set to 0")
        pred = np.where(bad, 0.0, pred)
    if expected_ids is not None:
        missing = set(expected_ids) - set(ids)
        extra = set(ids) - set(expected_ids)
        assert not missing and not extra, f"id mismatch: {len(missing)} missing, {len(extra)} extra"
    df = pd.DataFrame({CFG["sub_id"]: ids, CFG["sub_pred"]: pred})
    df.to_csv(path, index=False)
    print(f"[submit] wrote {path}: {len(df)} rows, pred std {pred.std():.4g}, nonzero {(pred != 0).mean():.2%}")
    return df


# ---------------------------------------------------------------- position helpers
def clip_sd(pred, ref=None, c=2.0):
    """Clip at c standard deviations of ref (defaults to pred itself)."""
    sd = np.std(pred if ref is None else ref) + 1e-12
    return np.clip(pred, -c * sd, c * sd)


def vol_size(pred, vol, power=0.5):
    """Divide position by vol**power. power=0.5 worked for the previous winner."""
    return pred / (np.maximum(vol, 1e-8) ** power)
