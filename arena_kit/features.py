"""Feature builders: price features and headline-drift features."""
import re

import numpy as np
import pandas as pd

from kit import CFG, close_of


# ---------------------------------------------------------------- price
def price_features(seen, cols):
    """seen: (S, n_seen, C). Returns DataFrame of per-session price features."""
    c = close_of(seen, cols)
    lp = np.log(np.maximum(c, 1e-12))
    r = np.diff(lp, axis=1)
    n = lp.shape[1]
    f = {}
    for k in (1, 2, 3, 5, 10, 20, n - 1):
        if k < n:
            f[f"mom_{k}"] = lp[:, -1] - lp[:, -1 - k]
    f["vol_all"] = r.std(1)
    f["vol_10"] = r[:, -10:].std(1)
    f["absret_mean"] = np.abs(r).mean(1)
    f["dist_max"] = lp[:, -1] - lp.max(1)
    f["dist_min"] = lp[:, -1] - lp.min(1)
    w = lp[:, -20:]
    f["z20"] = (lp[:, -1] - w.mean(1)) / (w.std(1) + 1e-12)
    f["ac1"] = (r[:, 1:] * r[:, :-1]).mean(1) / (r.var(1) + 1e-12)
    if "high" in cols and "low" in cols:
        hi, lo = seen[..., cols.index("high")], seen[..., cols.index("low")]
        rng = np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12))
        f["range_mean"] = rng.mean(1)
        f["range_10"] = rng[:, -10:].mean(1)
    if "open" in cols:
        o = seen[..., cols.index("open")]
        f["body_last"] = np.log(c[:, -1] / np.maximum(o[:, -1], 1e-12))
        f["gap_mean"] = np.log(o[:, 1:] / np.maximum(c[:, :-1], 1e-12)).mean(1)
    return pd.DataFrame(f)


def realized_vol(seen, cols):
    c = close_of(seen, cols)
    return np.diff(np.log(np.maximum(c, 1e-12)), axis=1).std(1)


# ---------------------------------------------------------------- headlines
_MONEY = re.compile(r"\$\s?\d[\d,]*(\.\d+)?\s?(k|m|b|bn|mn|million|billion)?\b", re.I)
_PCT = re.compile(r"\d[\d,]*(\.\d+)?\s?%")
_NUM = re.compile(r"\b\d[\d,]*(\.\d+)?\b")


def split_company(h):
    """Leading run of Capitalised words = company name (fictional names in the old data)."""
    toks = str(h).split()
    i = 0
    while i < len(toks) - 1 and toks[i][:1].isupper():
        i += 1
    return " ".join(toks[:i]), " ".join(toks[i:]) if i else str(h)


def normalise(text):
    t = _MONEY.sub("$X", text)
    t = _PCT.sub("N%", t)
    t = _NUM.sub("N", t)
    return re.sub(r"\s+", " ", t.lower()).strip()


def prep_heads(heads, ids):
    """Add sess_idx, company, type_full (numbers masked), type_short (first 3 words of action)."""
    if heads is None:
        return None
    h = heads.copy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    h["sess_idx"] = pos.reindex(h[CFG["session"]]).to_numpy()
    h = h.dropna(subset=["sess_idx"])
    h["sess_idx"] = h["sess_idx"].astype(int)
    h["t"] = h[CFG["bar"]].astype(int)
    comp_act = h[CFG["headline"]].map(split_company)
    h["company"] = comp_act.str[0]
    h["type_full"] = comp_act.str[1].map(normalise)
    h["type_short"] = h["type_full"].str.split().str[:3].str.join(" ")
    return h


def fit_rates(logp_full, heads, train_mask, type_col="type_full", lmax=10, shrink=20.0):
    """Average excess per-bar log return at lags 1..lmax after each headline type.
    Uses full price paths (ins bars past n_seen) of TRAIN sessions only."""
    h = heads[train_mask[heads["sess_idx"].to_numpy()]]
    T = logp_full.shape[1]
    r_all = np.diff(logp_full[train_mask], axis=1)
    mu = r_all.mean()
    parts = []
    s, t = h["sess_idx"].to_numpy(), h["t"].to_numpy()
    for L in range(1, lmax + 1):
        ok = t + L < T
        r = logp_full[s[ok], t[ok] + L] - logp_full[s[ok], t[ok] + L - 1]
        parts.append(pd.DataFrame({"type": h[type_col].to_numpy()[ok], "r": r - mu}))
    d = pd.concat(parts)
    g = d.groupby("type")["r"].agg(["sum", "count"])
    return g["sum"] / (g["count"] + shrink)


def drift_features(heads, n_sess, rates, logp_seen, type_col="type_full", lmax=10, prefix="d"):
    """Per session: drift still owed into the target window, drift already explained, the rest."""
    n, H = CFG["n_seen"], CFG["horizon"]
    s, t = heads["sess_idx"].to_numpy(), heads["t"].to_numpy()
    rate = heads[type_col].map(rates).fillna(0.0).to_numpy()
    lo = np.maximum(1, n - t)
    hi = np.minimum(lmax, n - 1 + H - t)
    owed_cnt = np.maximum(0, hi - lo + 1)
    done_cnt = np.maximum(0, np.minimum(lmax, n - 1 - t))
    owed = np.bincount(s, rate * owed_cnt, n_sess)
    done = np.bincount(s, rate * done_cnt, n_sess)
    recent = t >= n - lmax
    out = pd.DataFrame({
        f"{prefix}_owed": owed,
        f"{prefix}_done": done,
        f"{prefix}_unexplained": (logp_seen[:, -1] - logp_seen[:, 0]) - done,
        f"{prefix}_n_heads": np.bincount(s, minlength=n_sess).astype(float),
        f"{prefix}_n_recent": np.bincount(s[recent], minlength=n_sess).astype(float),
        f"{prefix}_abs_recent": np.bincount(s[recent], np.abs(rate[recent]), n_sess),
    })
    return out


def target_type_scores(heads, y, train_mask, type_col="type_short", window=10, shrink=30.0, prefix="tt"):
    """No-full-bars fallback: mean target of train sessions where a type appeared in the last `window` bars."""
    n = CFG["n_seen"]
    rec = heads[heads["t"] >= n - window]
    s = rec["sess_idx"].to_numpy()
    tr = rec[train_mask[s]]
    yt = y[tr["sess_idx"].to_numpy()] - y[train_mask].mean()
    g = pd.DataFrame({"type": tr[type_col].to_numpy(), "y": yt}).groupby("type")["y"].agg(["sum", "count"])
    sc = g["sum"] / (g["count"] + shrink)
    vals = rec[type_col].map(sc).fillna(0.0).to_numpy()
    return pd.DataFrame({f"{prefix}_score": np.bincount(s, vals, len(train_mask))})


def session_text(heads, n_sess, window=None):
    """Concatenate (recent) headlines per session, for TF-IDF models."""
    h = heads if window is None else heads[heads["t"] >= CFG["n_seen"] - window]
    txt = h.groupby("sess_idx")[CFG["headline"]].apply(" | ".join)
    return txt.reindex(range(n_sess)).fillna("").to_numpy()
