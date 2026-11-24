#!/usr/bin/env python3
"""Tools for Sharpe-scored prediction contests: score, paired comparison, sizing, blending and
leaderboard forensics. Import it, or run `python sharpe_tools.py selftest`.

All functions take numpy arrays aligned by session.
"""
from __future__ import annotations

import sys

import numpy as np


def sharpe(pos, ret, ann: float = 252.0) -> float:
    """sqrt(ann) * mean(pos*ret) / std(pos*ret), sample std (ddof=1) as pandas computes it."""
    pnl = np.asarray(pos, float) * np.asarray(ret, float)
    return float(np.sqrt(ann) * pnl.mean() / (pnl.std(ddof=1) + 1e-300))


def paired(p_new, p_base, ret, n_boot: int = 2000, seed: int = 0, ann: float = 252.0) -> dict:
    """Sharpe(new) - Sharpe(base) on identical sessions, with a session bootstrap.
    Two 99%-correlated models typically differ by a bootstrap sd of only ~0.01-0.02, far less than
    the ~0.15 standard error of either Sharpe on its own, which is why paired comparison matters."""
    a, b = np.asarray(p_new) * ret, np.asarray(p_base) * ret
    n = len(ret)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        d[i] = np.sqrt(ann) * (a[ix].mean() / a[ix].std() - b[ix].mean() / b[ix].std())
    return dict(diff=sharpe(p_new, ret, ann) - sharpe(p_base, ret, ann), boot_sd=float(d.std()),
                p_positive=float((d > 0).mean()), corr=float(np.corrcoef(p_new, p_base)[0, 1]))


def optimal_position(mu, var, k: float = 1.0):
    """First-order Sharpe-optimal position for forecast mu and per-session variance var:
    mu / (k*var + mu^2). Fit k by cross-validation; its optimum is usually flat."""
    mu = np.asarray(mu, float)
    return mu / (k * np.asarray(var, float) + mu ** 2)


def garman_klass(o, h, l, c, prev_close=None):
    """Per-row mean per-bar variance from OHLC arrays (rows = sessions), plus squared opening gaps.
    Pass only bars known at decision time. prev_close defaults to 1.0 before the first bar."""
    o, h, l, c = (np.log(np.asarray(x, float)) for x in (o, h, l, c))
    prev = np.concatenate([np.zeros((len(c), 1)) if prev_close is None else np.log(prev_close)[:, None], c[:, :-1]], 1)
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2 + (o - prev) ** 2
    return np.maximum(gk.mean(1), 1e-12)


def unit(x):
    x = np.asarray(x, float)
    return x / x.std()


def blend(components: dict, weights: dict):
    """Weighted sum of components, each scaled to unit sd first."""
    return sum(w * unit(components[k]) for k, w in weights.items())


def best_blend_weights(oof: dict, ret, shrink_to_equal: float = 0.5, nonneg: bool = True) -> dict:
    """Max-Sharpe weights (C^-1 m on unit-sd PnL streams), shrunk toward equal weights."""
    names = list(oof)
    P = np.column_stack([unit(oof[k]) * ret for k in names])
    w = np.linalg.solve(np.cov(P.T), P.mean(0))
    if nonneg:
        w = np.clip(w, 0, None)
    w = w / w.sum()
    w = (1 - shrink_to_equal) * w + shrink_to_equal * np.ones(len(names)) / len(names)
    return dict(zip(names, w / w.sum()))


def lb_forensics(oof: dict, ret, uploads: list, tau: float = 0.06, ann: float = 252.0) -> dict:
    """What leaderboard scores of known blends say about each component on the hidden set.

    oof     : {name: out-of-fold positions} on labelled data (same session order as ret)
    uploads : [(weights_dict, leaderboard_sharpe), ...] for files whose composition is known exactly
    tau     : prior sd of a common relative shift between in-sample and hidden-set mean PnL
    Returns implied hidden-set Sharpe per component (with sd), the best non-negative blend and its
    predicted score. Diagnostic: with a public/private split, do not tune uploads to it."""
    names = list(oof)
    n = len(ret)
    P = np.column_stack([unit(oof[k]) * ret for k in names])
    m0, C = P.mean(0), np.cov(P.T)
    W = np.array([[w.get(k, 0.0) for k in names] for w, _ in uploads], float)
    lb = np.array([s for _, s in uploads], float)
    y = lb / np.sqrt(ann) * np.sqrt(np.einsum("ji,ik,jk->j", W, C, W))
    S = C / n + tau ** 2 * np.outer(m0, m0)
    G = S @ W.T @ np.linalg.pinv(W @ S @ W.T)
    m1 = m0 + G @ (y - W @ m0)
    post = S - G @ W @ S
    sh = lambda w, m: float(np.sqrt(ann) * (w @ m) / np.sqrt(w @ C @ w))
    eye = np.eye(len(names))
    w_opt = np.clip(np.linalg.solve(C, m1), 0, None)
    w_opt = w_opt / w_opt.sum() if w_opt.sum() > 0 else np.ones(len(names)) / len(names)
    return dict(
        implied_component_sharpe={k: (sh(eye[i], m1), float(np.sqrt(max(post[i, i], 0)) / np.sqrt(C[i, i]) * np.sqrt(ann)))
                                  for i, k in enumerate(names)},
        cv_component_sharpe={k: sh(eye[i], m0) for i, k in enumerate(names)},
        uploads_cv_vs_lb=[(sh(W[j], m0), float(lb[j])) for j in range(len(lb))],
        best_blend=dict(zip(names, w_opt.round(4))),
        best_blend_predicted=sh(w_opt, m1),
        best_upload_predicted=max(sh(W[j], m1) for j in range(len(lb))),
    )


def _selftest() -> None:
    # strong signal and many sessions, so a single Sharpe is precise enough for the checks below
    rng = np.random.default_rng(0)
    n = 200_000
    sig = rng.uniform(0.5, 1.5, n)
    mu = rng.normal(0, 0.1, n) * sig
    ret = mu + rng.normal(0, 1, n) * sig
    a = mu + rng.normal(0, 0.1, n) * sig
    b = mu + rng.normal(0, 0.1, n) * sig
    s_raw, s_opt = sharpe(a, ret), sharpe(optimal_position(a, sig ** 2), ret)
    assert s_opt > s_raw, (s_raw, s_opt)
    pr = paired(optimal_position(a, sig ** 2), a, ret, n_boot=200)
    assert pr["diff"] > 0 and 0 < pr["boot_sd"] < 1
    w = best_blend_weights({"a": a, "b": b}, ret)
    assert abs(w["a"] - 0.5) < 0.1, w
    s_blend = sharpe(blend({"a": a, "b": b}, w), ret)
    assert s_blend > max(sharpe(a, ret), sharpe(b, ret)), (s_blend, sharpe(a, ret), sharpe(b, ret))
    fx = lb_forensics({"a": a, "b": b}, ret, [({"a": 1.0}, sharpe(a, ret) + 0.1), ({"a": 0.5, "b": 0.5}, sharpe(blend({"a": a, "b": b}, {"a": .5, "b": .5}), ret) + 0.1)])
    assert fx["implied_component_sharpe"]["a"][0] > fx["cv_component_sharpe"]["a"]
    print(f"selftest ok: raw {s_raw:.3f} -> optimal sizing {s_opt:.3f} (paired diff {pr['diff']:+.3f}, sd {pr['boot_sd']:.3f}); "
          f"blend weights {w}; forensics best blend predicted {fx['best_blend_predicted']:.3f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        _selftest()
    else:
        print(__doc__)
