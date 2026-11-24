"""Shared evaluation harness for research tracks: identical folds, leak-safe test blocks, uniform outputs.

A model is any object with fit(train: Block) and predict(test: Block) -> positions (one per session).
`train` holds full 100-bar sessions with labels; `test` physically contains only bars 0-49, headlines
with bar_ix <= 49 and no labels (built with Block.take(first_half=True)).

    from research.harness import run
    run(lambda: MyModel(...), tag="my_variant", outdir="/path/to/scratch")

`run` cross-validates on seeds 0 and 1 with the project's folds, compares against the cached baseline
(arena.latent.LatentTarget) on identical sessions, runs the look-ahead test, fits on all in-sample
sessions, writes out-of-sample positions, and saves everything as .npy plus a .json summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/HRT")
from arena.core import HALF, load, sharpe  # noqa: E402
from arena.latent import LatentTarget, all_sectors  # noqa: E402

CACHE = Path("/tmp/claude-0/-home-user-HRT/1b646c27-9e64-5f9e-b4f8-43257d3086ff/scratchpad/wf")
_STATE: dict = {}


def data():
    """(Data, sectors). Sectors come from headline co-occurrence only (no prices, no labels)."""
    if "d" not in _STATE:
        d = load()
        _STATE["d"], _STATE["sectors"] = d, all_sectors(d)
    return _STATE["d"], _STATE["sectors"]


def folds(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % 5


def oof_positions(make_model, seed: int) -> np.ndarray:
    d, _ = data()
    ins = d.ins
    n = len(ins)
    f = folds(n, seed)
    p = np.zeros(n)
    for k in range(5):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        m = make_model()
        m.fit(ins.take(tr, first_half=False))
        p[te] = m.predict(ins.take(te, first_half=True))
    return p


def leak_test(make_model, n_test: int = 1500, seed: int = 0) -> float:
    """Max change in predictions when bars 50-99 of the test sessions are overwritten with noise and
    later headlines dropped. Must be exactly 0.0."""
    d, _ = data()
    ins = d.ins
    n = len(ins)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, n_test, replace=False))
    tr = np.setdiff1d(np.arange(n), idx)
    m = make_model()
    m.fit(ins.take(tr, first_half=False))
    p1 = m.predict(ins.take(idx, first_half=True))
    cb = ins.take(idx, first_half=False)
    for x in (cb.open, cb.high, cb.low, cb.close):
        x[:, HALF + 1:] = rng.uniform(0.5, 2.0, x[:, HALF + 1:].shape)
    cb.headlines = cb.headlines[cb.headlines.bar_ix <= HALF].reset_index(drop=True)
    cb.returns = None
    return float(np.abs(p1 - m.predict(cb)).max())


def paired(p_new: np.ndarray, p_base: np.ndarray, r: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Sharpe(new) - Sharpe(base) on identical sessions, with a session bootstrap."""
    n = len(r)
    a, b = p_new * r, p_base * r
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        diffs[i] = np.sqrt(252) * (a[ix].mean() / a[ix].std() - b[ix].mean() / b[ix].std())
    return dict(diff=float(sharpe(p_new, r) - sharpe(p_base, r)), boot_sd=float(diffs.std()),
                p_positive=float((diffs > 0).mean()), corr=float(np.corrcoef(p_new, p_base)[0, 1]))


def baseline_oof(seed: int) -> np.ndarray:
    path = CACHE / f"baseline_oof_seed{seed}.npy"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        _, sectors = data()
        np.save(path, oof_positions(lambda: LatentTarget(sectors), seed))
    return np.load(path)


def run(make_model, tag: str, outdir: str, seeds=(0, 1), write_oos: bool = True, do_leak_test: bool = True) -> dict:
    d, _ = data()
    r = d.ins.returns
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"tag": tag}
    for s in seeds:
        p = oof_positions(make_model, s)
        np.save(out / f"{tag}_oof_seed{s}.npy", p)
        base = baseline_oof(s)
        res[f"seed{s}"] = dict(sharpe=float(sharpe(p, r)), baseline=float(sharpe(base, r)), **paired(p, base, r))
        print(f"[{tag}] seed {s}: Sharpe {res[f'seed{s}']['sharpe']:.3f} vs baseline {res[f'seed{s}']['baseline']:.3f} "
              f"(diff {res[f'seed{s}']['diff']:+.3f}, boot sd {res[f'seed{s}']['boot_sd']:.3f}, "
              f"P>0 {res[f'seed{s}']['p_positive']:.3f}, corr {res[f'seed{s}']['corr']:.3f})", flush=True)
    if do_leak_test:
        res["leak_test_max_change"] = leak_test(make_model)
        print(f"[{tag}] leak test max change: {res['leak_test_max_change']}", flush=True)
    if write_oos:
        m = make_model()
        m.fit(d.ins)
        p = m.predict(d.oos)
        np.save(out / f"{tag}_oos.npy", p)
        res["oos_finite"] = bool(np.isfinite(p).all())
        res["oos_len"] = int(len(p))
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------
# Fast path for variants that keep the EM stage (amplitudes, peer fraction) and change only the
# features / second stage / sizing: the fitted base model of every fold is cached on disk.

def _base_path(key: str) -> Path:
    return CACHE / f"base_{key}.pkl"


def get_base(seed: int | None = None, fold: int | None = None) -> LatentTarget:
    """Fitted arena.latent.LatentTarget for (seed, fold) training sessions, or for all in-sample
    sessions when called with no arguments. Cached on disk."""
    import pickle
    d, sectors = data()
    key = "full" if seed is None else f"s{seed}_f{fold}"
    path = _base_path(key)
    if path.exists():
        return pickle.loads(path.read_bytes())
    CACHE.mkdir(parents=True, exist_ok=True)
    ins = d.ins
    tr = np.arange(len(ins)) if seed is None else np.where(folds(len(ins), seed) != fold)[0]
    m = LatentTarget(sectors)
    m.fit(ins.take(tr, first_half=False))
    path.write_bytes(pickle.dumps(m))
    return m


def from_base(variant_cls, base: LatentTarget, train, **kw):
    """Instance of a LatentTarget subclass that reuses `base`'s EM fit and refits only the second
    stage (self.beta) on `train` with the subclass's own `_features`."""
    m = variant_cls(base.sectors, **kw)
    m.types, m.amp, m.peer, m.mu0 = base.types, base.amp, base.peer, base.mu0
    Z = m._features(train.take(np.arange(len(train)), first_half=True))
    m.beta = np.linalg.lstsq(Z, train.returns, rcond=None)[0]
    return m


def run_variant(variant_cls, tag: str, outdir: str, seeds=(0, 1), write_oos: bool = True, **kw) -> dict:
    """Like `run`, for LatentTarget subclasses that only override `_features` and/or `predict`.
    Uses the cached EM fits, so one variant takes roughly 20-40 s instead of minutes."""
    d, _ = data()
    ins, r = d.ins, d.ins.returns
    n = len(ins)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"tag": tag}
    for s in seeds:
        f = folds(n, s)
        p = np.zeros(n)
        for k in range(5):
            tr, te = np.where(f != k)[0], np.where(f == k)[0]
            m = from_base(variant_cls, get_base(s, k), ins.take(tr, first_half=False), **kw)
            p[te] = m.predict(ins.take(te, first_half=True))
        np.save(out / f"{tag}_oof_seed{s}.npy", p)
        base = baseline_oof(s)
        res[f"seed{s}"] = dict(sharpe=float(sharpe(p, r)), baseline=float(sharpe(base, r)), **paired(p, base, r))
        print(f"[{tag}] seed {s}: Sharpe {res[f'seed{s}']['sharpe']:.3f} vs baseline {res[f'seed{s}']['baseline']:.3f} "
              f"(diff {res[f'seed{s}']['diff']:+.3f}, boot sd {res[f'seed{s}']['boot_sd']:.3f}, "
              f"P>0 {res[f'seed{s}']['p_positive']:.3f}, corr {res[f'seed{s}']['corr']:.3f})", flush=True)
    # look-ahead test with the same reuse (fold 0 of seed 0 as the held-out set)
    f = folds(n, 0)
    tr, te = np.where(f != 0)[0], np.where(f == 0)[0]
    m = from_base(variant_cls, get_base(0, 0), ins.take(tr, first_half=False), **kw)
    p1 = m.predict(ins.take(te, first_half=True))
    cb = ins.take(te, first_half=False)
    rng = np.random.default_rng(0)
    for x in (cb.open, cb.high, cb.low, cb.close):
        x[:, HALF + 1:] = rng.uniform(0.5, 2.0, x[:, HALF + 1:].shape)
    cb.headlines = cb.headlines[cb.headlines.bar_ix <= HALF].reset_index(drop=True)
    cb.returns = None
    res["leak_test_max_change"] = float(np.abs(p1 - m.predict(cb)).max())
    print(f"[{tag}] leak test max change: {res['leak_test_max_change']}", flush=True)
    if write_oos:
        m = from_base(variant_cls, get_base(), ins, **kw)
        p = m.predict(d.oos)
        np.save(out / f"{tag}_oos.npy", p)
        res["oos_finite"] = bool(np.isfinite(p).all())
        res["oos_len"] = int(len(p))
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    d, sectors = data()
    for s in (0, 1, 2):
        print("baseline seed", s, round(float(sharpe(baseline_oof(s), d.ins.returns)), 3), flush=True)
        for k in range(5):
            get_base(s, k)
    get_base()
    print("base fits cached", flush=True)
