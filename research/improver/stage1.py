"""Stage 1: fit the unchanged LatentTarget EM per fold and cache an extended first-half feature
table for train (first-half view) and test. Usage: stage1.py <seed|full>"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, "/home/user/HRT")
import numpy as np
from arena.core import HALF, KERNEL, load
from arena.baseline import pending_weight
from arena.latent import LatentTarget, _Prep, _posterior, all_sectors, range_variance

OUT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["mu", "count", "unexpl", "live", "one", "s_raw", "s_dir", "mu_pos", "mu_neg", "conflict",
         "mu_own", "mu_peer", "mu_top", "ent_mu", "r1", "r5", "r10", "rv", "rv10", "ent"]


def rv_last10(block):
    o, h, l, c = (np.log(x[:, : HALF + 1]) for x in (block.open, block.high, block.low, block.close))
    prev = c[:, HALF - 10: HALF]
    o, h, l, c = (x[:, HALF - 9: HALF + 1] for x in (o, h, l, c))
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return np.maximum((gk + (o - prev) ** 2).mean(1), 1e-10)


def feats(m, test):
    """Everything from bars 0..HALF and headlines with bar_ix <= HALF only."""
    prep = _Prep(test, m.types, m.sectors)
    lr = test.logret()[:, : HALF + 1]
    y = lr - m.mu0
    sig = np.maximum(y.std(1), 1e-6)
    X = prep.paths(m.amp, HALF + 1)
    p, fit = _posterior(y, sig, X, prep, m.peer, with_unseen=True, temp=m.temp)
    pend = prep.pending(m.amp)
    peerf = prep.peer.astype(float)
    peer_pend = np.einsum("scj,sj->sc", peerf, pend)
    owed_if_target = pend + m.peer * peer_pend
    owed_if_unseen = m.peer * (prep.main * pend).sum(1)
    mu = (p[:, :4] * owed_if_target).sum(1) + p[:, 4] * owed_if_unseen
    n = len(test)
    count = np.bincount(prep.hs, minlength=n).astype(float)
    unexpl = y.sum(1) - np.einsum("sc,sct->s", p, fit)
    main_h = prep.main[prep.hs, prep.hj]
    live = np.bincount(prep.hs, weights=(main_h & (prep.hb >= HALF - 9)).astype(float), minlength=n)
    # expected role of each slot, then per headline
    role = p[:, :4] + m.peer * np.einsum("sc,scj->sj", p[:, :4], peerf) + p[:, 4:5] * m.peer * prep.main
    ok = prep.hb <= HALF
    a = np.where(prep.he >= 0, m.amp[np.maximum(prep.he, 0)], 0.0)
    pw = pending_weight(prep.hb)
    rh = role[prep.hs, prep.hj]
    o_h = np.where(ok, rh * a * pw, 0.0)
    mu_pos = np.bincount(prep.hs, weights=np.maximum(o_h, 0), minlength=n)
    mu_neg = np.bincount(prep.hs, weights=np.minimum(o_h, 0), minlength=n)
    conflict = np.minimum(mu_pos, -mu_neg)
    mu_own = (p[:, :4] * pend).sum(1)
    mu_peer = mu - mu_own
    top = p[:, :4].argmax(1)
    mu_top = pend[np.arange(n), top] * p[np.arange(n), top]
    ent = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    # surprise for main-sector headlines in bars 40..48
    cy = np.cumsum(y[:, ::-1], 1)[:, ::-1]  # cy[s, b] = sum y[s, b:HALF+1]
    hbc = np.clip(prep.hb, 0, HALF)
    cumw = np.array([KERNEL[: HALF - b + 1].sum() if b <= HALF else 0.0 for b in range(0, 100)])[np.clip(prep.hb, 0, 99)]
    sel = ok & main_h & (prep.hb >= HALF - 9) & (prep.hb <= HALF - 1)
    resid = cy[prep.hs, hbc] - rh * a * cumw
    s_raw = np.bincount(prep.hs, weights=np.where(sel, pw * resid, 0.0), minlength=n)
    s_dir = np.bincount(prep.hs, weights=np.where(sel, a * a * pw * cumw * resid, 0.0), minlength=n) / sig ** 2
    r1, r5, r10 = y[:, -1], y[:, -5:].sum(1), y[:, -10:].sum(1)
    return np.column_stack([mu, count, unexpl, live, np.ones(n), s_raw, s_dir, mu_pos, mu_neg, conflict,
                            mu_own, mu_peer, mu_top, ent * mu, r1, r5, r10, range_variance(test),
                            rv_last10(test), ent])


def main():
    arg = sys.argv[1]
    data = load()
    sectors = all_sectors(data)
    ins = data.ins
    n = len(ins)
    if arg == "full":
        m = LatentTarget(sectors); m.fit(ins)
        Ftr = feats(m, ins.take(np.arange(n), first_half=True))
        Foos = feats(m, data.oos.take(np.arange(len(data.oos)), first_half=True))
        base_oos = m.predict(data.oos.take(np.arange(len(data.oos)), first_half=True))
        np.savez(f"{OUT}/feat_full.npz", Ftr=Ftr, Foos=Foos, y=ins.returns, base_oos=base_oos, beta=m.beta,
                 oos_sessions=data.oos.sessions, ins_sessions=ins.sessions)
        print("full done", flush=True)
        return
    seed = int(arg)
    folds = np.random.default_rng(seed).permutation(n) % 5
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LatentTarget(sectors)
        train = ins.take(tr, first_half=False)
        m.fit(train)
        Ftr = feats(m, train.take(np.arange(len(train)), first_half=True))
        test = ins.take(te, first_half=True)
        Fte = feats(m, test)
        base = m.predict(test)
        extra = {}
        if f == 0:
            # leak test: full block for the test sessions, bars 50-99 replaced by random numbers,
            # headlines after bar 49 dropped
            full = ins.take(te, first_half=False)
            rng = np.random.default_rng(123)
            for nm in ("open", "high", "low", "close"):
                arr = getattr(full, nm).copy()
                arr[:, HALF + 1:] = np.exp(rng.normal(0, 1, arr[:, HALF + 1:].shape))
                setattr(full, nm, arr)
            full.headlines = full.headlines[full.headlines["bar_ix"] <= HALF].reset_index(drop=True)
            full.returns = None
            extra = dict(Fleak=feats(m, full), base_leak=m.predict(full))
        np.savez(f"{OUT}/feat_s{seed}_f{f}.npz", Ftr=Ftr, Fte=Fte, ytr=ins.returns[tr], tr=tr, te=te,
                 base=base, beta=m.beta, **extra)
        print(f"seed {seed} fold {f} done; base check {np.abs(Fte[:, :5] @ m.beta / (15 * Fte[:, 17] + (Fte[:, :5] @ m.beta) ** 2) - base).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
