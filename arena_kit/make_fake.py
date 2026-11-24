"""Synthetic data in the previous Algo Arena format, with a planted headline-drift signal.
Use it to smoke-test the pipeline before the real data arrives:  python make_fake.py && python baseline.py --data fake_data
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_INS, N_OOS, T, LMAX = 3000, 3000, 100, 10
COMPANIES = [f"{a} {b}" for a in ("Yorval", "Kestrin", "Dumora", "Pellix", "Ostrava", "Venmar", "Quillon", "Brasko")
             for b in ("Partners", "Systems", "Labs")]
EVENTS = [
    ("secures $%dM contract with a major logistics provider", 0.0012),
    ("announces %d%% workforce reduction", -0.0010),
    ("beats quarterly earnings estimates by %d%%", 0.0015),
    ("misses revenue guidance by %d%%", -0.0014),
    ("appoints new CEO after %d year search", 0.0004),
    ("faces regulatory probe in North America", -0.0008),
    ("expands into Scandinavia with $%dM investment", 0.0006),
    ("to present at industry conference", 0.0),
    ("recalls %d thousand units", -0.0011),
    ("partners with a top-tier research institute", 0.0007),
]


def make(n, start_id):
    rows, heads = [], []
    for k in range(n):
        sid = start_id + k
        r = rng.normal(0, 0.004, T)
        r[1:] += -0.15 * r[:-1]  # mild reversal
        for _ in range(rng.poisson(6)):
            t = int(rng.integers(0, T - 1))
            e = int(rng.integers(len(EVENTS)))
            txt, rate = EVENTS[e]
            if "%d" in txt:
                txt = txt % int(rng.integers(2, 500))
            heads.append((sid, t, f"{COMPANIES[rng.integers(len(COMPANIES))]} {txt}"))
            ker = np.array([0.5] + [1.0] * (LMAX - 1) + [0.5])
            end = min(T, t + 1 + len(ker))
            r[t + 1:end] += rate * ker[: end - t - 1]
        close = np.exp(np.cumsum(r))
        close = close / close[0]
        op = np.r_[1.0, close[:-1]] * np.exp(rng.normal(0, 0.0005, T))
        hi = np.maximum(op, close) * np.exp(np.abs(rng.normal(0, 0.001, T)))
        lo = np.minimum(op, close) * np.exp(-np.abs(rng.normal(0, 0.001, T)))
        for t in range(T):
            rows.append((sid, t, op[t], hi[t], lo[t], close[t]))
    bars = pd.DataFrame(rows, columns=["session", "bar_ix", "open", "high", "low", "close"])
    hd = pd.DataFrame(heads, columns=["session", "bar_ix", "headline"])
    return bars, hd


os.makedirs("fake_data", exist_ok=True)
ib, ih = make(N_INS, 1000)
ob, oh = make(N_OOS, 1000 + N_INS)
c = ib.pivot(index="session", columns="bar_ix", values="close")
ret = c[59] / c[49] - 1
ret = ret - ret.mean()
ib.to_parquet("fake_data/ins_bars.parquet")
ih.to_parquet("fake_data/ins_headlines.parquet")
pd.DataFrame({"session": ret.index, "returns": ret.values}).to_parquet("fake_data/ins_returns.parquet")
ob[ob.bar_ix < 50].to_parquet("fake_data/oos_bars.parquet")
oh[oh.bar_ix < 50].to_parquet("fake_data/oos_headlines.parquet")
print("wrote fake_data/:", len(ib), "ins bars,", len(ih), "ins headlines")
