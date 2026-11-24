"""Smoke check: data loads via sessions.py and the `returns` definition is what we think."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA))
import sessions  # noqa: E402

ins = sessions.get_ins_data()
oos = sessions.get_oos_data()
print("ins sessions:", len(ins), "oos sessions:", len(oos))
print("columns:", list(ins.columns))
print("close array lengths:", {len(a) for a in ins["close"]})

bars = pd.read_parquet(DATA / "ins_bars.parquet")
print("ins_bars shape:", bars.shape, "cols:", list(bars.columns))
close = bars.pivot(index="session", columns="bar_ix", values="close")
simple = close[59] / close[49] - 1
logr = np.log(close[59] / close[49])
ret = ins["returns"]
for name, raw in (("simple", simple), ("log", logr)):
    adj = raw - raw.mean()
    print(f"{name}: max|adj - returns| = {(adj - ret).abs().max():.3e}, corr = {adj.corr(ret):.6f}")
print("mean(returns) =", ret.mean(), " std =", ret.std())
print(ret.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print("kurtosis:", ret.kurt())
