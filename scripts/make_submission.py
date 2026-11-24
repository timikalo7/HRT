"""Fit on all in-sample data (+ OOS first-half events for the stem rates) and write predictions CSV.

Usage: python make_submission.py [feature_set] [out_path]
  feature_set: g_only | n_cum (default) | all
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import DATA, ROOT, load, sharpe
from pipeline import features, fit

SETS = {"g_only": ["g"], "n_cum": ["g", "n_first", "cum_noise"],
        "all": ["g", "n_first", "cum_noise", "noise5", "n_live"]}
name = sys.argv[1] if len(sys.argv) > 1 else "n_cum"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"predictions_{name}.csv"
cols = SETS[name]

ins, oos = load("ins"), load("oos")
rates = fit(ins, np.ones(len(ins.sessions), bool), sig_kind="park", extra=[oos])
Fi, vi = features(ins, rates, "park")
Fo, vo = features(oos, rates, "park")
Xi = np.column_stack([np.ones(len(Fi)), Fi[cols].to_numpy()])
beta = np.linalg.lstsq(Xi, ins.returns / vi, rcond=None)[0]
print("feature set:", name, " coef:", dict(zip(["const"] + cols, np.round(beta, 6))))

pi = Xi @ beta / vi
pi -= pi.mean()
print(f"in-sample FIT Sharpe (not evidence, rates+coefs fit on these sessions): {sharpe(pi, ins.returns):.3f}")

Xo = np.column_stack([np.ones(len(Fo)), Fo[cols].to_numpy()])
po = Xo @ beta / vo
po -= po.mean()  # returns are demeaned across sessions: a constant offset only adds variance
print(f"OOS pred: n={len(po)} mean={po.mean():.2e} sd={po.std():.4f} frac>0={(po > 0).mean():.3f} "
      f"finite={np.isfinite(po).all()}")
print(f"corr with IS-model-shape check: corr(pred, g/vs) = {np.corrcoef(po, Fo['g'] / vo)[0,1]:.4f}")

sys.path.insert(0, str(DATA))
import sessions  # noqa: E402

df = pd.DataFrame({"pred": po}, index=pd.Index(oos.sessions, name="session"))
sessions.write_submission(prediction_df=df, path=out)
