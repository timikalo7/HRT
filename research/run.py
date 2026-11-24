"""Usage: python run.py cv [v0|v1]      -> 5-fold CV Sharpe on INS (5 seeds of folds)
          python run.py submit [v0|v1]  -> fit on all INS, write submissions/<ver>.csv via sessions.write_submission
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib import DATA, folds, load, sharpe
from model import Model

mode, ver = sys.argv[1], sys.argv[2]
piv, hl, sess, ret = load("ins")
tmap = {t: i for i, t in enumerate(sorted(hl.tmpl.unique()))}
n = len(sess)

if mode == "cv":
    seeds = [0, 1, 2] if len(sys.argv) < 4 else range(int(sys.argv[3]))
    allf = []
    for seed in seeds:
        pred = np.zeros(n); fs = []
        for te in folds(n, 5, seed):
            tr = np.setdiff1d(np.arange(n), te)
            p = Model(ver).fit(piv, hl, tr, ret, tmap).predict(piv, hl, te)
            pred[te] = p; fs.append(sharpe(p, ret[te]))
        allf += fs
        print(f"seed {seed}: pooled CV sharpe {sharpe(pred, ret):.3f}  folds {np.round(fs, 2)}")
    print(f"{ver}: mean fold sharpe {np.mean(allf):.3f} +- {np.std(allf)/np.sqrt(len(allf)):.3f} (se)")
else:
    sys.path.insert(0, str(DATA))
    import sessions  # official harness: validates ids/rows and writes the arcade format

    po, ho, so, _ = load("oos")
    m = Model(ver).fit(piv, hl, np.arange(n), ret, tmap)
    p = m.predict(po, ho.assign(si=ho.si), np.arange(len(so)))
    df = pd.DataFrame({"session": so.to_numpy(), "pred": p})
    out = Path(__file__).resolve().parent.parent / "submissions"; out.mkdir(exist_ok=True)
    sessions.write_submission(prediction_df=df, path=out / f"{ver}.csv")
    ins_fit = m.predict(piv, hl, np.arange(n))
    print("in-sample (fit) sharpe, optimistic:", round(sharpe(ins_fit, ret), 3))
    print("oos pred stats:", pd.Series(p).describe().round(5).to_dict())
