"""Write the vol-scaled version of Timmy's c3 model and the c5-style three-way blend with it.

Output (git-excluded, in HRT_Hackathon/):
  predictions_zen_c3_vs.csv       VolScaled LatentTarget fit on all in-sample
  predictions_zen_c5_vs.csv       0.15 public + 0.15 m3564 + 0.70 c3_vs  (same recipe as c5)
"""
import os
import sys
from pathlib import Path

ZEN = Path(r"C:\Users\shriy\Claude_Code_Projects\HRT_branches\zen")
OUT = Path(r"C:\Users\shriy\Claude_Code_Projects\HRT_Hackathon")
os.environ["ARENA_DATA"] = str(OUT / "data")
sys.path.insert(0, str(ZEN))
sys.argv = [sys.argv[0], "vs", "0"]  # satisfy zen_volscale's arg parsing on import guard below

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from arena import latent  # noqa: E402
from arena.blend import blend  # noqa: E402
from arena.core import load, write_submission  # noqa: E402


class VolScaled(latent.LatentTarget):
    def _features(self, test):
        Z = super()._features(test)
        Z[:, 0] = Z[:, 0] * np.sqrt(latent.range_variance(test)) / self.sd_ref
        return Z

    def fit(self, train):
        self.sd_ref = float(np.sqrt(latent.range_variance(train)).mean())
        super().fit(train)


data = load()
sectors = latent.all_sectors(data)
m = VolScaled(sectors)
m.fit(data.ins)
pred = m.predict(data.oos)
p3 = write_submission(data.oos.sessions, pred, OUT / "predictions_zen_c3_vs.csv")
print("wrote", p3, "peer", round(m.peer, 3), "beta", np.round(m.beta, 5))

parts = [(0.15, pd.read_csv(ZEN / "submissions/predictions_public_improved_all_rerun.csv").set_index("headline_id")["pred"]),
         (0.15, pd.read_csv(ZEN / "submissions/predictions_public_m3564_rerun.csv").set_index("headline_id")["pred"]),
         (0.70, pd.read_csv(p3).set_index("headline_id")["pred"])]
b = blend(parts)
p5 = write_submission(b.index.to_numpy(), b.to_numpy(), OUT / "predictions_zen_c5_vs.csv")
old = pd.read_csv(ZEN / "submissions/predictions_c5_blend3.csv").set_index("headline_id")["pred"]
print("wrote", p5, " corr with Timmy's c5:", round(float(np.corrcoef(old.loc[b.index], b)[0, 1]), 4),
      " rows", len(b), " finite", bool(np.isfinite(b).all()))
