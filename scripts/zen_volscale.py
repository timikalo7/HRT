"""Test on Timmy's (zen) model: scale the owed-drift forecast by relative session vol.

Finding from our branch: drift amplitude is proportional to session sigma (slope 0.74 -> 1.31 across
vol quintiles). zen's mu ignores this. V1 multiplies mu by s = sigma_s / mean(sigma) before the
second stage, so E[y] ~ s * mu is what gets fitted and sized.
Usage: python zen_volscale.py <variant: base|vs> <seed>
"""
import os
import sys

ZEN = r"C:\Users\shriy\Claude_Code_Projects\HRT_branches\zen"
os.environ["ARENA_DATA"] = r"C:\Users\shriy\Claude_Code_Projects\HRT_Hackathon\data"
sys.path.insert(0, ZEN)

import numpy as np  # noqa: E402

from arena import latent  # noqa: E402
from arena.core import cross_validate, load  # noqa: E402


class VolScaled(latent.LatentTarget):
    def _features(self, test):
        Z = super()._features(test)
        sd = np.sqrt(latent.range_variance(test))
        s = sd / self.sd_ref
        Z[:, 0] = Z[:, 0] * s
        return Z

    def fit(self, train):
        self.sd_ref = float(np.sqrt(latent.range_variance(train)).mean())
        super().fit(train)


variant, seed = sys.argv[1], int(sys.argv[2])
data = load()
sectors = latent.all_sectors(data)
cls = VolScaled if variant == "vs" else latent.LatentTarget
res = cross_validate(lambda: cls(sectors), data.ins, seeds=(seed,))
print(variant, res.round(4).to_string(index=False))
