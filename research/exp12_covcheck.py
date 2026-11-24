"""Validate the leaderboard-fitting approximation on INS, where returns are known (OOF preds, 2 seeds)."""
import numpy as np
from lib import load
SC = "/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/"
piv, _, _, ret = load("ins")
H, L = piv["high"][:, :50], piv["low"][:, :50]; pk = (np.log(H / L) ** 2).mean(1) / (4 * np.log(2))
coef = np.linalg.lstsq(np.c_[np.ones(len(ret)), pk], ret ** 2, rcond=None)[0]; v = np.c_[np.ones(len(ret)), pk] @ coef
r = np.sqrt(252)
for s in (0, 1):
    a = np.load(SC + f"oof_latent_s{s}.npy"); b = np.load(SC + f"oof_ours_s{s}.npy"); a, b = a / a.std(), b / b.std()
    pa, pb = a * ret, b * ret
    act_corr = np.corrcoef(pa, pb)[0, 1]
    approx = np.mean(a * b * v) / np.sqrt(np.mean(a * a * v) * np.mean(b * b * v))
    print(f"seed {s}: pred corr {np.corrcoef(a, b)[0,1]:.4f}  PnL corr actual {act_corr:.4f}  approx {approx:.4f}")
    Sa, Sb = r * pa.mean() / pa.std(), r * pb.mean() / pb.std()
    for w in (0.25, 0.5):
        act = (1 - w) * pa + w * pb; act = r * act.mean() / act.std()
        # approx prediction from the two single scores + approx covariance
        sa, sb = np.sqrt(np.mean(a * a * v)), np.sqrt(np.mean(b * b * v)); ma, mb = Sa / r * sa, Sb / r * sb
        var = (1 - w) ** 2 * sa ** 2 + w ** 2 * sb ** 2 + 2 * w * (1 - w) * np.mean(a * b * v)
        print(f"   w={w}: actual blend {act:.4f}  approx-predicted {r * ((1 - w) * ma + w * mb) / np.sqrt(var):.4f}   (singles {Sa:.3f}, {Sb:.3f})")
