"""Is the hidden target company predictable from first-half headline structure (count, order)?"""
import os, sys, numpy as np, pandas as pd
os.environ["ARENA_DATA"] = "/home/user/HRT/data"; sys.path.insert(0, "/home/user/tm")
from arena.core import load as aload
from arena.latent import LatentTarget, all_sectors, _Prep, _posterior
data = aload(); sectors = all_sectors(data)
m = LatentTarget(sectors); m.fit(data.ins)
ins = data.ins
prep = _Prep(ins, m.types, sectors)
lr = ins.logret(); y = lr - m.mu0; sig = y.std(1)
X = prep.paths(m.amp, lr.shape[1])
p, _ = _posterior(y, sig, X, prep, m.peer, with_unseen=False, temp=1.0)
p = p[:, :4]
# first-half headline counts per slot
h1 = prep.hb <= 49
cnt = np.zeros((prep.n, 4)); np.add.at(cnt, (prep.hs[h1], prep.hj[h1]), 1)
cntall = np.zeros((prep.n, 4)); np.add.at(cntall, (prep.hs, prep.hj), 1)
main = prep.main
rows = []
for s in range(prep.n):
    for j in range(4):
        if main[s, j]:
            rows.append((s, j, p[s, j], cnt[s, j], cntall[s, j], main[s].sum()))
d = pd.DataFrame(rows, columns=["s", "slot", "p", "cnt1", "cntall", "nmain"])
d = d[d.nmain == 3]
d["rank1"] = d.groupby("s").cnt1.rank(ascending=False, method="first")
print("mean P(target) for main companies, 3-main sessions (uniform = 0.333)")
print(" by slot (order of first appearance):", d.groupby("slot").p.mean().round(3).to_dict())
print(" by first-half count rank:", d.groupby("rank1").p.mean().round(3).to_dict())
print(" by first-half count:", d.groupby(d.cnt1.clip(0, 8)).p.mean().round(3).to_dict())
print(" by full-session count:", d.groupby(d.cntall.clip(0, 12)).p.mean().round(3).to_dict())
# share of headlines: own vs peer by bar half
print(" corr(p, cnt1 share)", round(np.corrcoef(d.p, d.cnt1 / d.groupby('s').cnt1.transform('sum').clip(lower=1))[0, 1], 3))
print(" mean max posterior", round(p.max(1).mean(), 3))
