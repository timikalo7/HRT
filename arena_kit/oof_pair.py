"""OOF predictions of teammate latent model (arena/latent.py) and our model4 'ind' on identical folds."""
import os, sys, time, numpy as np
os.environ["ARENA_DATA"] = "/home/user/HRT/data"
sys.path.insert(0, "/home/user/tm")
from arena.core import load as aload, sharpe
from arena.latent import LatentTarget, all_sectors
import model4 as Q
from model2 import y, S_ins, ins as KINS
SC = "/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/"
data = aload(); sectors = all_sectors(data)
assert (np.asarray(data.ins.sessions) == np.asarray(KINS["ids"])).all()
assert np.allclose(data.ins.returns, y)
f_ours = Q.make()
n = len(y)
for seed in [int(s) for s in sys.argv[1:]] or [0]:
    folds = np.random.default_rng(seed).permutation(n) % 5
    A = np.zeros(n); B = np.zeros(n); t0 = time.time()
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LatentTarget(sectors); m.fit(data.ins.take(tr, first_half=False))
        A[te] = m.predict(data.ins.take(te, first_half=True))
        B[te] = f_ours(tr, te)
    np.save(SC + f"oof_latent_s{seed}.npy", A); np.save(SC + f"oof_ours_s{seed}.npy", B)
    print(f"seed {seed}: latent {sharpe(A, y):.3f}  ours {sharpe(B, y):.3f}  corr {np.corrcoef(A, B)[0,1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    a, b = A / A.std(), B / B.std()
    for w in [0, .2, .3, .4, .5, .6, .7, 1]:
        print(f"   w_ours={w:.1f}: {sharpe((1-w)*a + w*b, y):.3f}")
