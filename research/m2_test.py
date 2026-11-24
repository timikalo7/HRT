import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe
from arena.latent import LatentTarget, all_sectors, VAR_SCALE
d = load(); ins = d.ins; r = ins.returns; n = len(ins); sectors = all_sectors(d)
seed = int(sys.argv[1]); folds = np.random.default_rng(seed).permutation(n) % 5
MU = np.zeros(n); VAR = np.zeros(n); M2 = np.zeros(n)
for f in range(5):
    tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
    m = LatentTarget(sectors); m.fit(ins.take(tr, first_half=False))
    MU[te], VAR[te] = m.forecast(ins.take(te, first_half=True)); M2[te] = m.last_m2
np.savez(f'm2_seed{seed}.npz', MU=MU, VAR=VAR, M2=M2)
a = np.load(f'pub_oof_improved_all_seed{seed}.npy'); z = lambda x: x / x.std()
for name, pos in [('current mu/(15v+mu^2)', MU / (VAR_SCALE * VAR + MU ** 2)), ('second moment mu/(15v+m2)', MU / (VAR_SCALE * VAR + M2)), ('mu/(15v+0.5(mu^2+m2))', MU / (VAR_SCALE * VAR + 0.5 * (MU ** 2 + M2))), ('mu/(15v+2*m2)', MU / (VAR_SCALE * VAR + 2 * M2))]:
    print(f'seed {seed}  {name:28s} alone {sharpe(pos, r):.3f}   50/50 blend with public {sharpe(0.5 * z(a) + 0.5 * z(pos), r):.3f}', flush=True)
