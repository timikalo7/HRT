"""Forecast-level combination: one forecast from both models, then a single Sharpe-optimal sizing."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe
from arena.latent import VAR_SCALE
d = load(); r = d.ins.returns; n = len(r); z = lambda x: x / x.std()
for seed in (0, 1):
    a = np.load(f'pub_oof_improved_all_seed{seed}.npy'); P, V = np.load(f'pubraw_oof_seed{seed}.npy')
    m = np.load(f'm2_seed{seed}.npz'); MU, VAR = m['MU'], m['VAR']
    b = MU / (VAR_SCALE * VAR + MU ** 2)
    base = sharpe(0.5 * z(a) + 0.5 * z(b), r)
    out = [f'seed {seed}: position blend {base:.3f}']
    # calibrate each forecast to return units with a single slope (2 numbers; fit on all OOF -> tiny optimism)
    sP = np.dot(P, r) / np.dot(P, P); sM = np.dot(MU, r) / np.dot(MU, MU)
    for w in (0.3, 0.5, 0.7):
        f = w * sP * P + (1 - w) * sM * MU
        for k in (10, 15, 20):
            out.append(f'w_pub={w} k={k}: {sharpe(f / (k * VAR + f ** 2), r):.3f}')
    # public forecast alone with our variance and sizing
    fP = sP * P
    out.append('public forecast + our sizing: ' + ' '.join(f'k={k}: {sharpe(fP / (k * VAR + fP ** 2), r):.3f}' for k in (10, 15, 20)))
    # three-way: public position, our position, public-forecast-with-our-sizing
    c = fP / (15 * VAR + fP ** 2)
    out.append(f'blend(pub-our-sizing, mine) 50/50: {sharpe(0.5 * z(c) + 0.5 * z(b), r):.3f}  3-way equal: {sharpe(z(a) + z(b) + z(c), r):.3f}')
    print('\n   '.join(out))
