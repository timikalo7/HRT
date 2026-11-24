import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe
from arena.latent import range_variance, VAR_SCALE
d = load(); r = d.ins.returns; n = len(r)
var = range_variance(d.ins)
rows = []; rng = np.random.default_rng(1)
for seed in (0, 1):
    a = np.load(f'pub_oof_improved_all_seed{seed}.npy')
    P, V = np.load(f'pubraw_oof_seed{seed}.npy'); a2 = P / (V + P ** 2)
    mu = np.load(f'oof_seed{seed}.npy', allow_pickle=True).item()[('mu', 4.0)]
    b = mu / (VAR_SCALE * var + mu ** 2)
    z = lambda x: x / x.std()
    print(f'seed {seed}: pub {sharpe(a, r):.3f}  pub-resized {sharpe(a2, r):.3f}  mine {sharpe(b, r):.3f}  corr(pub, mine) {np.corrcoef(a, b)[0,1]:.3f}  corr(pub-resized, mine) {np.corrcoef(a2, b)[0,1]:.3f}')
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        rows.append((seed, w, sharpe(w * z(a) + (1 - w) * z(b), r), sharpe(w * z(a2) + (1 - w) * z(b), r)))
    for name, bl in [('blend(pub,mine)', 0.5 * z(a) + 0.5 * z(b)), ('blend(pub-resized,mine)', 0.5 * z(a2) + 0.5 * z(b))]:
        pa, pb = a * r, bl * r; diffs = []
        for _ in range(2000):
            ix = rng.integers(0, n, n)
            diffs.append(np.sqrt(252) * (pb[ix].mean() / pb[ix].std() - pa[ix].mean() / pa[ix].std()))
        diffs = np.array(diffs)
        print(f'   {name} vs public alone: {sharpe(bl, r) - sharpe(a, r):+.3f}  bootstrap sd {diffs.std():.3f}  P(>0) {(diffs > 0).mean():.3f}')
t = pd.DataFrame(rows, columns=['seed', 'w_public', 'blend_pub', 'blend_pub_resized']).set_index(['w_public', 'seed']).unstack()
print(t.round(3).to_string())
