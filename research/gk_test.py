import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, sharpe, HALF
d = load(); ins = d.ins; r = ins.returns
O, H, L, C = (np.log(x[:, :HALF + 1]) for x in (ins.open, ins.high, ins.low, ins.close))
prev = np.concatenate([np.zeros((len(C), 1)), C[:, :-1]], 1)
gk = (0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2).mean(1)
gap = ((O - prev) ** 2).mean(1)
cc = ((C - prev) ** 2).mean(1)
park = ((H - L) ** 2).mean(1) / (4 * np.log(2))
print('mean per-bar variance: cc %.3e  gk %.3e  gap %.3e  gk+gap %.3e  parkinson %.3e' % (cc.mean(), gk.mean(), gap.mean(), (gk + gap).mean(), park.mean()))
print('corr with r^2: cc %.3f  gk+gap %.3f  parkinson %.3f' % tuple(np.corrcoef(x, r ** 2)[0, 1] for x in (cc, gk + gap, park)))
rows = []
for seed in (0, 1):
    o = np.load(f'oof_seed{seed}.npy', allow_pickle=True).item(); mu = o[('mu', 4.0)]; sig = o['sig']
    for name, s2 in [('cc_std(current)', sig ** 2), ('gk+gap', (gk + gap) * cc.mean() / (gk + gap).mean()), ('parkinson', park * cc.mean() / park.mean()), ('avg(cc,gk+gap)', 0.5 * sig ** 2 + 0.5 * (gk + gap) * cc.mean() / (gk + gap).mean())]:
        for k in (5, 10, 15, 20):
            rows.append((seed, name, k, sharpe(mu / (k * s2 + mu ** 2), r)))
t = pd.DataFrame(rows, columns=['seed', 'var', 'k', 'sharpe']).pivot_table(index='var', columns=['seed', 'k'], values='sharpe')
print(t.round(3).to_string())
