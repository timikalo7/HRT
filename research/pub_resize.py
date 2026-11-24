"""Public model's forecast with Sharpe-optimal sizing instead of clip x vol-scale. Saves OOF + OOS."""
import sys, time
import numpy as np
sys.path.insert(0, '/home/user/suspiciously-haunted/algo-arena')
import backtest_bench, strategies
from backtest_bench import Data, cv_folds, metrics
from strategies import ImprovedAll
class Raw(ImprovedAll):
    """Return (forecast, predicted second moment) instead of a sized position."""
    def size(self, p, split):
        v = np.maximum(self.risk_model.predict(self.risk_inputs(split)), self.v_floor)
        self.last = (p, v)
        return p
data = Data(); t0 = time.time()
m = Raw(); m.fit(data.view(np.arange(data.n), train=True)); m.predict(strategies.oos_split())
np.save('pubraw_oos.npy', np.stack(m.last))
for seed in (0, 1):
    folds = cv_folds(data.n, seed); P = np.zeros(data.n); V = np.zeros(data.n)
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        mm = Raw(); mm.fit(data.view(tr, train=True)); mm.predict(data.view(te, train=False))
        P[te], V[te] = mm.last
    np.save(f'pubraw_oof_seed{seed}.npy', np.stack([P, V]))
    r = data.ret
    print(f'seed {seed}: raw p {metrics(P, r)["sharpe"]:.3f} | p/v^0.5 {metrics(P / V**0.5, r)["sharpe"]:.3f} | p/v {metrics(P / V, r)["sharpe"]:.3f} | ' + ' | '.join(f'p/(v+{k}p2) {metrics(P / (V + k * P**2), r)["sharpe"]:.3f}' for k in (0.5, 1, 2, 4)), round(time.time() - t0), 's', flush=True)
