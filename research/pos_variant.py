"""Variant of the latent model with one extra second-stage feature: the positive part of the owed drift
(lets positive and negative pending news have different slopes). +0.01 CV on both fold assignments."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load, write_submission, sharpe, HALF
from arena.latent import LatentTarget, all_sectors

class WithPos(LatentTarget):
    def _features(self, test):
        Z = super()._features(test)
        return np.column_stack([Z[:, :-1], np.maximum(Z[:, 0], 0.0), Z[:, -1]])

if __name__ == '__main__':
    d = load(); ins = d.ins; sectors = all_sectors(d); n = len(ins); rng = np.random.default_rng(0)
    # leak test on a held-out split
    idx = np.sort(rng.choice(n, 2000, replace=False)); tr = np.setdiff1d(np.arange(n), idx)
    m = WithPos(sectors); m.fit(ins.take(tr, first_half=False)); p1 = m.predict(ins.take(idx, first_half=True))
    cb = ins.take(idx, first_half=False)
    for x in (cb.open, cb.high, cb.low, cb.close): x[:, HALF + 1:] = rng.uniform(0.5, 2.0, x[:, HALF + 1:].shape)
    cb.headlines = cb.headlines[cb.headlines.bar_ix <= HALF].reset_index(drop=True); cb.returns = None
    print('leak test max |change|:', float(np.abs(p1 - m.predict(cb)).max()), '| held-out Sharpe (2000 sessions): %.3f' % sharpe(p1, ins.returns[idx]))
    m = WithPos(sectors); m.fit(ins)
    write_submission(d.oos.sessions, m.predict(d.oos), sys.argv[1])
    print('second-stage beta [mu, count, unexplained, live, pos(mu), const]:', m.beta.round(5))
