import sys, time
import numpy as np
sys.path.insert(0, '/home/user/suspiciously-haunted/algo-arena')
import backtest_bench, strategies
from backtest_bench import Data, MODELS, cv_folds, metrics
t0 = time.time(); data = Data()
m = MODELS['improved_all'](); m.fit(data.view(np.arange(data.n), train=True))
oos = m.predict(strategies.oos_split()); np.save('pub_oos_improved_all.npy', oos)
print('oos done', round(time.time() - t0), 's', flush=True)
for key in ['improved_all']:
    for seed in (0, 1):
        folds = cv_folds(data.n, seed); p = np.zeros(data.n)
        for f in range(5):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            mm = MODELS[key](); mm.fit(data.view(tr, train=True)); p[te] = mm.predict(data.view(te, train=False))
        np.save(f'pub_oof_{key}_seed{seed}.npy', p)
        print(key, 'seed', seed, 'cv sharpe', round(metrics(p, data.ret)['sharpe'], 3), round(time.time() - t0), 's', flush=True)
