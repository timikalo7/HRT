# Evaluation harness pattern

Build this in the first 15 minutes of any prediction competition, before the first serious model. It makes every later comparison honest and fast, and it lets independent agents and teammates evaluate on the same footing. Adapt the names; keep the guarantees.

## Guarantees

1. **Fixed folds.** `folds(n, seed) = np.random.default_rng(seed).permutation(n) % K`. Seeds 0 and 1 for development, 2 kept fresh for verification. Write the convention into the repo's CLAUDE.md so teammates use the same folds.
2. **Physical truncation.** The test block handed to `predict` *contains* only what is known at decision time: bars up to the decision bar, events up to then, no labels. Truncate the data itself; don't trust the model to ignore it.
3. **Models slice for themselves too.** Inside `predict`, select bars `0..decision` explicitly. A model that uses "all bars present" passes cross-validation and fails the leak test. That happened here (max change 79.7), and the leak test is what caught it.
4. **Leak test.** Fit on a training fold, predict a test fold. Then take the same test sessions as a *full-length* block, overwrite everything after the decision bar with random numbers, drop later events, and predict again. The maximum absolute change must be exactly 0.0.
5. **Always save out-of-fold and out-of-sample predictions** for every model as `<tag>_oof_seed{s}.npy` and `<tag>_oos.npy`, in a fixed session order. Without them a model cannot enter a cross-validated blend or leaderboard forensics.
6. **Paired comparison against a cached baseline.** Report the Sharpe difference on identical sessions with a session bootstrap (sd and P(diff > 0)). Accept a change only if it is positive on two fold assignments and larger than about one bootstrap sd.
7. **Cache expensive fits** (per seed and fold) so feature or sizing variants run in seconds.

## Skeleton

```python
def oof(make_model, data, seed, K=5):
    n = len(data.train); f = folds(n, seed); p = np.zeros(n)
    for k in range(K):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        m = make_model(); m.fit(data.train.take(tr, first_half=False))
        p[te] = m.predict(data.train.take(te, first_half=True))   # truncated block
    return p

def leak_test(make_model, data, idx, tr):
    m = make_model(); m.fit(data.train.take(tr, first_half=False))
    p1 = m.predict(data.train.take(idx, first_half=True))
    full = data.train.take(idx, first_half=False)                 # full-length block
    full.overwrite_after_decision(rng)                            # noise in bars > decision
    full.drop_events_after_decision(); full.labels = None
    return np.abs(p1 - m.predict(full)).max()                     # must be 0.0

def paired(p_new, p_base, r, n_boot=2000):
    a, b = p_new * r, p_base * r
    boots = [sharpe_diff(a[ix], b[ix]) for ix in bootstrap_indices(len(r), n_boot)]
    return sharpe(p_new, r) - sharpe(p_base, r), np.std(boots), np.mean(np.array(boots) > 0)
```

The HRT Algo Arena implementation of this pattern is in the HRT repo at `research/harness.py` (functions `run`, `run_variant`, `leak_test`, `paired`, `get_base`).

## The gate (run on every file before it reaches the user)

1. `scripts/validate_submission.py FILE --id-range A B` (or `--ids-from`) checks the upload format. Exit code 0 means valid.
2. Leak test of the model(s) behind the file is 0.0.
3. The cross-validated score was reproduced by a run you did not hand-edit.
4. `scripts/ledger.py add` records the hash, composition, CV and expected leaderboard score.

Steps 1 and 4 take seconds. Step 2 takes about one model fit. Run all four every time, and show the result next to the file when you hand it over.
