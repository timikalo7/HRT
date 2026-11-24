"""EXP-04: second stage — add price features to the mean, size by 1/variance. All OOF."""
import numpy as np
from lib import load, sharpe, folds
piv, hl, sess, ret = load("ins")
S='/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/'
irf = np.load(S+'pred_irf.npy')
C = piv["close"]; H=piv["high"]; Lo=piv["low"]
lc = np.log(np.c_[np.ones(len(C)), C]); br = np.diff(lc, axis=1)[:, :50]
# volatility estimate that excludes headline drift: use bar-return std after removing session mean
vol_cc = br.std(axis=1)
vol_pk = np.sqrt((np.log(H[:,:50]/Lo[:,:50])**2).mean(axis=1)/(4*np.log(2)))  # Parkinson
vol_rec = br[:, 30:].std(axis=1)
mom3 = lc[:,50]-lc[:,47]; rev50 = lc[:,50]
n=len(ret); F=folds(n, 5, seed=1)
def oof(X, y):
    out=np.zeros(n)
    for te in F:
        tr=np.setdiff1d(np.arange(n),te); A=np.c_[np.ones(len(tr)),X[tr]]
        b=np.linalg.lstsq(A,y[tr],rcond=None)[0]; out[te]=np.c_[np.ones(len(te)),X[te]]@b
    return out
mu0 = irf - irf.mean()
mu1 = oof(np.c_[irf, mom3, rev50], ret)
print("mean only    irf:", round(sharpe(mu0,ret),3), " +price:", round(sharpe(mu1-mu1.mean(),ret),3))
for name, v in [("cc",vol_cc),("parkinson",vol_pk),("recent",vol_rec)]:
    # variance model: E[res^2] ~ c0 + c1 v^2, fitted OOF on residual of mu1
    res2 = (ret - mu1)**2
    var = np.maximum(oof(v[:,None]**2, res2), 1e-8)
    for mu,lab in [(mu0,'irf'),(mu1,'irf+price')]:
        m = mu - mu.mean()
        print(f"  size 1/var[{name}] {lab}: {sharpe(m/var, ret):.3f}   1/vol: {sharpe(m/np.sqrt(var), ret):.3f}")
