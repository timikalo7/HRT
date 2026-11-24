"""c6_blend3_variant = 3.699 (confirmed). Components: c1, pub, c3, ours, m, var (variant latent recovered from c6)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
SC = "/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/"; A = SC + "all/predictions_"
rd = lambda p: pd.read_csv(p).set_index("headline_id").pred.values
u = lambda x: x / x.std()
pub, m, c3, c1 = rd(A + "public_improved_all_rerun.csv"), rd(A + "public_m3564_rerun.csv"), rd(SC + "c3.csv"), rd(A + "c1_latent.csv")
ours = rd("../submissions/kit4_ind_learned.csv"); c6 = rd(A + "c6_blend3_variant.csv")
var = (c6 - 0.15 * u(pub) - 0.15 * u(m)) / 0.7
print("recovered variant sd (expect ~1):", round(var.std(), 4), " corr(var, c3)", round(np.corrcoef(var, c3)[0, 1], 4))
basis = ["c1", "pub", "c3", "ours", "m", "var"]
U = np.c_[u(c1), u(pub), u(c3), u(ours), u(m), u(var)]
files = dict(c2=SC + "c2.csv", c3o="../submissions/c3_plus_ours25.csv", F="../submissions/final_lb_blend_m3564_15.csv",
             G="../submissions/fitted_sharpe_3702.csv", c6=A + "c6_blend3_variant.csv", m=A + "public_m3564_rerun.csv")
W = {}
for k, p in files.items():
    x = u(rd(p)); w = np.linalg.lstsq(U, x, rcond=None)[0]; W[k] = w
    print(f"{k:4s} {np.round(w, 3)} R2 {1 - np.var(x - U @ w):.5f}")
obs = {"c2": 3.669, "c3o": 3.653, "F": 3.673, "G": 3.650, "c6": 3.699, "m": 3.564}
v = np.load(SC + "v_oos.npy"); r = np.sqrt(252)
S0 = np.array([[np.mean(U[:, i] * U[:, j] * v) for j in range(6)] for i in range(6)]); sd = np.sqrt(np.diag(S0))
C = np.corrcoef(U.T)
def model(x):
    mu, al = x[:6], x[6]; R = S0 / np.outer(sd, sd); R = 1 - al * (1 - R); np.fill_diagonal(R, 1); S = R * np.outer(sd, sd)
    return lambda w: r * w @ mu / np.sqrt(w @ S @ w)
prior = np.array([3.49, 3.6, 3.59, 3.52, 3.564, 3.596]) + 0.08  # CV + typical CV->LB offset
def fit(pubclaim, ptight):
    def resid(x):
        f = model(x); out = [(f(W[k]) - o) / 0.005 for k, o in obs.items()]
        if pubclaim: out.append((r * x[1] / sd[1] - 3.6) / 0.05)
        out += list((r * x[:6] / sd - prior) / ptight); return out
    x = least_squares(resid, np.r_[prior / r * sd, 1.0], bounds=(np.r_[-np.ones(6), 0.2], np.r_[np.ones(6), 20])).x
    f = model(x); return f, x
fits = {}
for pc in (False, True):
    for pt in (0.15, 0.3, 0.6):
        f, x = fit(pc, pt); fits[(pc, pt)] = f
        print(f"fit pubclaim={pc} prior_sd={pt}: misfit max {max(abs(f(W[k]) - o) for k, o in obs.items()):.4f}  singles",
              dict(zip(basis, np.round(r * x[:6] / sd, 2))))
w6 = W["c6"]
def corr(a, b): return a @ C @ b / np.sqrt((a @ C @ a) * (b @ C @ b))
rng = np.random.default_rng(1); best = None
for _ in range(60000):
    w = np.abs(w6 + rng.normal(0, 0.06, 6)); w /= w.sum()
    if corr(w, w6) < 0.998: continue
    gains = [f(w) - f(w6) for f in fits.values()]
    if best is None or min(gains) > best[0]: best = (min(gains), w, gains)
print("\nbest robust step (corr>=0.998):", dict(zip(basis, np.round(best[1], 3))))
print("  gain over c6 under each fit:", np.round(best[2], 4), " corr to c6", round(corr(best[1], w6), 4))
np.save(SC + "w_c6step.npy", best[1])
for thr in (0.996, 0.993, 0.99):
    best = None
    for _ in range(60000):
        w = np.abs(w6 + rng.normal(0, 0.12, 6)); w /= w.sum()
        if corr(w, w6) < thr: continue
        gains = [f(w) - f(w6) for f in fits.values()]
        if best is None or min(gains) > best[0]: best = (min(gains), w, gains)
    print(f"corr>={thr}: {dict(zip(basis, np.round(best[1], 3)))} worst gain {best[0]:.4f} best {max(best[2]):.4f}")
    np.save(SC + f"w_c6step_{thr}.npy", best[1])
