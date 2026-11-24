"""Refit using only confirmed scores: c2 3.669, c3o 3.653, F 3.673, G(fitted) 3.650. No assumption about B."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
SC = "/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/"
rd = lambda p: pd.read_csv(p).set_index("headline_id").pred.values
c2 = rd(SC + "c2.csv"); c3o = rd("../submissions/c3_plus_ours25.csv"); m = rd(SC + "all/predictions_public_m3564_rerun.csv")
F = rd("../submissions/final_lb_blend_m3564_15.csv"); G = rd("../submissions/fitted_sharpe_3702.csv"); B = rd("../submissions/lb_blend_c2_c3o.csv")
v = np.load(SC + "v_oos.npy"); r = np.sqrt(252)
U = np.c_[c2 / c2.std(), c3o / c3o.std(), m / m.std()]
def wts(x): return np.linalg.lstsq(U, x, rcond=None)[0]
W = {"c2": [1, 0, 0], "c3o": [0, 1, 0], "F": wts(F), "G": wts(G), "B": wts(B)}
print({k: np.round(w, 4) for k, w in W.items()})
obs = {"c2": 3.669, "c3o": 3.653, "F": 3.673, "G": 3.650}
S0 = np.array([[np.mean(U[:, i] * U[:, j] * v) for j in range(3)] for i in range(3)])
def model(params):
    mu = params[:3]; al = params[3]
    sd = np.sqrt(np.diag(S0)); R = S0 / np.outer(sd, sd); R = 1 - al * (1 - R); np.fill_diagonal(R, 1)
    S = R * np.outer(sd, sd)
    return lambda w: r * np.dot(w, mu) / np.sqrt(np.dot(w, S @ w)), S
def resid(params):
    f, _ = model(params); return [f(np.array(W[k])) - obs[k] for k in obs]
sd = np.sqrt(np.diag(S0))
x0 = np.r_[np.array([3.669, 3.653, 3.4]) / r * sd, 1.0]
sol = least_squares(resid, x0, bounds=([-1, -1, -1, 0.2], [1, 1, 1, 20]))
f, S = model(sol.x)
print("fit residuals", np.round(resid(sol.x), 4), " alpha", round(sol.x[3], 3))
print("implied single Sharpes c2, c3o, m:", [round(r * sol.x[i] / np.sqrt(S[i, i]), 3) for i in range(3)])
print("predicted B (50/50):", round(f(np.array(W["B"])), 4))
mu = sol.x[:3]; wo = np.linalg.solve(S, mu); wo = wo / np.abs(wo).sum()
print("optimal weights (c2, c3o, m):", np.round(wo, 3), "pred", round(f(wo), 4))
for t in [0, 0.1, 0.2, 0.3, 0.4]:
    w = np.array([0.5, 0.5, t]); print(f"  0.5 c2 + 0.5 c3o + {t:.1f} m : {f(w):.4f}")

print("\n--- add public claim: m3564 alone = 3.564 on the leaderboard ---")
W["m"] = [0, 0, 1]; obs["m"] = 3.564
sol = least_squares(resid, sol.x, bounds=([-1, -1, -1, 0.2], [1, 1, 1, 20]))
f, S = model(sol.x)
print("fit residuals", dict(zip(obs, np.round(resid(sol.x), 4))), " alpha", round(sol.x[3], 3))
best = None
for a in np.arange(0, 1.01, 0.05):
    for b in np.arange(0, 1.01 - a, 0.05):
        w = np.array([a, b, 1 - a - b]); s = f(w)
        if best is None or s > best[0]: best = (s, w)
print("best non-negative weights (c2, c3o, m):", np.round(best[1], 2), "pred", round(best[0], 4))
for w in ([0.425, 0.425, 0.15], [0.4, 0.3, 0.3], [0.5, 0.2, 0.3], [0.45, 0.25, 0.3], [0.4, 0.2, 0.4], [0.35, 0.35, 0.3], [0.5, 0.25, 0.25]):
    print("  ", w, round(f(np.array(w)), 4))
