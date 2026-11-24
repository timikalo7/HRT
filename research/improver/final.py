"""Final: compare the combined candidate against base on identical folds, run the leak test,
save OOF positions (seeds 0, 1) and OOS positions for the chosen variant."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/user/HRT")
from stage2 import run, positions, BASE, OUT
from arena.core import sharpe

V = {"base": (BASE, "base"),
     "sign_conf": (BASE + ["mu_pos", "conflict"], "base"),
     "var_res_raw": (BASE, "res:raw"),
     "sign_conf+var_res_raw": (BASE + ["mu_pos", "conflict"], "res:raw"),
     "sign_conf+r5+var_res_raw": (BASE + ["mu_pos", "conflict", "r5"], "res:raw")}
res, oofs = run(V)
cands = [k for k in V if k != "base" and all(res[k][s] > res["base"][s] for s in (0, 1))]
best = max(cands, key=lambda k: min(res[k][s] - res["base"][s] for s in (0, 1))) if cands else "base"
print("chosen:", best)
cols, vm = V[best]
# leak test on seed 0 fold 0: same training-fold fit, test features from the truncated block vs
# from a full block whose bars 50-99 are random and whose later headlines are dropped
z = np.load(f"{OUT}/feat_s0_f0.npz")
p_clean = positions(z["Ftr"], z["ytr"], z["Fte"], cols, vm)
p_leak = positions(z["Ftr"], z["ytr"], z["Fleak"], cols, vm)
print("leak test: max |pos diff| =", np.abs(p_clean - p_leak).max(), " max |feature diff| =",
      np.abs(z["Fte"] - z["Fleak"]).max(), " base model max |pos diff| =", np.abs(z["base"] - z["base_leak"]).max())
full = np.load(f"{OUT}/feat_full.npz")
oos = positions(full["Ftr"], full["y"], full["Foos"], cols, vm)
chk = positions(full["Ftr"], full["y"], full["Foos"], BASE, "base")
print("base OOS reproduces original predict: max diff", np.abs(chk - full["base_oos"]).max(),
      " corr(best, base) oos", np.corrcoef(oos, full["base_oos"])[0, 1])
assert np.all(np.diff(full["oos_sessions"]) > 0) and np.all(np.diff(full["ins_sessions"]) > 0)
np.save(f"{OUT}/best_oof_seed0.npy", oofs[(best, 0)]); np.save(f"{OUT}/best_oof_seed1.npy", oofs[(best, 1)])
np.save(f"{OUT}/best_oos.npy", oos)
np.save(f"{OUT}/base_oof_seed0.npy", oofs[("base", 0)]); np.save(f"{OUT}/base_oof_seed1.npy", oofs[("base", 1)])
print("finite:", np.isfinite(oos).all(), np.isfinite(oofs[(best, 0)]).all(), "shapes", oos.shape, oofs[(best, 0)].shape)
json.dump({"chosen": best, "res": res}, open(f"{OUT}/final_results.json", "w"), indent=1)
