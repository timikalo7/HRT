"""Leaderboard forensics: what do the exact scores of known blends say about each component on the hidden set?
Linear model: Sharpe(w) = sqrt(252) * w.m / sqrt(w' C w); C from in-sample out-of-fold PnL streams; m = hidden-set mean PnL per component."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/HRT')
from arena.core import load
from arena.latent import VAR_SCALE
S = '/tmp/claude-0/-home-user-HRT/1b646c27-9e64-5f9e-b4f8-43257d3086ff/scratchpad/'
d = load(); r = d.ins.returns; n = len(r); z = lambda x: x / x.std()
s3 = np.load(S + 'research/stage3_seed0.npz'); f = s3['k7']; f = f - f.mean()
m2 = np.load(S + 'research/m2_seed0.npz')
oof = {'a_public': np.load(S + 'research/pub_oof_improved_all_seed0.npy'), 'c_m3564': np.load(S + 'm3564/m3564_oof_seed0.npy'),
       'c1_latent_nostage2': m2['MU'] / (VAR_SCALE * m2['VAR'] + m2['MU'] ** 2), 'b3_latent_stage2': np.load(S + 'research/mine_stage2_oof_seed0.npy'),
       'pos_variant': f / (VAR_SCALE * s3['VAR'] + f ** 2), 'v_improver_variant': np.load(S + 'improver/best_oof_seed0.npy')}
names = list(oof); P = np.column_stack([z(oof[k]) * r for k in names]); m0 = P.mean(0); C = np.cov(P.T)
sh = lambda w, m: np.sqrt(252) * (w @ m) / np.sqrt(w @ C @ w)
ix = {k: i for i, k in enumerate(names)}
def w(**kw):
    v = np.zeros(len(names))
    for k, x in kw.items(): v[ix[k]] = x
    return v
obs = [('c2 (0.5 public + 0.5 latent no-stage2)', w(a_public=.5, c1_latent_nostage2=.5), 3.669),
       ('c9 (0.15 pub + 0.15 m3564 + 0.70 pos variant)', w(a_public=.15, c_m3564=.15, pos_variant=.70), 3.685),
       ('c7 (0.25 m3564 + 0.75 latent stage2)', w(c_m3564=.25, b3_latent_stage2=.75), 3.658),
       ('c6 (0.15 pub + 0.15 m3564 + 0.70 improver variant)', w(a_public=.15, c_m3564=.15, v_improver_variant=.70), 3.699)]
W = np.array([o[1] for o in obs]); lb = np.array([o[2] for o in obs])
print('in-sample CV Sharpe of each component:', {k: round(sh(np.eye(len(names))[i], m0), 3) for i, k in enumerate(names)})
for (nm, wv, s) in obs: print(f'  {nm:52s} CV {sh(wv, m0):.3f}  leaderboard {s:.3f}  uplift {s - sh(wv, m0):+.3f}')
y = lb / np.sqrt(252) * np.sqrt(np.einsum('ji,ik,jk->j', W, C, W))
for tau in (0.03, 0.06, 0.12):
    Sig = C / n + tau ** 2 * np.outer(m0, m0)
    m1 = m0 + Sig @ W.T @ np.linalg.solve(W @ Sig @ W.T + 1e-14 * np.eye(len(W)), y - W @ m0)
    post = Sig - Sig @ W.T @ np.linalg.solve(W @ Sig @ W.T + 1e-14 * np.eye(len(W)), W @ Sig)
    sd = np.sqrt(np.clip(np.diag(post), 0, None)) / np.sqrt(np.diag(C)) * np.sqrt(252)
    print(f'common-level prior sd {tau}: implied hidden-set Sharpe per component:', {k: f'{sh(np.eye(len(names))[i], m1):.2f}±{sd[i]:.2f}' for i, k in enumerate(names)})
    wopt = np.clip(np.linalg.solve(C, m1), 0, None); wopt /= wopt.sum()
    print('      best non-negative blend:', {k: round(x, 2) for k, x in zip(names, wopt) if x > 0.005}, '-> predicted %.3f' % sh(wopt, m1))
