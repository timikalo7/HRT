"""Advisor item 6: drift vs session vol, generic momentum in quiet stretches, per-event heterogeneity."""
import copy

import numpy as np
import pandas as pd

from model import UNIT, bar_returns, load, template_rates_deconv

d = load("ins")
R = bar_returns(d)
S, NB = R.shape
ds = copy.copy(d)
ds.h = d.h.assign(templ=d.h["stem"])
rates = template_rates_deconv(ds, R, np.ones(S, bool), lam=400)
h = ds.h.copy()
h["rate"] = h["templ"].map(rates)

# expected drift path per session/bar from ALL headlines (full IS, research only)
E = np.zeros_like(R)
for L in range(11):
    t = h["bar_ix"].to_numpy() + L
    ok = t < NB
    np.add.at(E, (h["si"].to_numpy()[ok], t[ok]), h["rate"].to_numpy()[ok] * UNIT[L])
resid = R - E - (R - E).mean()

# ---- 1. drift vs session vol: regress R on E within sigma quintiles
sig_full = resid.std(axis=1)
sig_first = R[:, :50].std(axis=1)
q = pd.qcut(sig_first, 5, labels=False)
print("drift multiplier (slope of R on E) by first-half vol quintile:")
for k in range(5):
    m = q == k
    e, r = E[m].ravel(), R[m].ravel()
    e = e - e.mean()
    print(f"  q{k}: sigma {sig_first[m].mean()*1e4:5.1f}bps  slope {np.dot(e, r - r.mean())/np.dot(e, e):.3f}")

# ---- 2. generic momentum: residual autocorrelation on bars with no headline in the prior 11 bars
quiet = np.ones_like(R, bool)
for L in range(12):
    t = h["bar_ix"].to_numpy() + L
    ok = t < NB
    quiet[h["si"].to_numpy()[ok], t[ok]] = False
for lag in (1, 2, 5):
    a = resid[:, lag:][quiet[:, lag:] & quiet[:, :-lag]]
    b = resid[:, :-lag][quiet[:, lag:] & quiet[:, :-lag]]
    print(f"quiet-bar residual autocorr lag {lag}: {np.corrcoef(a, b)[0,1]:+.4f} (n={a.size})")
# also on all bars
for lag in (1, 2, 5):
    print(f"all-bar residual autocorr lag {lag}: {np.corrcoef(resid[:, lag:].ravel(), resid[:, :-lag].ravel())[0,1]:+.4f}")

# ---- 3. per-event heterogeneity: correlation between excess drift in lags 0-4 and lags 5-10 of the same event
# use isolated events (no other headline within +-11 bars in the same session) to avoid overlap
hb = h.sort_values(["si", "bar_ix"])
gap_prev = hb.groupby("si")["bar_ix"].diff().fillna(99)
gap_next = -hb.groupby("si")["bar_ix"].diff(-1).fillna(-99)
iso = hb[(gap_prev > 11) & (gap_next > 11) & (hb["bar_ix"] <= NB - 11)]
si, b = iso["si"].to_numpy(), iso["bar_ix"].to_numpy()
early = np.array([R[s, bb:bb + 5].sum() for s, bb in zip(si, b)]) - iso["rate"].to_numpy() * UNIT[0:5].sum()
late = np.array([R[s, bb + 5:bb + 11].sum() for s, bb in zip(si, b)]) - iso["rate"].to_numpy() * UNIT[5:11].sum()
print(f"\nisolated events: {len(iso)}; corr(early excess, late excess) = {np.corrcoef(early, late)[0,1]:+.4f}")
# control: same windows on quiet stretches (no events) -> pure noise autocorrelation
# implied event-rate sd: cov(early, late) / (units_early * units_late)
cov = np.cov(early, late)[0, 1]
print(f"implied per-event rate sd: {np.sqrt(max(cov, 0) / (UNIT[0:5].sum() * UNIT[5:11].sum()))*1e4:.2f} bps/bar"
      f"  (mean |rate| {np.abs(iso['rate']).mean()*1e4:.2f})")
# does heterogeneity scale with |rate|? split neutral vs non-neutral
for name, m in (("neutral |rate|<1bp", np.abs(iso["rate"].to_numpy()) < 1e-4),
                ("non-neutral", np.abs(iso["rate"].to_numpy()) >= 1e-4)):
    c = np.cov(early[m], late[m])[0, 1]
    print(f"  {name:20s} n={m.sum():6d} corr {np.corrcoef(early[m], late[m])[0,1]:+.4f}"
          f"  implied sd {np.sqrt(max(c,0)/(4.5*5.5))*1e4:.2f} bps/bar")
