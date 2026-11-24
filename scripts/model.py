"""Shared data loading, template parsing, drift estimation and features.

Generator structure found in event studies (scripts/lag_profile.py):
a headline at bar b adds a constant per-bar drift for lags 0..10 after it
(half weight at lags 0 and 10). So for the 49->59 window only headlines at
bars 40..49 matter, with overlap weight w(b) = b - 39.5.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HALF = 49

# lag -> drift units (half at the edges)
UNIT = np.zeros(200)
UNIT[0] = 0.5
UNIT[1:10] = 1.0
UNIT[10] = 0.5


def templ(headline: str) -> str:
    rest = headline.split(" ", 2)[2]
    rest = re.sub(r"\$[\d.,]+[MBK]?", "$X", rest)
    rest = re.sub(r"\d+(\.\d+)?%", "N%", rest)
    return re.sub(r"\b\d+(\.\d+)?\b", "N", rest)


PRODUCTS = ['automated logistics', 'cloud infrastructure', 'data analytics', 'digital payments',
            'enterprise software', 'precision manufacturing', 'process automation', 'renewable storage',
            'supply chain optimization', 'wireless connectivity']
REGIONS = ['Asia Pacific', 'Central Europe', 'Europe', 'Latin America', 'Middle East', 'North America',
           'Scandinavia', 'Southeast Asia']
PARTIES = ['a global retailer', 'a leading cloud platform', 'a major logistics provider',
           'a multinational manufacturer', 'a national infrastructure agency', 'a top-tier research institute',
           'an international consortium']
TITLES = ['Chief Operating Officer', 'Chief Strategy Officer', 'CEO', 'CFO', 'CTO']
EVENTS = ['Global Innovation Summit', 'Tech Leadership Conference', 'Annual Industry Forum',
          'Emerging Markets Forum', 'Corporate Strategy Symposium', 'Capital Markets Day']
_FILL = [(re.compile(r"\b" + "|".join(re.escape(x) for x in sorted(v, key=len, reverse=True)) + r"\b"), tag)
         for v, tag in ((PARTIES, "<C>"), (TITLES, "<T>"), (PRODUCTS, "<P>"), (REGIONS, "<R>"),
                        (EVENTS, "<E>"))]


def stem(template: str) -> str:
    # company names can carry a "Group"/"Co" suffix that the 2-token split leaves in the template
    template = re.sub(r"^(Group|Co) ", "", template)
    for rx, tag in _FILL:
        template = rx.sub(tag, template)
    return template


@dataclass
class Data:
    sessions: np.ndarray      # sorted session ids
    O: np.ndarray             # (S, nbars) open
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray             # close
    h: pd.DataFrame           # headlines: session, si, bar_ix, headline, company, templ
    returns: np.ndarray | None


def load(kind: str) -> Data:
    bars = pd.read_parquet(DATA / f"{kind}_bars.parquet")
    hl = pd.read_parquet(DATA / f"{kind}_headlines.parquet")
    sessions = np.sort(bars["session"].unique())
    mats = {c: bars.pivot(index="session", columns="bar_ix", values=c).reindex(sessions).to_numpy()
            for c in ("open", "high", "low", "close")}
    hl = hl.copy()
    hl["si"] = np.searchsorted(sessions, hl["session"].to_numpy())
    hl["company"] = hl["headline"].str.split().str[:2].str.join(" ")
    hl["templ"] = hl["headline"].map(templ)
    hl["stem"] = hl["templ"].map(stem)
    ret = None
    if kind == "ins":
        r = pd.read_parquet(DATA / "ins_returns.parquet").set_index("session")["returns"]
        ret = r.reindex(sessions).to_numpy()
    return Data(sessions, mats["open"], mats["high"], mats["low"], mats["close"], hl, ret)


def bar_returns(d: Data) -> np.ndarray:
    R = np.empty_like(d.C)
    R[:, 0] = d.C[:, 0] / d.O[:, 0] - 1
    R[:, 1:] = d.C[:, 1:] / d.C[:, :-1] - 1
    return R


def template_rates(d: Data, R: np.ndarray, train_mask: np.ndarray, lags=range(1, 10),
                   shrink: float = 0.0) -> pd.Series:
    """Per-template mean per-bar return over the given lags, estimated on training sessions.

    Uses all 100 bars (research use of IS second half is allowed; rates are then applied to
    first-half headlines only).
    """
    h = d.h[train_mask[d.h["si"].to_numpy()]]
    si, b = h["si"].to_numpy(), h["bar_ix"].to_numpy()
    nb = R.shape[1]
    vals, cnts = np.zeros(len(h)), np.zeros(len(h))
    for L in lags:
        t = b + L
        ok = t < nb
        vals[ok] += R[si[ok], t[ok]]
        cnts[ok] += 1
    df = pd.DataFrame({"templ": h["templ"].to_numpy(), "v": vals, "c": cnts})
    agg = df.groupby("templ").agg(v=("v", "sum"), c=("c", "sum"), n=("v", "size"))
    rate = agg["v"] / agg["c"]
    if shrink:
        rate = rate * agg["n"] / (agg["n"] + shrink)
    return rate


def template_rates_deconv(d: Data, R: np.ndarray, train_mask: np.ndarray, lam: float = 0.0,
                          prior: pd.Series | None = None, extra: list[tuple[Data, np.ndarray]] = ()
                          ) -> pd.Series:
    """Least-squares deconvolution: R[s,t] = mu + sum_e rate[templ_e] * UNIT[t - b_e] + noise.

    Removes the overlap bias of the simple lag-mean estimator. Ridge `lam` shrinks each rate
    towards `prior` (per-template, default 0). `extra` adds (Data, R) pairs whose events are
    used too (e.g. OOS first half, where R only covers bars 0..49).
    """
    from scipy import sparse

    blocks = [(d, R, train_mask)] + [(e, Re, np.ones(len(e.sessions), bool)) for e, Re in extra]
    templs = pd.Index(sorted(set().union(*[set(x.h["templ"]) for x, _, _ in blocks])))
    T = len(templs)
    XtX = np.zeros((T + 1, T + 1))
    Xty = np.zeros(T + 1)
    for dd, RR, mask in blocks:
        sess = np.flatnonzero(mask)
        nb = RR.shape[1]
        row_of = -np.ones(len(dd.sessions), int)
        row_of[sess] = np.arange(len(sess))
        h = dd.h[mask[dd.h["si"].to_numpy()]]
        hs, hb = row_of[h["si"].to_numpy()], h["bar_ix"].to_numpy()
        hj = templs.get_indexer(h["templ"])
        rows, cols, vals = [], [], []
        for L in range(11):
            t = hb + L
            ok = t < nb
            rows.append(hs[ok] * nb + t[ok])
            cols.append(hj[ok])
            vals.append(np.full(ok.sum(), UNIT[L]))
        n = len(sess) * nb
        X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, T))
        yv = RR[sess].reshape(-1)
        ones = np.ones(n)
        XtX[:T, :T] += (X.T @ X).toarray()
        XtX[:T, T] += X.T @ ones
        XtX[T, :T] += X.T @ ones
        XtX[T, T] += n
        Xty[:T] += X.T @ yv
        Xty[T] += yv.sum()
    p = np.zeros(T + 1)
    if prior is not None:
        p[:T] = prior.reindex(templs).fillna(0).to_numpy()
    A = XtX.copy()
    A[np.arange(T), np.arange(T)] += lam
    coef = np.linalg.solve(A, Xty + lam * p)
    return pd.Series(coef[:T], index=templs)


def news_feature(d: Data, rates: pd.Series, lo: int = 40) -> np.ndarray:
    """Expected drift over bars 50..59 from first-half headlines: sum rate * overlap units."""
    h = d.h[d.h["bar_ix"] <= HALF]
    b = h["bar_ix"].to_numpy()
    # units remaining inside window t=50..59 for a headline at bar b
    w = np.array([UNIT[max(0, 50 - bb): 60 - bb].sum() if 60 - bb > 0 else 0.0 for bb in range(HALF + 1)])
    r = h["templ"].map(rates).fillna(0.0).to_numpy()
    f = np.zeros(len(d.sessions))
    np.add.at(f, h["si"].to_numpy(), r * w[b])
    return f


def sharpe(pred: np.ndarray, ret: np.ndarray) -> float:
    pnl = pred * ret
    return float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(252))


def folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % k
