"""Shared data loading, headline parsing and scoring for the HRT Algo Arena
synthetic stock/news prediction challenge.

Data contract (organiser's starter kit): five parquet files in one folder --
ins_bars, ins_headlines, ins_returns, oos_bars, oos_headlines. The folder is taken
from the ARENA_DATA environment variable, defaulting to ./data (git-ignored).

Nothing here imports the organiser's sessions.py; the scoring formula and the
submission format are re-implemented to match it exactly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HALF = 49  # trade at the close of this bar
HORIZON = 10  # hold until the close of bar HALF + HORIZON
# Verified by event study on the in-sample data: a headline moves price at a constant
# rate over lags 0..10, with half weight at lags 0 and 10, and nothing outside.
KERNEL = np.array([0.5] + [1.0] * 9 + [0.5])

_TITLES = {"Chief", "CTO", "CFO", "CEO", "COO"}
_SLOTS = {
    "SEG": ["automated logistics", "wireless connectivity", "digital payments", "renewable storage",
            "cloud infrastructure", "supply chain optimization", "data analytics", "enterprise software",
            "precision manufacturing", "process automation"],
    "CP": ["a top-tier research institute", "a global retailer", "a leading cloud platform",
           "an international consortium", "a major logistics provider", "a multinational manufacturer",
           "a national infrastructure agency"],
    "REG": ["Southeast Asia", "Asia Pacific", "Latin America", "Central Europe", "Middle East",
            "North America", "Scandinavia", "Europe"],
    "CONF": ["Global Innovation Summit", "Annual Industry Forum", "Emerging Markets Forum",
             "Corporate Strategy Symposium", "Tech Leadership Conference", "Capital Markets Day"],
    "EXEC": ["Chief Strategy Officer", "Chief Operating Officer", "CTO", "CFO", "CEO"],
}


def data_dir() -> Path:
    return Path(os.environ.get("ARENA_DATA", "data")).resolve()


def split_company(headline: str) -> tuple[str, str]:
    """Company = leading capitalised tokens that are not an executive title."""
    toks = headline.split()
    i = 0
    while i < len(toks) and toks[i][:1].isupper() and toks[i] not in _TITLES:
        i += 1
    return " ".join(toks[:i]), " ".join(toks[i:])


def event_type(rest: str) -> str:
    """Headline text after the company, with numbers and slot fillers masked.

    A filler missing from _SLOTS is left in place, so an unseen vocabulary only
    splits a type into several rather than breaking anything; `Data.summary()`
    reports the type count (50 in the edition this was checked against).
    """
    t = re.sub(r"\$[\d.]+[MBK]?", "$X", rest)
    t = re.sub(r"\d+(?:\.\d+)?%", "N%", t)
    for tag, fillers in _SLOTS.items():
        for f in fillers:
            if f in t:
                t = t.replace(f, f"<{tag}>")
                break
    return t


def _parse(h: pd.DataFrame, sessions: np.ndarray) -> pd.DataFrame:
    cache: dict[str, tuple[str, str]] = {}
    for s in h["headline"].unique():
        company, rest = split_company(s)
        cache[s] = (company, event_type(rest))
    h = h.copy()
    h["company"] = h["headline"].map(lambda s: cache[s][0])
    h["ev"] = h["headline"].map(lambda s: cache[s][1])
    h["si"] = h["session"].map(pd.Series(np.arange(len(sessions)), index=sessions)).to_numpy()
    return h.sort_values(["si", "bar_ix"], kind="stable").reset_index(drop=True)


def _wide(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    return bars.pivot(index="session", columns="bar_ix", values=col).sort_index()


@dataclass
class Block:
    """One set of sessions. Arrays are (n_sessions, n_bars); headlines carry `si`, the row."""
    sessions: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    headlines: pd.DataFrame
    returns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.sessions)

    def logret(self) -> np.ndarray:
        """Per-bar log return; bar 0 is close0 / open0."""
        lc = np.log(self.close)
        return np.diff(np.concatenate([np.log(self.open[:, :1]), lc], axis=1), axis=1)

    def take(self, idx: np.ndarray, *, first_half: bool) -> "Block":
        """Subset of sessions. `first_half=True` hides everything after bar HALF and the labels."""
        idx = np.sort(np.asarray(idx))
        remap = np.full(len(self), -1)
        remap[idx] = np.arange(len(idx))
        h = self.headlines[remap[self.headlines["si"].to_numpy()] >= 0].copy()
        h["si"] = remap[h["si"].to_numpy()]
        cut = slice(0, HALF + 1) if first_half else slice(None)
        if first_half:
            h = h[h["bar_ix"] <= HALF]
        return Block(self.sessions[idx], self.open[idx, cut], self.high[idx, cut], self.low[idx, cut],
                     self.close[idx, cut], h.reset_index(drop=True),
                     None if first_half or self.returns is None else self.returns[idx])


@dataclass
class Data:
    ins: Block
    oos: Block

    def summary(self) -> str:
        allh = pd.concat([self.ins.headlines, self.oos.headlines])
        return (f"in-sample {len(self.ins)} sessions x {self.ins.close.shape[1]} bars, "
                f"out-of-sample {len(self.oos)} sessions x {self.oos.close.shape[1]} bars, "
                f"{allh['company'].nunique()} companies, {allh['ev'].nunique()} event types")


def load(folder: Path | None = None) -> Data:
    folder = folder or data_dir()
    missing = [f for f in ("ins_bars", "ins_headlines", "ins_returns", "oos_bars", "oos_headlines")
               if not (folder / f"{f}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}; put the organiser's parquet files there "
                                "or set ARENA_DATA")

    def block(prefix: str, returns: bool) -> Block:
        bars = pd.read_parquet(folder / f"{prefix}_bars.parquet")
        wide = {c: _wide(bars, c) for c in ("open", "high", "low", "close")}
        sessions = wide["close"].index.to_numpy()
        h = _parse(pd.read_parquet(folder / f"{prefix}_headlines.parquet"), sessions)
        r = None
        if returns:
            r = (pd.read_parquet(folder / f"{prefix}_returns.parquet").set_index("session")["returns"]
                 .reindex(sessions).to_numpy())
        return Block(sessions, *(wide[c].to_numpy(float) for c in ("open", "high", "low", "close")), h, r)

    return Data(block("ins", True), block("oos", False))


def sharpe(pred: np.ndarray, returns: np.ndarray) -> float:
    """The competition score: sqrt(252) * mean(pnl) / std(pnl), pnl = pred * return.

    Uses the sample standard deviation (ddof=1), as the organiser's pandas code does.
    """
    pnl = np.asarray(pred, float) * np.asarray(returns, float)
    return float(np.sqrt(252) * pnl.mean() / (pnl.std(ddof=1) + 1e-12))


def cross_validate(make_model, ins: Block, *, k: int = 5, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Out-of-fold Sharpe. A model has fit(train: Block) and predict(test: Block) -> array;
    `test` only ever contains bars 0..HALF, their headlines, and no labels."""
    rows = []
    n = len(ins)
    for seed in seeds:
        folds = np.random.default_rng(seed).permutation(n) % k
        oof = np.zeros(n)
        per_fold = []
        for f in range(k):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            model = make_model()
            model.fit(ins.take(tr, first_half=False))
            oof[te] = model.predict(ins.take(te, first_half=True))
            per_fold.append(sharpe(oof[te], ins.returns[te]))
        rows.append(dict(seed=seed, sharpe=sharpe(oof, ins.returns), fold_min=min(per_fold),
                         fold_max=max(per_fold), fold_sd=float(np.std(per_fold)),
                         corr=float(np.corrcoef(oof, ins.returns)[0, 1])))
    return pd.DataFrame(rows)


def write_submission(sessions: np.ndarray, pred: np.ndarray, path: str | Path) -> Path:
    """CSV in the upload format: columns headline_id,pred -- headline_id is the session number."""
    pred = np.asarray(pred, float)
    if len(pred) != len(sessions) or len(set(sessions.tolist())) != len(sessions):
        raise ValueError("need exactly one prediction per out-of-sample session")
    out = pd.DataFrame({"headline_id": sessions.astype(np.int64), "pred": np.where(np.isfinite(pred), pred, 0.0)})
    path = Path(path)
    out.to_csv(path, index=False, lineterminator="\n")
    return path
