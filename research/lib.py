"""Shared research utilities: fast data loading, headline parsing, Sharpe, CV folds.

Everything here works on whole arrays (no per-row apply) so experiments run in seconds.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
HALF = 49
NBARS = 100
EXEC_TITLES = {"CEO", "CTO", "CFO", "COO", "Chief"}


def company_of(headline: str) -> str:
    """Company name = leading capitalised words, stopping at a verb, number or exec title."""
    out = []
    for w in headline.split():
        if w[0].islower() or w in EXEC_TITLES or w[0] == "$" or w[0].isdigit():
            break
        out.append(w)
    return " ".join(out)


def template_of(headline: str) -> str:
    """Headline with the company name stripped (numbers kept: they change the effect size)."""
    return headline[len(company_of(headline)):].strip()


@functools.cache
def load(split: str = "ins"):
    """Return (closes[n,B], opens, highs, lows, headlines df, sessions index, returns or None)."""
    bars = pd.read_parquet(DATA / f"{split}_bars.parquet")
    hl = pd.read_parquet(DATA / f"{split}_headlines.parquet")
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    piv = {c: bars.pivot(index="session", columns="bar_ix", values=c).reindex(sessions).to_numpy()
           for c in ("open", "high", "low", "close")}
    hl = hl.copy()
    hl["co"] = hl["headline"].map(company_of)
    hl["tmpl"] = hl["headline"].map(template_of)
    hl["si"] = hl["session"].map({s: i for i, s in enumerate(sessions)}).astype(int)
    ret = None
    if split == "ins":
        ret = (pd.read_parquet(DATA / "ins_returns.parquet").set_index("session")["returns"]
               .reindex(sessions).to_numpy())
    return piv, hl, sessions, ret


def sharpe(pred: np.ndarray, ret: np.ndarray) -> float:
    pnl = pred * ret
    return float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(252))


def folds(n: int, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(idx[i::k]) for i in range(k)]
