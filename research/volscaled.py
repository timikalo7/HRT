"""Volatility-normalised model.

Measured on in-sample data: a headline's price effect is proportional to the session's volatility (response
slope by volatility quintile 0.96 / 1.08 / 1.24 / 1.47 / 1.60; interaction ratio 1.03-1.06 where 1 = fully
proportional). So the whole problem is solved in volatility units: every session's log-price path is rescaled
to a common bar volatility, the inner model is fitted and applied there, and the position is converted back.

In normalised units the forecast f' and the noise are comparable across sessions; the real-return forecast is
f' * k with k = sigma_session / SIGMA_BAR, so the Sharpe-optimal position f'k / (c k^2 SIGMA_BAR^2 + f'^2 k^2)
equals [f' / (c SIGMA_BAR^2 + f'^2)] / k.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/user/HRT")
from arena.core import Block  # noqa: E402
from arena.latent import VAR_SCALE, LatentTarget  # noqa: E402

SIGMA_BAR = 0.00245


def bar_sigma(block: Block) -> np.ndarray:
    """Per-session bar volatility from every bar the block contains: Garman-Klass plus opening gaps."""
    o, h, l, c = (np.log(x) for x in (block.open, block.high, block.low, block.close))
    prev = np.concatenate([np.zeros((len(c), 1)), c[:, :-1]], 1)
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2 + (o - prev) ** 2
    return np.sqrt(np.maximum(gk.mean(1), 1e-10))


def scale_block(block: Block, k: np.ndarray) -> Block:
    """Divide every log return of each session by k (log prices start at 0, so scaling the log path does it)."""
    f = (1.0 / k)[:, None]
    o, h, l, c = (np.exp(np.log(x) * f) for x in (block.open, block.high, block.low, block.close))
    return Block(block.sessions, o, h, l, c, block.headlines, None if block.returns is None else block.returns / k)


class VolNormalised:
    name = "vol_normalised"

    def __init__(self, make_inner):
        self.make_inner = make_inner

    def fit(self, train: Block) -> None:
        k = bar_sigma(train) / SIGMA_BAR          # training sessions: all 100 bars
        self.inner = self.make_inner()
        self.inner.fit(scale_block(train, k))

    def forecast_normalised(self, test: Block):
        k = bar_sigma(test) / SIGMA_BAR           # test sessions: bars 0-49 are all the block holds
        f, var = self.inner.forecast(scale_block(test, k))
        return f, var, k

    def predict(self, test: Block) -> np.ndarray:
        f, var, k = self.forecast_normalised(test)
        return f / (VAR_SCALE * var + f ** 2) / k


if __name__ == "__main__":
    from research.harness import data, run
    _, sectors = data()
    which, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if which == "base":
        make = lambda: VolNormalised(lambda: LatentTarget(sectors))
    else:
        from research.final_variant import Plus
        make = lambda: VolNormalised(lambda: Plus(sectors))
    first = seed == 0
    run(make, f"volnorm_{which}" + ("" if first else f"_s{seed}"), out, seeds=(seed,), write_oos=first, do_leak_test=first)
