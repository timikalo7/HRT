"""Blend prediction files: each is scaled to unit standard deviation, then averaged with weights.

  python -m arena.blend out.csv 0.5:a.csv 0.5:b.csv

Scaling uses only the predictions themselves, so no labels are involved.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from arena.core import write_submission


def blend(parts: list[tuple[float, pd.Series]]) -> pd.Series:
    index = parts[0][1].index
    total = sum(w for w, _ in parts)
    out = np.zeros(len(index))
    for w, s in parts:
        if not s.index.equals(index):
            raise ValueError("prediction files cover different sessions or orders")
        out += (w / total) * s.to_numpy() / s.to_numpy().std()
    return pd.Series(out, index=index)


def main(argv: list[str]) -> None:
    out_path, specs = argv[0], argv[1:]
    parts = []
    for spec in specs:
        w, path = spec.split(":", 1)
        parts.append((float(w), pd.read_csv(path).set_index("headline_id")["pred"]))
    b = blend(parts)
    write_submission(b.index.to_numpy(), b.to_numpy(), out_path)
    print(f"wrote {len(b)} blended predictions to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
