#!/usr/bin/env python3
"""Validate a prediction CSV before upload. Exit code 0 = valid, 1 = invalid.

Checks: exact header, column count, one row per expected id (no duplicates, none missing or extra),
all predictions finite, file size; reports position statistics and the SHA-256 for the ledger.

Examples
  python validate_submission.py preds.csv --id-range 11000 20999
  python validate_submission.py preds.csv --ids-from oos_bars.parquet --ids-column session
  python validate_submission.py preds.csv --id-range 11000 20999 --compare best.csv
  python validate_submission.py preds.csv --header id,prediction --max-bytes 2000000
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path


def read_ids(path: str, column: str) -> list[int]:
    if path.endswith(".parquet"):
        import pandas as pd
        return sorted(set(int(x) for x in pd.read_parquet(path, columns=[column])[column]))
    import csv
    with open(path, newline="") as fh:
        return sorted(set(int(row[column]) for row in csv.DictReader(fh)))


def parse(path: Path, header: list[str]):
    errors: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file starts with a UTF-8 byte-order mark")
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")
    if "\r" in text:
        errors.append("file contains carriage returns (Windows line endings)")
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n")]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return [], [], ["file is empty"]
    got = [h.strip() for h in lines[0].split(",")]
    if got != header:
        errors.append(f"line 1: header must be {','.join(header)!r}, got {lines[0]!r}")
    ids, preds = [], []
    for i, ln in enumerate(lines[1:], start=2):
        parts = ln.split(",")
        if len(parts) != len(header):
            errors.append(f"line {i}: expected {len(header)} columns, got {len(parts)}")
            continue
        try:
            ids.append(int(parts[0]))
        except ValueError:
            errors.append(f"line {i}: id {parts[0]!r} is not an integer")
            continue
        try:
            v = float(parts[1])
        except ValueError:
            errors.append(f"line {i}: prediction {parts[1]!r} is not a number")
            preds.append(float("nan"))
            continue
        if not math.isfinite(v):
            errors.append(f"line {i}: prediction {parts[1]!r} is not finite")
        preds.append(v)
    if len(ids) == 0:
        errors.append("the file has no prediction rows")
    return ids, preds, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--header", default="headline_id,pred", help="exact comma-separated header")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--id-range", nargs=2, type=int, metavar=("FIRST", "LAST"), help="expected ids FIRST..LAST inclusive")
    g.add_argument("--ids-from", help="CSV or parquet file holding the expected ids")
    ap.add_argument("--ids-column", default="session", help="id column in --ids-from (default: session)")
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--compare", help="another submission to report correlation with")
    a = ap.parse_args(argv)

    path = Path(a.file)
    if not path.exists():
        print(f"INVALID: {path} does not exist")
        return 1
    header = [h.strip() for h in a.header.split(",")]
    ids, preds, errors = parse(path, header)
    size = path.stat().st_size
    if size > a.max_bytes:
        errors.append(f"file is {size} bytes; the limit is {a.max_bytes}")
    seen, dups = set(), set()
    for i in ids:
        (dups if i in seen else seen).add(i)
    if dups:
        errors.append(f"{len(dups)} duplicated ids, e.g. {sorted(dups)[:5]}")
    if a.id_range or a.ids_from:
        expected = set(range(a.id_range[0], a.id_range[1] + 1)) if a.id_range else set(read_ids(a.ids_from, a.ids_column))
        missing, extra = expected - seen, seen - expected
        if missing:
            errors.append(f"{len(missing)} expected ids missing, e.g. {sorted(missing)[:5]}")
        if extra:
            errors.append(f"{len(extra)} unexpected ids, e.g. {sorted(extra)[:5]}")
    else:
        print("note: no --id-range/--ids-from given, so the id set was not checked")

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    finite = [p for p in preds if math.isfinite(p)]
    if finite:
        n = len(finite)
        mean = sum(finite) / n
        sd = math.sqrt(sum((p - mean) ** 2 for p in finite) / max(n - 1, 1))
        sq = sorted((p * p for p in finite), reverse=True)
        top = sum(sq[: max(1, n // 100)]) / max(sum(sq), 1e-300)
        zeros = sum(1 for p in finite if p == 0.0) / n
        print(f"rows {len(ids)} | mean {mean:.4g} | sd {sd:.4g} | max|p|/sd {max(abs(p) for p in finite) / max(sd, 1e-300):.2f} | "
              f"exact zeros {zeros:.3f} | top-1% share of sum p^2 {top:.3f}")
        if sd == 0:
            errors.append("all predictions are identical: the Sharpe score is undefined or zero")
    if a.compare and finite:
        import csv
        with open(a.compare, newline="") as fh:
            other = {int(r[header[0]]): float(r[header[1]]) for r in csv.DictReader(fh)}
        common = [(p, other[i]) for i, p in zip(ids, preds) if i in other and math.isfinite(p)]
        if len(common) > 2:
            xs, ys = zip(*common)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((x - mx) * (y - my) for x, y in common)
            vx, vy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
            print(f"correlation with {a.compare}: {cov / math.sqrt(vx * vy):.4f} over {len(common)} common ids")
    print(f"sha256 {sha}")
    if errors:
        print("INVALID:")
        for e in errors[:20]:
            print("  -", e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
