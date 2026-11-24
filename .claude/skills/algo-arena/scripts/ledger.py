#!/usr/bin/env python3
"""Submission ledger keyed by file hash, so every leaderboard score is tied to the exact file that earned it.

  python ledger.py add FILE --composition "0.7 latent_v2 + 0.15 public_a + 0.15 public_b" --cv 3.61 --expected 3.70 --gate "valid; leak 0.0; cv reproduced"
  python ledger.py uploaded FILE                 # mark as uploaded now
  python ledger.py score FILE 3.699              # record the leaderboard score against FILE's hash
  python ledger.py best [--copy-to submissions/CURRENT_BEST.csv]
  python ledger.py show

The ledger lives at $LEDGER or submissions/LEDGER.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import shutil
import sys
from pathlib import Path

FIELDS = ["sha256", "file", "created_utc", "composition", "cv_sharpe", "expected_lb", "gate", "uploaded_utc", "lb_score", "notes"]


def path() -> Path:
    return Path(os.environ.get("LEDGER", "submissions/LEDGER.csv"))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def sha(f: str) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def load() -> list[dict]:
    p = path()
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def save(rows: list[dict]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def find(rows, f):
    h = sha(f)
    for r in rows:
        if r["sha256"] == h:
            return r
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("file"); a.add_argument("--composition", default=""); a.add_argument("--cv", default="")
    a.add_argument("--expected", default=""); a.add_argument("--gate", default=""); a.add_argument("--notes", default="")
    u = sub.add_parser("uploaded"); u.add_argument("file")
    s = sub.add_parser("score"); s.add_argument("file"); s.add_argument("lb", type=float)
    b = sub.add_parser("best"); b.add_argument("--copy-to")
    sub.add_parser("show")
    args = ap.parse_args(argv)
    rows = load()

    if args.cmd == "add":
        r = find(rows, args.file)
        if r:
            print(f"already in ledger as {r['file']} (same bytes); not added again")
            return 0
        rows.append(dict(sha256=sha(args.file), file=args.file, created_utc=now(), composition=args.composition, cv_sharpe=args.cv,
                         expected_lb=args.expected, gate=args.gate, uploaded_utc="", lb_score="", notes=args.notes))
        save(rows); print(f"added {args.file} ({rows[-1]['sha256'][:12]})")
    elif args.cmd in ("uploaded", "score"):
        r = find(rows, args.file)
        if r is None:
            print(f"{args.file} is not in the ledger (by hash). Add it first."); return 1
        if args.cmd == "uploaded":
            r["uploaded_utc"] = now()
        else:
            r["lb_score"] = f"{args.lb:.4f}"
            if not r["uploaded_utc"]:
                r["uploaded_utc"] = now()
        save(rows); print(f"{r['file']}: uploaded {r['uploaded_utc']} lb {r['lb_score'] or '-'}")
    elif args.cmd == "best":
        scored = [r for r in rows if r["lb_score"]]
        if not scored:
            print("no scored uploads yet"); return 1
        r = max(scored, key=lambda x: float(x["lb_score"]))
        print(f"best: {r['file']} lb {r['lb_score']} ({r['sha256'][:12]}) — {r['composition']}")
        if args.copy_to:
            if sha(r["file"]) != r["sha256"]:
                print(f"WARNING: {r['file']} has changed on disk since it was scored; not copied"); return 1
            shutil.copyfile(r["file"], args.copy_to); print(f"copied to {args.copy_to}")
    else:
        w = csv.writer(sys.stdout)
        w.writerow(["file", "sha", "cv", "expected", "lb", "uploaded", "gate", "composition"])
        for r in rows:
            w.writerow([r["file"], r["sha256"][:10], r["cv_sharpe"], r["expected_lb"], r["lb_score"], r["uploaded_utc"], r["gate"], r["composition"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
