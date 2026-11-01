"""
Merge fetched block ranges into one dataset.

Both collectors are supported. Log traces are de-duplicated on
``(block, log_index)`` and direct-call traces on ``(block, tx_index)``.

    python merge_parts.py data/part_*.csv data/eth_transfers.csv -o data/eth_all.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="data/eth_all.csv")
    a = ap.parse_args()

    paths = []
    for pat in a.inputs:
        paths.extend(sorted(glob.glob(pat)))

    if not paths:
        raise SystemExit("no input files matched")

    with open(paths[0], newline="") as f:
        first = csv.DictReader(f)
        fields = first.fieldnames or []
    if "tx_index" in fields:
        pos = "tx_index"
        expected = ["block", "tx_index", "token", "symbol", "from",
                    "nonce", "to", "value"]
    elif "log_index" in fields:
        pos = "log_index"
        expected = ["block", "log_index", "token", "symbol", "from",
                    "to", "value"]
    else:
        raise SystemExit("inputs must contain tx_index or log_index")

    seen = {}
    per_file = []
    for p in paths:
        if not os.path.exists(p):
            continue
        n = 0
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise SystemExit(f"schema mismatch: {p}")
            for r in reader:
                key = (int(r["block"]), int(r[pos]))
                if key not in seen:
                    seen[key] = r
                    n += 1
        per_file.append((p, n))

    rows = [seen[k] for k in sorted(seen)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=expected)
        w.writeheader()
        w.writerows(rows)

    for p, n in per_file:
        print(f"  {os.path.basename(p):<24} +{n} new")

    blocks = {int(r["block"]) for r in rows}
    accts = {r["from"] for r in rows} | {r["to"] for r in rows}
    toks = {r["symbol"] for r in rows}
    print(f"\nwrote {a.out}")
    print(f"  {len(rows)} transfers, {len(blocks)} distinct blocks "
          f"({min(blocks)}..{max(blocks)})")
    print(f"  {len(accts)} accounts, {len(toks)} tokens")


if __name__ == "__main__":
    main()
