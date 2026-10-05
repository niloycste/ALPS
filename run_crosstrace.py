"""
Cross-trace generalisation study.

The headline results are measured on one Ethereum time slice. This script
re-measures them on traces the parameters were *not* chosen on:

  eth_aug14  the primary trace (parameters were selected here)
  eth_aug13  a non-overlapping Ethereum range from the previous day
  eth_aug07  a non-overlapping Ethereum range from the previous week
  bsc        BNB Smart Chain: disjoint token set and shorter block time

Two questions are answered separately.

  frozen   Every ALPS parameter is held at the value chosen on eth_w1
           (lambda_g = 0.15, alpha = 1.15). This is the honest generalisation
           test: no re-tuning of any kind.

  retuned  The lambda_g frontier is re-swept on each trace, so that a knee that
           moves is visible rather than hidden. The break-even ratio kappa* is
           re-derived per trace, since the deployment rule in the paper depends
           on it.

    python run_crosstrace.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from sim.native import NativeWorkload
from sim.chain import fully_funded_state, uniform_state
from sim.experiment import run_parallel, run_sequential
from sim.partition import (
    STRATEGIES,
    alps,
    cross_region_ratio,
    load_balance,
    resource_locality_violations,
    semantic_locality,
    sender_locality_violations,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BLOCK = 250
REGIONS = (6, 11)
SEEDS = (0, 1, 2)

# Values selected on the primary trace; held fixed for the frozen study.
FROZEN_LAMBDA_G = 0.15
FROZEN_ALPHA = 1.15
LAMBDA_GRID = (0.0, 0.05, 0.08, 0.15, 0.35, 0.60)

TRACES = {
    # Parameters were selected on eth_aug14 only; the rest are held-out.
    "eth_aug14": "data/native_all.csv",
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}
TUNED_ON = "eth_aug14"

SCHEMES = ["ADDR-MOD", "TX-KMEANS", "SENDER-KMEANS", "SENDER-KMEANS-BAL", "ALPS"]


def write(name: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def describe(tag: str, wl: NativeWorkload, txs) -> Dict[str, object]:
    """Population statistics, including the token-dominance figure that the
    subdivision mechanism depends on."""
    from collections import Counter

    d = wl.describe(txs)
    cats = Counter(t.category for t in txs)
    top = max(cats.values()) / max(len(txs), 1)
    manifest_path = TRACES[tag] + ".manifest.json"
    provenance = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        provenance = {
            "chain": m.get("chain"),
            "chain_id": m.get("chain_id"),
            "first_block": m.get("requested_first_block"),
            "last_block": m.get("requested_last_block"),
            "first_utc": m.get("first_block_utc"),
            "last_utc": m.get("last_block_utc"),
            "csv_sha256": (m.get("csv_sha256") or "")[:16],
        }
    return {"trace": tag, **provenance, **d,
            "dominant_token_share": round(top, 4)}


def measure(txs, genesis, funded, sf, regions_map, n) -> Dict[str, float]:
    v = sender_locality_violations(txs, regions_map)
    rv = resource_locality_violations(txs, regions_map)
    s = semantic_locality(txs, regions_map, n)
    lb = load_balance(regions_map, n)
    cr = cross_region_ratio(txs, regions_map)
    g = run_parallel(txs, funded, regions_map, n, 0, BLOCK, "x",
                     nonce_mode="global")
    rr = run_parallel(txs, genesis, regions_map, n, 0, BLOCK, "x",
                      nonce_mode="regional")
    return {
        "split_senders": v["split_senders"],
        "split_resources": rv["split_resources"],
        "exposed_tx_frac": round(v["exposed_tx_frac"], 4),
        "commit_global": round(g.committed / len(txs), 4),
        "rounds": g.rounds,
        "tx_per_round": round(g.committed / max(g.rounds, 1), 2),
        "deferred_checks": g.deferred_attempts,
        "overdrawn_accounts": rr.double_spend_accounts,
        "purity": round(s["purity"], 4),
        "r90": round(s["r90"], 3),
        "load": round(lb["max_over_mean"], 3),
        "cross_region": round(cr["cross_region_frac"], 4),
    }


def evaluate_trace(tag: str, path: str):
    """Evaluate one trace independently so complete traces can run in parallel."""
    desc_rows, frozen_rows, frontier_rows = [], [], []
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, genesis = wl.generate()
    sf = wl.sender_feature_matrix(txs)
    funded = fully_funded_state(txs)
    seq = run_sequential(txs, funded, 0, BLOCK)

    d = describe(tag, wl, txs)
    d["seq_commit"] = round(seq.committed / len(txs), 4)
    desc_rows.append(d)
    print(f"\n[{tag}] {d.get('chain','?')} "
          f"tx={d['transactions']} senders={d['senders']} "
          f"dominant_token={d['dominant_token_share']:.1%}", flush=True)

    for n in REGIONS:
        # --- frozen parameters, averaged over clustering seeds ---
        for name in SCHEMES:
            for seed in SEEDS:
                kw = dict(sender_features=sf, seed=seed)
                if name == "ALPS":
                    kw.update(graph_weight=FROZEN_LAMBDA_G,
                              over=FROZEN_ALPHA)
                elif name == "SENDER-KMEANS-BAL":
                    kw.update(over=FROZEN_ALPHA)
                reg = STRATEGIES[name](txs, n, **kw)
                frozen_rows.append({
                    "trace": tag, "n_regions": n, "strategy": name,
                    "seed": seed, "lambda_g": (FROZEN_LAMBDA_G
                                               if name == "ALPS" else ""),
                    **measure(txs, genesis, funded, sf, reg, n),
                })
        print(f"[{tag}] n={n} frozen done", flush=True)

        # --- retuned frontier, so a moved knee is visible ---
        for lg in LAMBDA_GRID:
            for seed in SEEDS:
                reg = alps(txs, n, sender_features=sf, seed=seed,
                           graph_weight=lg, over=FROZEN_ALPHA)
                frontier_rows.append({
                    "trace": tag, "n_regions": n, "lambda_g": lg,
                    "seed": seed,
                    **measure(txs, genesis, funded, sf, reg, n),
                })
        print(f"[{tag}] n={n} frontier done", flush=True)

    return desc_rows, frozen_rows, frontier_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="number of traces to evaluate concurrently (default: 1)",
    )
    args = parser.parse_args()

    available = {k: v for k, v in TRACES.items() if os.path.exists(v)}
    missing = sorted(set(TRACES) - set(available))
    if missing:
        print(f"  (missing, skipping: {', '.join(missing)})")

    desc_rows, frozen_rows, frontier_rows = [], [], []
    workers = max(1, min(args.workers, len(available)))
    if workers == 1:
        batches = [evaluate_trace(tag, path) for tag, path in available.items()]
    else:
        batches = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(evaluate_trace, tag, path): tag
                for tag, path in available.items()
            }
            for future in as_completed(pending):
                tag = pending[future]
                batches.append(future.result())
                print(f"[{tag}] trace complete", flush=True)

    for desc, frozen, frontier in batches:
        desc_rows.extend(desc)
        frozen_rows.extend(frozen)
        frontier_rows.extend(frontier)

    trace_order = {tag: i for i, tag in enumerate(TRACES)}
    strategy_order = {name: i for i, name in enumerate(SCHEMES)}
    desc_rows.sort(key=lambda r: trace_order[r["trace"]])
    frozen_rows.sort(key=lambda r: (
        trace_order[r["trace"]], r["n_regions"],
        strategy_order[r["strategy"]], r["seed"],
    ))
    frontier_rows.sort(key=lambda r: (
        trace_order[r["trace"]], r["n_regions"], r["lambda_g"], r["seed"],
    ))

    write("crosstrace_describe.csv", desc_rows)
    write("crosstrace_frozen.csv", frozen_rows)
    write("crosstrace_frontier.csv", frontier_rows)
    print("\ndone")


if __name__ == "__main__":
    main()
