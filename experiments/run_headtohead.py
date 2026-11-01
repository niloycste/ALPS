"""
Head-to-head against a reimplemented TxAllo allocation objective.

Runs ADDR-MOD, TXALLO, SENDER-KMEANS-BAL and ALPS on every direct-call trace
at every region count, with all ALPS weights frozen at their D1 values.

    python -m experiments.run_headtohead
"""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.chain import fully_funded_state
from sim.experiment import run_parallel
from sim.partition import (
    STRATEGIES,
    cross_region_ratio,
    load_balance,
    resource_locality_violations,
    semantic_locality,
    sender_locality_violations,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
BLOCK = 250
REGIONS = (2, 3, 4, 6, 8, 11)
SEEDS = (0, 1, 2)
DEFAULT_SCHEMES = [
    "ADDR-MOD", "METIS-SENDER", "TXALLO", "SENDER-KMEANS-BAL", "ALPS",
]
SCHEMES = [name for name in os.environ.get(
    "ALPS_HEADTOHEAD_METHODS", ",".join(DEFAULT_SCHEMES)).split(",") if name]
TXALLO_ETA = 2.0
TXALLO_MAX_ITER = 8
TRACES = {
    "eth_aug14": "data/native_all.csv",
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}


def run_trace(tag: str, path: str) -> list:
    rows = []
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, genesis = wl.generate()
    sf = wl.sender_feature_matrix(txs)
    funded = fully_funded_state(txs)
    print(f"\n[{tag}] {len(txs)} transactions", flush=True)

    for n in REGIONS:
        for name in SCHEMES:
            t0 = time.perf_counter()
            for seed in SEEDS:
                extra = ({"eta": TXALLO_ETA,
                          "max_iter": TXALLO_MAX_ITER}
                         if name == "TXALLO" else {})
                allocation_start = time.perf_counter()
                reg = STRATEGIES[name](
                    txs, n, sender_features=sf, seed=seed, **extra)
                allocation_seconds = time.perf_counter() - allocation_start
                v = sender_locality_violations(txs, reg)
                rv = resource_locality_violations(txs, reg)
                s = semantic_locality(txs, reg, n)
                lb = load_balance(reg, n)
                cr = cross_region_ratio(txs, reg)
                g = run_parallel(txs, funded, reg, n, 0, BLOCK, name,
                                 nonce_mode="global")
                rows.append({
                    "trace": tag, "n_regions": n, "strategy": name,
                    "seed": seed,
                    "split_senders": v["split_senders"],
                    "split_resources": rv["split_resources"],
                    "commit_global": round(g.committed / len(txs), 4),
                    "rounds": g.rounds,
                    "tx_per_round": round(g.committed / max(g.rounds, 1), 2),
                    "purity": round(s["purity"], 4),
                    "r90": round(s["r90"], 3),
                    "load": round(lb["max_over_mean"], 3),
                    "cross_region": round(cr["cross_region_frac"], 4),
                    "interaction_coverage": round(
                        cr["interaction_coverage"], 4),
                    "allocation_seconds": round(allocation_seconds, 6),
                    "txallo_eta": TXALLO_ETA if name == "TXALLO" else "",
                    "txallo_max_iter": (TXALLO_MAX_ITER
                                        if name == "TXALLO" else ""),
                })
            print(f"   [{tag}] n={n:<3} {name:<18} "
                  f"{time.perf_counter()-t0:5.0f}s", flush=True)
    return rows


def main() -> None:
    rows = []
    jobs = [(tag, path) for tag, path in TRACES.items()
            if os.path.exists(path)]
    missing = [path for path in TRACES.values() if not os.path.exists(path)]
    if missing:
        print(f"  (missing: {', '.join(missing)})")

    workers = max(1, int(os.environ.get("ALPS_HEADTOHEAD_WORKERS", "4")))
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futures = {ex.submit(run_trace, tag, path): tag for tag, path in jobs}
        for future in as_completed(futures):
            rows.extend(future.result())

    trace_order = {tag: i for i, tag in enumerate(TRACES)}
    scheme_order = {name: i for i, name in enumerate(SCHEMES)}
    rows.sort(key=lambda r: (trace_order[r["trace"]], r["n_regions"],
                             scheme_order[r["strategy"]], r["seed"]))

    os.makedirs(RESULTS, exist_ok=True)
    output_name = os.environ.get("ALPS_HEADTOHEAD_OUTPUT", "headtohead.csv")
    p = os.path.join(RESULTS, output_name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {output_name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
