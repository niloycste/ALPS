"""One-factor sensitivity audit for the reimplemented TxAllo allocator.

The main comparison freezes ``eta=2`` and eight coordinate sweeps.  This
driver varies eta with eight sweeps and varies the sweep cap with eta fixed at
two.  It evaluates the paper's six-region headline on every direct-call trace
and three seeds.  Only allocation metrics are measured; this is not a TxAllo
protocol or throughput benchmark.

    python -m experiments.run_txallo_sensitivity
"""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Avoid NetworkX/scikit-learn's deprecated WMIC probe on current Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.partition import (
    cross_region_ratio,
    load_balance,
    resource_locality_violations,
    semantic_locality,
    sender_locality_violations,
)
from sim.txallo import txallo

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
HEADTOHEAD = os.path.join(RESULTS, "headtohead.csv")
REGIONS = (6,)
SEEDS = (0, 1, 2)
CONFIGS = (
    (1.25, 8), (1.5, 8), (2.0, 8), (3.0, 8), (4.0, 8),
    (2.0, 4), (2.0, 16), (2.0, 32),
)
TRACES = {
    "eth_aug14": "data/native_all.csv",
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}


def run_case(args: tuple[str, str, int, float, int, int]) -> dict:
    tag, path, n_regions, eta, max_iter, seed = args
    workload = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = workload.generate()
    start = time.perf_counter()
    regions = txallo(
        txs, n_regions, seed=seed, eta=eta, max_iter=max_iter,
    )
    elapsed = time.perf_counter() - start
    sender = sender_locality_violations(txs, regions)
    resource = resource_locality_violations(txs, regions)
    semantic = semantic_locality(txs, regions, n_regions)
    load = load_balance(regions, n_regions)
    cut = cross_region_ratio(txs, regions)
    return {
        "trace": tag,
        "n_regions": n_regions,
        "seed": seed,
        "eta": eta,
        "max_iter": max_iter,
        "split_senders": sender["split_senders"],
        "split_resources": resource["split_resources"],
        "r90": round(semantic["r90"], 6),
        "purity": round(semantic["purity"], 6),
        "load": round(load["max_over_mean"], 6),
        "cross_region": round(cut["cross_region_frac"], 6),
        "interaction_coverage": round(cut["interaction_coverage"], 6),
        "allocation_seconds": round(elapsed, 6),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict]) -> list[dict]:
    with open(HEADTOHEAD, newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    alps = {
        (row["trace"], int(row["n_regions"]), int(row["seed"])): row
        for row in baseline_rows if row["strategy"] == "ALPS"
    }

    summary: list[dict] = []
    for eta, max_iter in CONFIGS:
        group = [row for row in rows
                 if row["eta"] == eta and row["max_iter"] == max_iter]
        lower_cut = lower_r90 = cap_violations = 0
        for row in group:
            ref = alps[(row["trace"], row["n_regions"], row["seed"])]
            lower_cut += row["cross_region"] < float(ref["cross_region"])
            lower_r90 += float(ref["r90"]) < row["r90"]
            cap_violations += row["load"] > 1.15 + 1e-9
        summary.append({
            "eta": eta,
            "max_iter": max_iter,
            "settings": len(group),
            "txallo_lower_cut": lower_cut,
            "alps_lower_r90": lower_r90,
            "txallo_load_over_1_15": cap_violations,
            "txallo_mean_cut": round(
                sum(row["cross_region"] for row in group) / len(group), 6),
            "txallo_mean_r90": round(
                sum(row["r90"] for row in group) / len(group), 6),
            "txallo_max_load": round(max(row["load"] for row in group), 6),
            "mean_allocation_seconds": round(
                sum(row["allocation_seconds"] for row in group) / len(group),
                6,
            ),
        })
    return summary


def main() -> None:
    if not os.path.exists(HEADTOHEAD):
        raise FileNotFoundError(
            "headtohead.csv is required for the frozen ALPS comparison")
    paths = {tag: os.path.join(HERE, path) for tag, path in TRACES.items()}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing traces: {missing}")

    rows: list[dict] = []
    jobs = [
        (tag, path, n_regions, eta, max_iter, seed)
        for tag, path in paths.items()
        for n_regions in REGIONS
        for eta, max_iter in CONFIGS
        for seed in SEEDS
    ]
    partial = os.path.join(RESULTS, "txallo_sensitivity.partial.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_case, job): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                write_csv(partial, rows)
                print(f"completed {len(rows)}/{len(jobs)}", flush=True)

    trace_order = {tag: i for i, tag in enumerate(TRACES)}
    config_order = {config: i for i, config in enumerate(CONFIGS)}
    rows.sort(key=lambda row: (
        trace_order[row["trace"]], row["n_regions"],
        config_order[(row["eta"], row["max_iter"])], row["seed"],
    ))
    write_csv(os.path.join(RESULTS, "txallo_sensitivity.csv"), rows)
    summary = make_summary(rows)
    write_csv(os.path.join(RESULTS, "txallo_sensitivity_summary.csv"), summary)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"wrote txallo_sensitivity.csv ({len(rows)} rows)")
    print(f"wrote txallo_sensitivity_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
