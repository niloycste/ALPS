"""Five-seed sensitivity of ALPS to the hard-cap multiplier alpha.

Runs the six-region headline on every direct-call trace.  Raw outputs retain
each initialization; the summary reports means, sample standard deviations,
and 95% t-interval half-widths over allocator initializations.

    python -B run_alpha_sensitivity.py
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.partition import alps, cross_region_ratio, load_balance, semantic_locality

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N_REGIONS = 6
ALPHAS = (1.05, 1.10, 1.15, 1.25, 1.50)
SEEDS = (0, 1, 2, 3, 4)
TRACES = {
    "eth_aug14": "data/native_all.csv",
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(args: tuple[str, str, float, int]) -> dict:
    tag, path, alpha, seed = args
    workload = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = workload.generate()
    start = time.perf_counter()
    sender_features = workload.sender_feature_matrix(txs)
    regions, diagnostics = alps(
        txs, N_REGIONS, sender_features=sender_features, seed=seed,
        over=alpha, return_diagnostics=True,
    )
    elapsed = time.perf_counter() - start
    semantic = semantic_locality(txs, regions, N_REGIONS)
    cut = cross_region_ratio(txs, regions)
    load = load_balance(regions, N_REGIONS)
    return {
        "trace": tag,
        "n_regions": N_REGIONS,
        "alpha": alpha,
        "seed": seed,
        "r90": round(semantic["r90"], 6),
        "cross_region": round(cut["cross_region_frac"], 6),
        "interaction_coverage": round(cut["interaction_coverage"], 6),
        "load": round(load["max_over_mean"], 6),
        "target_capacity": round(diagnostics["target_capacity"], 6),
        "effective_capacity": round(diagnostics["capacity"], 6),
        "capacity_relaxed": diagnostics["capacity_relaxed"],
        "allocation_seconds": round(elapsed, 6),
    }


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for tag in TRACES:
        for alpha in ALPHAS:
            group = [r for r in rows
                     if r["trace"] == tag and r["alpha"] == alpha]
            row = {
                "trace": tag,
                "n_regions": N_REGIONS,
                "alpha": alpha,
                "seeds": len(group),
                "capacity_relaxed_runs": sum(
                    bool(r["capacity_relaxed"]) for r in group),
            }
            critical = T_975[len(group)]
            for metric in ("r90", "cross_region", "load",
                           "allocation_seconds"):
                values = [float(r[metric]) for r in group]
                mean = statistics.mean(values)
                std = statistics.stdev(values)
                row[f"{metric}_mean"] = round(mean, 6)
                row[f"{metric}_std"] = round(std, 6)
                row[f"{metric}_ci95_halfwidth"] = round(
                    critical * std / math.sqrt(len(values)), 6)
            output.append(row)
    return output


def main() -> None:
    paths = {tag: os.path.join(HERE, rel) for tag, rel in TRACES.items()}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing traces: {missing}")

    jobs = [(tag, path, alpha, seed)
            for tag, path in paths.items()
            for alpha in ALPHAS for seed in SEEDS]
    rows: list[dict] = []
    os.makedirs(RESULTS, exist_ok=True)
    partial = os.path.join(RESULTS, "alpha_sensitivity.partial.csv")
    with ProcessPoolExecutor(max_workers=min(4, len(paths))) as executor:
        futures = [executor.submit(run_case, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0:
                write_csv(partial, rows)
                print(f"completed {len(rows)}/{len(jobs)}", flush=True)

    trace_order = {tag: i for i, tag in enumerate(TRACES)}
    rows.sort(key=lambda r: (trace_order[r["trace"]], r["alpha"], r["seed"]))
    write_csv(os.path.join(RESULTS, "alpha_sensitivity.csv"), rows)
    summary = summarize(rows)
    write_csv(os.path.join(RESULTS, "alpha_sensitivity_summary.csv"), summary)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"wrote alpha_sensitivity.csv ({len(rows)} rows)")
    print(f"wrote alpha_sensitivity_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
