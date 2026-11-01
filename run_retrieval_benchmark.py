"""Benchmark oracle-90%, history-ranked 90%, and complete token retrieval."""

from __future__ import annotations

import csv
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np

from sim.native import NativeWorkload
from sim.partition import STRATEGIES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRACES = {"eth_aug14": "data/native_all.csv", "eth_aug13": "data/native_day2.csv",
          "eth_aug07": "data/eth_week2.csv", "bsc": "data/bsc_trace.csv"}
METHODS = tuple(os.environ.get(
    "ALPS_BENCH_METHODS",
    "SENDER-KMEANS-BAL,METIS-SENDER,TXALLO,ALPS").split(","))
SEEDS = (0, 1, 2)
N_REGIONS, HISTORY_FRACTION, REPETITIONS = 6, 0.7, 250


def stores_for(txs, regions):
    stores = [[] for _ in range(N_REGIONS)]
    for tx, region in zip(txs, regions):
        stores[int(region)].append(tx.category)
    return [np.asarray(values, dtype=np.int16) for values in stores]


def counts_for(txs, regions, n_categories):
    counts = np.zeros((n_categories, N_REGIONS), dtype=np.int64)
    for tx, region in zip(txs, regions):
        counts[tx.category, int(region)] += 1
    return counts


def ranked_plan(training_counts, coverage):
    plans = {}
    for category, counts in enumerate(training_counts):
        total = int(counts.sum())
        if not total:
            continue
        order = np.argsort(-counts, kind="stable")
        contacted = (N_REGIONS if coverage >= 1.0 else
                     int(np.searchsorted(np.cumsum(counts[order]),
                                         coverage * total) + 1))
        plans[category] = order[:contacted]
    return plans


def benchmark_mode(mode, training_txs, training_regions, query_txs,
                   query_regions, n_categories, coverage):
    training_counts = counts_for(training_txs, training_regions, n_categories)
    query_counts = counts_for(query_txs, query_regions, n_categories)
    stores = stores_for(query_txs, query_regions)
    plans = ranked_plan(training_counts, coverage)
    samples = {category: [] for category in plans
               if query_counts[category].sum() > 0}
    for category in samples:
        chosen = plans[category]
        sum(np.count_nonzero(stores[r] == category) for r in chosen)
    for _ in range(REPETITIONS):
        for category in samples:
            chosen = plans[category]
            start = time.perf_counter_ns()
            matches = sum(np.count_nonzero(stores[r] == category) for r in chosen)
            samples[category].append((time.perf_counter_ns() - start) / 1000.0)
    rows = []
    for category, timings in samples.items():
        chosen = plans[category]
        future_total = int(query_counts[category].sum())
        returned = int(query_counts[category, chosen].sum())
        rows.append({
            "query_mode": mode, "category": category,
            "contacted_regions": len(chosen),
            "rows_scanned": sum(stores[r].size for r in chosen),
            "result_rows": returned, "result_recall": returned / future_total,
            "median_latency_us": round(float(np.median(timings)), 4),
            "latency_iqr_us": round(float(np.percentile(timings, 75)
                                           - np.percentile(timings, 25)), 4),
        })
    return rows


def run_trace(tag, path):
    workload = NativeWorkload(path)
    txs, _ = workload.generate()
    features = workload.sender_feature_matrix(txs)
    split_at = int(len(txs) * HISTORY_FRACTION)
    history, future = txs[:split_at], txs[split_at:]
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            extra = {"eta": 2.0, "max_iter": 8} if method == "TXALLO" else {}
            regions = STRATEGIES[method](txs, N_REGIONS,
                                         sender_features=features,
                                         seed=seed, **extra)
            cases = (("oracle90", txs, regions, txs, regions, 0.9),
                     ("history90", history, regions[:split_at], future,
                      regions[split_at:], 0.9),
                     ("complete", txs, regions, txs, regions, 1.0))
            for mode, train_txs, train_regions, query_txs, query_regions, cov in cases:
                for row in benchmark_mode(mode, train_txs, train_regions,
                                          query_txs, query_regions,
                                          workload.n_categories, cov):
                    rows.append({
                        "trace": tag, "method": method, "seed": seed,
                        "n_regions": N_REGIONS,
                        "history_fraction": (HISTORY_FRACTION
                                             if mode == "history90" else ""),
                        "target_coverage": cov, "repetitions": REPETITIONS,
                        **row,
                    })
    return rows


def main():
    rows = []
    jobs = [(tag, path) for tag, path in TRACES.items() if os.path.exists(path)]
    for tag, path in jobs:
        result = run_trace(tag, path)
        rows.extend(result)
        print(f"[{tag}] {len(result)} measurements", flush=True)
    rows.sort(key=lambda row: (row["trace"], row["method"], row["seed"],
                               row["query_mode"], row["category"]))
    os.makedirs(RESULTS, exist_ok=True)
    output_name = os.environ.get("ALPS_BENCH_OUTPUT",
                                 "retrieval_benchmark.csv")
    path = os.path.join(RESULTS, output_name)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
