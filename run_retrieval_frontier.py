"""Frozen historical coverage plans: achieved suffix recall versus scan work.

Allocations are retrospective, as in run_retrieval_benchmark.py. No suffix
counts select a ranking, coverage target, or prefix length. No latency is timed.
Run from paper/; ALPS_BENCH_METHODS and ALPS_FRONTIER_OUTPUT allow split runs.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from run_retrieval_benchmark import (
    TRACES, METHODS, SEEDS, N_REGIONS, HISTORY_FRACTION,
    NativeWorkload, STRATEGIES, counts_for, ranked_plan,
)

TARGETS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def main():
    rows, populations = [], []
    for trace, path in TRACES.items():
        workload = NativeWorkload(path)
        txs, _ = workload.generate()
        cut = int(len(txs) * HISTORY_FRACTION)
        history, future = txs[:cut], txs[cut:]
        hcats, fcats = {t.category for t in history}, {t.category for t in future}
        common = sorted(hcats & fcats)
        populations.append(dict(
            trace=trace, total_categories=workload.n_categories,
            history_categories=len(hcats), suffix_categories=len(fcats),
            evaluated_categories=len(common),
            suffix_categories_without_history=len(fcats - hcats),
            history_categories_without_suffix=len(hcats - fcats),
            suffix_rows_without_history=sum(t.category not in hcats for t in future),
        ))
        features = workload.sender_feature_matrix(txs)
        for method in METHODS:
            for seed in SEEDS:
                extra = {"eta": 2.0, "max_iter": 8} if method == "TXALLO" else {}
                regions = STRATEGIES[method](txs, N_REGIONS,
                    sender_features=features, seed=seed, **extra)
                hc = counts_for(history, regions[:cut], workload.n_categories)
                fc = counts_for(future, regions[cut:], workload.n_categories)
                loads = np.bincount(regions[cut:], minlength=N_REGIONS)
                for target in TARGETS:
                    plans = ranked_plan(hc, target)
                    for category in common:
                        chosen = plans[category]
                        total = int(fc[category].sum())
                        returned = int(fc[category, chosen].sum())
                        rows.append(dict(trace=trace, method=method, seed=seed,
                            n_regions=N_REGIONS, history_fraction=HISTORY_FRACTION,
                            allocation_scope="full_trace_retrospective",
                            target_coverage=target, category=category,
                            contacted_regions=len(chosen),
                            rows_scanned=int(loads[chosen].sum()),
                            suffix_rows=len(future), result_rows=returned,
                            result_recall=returned / total))
                print(f"{trace} {method} seed={seed}", flush=True)
    output = Path("results") / os.environ.get("ALPS_FRONTIER_OUTPUT", "retrieval_frontier.csv")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    population_output = output.with_name(output.stem + "_population.csv")
    with population_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(populations[0]))
        writer.writeheader()
        writer.writerows(populations)
    print(f"wrote {output}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
