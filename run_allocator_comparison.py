"""Fair allocator-only runtime comparison on the four direct-call traces.

CSV loading is excluded.  Algorithm-required preprocessing is included in
``total_seconds``: sender-profile construction for S-KM-BAL and ALPS, and graph
construction inside METIS-SENDER and TXALLO.  Every measurement runs in a
fresh process, sequentially, to avoid CPU contention and stale peak-memory
counters.

    python -B run_allocator_comparison.py
"""

from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.partition import STRATEGIES, cross_region_ratio, load_balance, semantic_locality

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N_REGIONS = 6
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "ADDR-MOD", "TX-KMEANS", "SENDER-KMEANS-BAL",
    "METIS-SENDER", "TXALLO", "ALPS",
)
PROFILE_METHODS = {"SENDER-KMEANS-BAL", "ALPS"}
TRACES = {
    "eth_aug14": "data/native_all.csv",
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}


def peak_working_set_mib() -> float:
    if os.name != "nt":
        return float("nan")
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_current = ctypes.windll.kernel32.GetCurrentProcess
    get_current.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    if not get_memory(get_current(), ctypes.byref(counters), counters.cb):
        return float("nan")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def processor_name() -> str:
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def run_case(args: tuple[str, str, str, int]) -> dict:
    tag, path, method, seed = args
    workload = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = workload.generate()
    sender_features = None
    profile_seconds = 0.0
    if method in PROFILE_METHODS:
        start = time.perf_counter()
        sender_features = workload.sender_feature_matrix(txs)
        profile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    regions = STRATEGIES[method](
        txs, N_REGIONS, sender_features=sender_features, seed=seed)
    allocation_seconds = time.perf_counter() - start
    semantic = semantic_locality(txs, regions, N_REGIONS)
    cut = cross_region_ratio(txs, regions)
    load = load_balance(regions, N_REGIONS)
    return {
        "trace": tag,
        "transactions": len(txs),
        "active_senders": len({tx.sender for tx in txs}),
        "n_regions": N_REGIONS,
        "method": method,
        "seed": seed,
        "profile_seconds": round(profile_seconds, 6),
        "allocation_seconds": round(allocation_seconds, 6),
        "total_seconds": round(profile_seconds + allocation_seconds, 6),
        "peak_working_set_mib": round(peak_working_set_mib(), 3),
        "r90": round(semantic["r90"], 6),
        "cross_region": round(cut["cross_region_frac"], 6),
        "load": round(load["max_over_mean"], 6),
        "cpu": processor_name(),
        "python": sys.version.split()[0],
    }


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for tag in TRACES:
        for method in METHODS:
            group = [r for r in rows
                     if r["trace"] == tag and r["method"] == method]
            output.append({
                "trace": tag,
                "transactions": group[0]["transactions"],
                "n_regions": N_REGIONS,
                "method": method,
                "runs": len(group),
                "median_total_seconds": round(statistics.median(
                    r["total_seconds"] for r in group), 6),
                "std_total_seconds": round(statistics.stdev(
                    r["total_seconds"] for r in group), 6),
                "median_peak_working_set_mib": round(statistics.median(
                    r["peak_working_set_mib"] for r in group), 3),
                "mean_r90": round(statistics.mean(r["r90"] for r in group), 6),
                "mean_cross_region": round(statistics.mean(
                    r["cross_region"] for r in group), 6),
                "max_load": round(max(r["load"] for r in group), 6),
                "cpu": group[0]["cpu"],
                "python": group[0]["python"],
            })
    return output


def main() -> None:
    paths = {tag: os.path.join(HERE, rel) for tag, rel in TRACES.items()}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing traces: {missing}")
    jobs = [(tag, path, method, seed)
            for tag, path in paths.items()
            for method in METHODS for seed in SEEDS]
    rows: list[dict] = []
    os.makedirs(RESULTS, exist_ok=True)
    partial = os.path.join(RESULTS, "allocator_comparison.partial.csv")
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        for i, row in enumerate(executor.map(run_case, jobs), 1):
            rows.append(row)
            write_csv(partial, rows)
            print(
                f"{i:>3}/{len(jobs)} {row['trace']:<10} "
                f"{row['method']:<20} seed={row['seed']} "
                f"{row['total_seconds']:.3f}s",
                flush=True,
            )

    trace_order = {tag: i for i, tag in enumerate(TRACES)}
    method_order = {method: i for i, method in enumerate(METHODS)}
    rows.sort(key=lambda r: (
        trace_order[r["trace"]], method_order[r["method"]], r["seed"]))
    write_csv(os.path.join(RESULTS, "allocator_comparison.csv"), rows)
    write_csv(os.path.join(RESULTS, "allocator_comparison_summary.csv"),
              summarize(rows))
    if os.path.exists(partial):
        os.remove(partial)
    print(f"wrote allocator_comparison.csv ({len(rows)} rows)")
    print("wrote allocator_comparison_summary.csv (24 rows)")


if __name__ == "__main__":
    main()
