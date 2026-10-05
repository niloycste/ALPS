"""Measure ALPS allocation cost on real prefixes of the D1 trace.

Each sample runs in a fresh process.  CSV loading is excluded; sender-profile
construction and ALPS allocation are timed separately.  Peak working set is
the process-wide Windows peak and therefore includes the loaded trace and
Python runtime, making it a conservative deployment figure.

    python -B run_allocator_scaling.py
"""

from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Avoid scikit-learn's deprecated WMIC probe on current Windows releases.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.partition import alps

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "data", "native_all.csv")
RESULTS = os.path.join(HERE, "results")
SIZES = (5_000, 10_000, 20_000, 34_326)
SEEDS = (0, 1, 2)
N_REGIONS = 6


def processor_name() -> str:
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or "unknown"


def peak_working_set_mib() -> float:
    if os.name != "nt":
        return float("nan")
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
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
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    handle = get_current()
    ok = get_memory(
        handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return float("nan")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def count_active_edges(txs) -> int:
    active = {tx.sender for tx in txs}
    edges = set()
    for tx in txs:
        if tx.receiver in active and tx.receiver != tx.sender:
            edges.add(tuple(sorted((tx.sender, tx.receiver))))
    return len(edges)


def run_sample(args: tuple[int, int]) -> dict:
    size, seed = args
    workload = NativeWorkload(TRACE, initial_balance=1000.0)
    all_txs, _ = workload.generate()
    txs = all_txs[:size]

    start = time.perf_counter()
    sender_features = workload.sender_feature_matrix(txs)
    profile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    _, diagnostics = alps(
        txs, N_REGIONS, sender_features=sender_features, seed=seed,
        return_diagnostics=True,
    )
    allocation_seconds = time.perf_counter() - start
    return {
        "transactions": len(txs),
        "active_senders": len({tx.sender for tx in txs}),
        "active_edges": count_active_edges(txs),
        "n_regions": N_REGIONS,
        "seed": seed,
        "profile_seconds": round(profile_seconds, 6),
        "allocation_seconds": round(allocation_seconds, 6),
        "total_seconds": round(profile_seconds + allocation_seconds, 6),
        "peak_working_set_mib": round(peak_working_set_mib(), 3),
        "accepted_moves": diagnostics["accepted_moves"],
        "load": round(diagnostics["max_over_mean"], 6),
        "capacity_relaxed": diagnostics["capacity_relaxed"],
        "cpu": processor_name(),
        "python": sys.version.split()[0],
    }


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not os.path.exists(TRACE):
        raise FileNotFoundError(TRACE)
    tasks = [(size, seed) for size in SIZES for seed in SEEDS]
    rows: list[dict] = []
    # One worker avoids CPU contention; one task per child resets peak RSS.
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        for row in executor.map(run_sample, tasks):
            rows.append(row)
            print(
                f"N={row['transactions']:<6} seed={row['seed']} "
                f"total={row['total_seconds']:.2f}s "
                f"peak={row['peak_working_set_mib']:.1f} MiB",
                flush=True,
            )

    rows.sort(key=lambda row: (row["transactions"], row["seed"]))
    os.makedirs(RESULTS, exist_ok=True)
    write_csv(os.path.join(RESULTS, "allocator_scaling.csv"), rows)

    summary = []
    for size in SIZES:
        group = [row for row in rows if row["transactions"] == size]
        summary.append({
            "transactions": size,
            "active_senders": group[0]["active_senders"],
            "active_edges": group[0]["active_edges"],
            "median_profile_seconds": round(statistics.median(
                row["profile_seconds"] for row in group), 6),
            "median_allocation_seconds": round(statistics.median(
                row["allocation_seconds"] for row in group), 6),
            "median_total_seconds": round(statistics.median(
                row["total_seconds"] for row in group), 6),
            "max_peak_working_set_mib": round(max(
                row["peak_working_set_mib"] for row in group), 3),
            "max_load": round(max(row["load"] for row in group), 6),
            "capacity_relaxed_runs": sum(
                bool(row["capacity_relaxed"]) for row in group),
            "seeds": len(group),
            "cpu": group[0]["cpu"],
            "python": group[0]["python"],
        })
    write_csv(os.path.join(RESULTS, "allocator_scaling_summary.csv"), summary)
    print(f"wrote allocator_scaling.csv ({len(rows)} rows)")
    print(f"wrote allocator_scaling_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
