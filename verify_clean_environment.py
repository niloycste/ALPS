"""Run the manuscript artifact end to end and compare scientific outputs.

Run this from an extracted anonymous artifact in a fresh environment containing
requirements-metis.txt plus pypdf. Timing columns are excluded because they are
environment dependent. The script writes CLEAN_REPRODUCTION.json.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def run(python, *args, extra_env=None):
    env = dict(ENV, **(extra_env or {}))
    completed = subprocess.run([python, "-B", *args], cwd=ROOT, env=env,
                               text=True, capture_output=True, timeout=3600)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    print(completed.stdout, end="")


def same_csv(actual, expected, keys, columns):
    a = pd.read_csv(actual).set_index(keys)[columns].sort_index()
    e = pd.read_csv(expected).set_index(keys)[columns].sort_index()
    pd.testing.assert_frame_equal(a, e, check_exact=True)


def package_versions(python):
    code = """import importlib.metadata as m, json
names = ['numpy','pandas','scikit-learn','scipy','matplotlib','networkx','pymetis','pypdf']
out = {}
for name in names:
    try:
        out[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
print(json.dumps(out))
"""
    return json.loads(subprocess.check_output([python, "-c", code], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metis-python", required=True,
                        help="Python from a second fresh environment containing PyMetis")
    args = parser.parse_args()
    checks = []
    run(sys.executable, "verify_semantic_scale.py")
    checks.append("Equation (1) semantic scale: all initial sender/prototype pairs, computed once")
    run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    checks.append("15 unit tests")

    expected_static = pd.read_csv(RESULTS / "all_static.csv")
    expected_horizon = pd.read_csv(RESULTS / "all_horizon.csv")
    expected_head = pd.read_csv(RESULTS / "headtohead.csv")
    expected_benchmark = pd.read_csv(RESULTS / "retrieval_benchmark.csv")
    expected_frontier = pd.read_csv(RESULTS / "retrieval_frontier.csv")

    run(sys.executable, "run_all_traces.py", "--only", "static,horizon",
        "--workers", "4")
    pd.testing.assert_frame_equal(pd.read_csv(RESULTS / "all_static.csv"),
                                  expected_static, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(RESULTS / "all_horizon.csv"),
                                  expected_horizon, check_exact=True)
    checks.append("Tables II-III: all 150 static rows and 48 horizon rows")

    stable_head = [c for c in expected_head.columns if c != "allocation_seconds"]
    head_parts = ((sys.executable, "core",
                   "ADDR-MOD,TXALLO,SENDER-KMEANS-BAL,ALPS"),
                  (args.metis_python, "metis", "METIS-SENDER"))
    for python, tag, methods in head_parts:
        output_name = f"clean_headtohead_{tag}.csv"
        run(python, "run_headtohead.py", extra_env={
            "ALPS_HEADTOHEAD_WORKERS": "2",
            "ALPS_HEADTOHEAD_METHODS": methods,
            "ALPS_HEADTOHEAD_OUTPUT": output_name})
        actual_head = pd.read_csv(RESULTS / output_name)
        expected_part = expected_head[
            expected_head.strategy.isin(methods.split(","))].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_head[stable_head],
                                      expected_part[stable_head], check_exact=True)
    checks.append("Table IV: all 360 head-to-head configurations (timings excluded)")

    bench_keys = ["trace", "method", "seed", "query_mode", "category"]
    bench_fields = ["n_regions", "history_fraction", "target_coverage",
                    "repetitions", "contacted_regions", "rows_scanned",
                    "result_rows", "result_recall"]
    frontier_keys = ["trace", "method", "seed", "target_coverage", "category"]
    frontier_fields = ["n_regions", "history_fraction", "allocation_scope",
                       "contacted_regions", "rows_scanned", "suffix_rows",
                       "result_rows", "result_recall"]
    parts = ((sys.executable, "core", "SENDER-KMEANS-BAL,TXALLO,ALPS"),
             (args.metis_python, "metis", "METIS-SENDER"))
    for python, tag, methods in parts:
        bench_name = f"clean_retrieval_{tag}.csv"
        frontier_name = f"clean_frontier_{tag}.csv"
        run(python, "run_retrieval_benchmark.py", extra_env={
            "ALPS_BENCH_METHODS": methods, "ALPS_BENCH_OUTPUT": bench_name})
        run(python, "run_retrieval_frontier.py", extra_env={
            "ALPS_BENCH_METHODS": methods, "ALPS_FRONTIER_OUTPUT": frontier_name})
        actual_bench = RESULTS / bench_name
        expected_bench = RESULTS / f"clean_retrieval_{tag}_expected.csv"
        actual_frontier = RESULTS / frontier_name
        expected_frontier_part = RESULTS / f"clean_frontier_{tag}_expected.csv"
        method_names = methods.split(",")
        expected_benchmark[expected_benchmark.method.isin(method_names)].to_csv(
            expected_bench, index=False)
        expected_frontier[expected_frontier.method.isin(method_names)].to_csv(
            expected_frontier_part, index=False)
        same_csv(actual_bench, expected_bench, bench_keys, bench_fields)
        same_csv(actual_frontier, expected_frontier_part,
                 frontier_keys, frontier_fields)
    expected_benchmark.to_csv(RESULTS / "retrieval_benchmark.csv", index=False)
    expected_frontier.to_csv(RESULTS / "retrieval_frontier.csv", index=False)
    run(sys.executable, "plot_retrieval_frontier.py")
    run(sys.executable, "make_figures.py")
    for name in ("fig_sweep.pdf", "fig_locality.pdf", "fig_retrieval_frontier.pdf"):
        assert (ROOT / "ALPS_conference_submission" / "figures" / name).is_file()
    checks.append("Table V and experimental Figures 2-4: all 1,140 benchmark and 2,604 frontier rows; plots regenerated")

    run(sys.executable, "run_prospective_retrieval.py", "--methods",
        "SENDER-KMEANS-BAL,TXALLO,ALPS", "--repetitions", "1",
        "--output", "clean_full_core")
    run(args.metis_python, "run_prospective_retrieval.py", "--methods", "METIS-SENDER",
        "--repetitions", "1", "--output", "clean_full_metis")
    query_keys = ["trace", "method", "seed", "target_coverage", "category"]
    query_fields = ["cold_token", "contacted_regions", "rows_scanned",
                    "result_rows", "total_matching_rows", "result_recall"]
    allocation_keys = ["trace", "method", "seed"]
    allocation_fields = ["future_calls", "cold_senders", "cold_sender_tx_fraction",
                         "interaction_coverage", "active_cut", "max_mean_load",
                         "admission_fallbacks", "split_senders"]
    for prefix, methods in (("clean_full_core", ["SENDER-KMEANS-BAL", "TXALLO+CAUSAL", "ALPS"]),
                            ("clean_full_metis", ["METIS-SENDER+CAUSAL"])):
        official_query = pd.read_csv(RESULTS / "prospective_retrieval.csv")
        official_query = official_query[official_query.method.isin(methods)]
        official_alloc = pd.read_csv(RESULTS / "prospective_retrieval_allocation.csv")
        official_alloc = official_alloc[official_alloc.method.isin(methods)]
        official_query.to_csv(RESULTS / (prefix + "_expected.csv"), index=False)
        official_alloc.to_csv(RESULTS / (prefix + "_expected_allocation.csv"), index=False)
        same_csv(RESULTS / f"{prefix}.csv", RESULTS / f"{prefix}_expected.csv",
                 query_keys, query_fields)
        same_csv(RESULTS / f"{prefix}_allocation.csv",
                 RESULTS / f"{prefix}_expected_allocation.csv",
                 allocation_keys, allocation_fields)
    checks.append("all 48 prospective allocations and 2,688 query rows")

    for weight, tag in ((.15, "0p15"), (.6, "0p6"), (2.4, "2p4")):
        prefix = "clean_graph_" + tag
        run(sys.executable, "run_prospective_retrieval.py", "--methods", "ALPS",
            "--graph-weight", str(weight), "--repetitions", "1", "--output", prefix)
        same_csv(RESULTS / f"{prefix}.csv", RESULTS / f"prospective_graph_{tag}.csv",
                 query_keys, query_fields)
        same_csv(RESULTS / f"{prefix}_allocation.csv",
                 RESULTS / f"prospective_graph_{tag}_allocation.csv",
                 allocation_keys, allocation_fields)
    checks.append("all 36 graph-weight allocations and 2,016 query rows")

    run(sys.executable, "audit_receipts.py", "--offline", "--traces", "bsc", "--limit", "100",
        "--bsc-rpc", "https://bsc-dataseed.bnbchain.org", "--output", "clean_receipts")
    receipt = json.loads((RESULTS / "clean_receipts_summary.json").read_text())
    assert receipt["sample_fully_resolved"]
    assert receipt["traces"][0]["classifications"] == {
        "success_matching_transfer": 100}
    checks.append("100 cached BSC receipts")

    report = {
        "status": "passed",
        "date": "2026-10-17",
        "core_environment": {"python": sys.version.split()[0],
                             "packages": package_versions(sys.executable)},
        "metis_environment": {"python": subprocess.check_output(
            [args.metis_python, "--version"], text=True).strip().removeprefix("Python "),
                              "packages": package_versions(args.metis_python)},
        "checks": checks,
        "scientific_fields": "exact match",
        "timing_comparison": "excluded",
        "scope": "fresh core venv plus fresh Conda METIS environment; extracted anonymous artifact; local independent rerun, not third-party audit",
    }
    (ROOT / "CLEAN_REPRODUCTION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
