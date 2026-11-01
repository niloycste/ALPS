"""Verify an extracted anonymous archive and reproduce one configuration per method.

This is an isolated local check, not third-party replication or a full sweep.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--metis-python", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH="")
    checks = []
    with tempfile.TemporaryDirectory(prefix="alps-artifact-") as temp:
        dest = Path(temp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert (dest / name).resolve().is_relative_to(dest.resolve())
            z.extractall(dest)
        root = dest / "ALPS"
        inventory = json.loads((root / "SHA256SUMS.json").read_text())
        for name, digest in inventory.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        checks.append(dict(check="archive_hashes", files=len(inventory), passed=True))

        def run(python, arguments):
            result = subprocess.run([python, "-B", *arguments], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(result.stdout + result.stderr, flush=True)
            checks.append(dict(command=arguments, passed=True))

        run(sys.executable, ["-m", "unittest", "discover", "-s", "tests", "-v"])
        run(sys.executable, ["-m", "tools.verification.verify_semantic_scale"])
        original = pd.read_csv(root / "results/prospective_retrieval_headline.csv")
        run(sys.executable, ["-m", "experiments.summarize_prospective", "--parts", "prospective_core,prospective_metis"])
        pd.testing.assert_frame_equal(original, pd.read_csv(root / "results/prospective_retrieval_headline.csv"))
        expected_equal = pd.read_csv(root / "results/prospective_equal_recall_summary.csv")
        run(sys.executable, ["-m", "experiments.analyze_prospective_equal_recall"])
        pd.testing.assert_frame_equal(
            expected_equal,
            pd.read_csv(root / "results/prospective_equal_recall_summary.csv"),
            check_exact=True)
        run(sys.executable, ["-m", "experiments.summarize_graph_sensitivity"])
        run(sys.executable, ["-m", "tools.audit_receipts", "--offline", "--traces", "bsc", "--limit", "100",
                             "--bsc-rpc", "https://bsc-dataseed.bnbchain.org", "--output", "isolated_receipts"])
        receipt = json.loads((root / "results/isolated_receipts_summary.json").read_text())
        assert receipt["sample_fully_resolved"]
        assert receipt["traces"][0]["classifications"] == {"success_matching_transfer": 100}
        for python, methods, output in ((sys.executable, "ALPS,SENDER-KMEANS-BAL,TXALLO", "isolated_core"),
                                        (args.metis_python, "METIS-SENDER", "isolated_metis")):
            run(python, ["-m", "experiments.run_prospective_retrieval", "--methods", methods, "--traces", "eth_aug14",
                         "--seeds", "0", "--repetitions", "1", "--output", output])
            fresh = pd.read_csv(root / "results" / (output + ".csv"))
            baseline = pd.read_csv(root / "results/prospective_retrieval.csv")
            baseline = baseline[(baseline.trace == "eth_aug14") & (baseline.seed == 0)
                                & baseline.method.isin(fresh.method.unique())]
            keys = ["method", "target_coverage", "category"]
            fields = ["result_recall", "rows_scanned", "contacted_regions"]
            pd.testing.assert_frame_equal(fresh.set_index(keys)[fields].sort_index(),
                                          baseline.set_index(keys)[fields].sort_index())
        checks.append(dict(check="four_real_allocators_D1_seed0_scan_metrics_reproduced", passed=True))
    report = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), checks=checks,
                  scope="isolated local extraction; existing Python environments; no external reviewer, full sweep, or latency replication")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
