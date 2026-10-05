"""
Run the complete experiment suite on every trace.

run_final.py applies the full battery to the primary trace and a reduced
battery elsewhere. This driver applies *all* experiments to *all* direct-call
traces, so each dataset receives the same treatment the primary one did:

  static      region sweep over n in {2,3,4,6,8,11}, all five schemes
  headroom    balance-only stress across seven starting balances at n=6
  population  measured-population stratification sweep (RealSweep)
  temporal    chronological prefix-to-suffix allocation
  tempsens    graph-weight sensitivity under temporal separation
  analysis    ALPS lambda_g frontier and component ablations

Dataset spans are evaluated in parallel processes; disjoint held-out spans are
not treated as statistically independent replications.

    python run_all_traces.py --workers 4
    python run_all_traces.py --only static,headroom
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sim.native import NativeWorkload
from sim.ethereum import EthereumWorkload

import run_final as R

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Direct-call traces carry real signer nonces and support every experiment.
DIRECT = {
    "eth_aug14": "data/native_all.csv",   # primary; parameters chosen here
    "eth_aug13": "data/native_day2.csv",
    "eth_aug07": "data/eth_week2.csv",
    "bsc": "data/bsc_trace.csv",
}
# The log trace cannot identify the outer signer, so it is limited to the
# experiments that do not depend on nonce ownership.
LOG = {"eth_logs": "data/eth_all.csv"}

BALANCES = [1000.0, 600.0, 400.0, 250.0, 150.0, 100.0, 60.0]
STAGES = ("static", "headroom", "population", "temporal", "tempsens", "analysis")
LOG_STAGES = ("static", "headroom")


def _tag_rows(rows, tag):
    out = []
    for r in rows:
        d = {"trace": tag}
        d.update({k: v for k, v in r.items() if k != "trace"})
        out.append(d)
    return out


def run_one(tag: str, path: str, is_log: bool, stages) -> dict:
    """Every experiment for a single trace. Returns {stage: rows}."""
    t0 = time.perf_counter()
    cls = EthereumWorkload if is_log else NativeWorkload
    allowed = LOG_STAGES if is_log else stages
    out: dict = {}

    def stage(name, fn):
        if name not in stages or name not in allowed:
            return
        s = time.perf_counter()
        try:
            out[name] = _tag_rows(fn(), tag)
            print(f"  [{tag}] {name:<11} {len(out[name]):>5} rows "
                  f"({time.perf_counter()-s:.0f}s)", flush=True)
        except Exception:
            print(f"  [{tag}] {name:<11} FAILED\n{traceback.format_exc()}",
                  flush=True)

    def _static():
        rows, desc = R.sweep(cls(path), tag)
        out["describe"] = [{"trace": tag, **desc}]
        return rows

    stage("static", _static)
    stage("headroom", lambda: R.headroom(cls, path, tag, BALANCES))
    stage("population", lambda: R.conflict_sweep(path))
    stage("temporal", lambda: R.temporal(path))
    stage("tempsens", lambda: R.temporal_sensitivity(path))
    stage("analysis", lambda: R.alps_analysis(path))

    print(f"  [{tag}] complete in {time.perf_counter()-t0:.0f}s", flush=True)
    return out


def write(name: str, rows: list) -> None:
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


def _read(name: str) -> list:
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _view(source: str, trace: str, label: str,
          region_filter=None) -> list:
    rows = []
    for row in _read(source):
        if row.get("trace") != trace:
            continue
        if region_filter is not None and row.get("n_regions") not in region_filter:
            continue
        rows.append({**row, "trace": label})
    return rows


def sync_final_views() -> None:
    """Refresh the smaller ``final_*`` compatibility views from all-results.

    The manuscript and figures use ``all_*``. Keeping these historical filenames
    synchronized prevents readers with older notebooks from seeing stale metric
    definitions after a cross-trace rerun.
    """
    views = {
        "final_native.csv": _view(
            "all_static.csv", "eth_aug14", "native"),
        "final_validation.csv": _view(
            "all_static.csv", "eth_aug13", "native-day2", {"6", "11"}),
        "final_log.csv": _view(
            "all_static.csv", "eth_logs", "log"),
        "final_online.csv": _view(
            "all_temporal.csv", "eth_aug14", "native"),
        "final_temporal_sensitivity.csv": _view(
            "all_tempsens.csv", "eth_aug14", "native"),
        "final_alps_analysis.csv": _view(
            "all_analysis.csv", "eth_aug14", "native"),
    }
    for name, rows in views.items():
        if rows:
            write(name, rows)

    desc = []
    for source, label in (("eth_logs", "log"), ("eth_aug14", "native"),
                          ("eth_aug13", "native-day2")):
        desc.extend(_view("all_describe.csv", source, label))
    if desc:
        write("final_describe.csv", desc)
        write("final_validation_describe.csv",
              [r for r in desc if r["trace"] == "native-day2"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sync-views-only", action="store_true",
                    help="refresh final_* compatibility CSVs from all_* files")
    ap.add_argument("--only", default=",".join(STAGES),
                    help="comma-separated subset of: " + ",".join(STAGES))
    ap.add_argument("--traces",
                    help="comma-separated trace IDs to run (default: all)")
    ap.add_argument("--merge-existing", action="store_true",
                    help="retain rows for traces not selected in this rerun")
    a = ap.parse_args()
    if a.sync_views_only:
        sync_final_views()
        return
    stages = tuple(s.strip() for s in a.only.split(",") if s.strip())

    selected = ({item.strip() for item in a.traces.split(",") if item.strip()}
                if a.traces else None)
    known = set(DIRECT) | set(LOG)
    if selected is not None and not selected <= known:
        ap.error(f"unknown traces: {sorted(selected - known)}")
    jobs = [(t, p, False) for t, p in DIRECT.items()
            if os.path.exists(p) and (selected is None or t in selected)]
    jobs += [(t, p, True) for t, p in LOG.items()
             if os.path.exists(p) and (selected is None or t in selected)]
    missing = [p for _, p in {**DIRECT, **LOG}.items() if not os.path.exists(p)]
    if missing:
        print(f"  (missing: {', '.join(missing)})")

    print(f"=== all experiments x {len(jobs)} traces, stages: {', '.join(stages)}")
    t0 = time.perf_counter()
    collected: dict = {}

    if a.workers <= 1:
        results = [(t, run_one(t, p, l, stages)) for t, p, l in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
            fut = {ex.submit(run_one, t, p, l, stages): t for t, p, l in jobs}
            for f in as_completed(fut):
                results.append((fut[f], f.result()))

    for _, out in results:
        for stage_name, rows in out.items():
            collected.setdefault(stage_name, []).extend(rows)

    order = {t: i for i, t in enumerate(list(DIRECT) + list(LOG))}
    for stage_name, rows in collected.items():
        if a.merge_existing:
            rerun_traces = {tag for tag, _, _ in jobs}
            rows = [row for row in _read(f"all_{stage_name}.csv")
                    if row.get("trace") not in rerun_traces] + rows
        rows.sort(key=lambda r: order.get(r.get("trace"), 99))
        write(f"all_{stage_name}.csv", rows)

    sync_final_views()

    print(f"\ndone in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
