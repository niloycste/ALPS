"""
Definitive run over three traces, each used for what it can support.

  log trace     every ERC-20 Transfer event. ``from`` is the account whose
                balance is debited, which is exactly the entity sender locality
                concerns, so this trace is authoritative for violation counts,
                retrieval and load. It cannot support the nonce experiment,
                because a token holder has no nonce -- the signer does.

  native trace  direct transfer() calls only, where the signer *is* the token
                holder and the real Ethereum nonce is available. Authoritative
                for anything involving nonce enforcement.

  validation    a fixed, non-overlapping prior-day direct-call trace. It uses
                the primary parameters unchanged and tests cross-day validity
                at six and eleven regions.

Writes the complete ``results/final_*.csv`` evidence bundle, including the
ALPS component ablation and graph-weight/deployment-cost analysis.
"""

from __future__ import annotations

import csv
import os

from sim.ethereum import EthereumWorkload
from sim.native import NativeWorkload
from sim.online import run_temporal_allocator
from sim.realsweep import RealSweep
from sim.partition import (
    STRATEGIES,
    alps,
    cross_region_ratio,
    load_balance,
    semantic_locality,
    sender_locality_violations,
    resource_locality_violations,
)
from sim.experiment import run_parallel, run_sequential
from sim.chain import fully_funded_state

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
BLOCK = 250
SCHEMES = [
    "ADDR-MOD", "TX-KMEANS", "SENDER-KMEANS",
    "SENDER-KMEANS-BAL", "ALPS",
]

NATIVE = "data/native_all.csv"      # primary trace: real nonces
VALIDATION = "data/native_day2.csv" # disjoint prior-day trace: real nonces
LOG = "data/eth_all.csv"            # secondary trace: all Transfer events


def write(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def sweep(wl, tag, regions=(2, 3, 4, 6, 8, 11)):
    txs, genesis = wl.generate()
    sf = wl.sender_feature_matrix(txs)
    desc = wl.describe(txs)
    print(f"\n[{tag}] " + ", ".join(f"{k}={v}" for k, v in desc.items()))

    funded = fully_funded_state(txs)
    seq = run_sequential(txs, funded, 0, BLOCK)
    print(f"  sequential commit: {seq.committed}/{len(txs)} = "
          f"{seq.committed/len(txs):.4f}")

    rows = []
    for n in regions:
        for name in SCHEMES:
            reg = STRATEGIES[name](txs, n, sender_features=sf, seed=0)
            v = sender_locality_violations(txs, reg)
            rv = resource_locality_violations(txs, reg)
            s = semantic_locality(txs, reg, n)
            lb = load_balance(reg, n)
            cr = cross_region_ratio(txs, reg)
            g = run_parallel(txs, funded, reg, n, 0, BLOCK, name,
                             nonce_mode="global")
            rr = run_parallel(txs, genesis, reg, n, 0, BLOCK, name,
                              nonce_mode="regional")
            rows.append({
                "trace": tag, "strategy": name, "n_regions": n,
                "split_senders": v["split_senders"],
                "exposed_tx_frac": round(v["exposed_tx_frac"], 4),
                "split_resources": rv["split_resources"],
                "resource_exposed_tx_frac": round(
                    rv["resource_exposed_tx_frac"], 4),
                "commit_global": round(g.committed / len(txs), 4),
                "global_rounds": g.rounds,
                "tx_per_round": round(g.committed / max(g.rounds, 1), 2),
                "deferred_attempts": g.deferred_attempts,
                "pending": g.pending,
                "commit_regional": round(rr.committed / len(txs), 4),
                "seq_commit": round(seq.committed / len(txs), 4),
                "conflicting_accounts": rr.conflicting_accounts,
                "overdrawn_accounts": rr.double_spend_accounts,
                "phantom_value": round(rr.phantom_value, 2),
                "purity": round(s["purity"], 4),
                "r90": round(s["r90"], 3),
                "load": round(lb["max_over_mean"], 3),
                "cross_region": round(cr["cross_region_frac"], 4),
                "interaction_coverage": round(cr["interaction_coverage"], 4),
            })
    return rows, desc


def headroom(wl_cls, path, tag, balances, n=6):
    rows = []
    for bal in balances:
        wl = wl_cls(path, initial_balance=bal)
        txs, genesis = wl.generate()
        sf = wl.sender_feature_matrix(txs)
        seq = run_sequential(txs, genesis, 0, BLOCK)
        supply = genesis.total_supply()
        for name in SCHEMES:
            reg = STRATEGIES[name](txs, n, sender_features=sf, seed=0)
            r = run_parallel(txs, genesis, reg, n, 0, BLOCK, name,
                             nonce_mode="regional")
            rows.append({
                "trace": tag, "strategy": name, "initial_balance": bal,
                "seq_commit": round(seq.committed / len(txs), 4),
                "overdrawn_accounts": r.double_spend_accounts,
                "phantom_value": round(r.phantom_value, 2),
                "phantom_pct_supply": round(100 * r.phantom_value / supply, 4),
            })
    return rows


# --------------------------------------------------------------------------
# Conflict-rate sweep: stratify measured account populations (Fig. 2)
# --------------------------------------------------------------------------

def conflict_sweep(path, n=6, senders=500, budget=2000, draws=5,
                   mixes=(0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)):
    rs = RealSweep(path, initial_balance=1000.0)
    print(f"\n[sweep] {rs.population()}")
    rows = []
    for m in mixes:
        for draw in range(draws):
            txs, genesis, info = rs.build(m, senders, seed=100 + draw,
                                          max_tx=budget)
            n_users = max(max(t.sender, t.receiver) for t in txs) + 1
            sf = rs.sender_feature_matrix(txs, n_users)
            funded = fully_funded_state(txs)
            for name in SCHEMES:
                reg = STRATEGIES[name](txs, n, sender_features=sf, seed=0)
                v = sender_locality_violations(txs, reg)
                rv = resource_locality_violations(txs, reg)
                for mode in ("global", "regional"):
                    state = funded if mode == "global" else genesis
                    r = run_parallel(txs, state, reg, n, 0, BLOCK, name,
                                     nonce_mode=mode)
                    rows.append({
                        "strategy": name, "nonce_mode": mode, "draw": draw,
                        "requested_multi_frac": m,
                        "multi_frac": info["realised_multi_frac"],
                        "exposed_tx_frac": round(v["exposed_tx_frac"], 4),
                        "resource_exposed_tx_frac": round(
                            rv["resource_exposed_tx_frac"], 4),
                        "split_senders": v["split_senders"],
                        "split_resources": rv["split_resources"],
                        "commit_rate": round(r.committed / len(txs), 4),
                        "rounds": r.rounds,
                        "tx_per_round": round(
                            r.committed / max(r.rounds, 1), 2),
                        "deferred_attempts": r.deferred_attempts,
                        "pending": r.pending,
                        "conflicting_accounts": r.conflicting_accounts,
                        "overdrawn_accounts": r.double_spend_accounts,
                    })
    return rows


# --------------------------------------------------------------------------
# Temporal allocation: assign from history, score on the future (Fig. 5)
# --------------------------------------------------------------------------

def temporal(path, regions=(6,), fracs=(0.5, 0.6, 0.7, 0.8),
             seeds=(0, 1, 2),
             modes=("HASH", "SENDER-KMEANS", "SENDER-KMEANS-BAL", "ALPS")):
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = wl.generate_temporal(frac=min(fracs))
    print(f"\n[temporal] {len(txs)} transactions")
    rows = []
    for frac in fracs:
        for n in regions:
            for mode in modes:
                for seed in seeds:
                    r = run_temporal_allocator(
                        txs, n, mode, frac=frac, seed=seed)
                    rows.append(r)
    return rows


def temporal_sensitivity(path, n=6, fracs=(0.5, 0.6, 0.7, 0.8),
                         seeds=(0, 1, 2),
                         graph_weights=(0.08, 0.15, 0.35)):
    """Check that the ALPS graph-weight choice survives temporal separation."""
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = wl.generate_temporal(frac=min(fracs))
    print(f"\n[temporal-sensitivity] {len(txs)} transactions")
    rows = []
    for weight in graph_weights:
        for frac in fracs:
            for seed in seeds:
                row = run_temporal_allocator(
                    txs, n, "ALPS", frac=frac, seed=seed,
                    graph_weight=weight)
                row["variant"] = f"GW-{weight:g}"
                rows.append(row)
    return rows


def temporal_horizon_sensitivity(path, n=6, frac=0.7,
                                 multipliers=(0.5, 1.0, 1.5, 2.0),
                                 seeds=(0, 1, 2)):
    """Vary the declared placement horizon around the observed suffix size."""
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = wl.generate_temporal(frac=frac)
    suffix_size = len(txs) - int(len(txs) * frac)
    rows = []
    for multiplier in multipliers:
        horizon = max(1, round(suffix_size * multiplier))
        for seed in seeds:
            row = run_temporal_allocator(
                txs, n, "ALPS", frac=frac, seed=seed,
                horizon_size=horizon)
            row["horizon_multiplier"] = multiplier
            row["horizon_size"] = horizon
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# ALPS mechanism audit: exact-objective ablations and deployment threshold
# --------------------------------------------------------------------------

def alps_analysis(path, n=6, seeds=(0, 1, 2),
                  graph_weights=(0.0, 0.05, 0.08, 0.15, 0.35, 0.60)):
    """Isolate ALPS components and derive its break-even deployment cost.

    For a deployment cost ``D_k = R90 + k * cross_region_fraction``, the
    reported threshold is the smallest ``k`` at which the candidate is cheaper
    than SENDER-KMEANS-BAL.  It exposes rather than hides the fact that the
    preferred allocator depends on query and cross-region settlement costs.
    """
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = wl.generate()
    sf = wl.sender_feature_matrix(txs)
    print(f"\n[alps-analysis] {len(txs)} transactions, n={n}")

    configs = [
        (f"GW-{weight:g}", "frontier", 1.0, weight, 0.20, 1.15)
        for weight in graph_weights
    ] + [
        # One-factor ablations retain the deployed graph weight (0.15).
        ("NO-SEMANTIC", "ablation", 0.0, 0.15, 0.20, 1.15),
        ("NO-CAPACITY", "ablation", 1.0, 0.15, 0.20, 1e9),
    ]

    rows = []
    for seed in seeds:
        base = STRATEGIES["SENDER-KMEANS-BAL"](
            txs, n, sender_features=sf, seed=seed)
        base_sem = semantic_locality(txs, base, n)
        base_cr = cross_region_ratio(txs, base)
        base_cross = base_cr["cross_region_frac"]

        for variant, analysis, sw, gw, lw, over in configs:
            regions, diagnostics = alps(
                txs, n, sender_features=sf, seed=seed, over=over,
                semantic_weight=sw, graph_weight=gw, load_weight=lw,
                return_diagnostics=True)
            sem = semantic_locality(txs, regions, n)
            balance = load_balance(regions, n)
            cr = cross_region_ratio(txs, regions)
            cross = cr["cross_region_frac"]
            violations = sender_locality_violations(txs, regions)

            cross_gain = base_cross - cross
            retrieval_penalty = sem["r90"] - base_sem["r90"]
            if cross_gain <= 0:
                threshold = ""
            else:
                threshold = max(0.0, retrieval_penalty / cross_gain)

            comp = diagnostics["components"]
            rows.append({
                "analysis": analysis,
                "variant": variant,
                "seed": seed,
                "n_regions": n,
                "semantic_weight": sw,
                "graph_weight": gw,
                "load_weight": lw,
                "capacity_multiplier": "none" if over > 1e6 else over,
                "r90": round(sem["r90"], 4),
                "purity": round(sem["purity"], 4),
                "load": round(balance["max_over_mean"], 4),
                "cross_region": round(cross, 5),
                "interaction_coverage": round(cr["interaction_coverage"], 5),
                "split_senders": violations["split_senders"],
                "objective": round(comp["total"], 8),
                "semantic_component": round(comp["semantic"], 8),
                "graph_component": round(comp["graph"], 8),
                "load_component": round(comp["load"], 8),
                "accepted_moves": diagnostics["accepted_moves"],
                "capacity_relaxed": diagnostics["capacity_relaxed"],
                "balanced_r90": round(base_sem["r90"], 4),
                "balanced_cross_region": round(base_cross, 5),
                "balanced_interaction_coverage": round(
                    base_cr["interaction_coverage"], 5),
                "break_even_kappa": (round(threshold, 4)
                                      if threshold != "" else ""),
            })
    return rows


# --------------------------------------------------------------------------
# Wall-clock throughput with genuine proof-of-work (optional; needs an idle
# machine, since competing load distorts the parallel speedup being measured)
# --------------------------------------------------------------------------

def throughput(path, regions=(2, 3, 4, 6, 8, 11), reps=6, difficulty=5):
    from concurrent.futures import ProcessPoolExecutor

    # Use direct calls: only they expose the outer signer and genuine account
    # nonce order. Transfer logs cannot support an Ethereum nonce benchmark.
    wl = NativeWorkload(path, initial_balance=1000.0)
    txs, _ = wl.generate()
    genesis = fully_funded_state(txs)
    sf = wl.sender_feature_matrix(txs)
    print(f"\n[throughput] difficulty={difficulty}, {reps} reps -- slow")

    rows = []
    with ProcessPoolExecutor(max_workers=max(regions)) as ex:
        for rep in range(reps):
            s = run_sequential(txs, genesis, difficulty, BLOCK)
            rows.append({"strategy": "SEQUENTIAL", "n_regions": 1, "rep": rep,
                         **{k: v for k, v in s.row().items() if k != "label"}})
            print(f"   rep{rep} SEQUENTIAL {s.tps:8.1f} TPS", flush=True)
            for n in regions:
                for name in SCHEMES:
                    reg = STRATEGIES[name](txs, n, sender_features=sf, seed=0)
                    r = run_parallel(txs, genesis, reg, n, difficulty, BLOCK,
                                     name, nonce_mode="global", executor=ex)
                    rows.append({"strategy": name, "n_regions": n, "rep": rep,
                                 **{k: v for k, v in r.row().items()
                                    if k != "label"}})
                print(f"   rep{rep} n={n} done", flush=True)
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--throughput-only", action="store_true",
                    help="just the proof-of-work benchmark (run on an idle machine)")
    ap.add_argument("--analysis-only", action="store_true",
                    help="just the ALPS component/frontier analysis")
    ap.add_argument("--validation-only", action="store_true",
                    help="just the frozen-parameter prior-day validation sweep")
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()

    if a.throughput_only:
        write("throughput.csv", throughput(NATIVE, reps=a.reps))
        raise SystemExit(0)
    if a.analysis_only:
        write("final_alps_analysis.csv", alps_analysis(NATIVE))
        raise SystemExit(0)
    if a.validation_only:
        val_rows, val_desc = sweep(
            NativeWorkload(VALIDATION), "native-day2", regions=(6, 11))
        write("final_validation.csv", val_rows)
        write("final_validation_describe.csv", [
            {"trace": "native-day2", **val_desc},
        ])
        raise SystemExit(0)

    print("=== definitive run (all results the paper reports)")

    log_rows, log_desc = sweep(EthereumWorkload(LOG), "log")
    nat_rows, nat_desc = sweep(NativeWorkload(NATIVE), "native")
    val_rows, val_desc = sweep(
        NativeWorkload(VALIDATION), "native-day2", regions=(6, 11))
    write("final_log.csv", log_rows)
    write("final_native.csv", nat_rows)
    write("final_validation.csv", val_rows)
    write("final_describe.csv", [
        {"trace": "log", **log_desc},
        {"trace": "native", **nat_desc},
        {"trace": "native-day2", **val_desc},
    ])

    bals = [1000.0, 600.0, 400.0, 250.0, 150.0, 100.0, 60.0]
    write("final_headroom.csv",
          headroom(EthereumWorkload, LOG, "log", bals)
          + headroom(NativeWorkload, NATIVE, "native", bals))

    write("final_sweep.csv", conflict_sweep(NATIVE))
    write("final_online.csv", temporal(NATIVE))
    write("final_temporal_sensitivity.csv", temporal_sensitivity(NATIVE))
    write("final_alps_analysis.csv", alps_analysis(NATIVE))
    print("\ndone -- run python -m experiments.make_figures next")
