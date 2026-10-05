"""
Experiment harness.

Two nonce regimes are evaluated, because they are the only two ways to run a
partition that violates sender locality:

  global   -- one monotonic nonce per account, as in the base paper.  A region
              can only admit a transaction whose nonce matches the account's
              current value.  A future-nonce transaction is deferred and retried
              rather than incorrectly counted as permanently rejected.

  regional -- a balance-only stress mode representing independent region-local
              sequencing.  Throughput is recovered, but two regions now
              validate debits of the same account against the same stale
              snapshot, which admits double spends.

The two modes answer different questions: global mode measures the scheduling
cost of preserving account order; regional mode isolates balance-safety risk.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .chain import (
    Block,
    RegionDelta,
    State,
    Transaction,
    merge_deltas,
    mine,
)


# --------------------------------------------------------------------------
# Worker (module level so it can be pickled by ProcessPoolExecutor)
# --------------------------------------------------------------------------

def _mine_worker(args):
    index, region, prev_hash, txs, difficulty = args
    b = Block(index=index, region=region, prev_hash=prev_hash, transactions=list(txs))
    b = mine(b, difficulty)
    return b.region, b.block_hash, b.hash_attempts


@dataclass
class Result:
    label: str
    n_regions: int
    wall_time: float = 0.0
    committed: int = 0
    rejected: int = 0
    blocks: int = 0
    hash_attempts: int = 0
    double_spend_accounts: int = 0
    phantom_value: float = 0.0
    conflicting_accounts: int = 0
    rounds: int = 0
    deferred_attempts: int = 0
    pending: int = 0
    state: State | None = None
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def tps(self) -> float:
        return self.committed / self.wall_time if self.wall_time > 0 else 0.0

    def row(self) -> Dict[str, float]:
        d = {
            "label": self.label,
            "n_regions": self.n_regions,
            "wall_time": round(self.wall_time, 4),
            "tps": round(self.tps, 3),
            "committed": self.committed,
            "rejected": self.rejected,
            "blocks": self.blocks,
            "rounds": self.rounds,
            "deferred_attempts": self.deferred_attempts,
            "pending": self.pending,
            "hash_attempts": self.hash_attempts,
            "double_spend_accounts": self.double_spend_accounts,
            "phantom_value": round(self.phantom_value, 3),
            "conflicting_accounts": self.conflicting_accounts,
        }
        d.update(self.extra)
        return d


# --------------------------------------------------------------------------
# Region execution (in-process; cheap relative to PoW)
# --------------------------------------------------------------------------

def _execute(
    txs: Sequence[Transaction],
    snapshot: State,
    region: int,
    nonce_mode: str,
    commit_limit: int | None = None,
) -> RegionDelta:
    local_bal = dict(snapshot.balances)
    local_nonce = dict(snapshot.nonces)
    delta = RegionDelta(region=region)

    for i, tx in enumerate(txs):
        if commit_limit is not None and len(delta.committed) >= commit_limit:
            delta.deferred.extend(txs[i:])
            break
        expected = local_nonce.get(tx.sender, 0)
        if nonce_mode == "global" and tx.nonce > expected:
            delta.deferred.append(tx)
            delta.deferred_checks += 1
            continue

        ok = local_bal.get(tx.debit_key, 0.0) >= tx.amount
        if nonce_mode == "global":
            ok = ok and expected == tx.nonce
        if ok:
            local_bal[tx.debit_key] -= tx.amount
            local_bal[tx.credit_key] = (
                local_bal.get(tx.credit_key, 0.0) + tx.amount
            )
            local_nonce[tx.sender] = local_nonce.get(tx.sender, 0) + 1
            delta.record(tx)
        else:
            delta.rejected.append(tx)
    return delta


# --------------------------------------------------------------------------
# Sequential baseline
# --------------------------------------------------------------------------

def run_sequential(
    txs: Sequence[Transaction],
    genesis: State,
    difficulty: int,
    block_size: int,
) -> Result:
    state = genesis.clone()
    res = Result(label="SEQUENTIAL", n_regions=1)
    prev = "0" * 64
    idx = 0

    t0 = time.perf_counter()
    for start in range(0, len(txs), block_size):
        batch = txs[start : start + block_size]
        included = []
        for tx in batch:
            if state.apply(tx):
                included.append(tx)
            else:
                res.rejected += 1
        b = Block(index=idx, region=0, prev_hash=prev, transactions=included)
        b = mine(b, difficulty)
        prev = b.block_hash
        idx += 1
        res.blocks += 1
        res.hash_attempts += b.hash_attempts
        res.committed += len(included)
    res.wall_time = time.perf_counter() - t0
    res.rounds = res.blocks
    res.state = state
    return res


# --------------------------------------------------------------------------
# Parallel region-based mining
# --------------------------------------------------------------------------

def run_parallel(
    txs: Sequence[Transaction],
    genesis: State,
    regions: Sequence[int],
    n_regions: int,
    difficulty: int,
    block_size: int,
    label: str,
    nonce_mode: str = "global",
    executor: ProcessPoolExecutor | None = None,
    retry_future_nonces: bool = True,
) -> Result:
    queues: List[List[Transaction]] = [[] for _ in range(n_regions)]
    for tx, r in zip(txs, regions):
        queues[r].append(tx)

    state = genesis.clone()
    res = Result(label=label, n_regions=n_regions)
    prev_super = "0" * 64
    idx = 0

    # difficulty 0 skips proof-of-work entirely; used for the safety sweeps,
    # where only the state outcome matters and mining cost would dominate.
    use_pool = difficulty > 0
    owns_executor = use_pool and executor is None
    if owns_executor:
        executor = ProcessPoolExecutor(max_workers=n_regions)

    try:
        t0 = time.perf_counter()
        while any(queues):
            snapshot = state.clone()
            deltas: List[RegionDelta] = []
            jobs = []
            round_progress = 0

            for r in range(n_regions):
                if not queues[r]:
                    continue
                if nonce_mode == "global" and retry_future_nonces:
                    # A mempool can index transactions by sender/nonce.  Scan
                    # the region queue for currently ready transactions rather
                    # than dropping future nonces or imposing an artificial
                    # FIFO head-of-line stall.
                    batch = queues[r]
                    queues[r] = []
                    d = _execute(batch, snapshot, r, nonce_mode,
                                 commit_limit=block_size)
                    queues[r].extend(d.deferred)
                else:
                    batch = queues[r][:block_size]
                    queues[r] = queues[r][block_size:]
                    d = _execute(batch, snapshot, r, nonce_mode)
                deltas.append(d)
                res.rejected += len(d.rejected)
                res.deferred_attempts += d.deferred_checks
                round_progress += len(d.committed) + len(d.rejected)
                if d.committed:
                    jobs.append((idx, r, prev_super, d.committed, difficulty))
                    idx += 1

            if jobs:
                # All regions mine concurrently; the round costs as long as its
                # slowest region.  This is the real wall-clock parallel speedup.
                if use_pool:
                    mined = list(executor.map(_mine_worker, jobs))
                else:
                    mined = [_mine_worker(j) for j in jobs]
                hashes = [m[2] for m in mined]
                res.hash_attempts += sum(hashes)
                res.blocks += len(mined)
                prev_super = _combine(prev_super, [m[1] for m in sorted(mined)])

            ds, phantom, conflicting = merge_deltas(state, deltas)
            res.double_spend_accounts += ds
            res.phantom_value += phantom
            res.conflicting_accounts += conflicting
            res.committed += sum(len(d.committed) for d in deltas)
            res.rounds += 1

            # All remaining transactions wait on nonces that no region can
            # advance.  Stop with an explicit pending count instead of looping
            # forever or mislabelling them as rejected.
            if round_progress == 0:
                break

        res.wall_time = time.perf_counter() - t0
    finally:
        if owns_executor:
            executor.shutdown()

    res.state = state
    res.pending = sum(len(q) for q in queues)
    return res


def _combine(prev: str, hashes: List[str]) -> str:
    """PBH = H(B_1 || B_2 || ... || B_N), eq. (2) of the base paper."""
    import hashlib

    return hashlib.sha256((prev + "".join(hashes)).encode()).hexdigest()
