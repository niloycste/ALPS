"""
State-based blockchain core with real Proof-of-Work.

No simulated mining delays: every block header is hashed with SHA-256 until a
nonce satisfying the difficulty target is found.  All timing reported by the
experiments is therefore wall-clock time spent doing real cryptographic work.

State model follows the transaction-based state machine of Chishti & Banerjee
(Internet Technology Letters, 2021):  sigma_{t+1} = Upsilon(sigma_t, T).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


BalanceKey = Tuple[int, int]  # (account, asset/category)


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Transaction:
    """A single state-transition request.

    ``category`` and ``features`` model the ``data`` field of the transaction,
    which the thesis under study uses as the clustering key.
    """

    tid: int
    sender: int
    receiver: int
    amount: float
    nonce: int
    category: int                      # semantic label of the payload
    features: Tuple[float, ...]        # numeric embedding of the data field

    @property
    def debit_key(self) -> BalanceKey:
        """Non-commutative state object consumed by this transfer."""
        return (self.sender, self.category)

    @property
    def credit_key(self) -> BalanceKey:
        return (self.receiver, self.category)

    def serialize(self) -> bytes:
        return (
            f"{self.tid}|{self.sender}|{self.receiver}|"
            f"{self.amount:.6f}|{self.nonce}|{self.category}"
        ).encode()


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

@dataclass
class Block:
    index: int
    region: int
    prev_hash: str
    transactions: List[Transaction]
    nonce: int = 0
    block_hash: str = ""
    timestamp: float = 0.0
    hash_attempts: int = 0

    def merkle_root(self) -> str:
        """Standard binary Merkle root over the transaction serializations."""
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
        layer = [hashlib.sha256(t.serialize()).hexdigest() for t in self.transactions]
        while len(layer) > 1:
            if len(layer) % 2:
                layer.append(layer[-1])
            layer = [
                hashlib.sha256((layer[i] + layer[i + 1]).encode()).hexdigest()
                for i in range(0, len(layer), 2)
            ]
        return layer[0]

    def header(self, nonce: int, merkle: str) -> bytes:
        return f"{self.index}|{self.region}|{self.prev_hash}|{merkle}|{nonce}".encode()


def mine(block: Block, difficulty: int) -> Block:
    """Search for a nonce whose header hash has ``difficulty`` leading zeros.

    This is genuine proof-of-work; the loop is the dominant cost of every
    experiment in this study.
    """
    target = "0" * difficulty
    merkle = block.merkle_root()
    nonce = 0
    while True:
        h = hashlib.sha256(block.header(nonce, merkle)).hexdigest()
        if h.startswith(target):
            block.nonce = nonce
            block.block_hash = h
            block.hash_attempts = nonce + 1
            block.timestamp = time.time()
            return block
        nonce += 1


def verify(block: Block, difficulty: int) -> bool:
    merkle = block.merkle_root()
    h = hashlib.sha256(block.header(block.nonce, merkle)).hexdigest()
    return h == block.block_hash and h.startswith("0" * difficulty)


# --------------------------------------------------------------------------
# World state
# --------------------------------------------------------------------------

class State:
    """Per-asset account balances plus global per-account transaction nonces.

    ERC-20 contracts maintain independent balances, so USDC and WETH owned by
    the same address are different state objects.  The Ethereum transaction
    nonce remains account-wide.  Keeping those keys separate prevents the
    simulator from inventing cross-token balance conflicts.
    """

    def __init__(self, balances: Dict[BalanceKey, float]):
        self.balances: Dict[BalanceKey, float] = dict(balances)
        self.nonces: Dict[int, int] = {a: 0 for a, _ in balances}

    def clone(self) -> "State":
        s = State(self.balances)
        s.nonces = dict(self.nonces)
        return s

    def total_supply(self) -> float:
        return sum(self.balances.values())

    def applicable(self, tx: Transaction) -> bool:
        """Validity predicate: sufficient funds and correct transactional nonce."""
        return (
            self.balances.get(tx.debit_key, 0.0) >= tx.amount
            and self.nonces.get(tx.sender, 0) == tx.nonce
        )

    def apply(self, tx: Transaction) -> bool:
        if not self.applicable(tx):
            return False
        self.balances[tx.debit_key] -= tx.amount
        self.balances[tx.credit_key] = (
            self.balances.get(tx.credit_key, 0.0) + tx.amount
        )
        self.nonces[tx.sender] = self.nonces.get(tx.sender, 0) + 1
        return True

    def divergence(self, other: "State") -> Tuple[int, float]:
        """Compare against a reference state.

        Returns ``(n_balance_keys_wrong, total_absolute_error)``.
        """
        keys = set(self.balances) | set(other.balances)
        wrong = 0
        err = 0.0
        for k in keys:
            d = abs(self.balances.get(k, 0.0) - other.balances.get(k, 0.0))
            if d > 1e-6:
                wrong += 1
                err += d
        return wrong, err


def uniform_state(txs: Sequence[Transaction], initial_balance: float) -> State:
    """Give every observed account/asset state key the same stress balance."""
    keys = {t.debit_key for t in txs} | {t.credit_key for t in txs}
    return State({k: float(initial_balance) for k in keys})


def fully_funded_state(txs: Sequence[Transaction], cushion: float = 1.0) -> State:
    """Construct a state in which balance headroom cannot confound nonce tests.

    Every debit key receives its complete observed outgoing amount plus a small
    cushion.  Thus a sequential execution can commit the entire trace, and any
    extra rounds in a parallel run come from nonce dependencies rather than an
    arbitrary assumed balance.
    """
    balances: Dict[BalanceKey, float] = {}
    for t in txs:
        balances[t.debit_key] = balances.get(t.debit_key, cushion) + t.amount
        balances.setdefault(t.credit_key, cushion)
    return State(balances)


# --------------------------------------------------------------------------
# Region delta: the unit merged at the end-of-round barrier
# --------------------------------------------------------------------------

@dataclass
class RegionDelta:
    """State changes produced by one region during one round.

    Debits and credits are kept apart because the safety argument of the base
    paper depends on them behaving differently: a debit is guarded by a balance
    precondition, whereas a credit is an unconditional commutative addition.
    """

    region: int
    debits: Dict[BalanceKey, float] = field(default_factory=dict)
    credits: Dict[BalanceKey, float] = field(default_factory=dict)
    nonce_bumps: Dict[int, int] = field(default_factory=dict)
    committed: List[Transaction] = field(default_factory=list)
    rejected: List[Transaction] = field(default_factory=list)
    deferred: List[Transaction] = field(default_factory=list)
    deferred_checks: int = 0

    def record(self, tx: Transaction) -> None:
        self.debits[tx.debit_key] = self.debits.get(tx.debit_key, 0.0) + tx.amount
        self.credits[tx.credit_key] = self.credits.get(tx.credit_key, 0.0) + tx.amount
        self.nonce_bumps[tx.sender] = self.nonce_bumps.get(tx.sender, 0) + 1
        self.committed.append(tx)


def execute_region(
    txs: Sequence[Transaction], snapshot: State
) -> RegionDelta:
    """Execute a region's transactions against the start-of-round snapshot.

    Miners in different regions work concurrently and cannot observe one
    another's uncommitted work, so each region validates against the same
    snapshot.  This is precisely the condition under which a broken partition
    admits a double-spend.
    """
    local = snapshot.clone()
    delta = RegionDelta(region=-1)
    for tx in txs:
        if local.apply(tx):
            delta.record(tx)
        else:
            delta.rejected.append(tx)
    return delta


def merge_deltas(
    state: State, deltas: Iterable[RegionDelta]
) -> Tuple[int, float, int]:
    """Apply all region deltas at the end-of-round barrier.

    Returns ``(n_overdrawn_accounts, phantom_value, n_conflicting_accounts)``.

    A guarded balance key is *conflicting* when more than one region debited it
    during the round.  Counts are reported as unique affected accounts for
    compatibility with the paper.  An account is *overdrawn* when any of its
    token balances is negative after the merge; the deficit magnitude is the
    invalid value admitted by the stale-snapshot execution.

    A negative balance is an exact witness of a broken partition.  If every
    debit of a guarded key lies in a single region r, then that region executed
    those debits against a running local balance that never went below zero, so

        final = snapshot + credits(all regions) - debits(r) >= 0,

    since credits only add.  A balance can therefore go negative only when two
    or more regions debited the same account against the same stale snapshot.
    """
    deltas = list(deltas)

    debit_regions: Dict[BalanceKey, int] = {}
    for d in deltas:
        for acct in d.debits:
            debit_regions[acct] = debit_regions.get(acct, 0) + 1
    conflicting_accounts = {
        key[0] for key, n in debit_regions.items() if n > 1
    }

    for d in deltas:
        for acct, amt in d.debits.items():
            state.balances[acct] = state.balances.get(acct, 0.0) - amt
        for acct, amt in d.credits.items():
            state.balances[acct] = state.balances.get(acct, 0.0) + amt
        for acct, n in d.nonce_bumps.items():
            state.nonces[acct] = state.nonces.get(acct, 0) + n

    overdrawn_accounts = set()
    phantom = 0.0
    for key, bal in state.balances.items():
        if bal < -1e-9:
            overdrawn_accounts.add(key[0])
            phantom += -bal
            # The overdraft is realised; the deficit is not carried forward, as
            # the network would have accepted these blocks as valid.
            state.balances[key] = 0.0

    return len(overdrawn_accounts), phantom, len(conflicting_accounts)
