"""
Adapter for the native-transaction trace, where sender, signer and nonce owner
are the same account.

The trace keeps only direct ``transfer(address,uint256)`` calls, so the
externally-owned account that signed the transaction is also the token holder
being debited and the owner of the Ethereum nonce. That removes the conflation
present when Transfer logs are read directly: a log names the token holder, but
the nonce belongs to whoever signed, and for router-mediated transfers the two
differ.

One quantity is re-based rather than used verbatim. Real nonces count *all* of
an account's activity, while the trace holds only its ERC-20 transfers, so the
observed values contain gaps. We therefore replace each sender's nonces by their
rank in real nonce order. The ordering -- which is what a nonce constraint
enforces -- is measured; only the absolute values are re-based, so that a
sequential baseline is not penalised for transactions outside the trace.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .chain import State, Transaction, uniform_state


class NativeWorkload:
    def __init__(
        self,
        path: str,
        max_tx: int | None = None,
        initial_balance: float = 1000.0,
        amount_clip: Tuple[float, float] = (0.1, 20.0),
        amount_scale: float = 8.0,
        seed: int = 7,
    ):
        # ``seed`` is retained for API compatibility.  Loading and feature
        # construction are deterministic; no synthetic jitter is sampled.
        self.path = path
        self.max_tx = max_tx
        self.initial_balance = initial_balance
        self.amount_clip = amount_clip
        self.amount_scale = amount_scale
        self._load()

    def _load(self) -> None:
        rows = []
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                r["block"] = int(r["block"])
                r["tx_index"] = int(r["tx_index"])
                r["nonce"] = int(r["nonce"])
                r["value"] = int(r["value"])
                rows.append(r)
        rows.sort(key=lambda r: (r["block"], r["tx_index"]))
        if self.max_tx:
            rows = rows[: self.max_tx]
        self.rows = rows

        self.symbols = sorted({r["symbol"] for r in rows})
        self.cat_of = {s: i for i, s in enumerate(self.symbols)}
        self.n_categories = len(self.symbols)

        addrs = sorted({r["from"] for r in rows} | {r["to"] for r in rows})
        self.id_of = {a: i for i, a in enumerate(addrs)}
        self.n_users = len(addrs)

        by_tok: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            by_tok[r["symbol"]].append(float(r["value"]))
        self.median = {k: (np.median(v) or 1.0) for k, v in by_tok.items()}

        # Rank-normalise each sender's real nonces, preserving their order.
        seen: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            seen[r["from"]].append(r["nonce"])
        self.rank_of: Dict[Tuple[str, int], int] = {}
        for a, ns in seen.items():
            for rank, n in enumerate(sorted(set(ns))):
                self.rank_of[(a, n)] = rank

    def generate(self) -> Tuple[List[Transaction], State]:
        lo, hi = self.amount_clip
        # Real payload features: exact token identity plus the log-normalised
        # amount decoded from calldata.  No random jitter or synthetic feature
        # is introduced.  The continuous amount coordinate lets a transaction-
        # level clustering split a popular token only when the measured payload
        # itself supports that split.
        eye = np.eye(self.n_categories) * 4.0
        txs: List[Transaction] = []
        used: set = set()

        for i, r in enumerate(self.rows):
            s, d = self.id_of[r["from"]], self.id_of[r["to"]]
            if s == d:
                continue
            key = (r["from"], r["nonce"])
            if key in used:            # same nonce cannot appear twice
                continue
            used.add(key)
            cat = self.cat_of[r["symbol"]]
            rel = float(r["value"]) / self.median[r["symbol"]]
            amt = float(np.clip(rel, lo, hi)) * self.amount_scale
            log_amount = float(np.log1p(np.clip(rel, lo, hi)))
            feat = np.concatenate((eye[cat], [log_amount]))
            txs.append(Transaction(
                tid=i, sender=s, receiver=d, amount=amt,
                nonce=self.rank_of[key], category=cat,
                features=tuple(float(x) for x in feat),
            ))

        genesis = uniform_state(txs, self.initial_balance)
        return txs, genesis

    def sender_feature_matrix(self, txs: List[Transaction]) -> np.ndarray:
        dim = len(txs[0].features)
        acc = np.zeros((self.n_users, dim)); cnt = np.zeros(self.n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features); cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]

    def describe(self, txs: List[Transaction]) -> Dict[str, float]:
        from collections import Counter

        per = Counter(t.sender for t in txs)
        toks: Dict[int, set] = defaultdict(set)
        for t in txs:
            toks[t.sender].add(t.category)
        repeat = {s for s, c in per.items() if c > 1}
        multi = {s for s in repeat if len(toks[s]) > 1}
        blocks = sorted({r["block"] for r in self.rows})
        return {
            "transactions": len(txs),
            "nonempty_blocks": len(blocks),
            "first_block": min(blocks),
            "last_block": max(blocks),
            "height_span": max(blocks) - min(blocks) + 1,
            "accounts": self.n_users,
            "categories": self.n_categories,
            "senders": len(per),
            "repeat_senders": len(repeat),
            "multi_category_senders": len(multi),
            "multi_category_tx_frac": sum(per[s] for s in multi) / max(len(txs), 1),
            "max_sender_tx": max(per.values()),
        }
