"""
Adapter turning real Ethereum ERC-20 transfers into the simulator's workload.

Everything structural is measured, not invented: senders, receivers, the
sender-repeat distribution, which accounts trade across several tokens, and the
order of events. The token contract is the payload *category* -- a real
semantic label, since USDC and WETH are genuinely different transferred goods.

Two quantities must be derived rather than taken verbatim, and both are
documented in the paper:

  amount   ERC-20 values are not comparable across tokens (different decimals
           and unit prices), so each transfer is normalised by the median
           transfer of its own token. This preserves the heavy tail *within* a
           token while making magnitudes comparable across them.

  nonce    Transfer logs do not expose the originating account nonce, so we
           assign per-sender sequence numbers in observed (block, log_index)
           order -- exactly the order a validator would have applied them.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from .chain import State, Transaction, uniform_state


class EthereumWorkload:
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
                rows.append(r)
        rows.sort(key=lambda r: (int(r["block"]), int(r["log_index"])))
        if self.max_tx:
            rows = rows[: self.max_tx]
        self.rows = rows

        symbols = sorted({r["symbol"] for r in rows})
        self.symbols = symbols
        self.cat_of = {s: i for i, s in enumerate(symbols)}
        self.n_categories = len(symbols)

        addrs = sorted({r["from"] for r in rows} | {r["to"] for r in rows})
        self.id_of = {a: i for i, a in enumerate(addrs)}
        self.n_users = len(addrs)

        # Per-token median, for cross-token comparable magnitudes.
        by_tok: Dict[str, List[float]] = {}
        for r in rows:
            by_tok.setdefault(r["symbol"], []).append(float(r["value"]))
        self.median = {k: (np.median(v) or 1.0) for k, v in by_tok.items()}

    def generate(self) -> Tuple[List[Transaction], State]:
        lo, hi = self.amount_clip
        nonces: Dict[int, int] = {}
        txs: List[Transaction] = []

        # Exact token identity plus a real, decoded amount coordinate.  This
        # replaces the former Gaussian jitter, which could manufacture cluster
        # boundaries between otherwise identical payloads.
        eye = np.eye(self.n_categories) * 4.0

        for i, r in enumerate(self.rows):
            s = self.id_of[r["from"]]
            d = self.id_of[r["to"]]
            if s == d:
                continue
            cat = self.cat_of[r["symbol"]]
            rel = float(r["value"]) / self.median[r["symbol"]]
            amt = float(np.clip(rel, lo, hi)) * self.amount_scale
            log_amount = float(np.log1p(np.clip(rel, lo, hi)))
            feat = np.concatenate((eye[cat], [log_amount]))
            n = nonces.get(s, 0)
            txs.append(
                Transaction(
                    tid=i,
                    sender=s,
                    receiver=d,
                    amount=amt,
                    nonce=n,
                    category=cat,
                    features=tuple(float(x) for x in feat),
                )
            )
            nonces[s] = n + 1

        genesis = uniform_state(txs, self.initial_balance)
        return txs, genesis

    def sender_feature_matrix(self, txs: List[Transaction]) -> np.ndarray:
        dim = len(txs[0].features)
        acc = np.zeros((self.n_users, dim))
        cnt = np.zeros(self.n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features)
            cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]

    def describe(self, txs: List[Transaction]) -> Dict[str, float]:
        from collections import Counter

        per_sender = Counter(t.sender for t in txs)
        tok_per_sender: Dict[int, set] = {}
        for t in txs:
            tok_per_sender.setdefault(t.sender, set()).add(t.category)

        repeat = {s for s, c in per_sender.items() if c > 1}
        multi = {s for s in repeat if len(tok_per_sender[s]) > 1}
        multi_tx = sum(per_sender[s] for s in multi)

        blocks = sorted({int(r["block"]) for r in self.rows})
        return {
            "transactions": len(txs),
            "nonempty_blocks": len(blocks),
            "first_block": min(blocks),
            "last_block": max(blocks),
            "height_span": max(blocks) - min(blocks) + 1,
            "accounts": self.n_users,
            "categories": self.n_categories,
            "senders": len(per_sender),
            "repeat_senders": len(repeat),
            "multi_category_senders": len(multi),
            "multi_category_tx_frac": multi_tx / max(len(txs), 1),
            "max_sender_tx": max(per_sender.values()),
        }
