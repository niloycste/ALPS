"""
Conflict-rate sweep built entirely from real Ethereum transactions.

The question the sweep answers is how the failure scales with the share of
accounts that trade in more than one category. On a fixed trace that share is a
constant, so instead of generating accounts we *stratify real ones*.

Every sender in the trace is classified by measurement:

  specialist         issues >1 transfer, all of a single token
  multi-token trader issues >1 transfer, spanning two or more tokens

A workload at mix m is then assembled by drawing real senders so that a
fraction m are multi-token traders, and taking **all of their real transfers**
in their real on-chain order. No transaction, address, value, or category is
synthesised; only the composition of the account population is controlled.

Single-transfer senders are excluded from the sampled population because one
transaction cannot straddle two regions -- they can never violate sender
locality, and including them would only dilute the reported rates.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .chain import State, Transaction, uniform_state


class RealSweep:
    def __init__(
        self,
        path: str,
        initial_balance: float = 1000.0,
        amount_clip: Tuple[float, float] = (0.1, 20.0),
        amount_scale: float = 8.0,
        seed: int = 7,
    ):
        self.path = path
        self.initial_balance = initial_balance
        self.amount_clip = amount_clip
        self.amount_scale = amount_scale
        self.seed = seed
        self._load()

    def _load(self) -> None:
        rows = []
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        # Log traces order by log index, native traces by transaction index.
        pos = "log_index" if "log_index" in rows[0] else "tx_index"
        rows.sort(key=lambda r: (int(r["block"]), int(r[pos])))
        self.rows = rows

        self.symbols = sorted({r["symbol"] for r in rows})
        self.cat_of = {s: i for i, s in enumerate(self.symbols)}
        self.n_categories = len(self.symbols)

        # Measured classification of real senders.
        by_sender: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            by_sender[r["from"]].append(r)
        self.by_sender = by_sender

        self.specialists, self.multi = [], []
        for s, rs in by_sender.items():
            if len(rs) < 2:
                continue
            (self.multi if len({r["symbol"] for r in rs}) > 1
             else self.specialists).append(s)

        by_tok: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            by_tok[r["symbol"]].append(float(r["value"]))
        self.median = {k: (np.median(v) or 1.0) for k, v in by_tok.items()}

    # ------------------------------------------------------------------
    def population(self) -> Dict[str, int]:
        return {
            "transfers": len(self.rows),
            "senders": len(self.by_sender),
            "specialists": len(self.specialists),
            "multi_token_traders": len(self.multi),
        }

    def max_mix(self, n_senders: int) -> Tuple[float, float]:
        """Range of mixes achievable with the real populations available."""
        lo = max(0.0, 1.0 - len(self.specialists) / n_senders)
        hi = min(1.0, len(self.multi) / n_senders)
        return lo, hi

    def build(
        self, multi_frac: float, n_senders: int, seed: int | None = None,
        max_tx: int | None = None,
    ) -> Tuple[List[Transaction], State, Dict[str, float]]:
        rng = np.random.default_rng(self.seed if seed is None else seed)

        k_multi = int(round(n_senders * multi_frac))
        k_multi = min(k_multi, len(self.multi))
        k_spec = min(n_senders - k_multi, len(self.specialists))

        chosen = list(rng.choice(self.multi, k_multi, replace=False)) + \
                 list(rng.choice(self.specialists, k_spec, replace=False))
        chosen_set = set(chosen)

        # All real transfers issued by the chosen accounts, in on-chain order.
        sel = [r for r in self.rows if r["from"] in chosen_set]

        # Multi-token traders are more active than specialists, so an unbounded
        # selection makes workload size covary with the mix. Truncating to a
        # fixed budget (still a contiguous on-chain window) removes that
        # confound, leaving the account composition as the only variable.
        if max_tx is not None:
            sel = sel[:max_tx]

        addrs = sorted({r["from"] for r in sel} | {r["to"] for r in sel})
        idx = {a: i for i, a in enumerate(addrs)}

        lo, hi = self.amount_clip
        eye = np.eye(self.n_categories) * 4.0
        nonces: Dict[int, int] = {}
        txs: List[Transaction] = []

        for i, r in enumerate(sel):
            s, d = idx[r["from"]], idx[r["to"]]
            if s == d:
                continue
            cat = self.cat_of[r["symbol"]]
            rel = float(r["value"]) / self.median[r["symbol"]]
            amt = float(np.clip(rel, lo, hi)) * self.amount_scale
            log_amount = float(np.log1p(np.clip(rel, lo, hi)))
            feat = np.concatenate((eye[cat], [log_amount]))
            n = nonces.get(s, 0)
            txs.append(Transaction(
                tid=i, sender=s, receiver=d, amount=amt, nonce=n,
                category=cat, features=tuple(float(x) for x in feat),
            ))
            nonces[s] = n + 1

        genesis = uniform_state(txs, self.initial_balance)

        counts = Counter(t.sender for t in txs)
        toks: Dict[int, set] = defaultdict(set)
        for t in txs:
            toks[t.sender].add(t.category)
        realised_multi = [s for s in counts if len(toks[s]) > 1]

        info = {
            "requested_multi_frac": multi_frac,
            "sampled_senders": len(chosen),
            "realised_multi_frac": round(
                len(realised_multi) / max(len(counts), 1), 4),
            "transactions": len(txs),
            "accounts": len(addrs),
        }
        return txs, genesis, info

    def sender_feature_matrix(self, txs: List[Transaction],
                              n_users: int) -> np.ndarray:
        dim = len(txs[0].features)
        acc = np.zeros((n_users, dim))
        cnt = np.zeros(n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features)
            cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]
