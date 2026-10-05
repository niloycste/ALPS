import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.chain import Transaction, fully_funded_state, uniform_state
from sim.experiment import run_parallel
from sim.online import run_temporal_allocator
from sim.partition import (
    alps,
    cross_region_ratio,
    load_balance,
    resource_locality_violations,
    sender_locality_violations,
)
from sim.metis import metis_sender, sender_interaction_graph


def tx(tid, sender, receiver, amount, nonce, category, feature):
    return Transaction(tid, sender, receiver, amount, nonce, category,
                       tuple(feature))


class ValidityTests(unittest.TestCase):
    def test_metis_graph_uses_active_senders_and_symmetric_weights(self):
        txs = [
            tx(0, 0, 1, 1, 0, 0, (1, 0)),
            tx(1, 0, 1, 1, 1, 0, (1, 0)),
            tx(2, 1, 0, 1, 0, 0, (1, 0)),
            tx(3, 2, 99, 1, 0, 0, (1, 0)),
        ]
        nodes, adjacency, vertex_weights, edge_weights = (
            sender_interaction_graph(txs))
        self.assertEqual(nodes, [0, 1, 2])
        self.assertEqual(vertex_weights, [2, 1, 1])
        self.assertEqual(adjacency, [[1], [0], []])
        self.assertEqual(edge_weights, [3, 3])

    def test_metis_assignment_is_sender_local(self):
        txs = [
            tx(0, 0, 1, 1, 0, 0, (1, 0)),
            tx(1, 0, 2, 1, 1, 1, (0, 1)),
            tx(2, 1, 0, 1, 0, 0, (1, 0)),
            tx(3, 2, 0, 1, 0, 1, (0, 1)),
        ]

        fake = types.SimpleNamespace()
        fake.Options = lambda **kw: kw
        fake.part_graph = lambda n, **kw: (0, [0, 1, 1])
        previous = sys.modules.get("pymetis")
        sys.modules["pymetis"] = fake
        try:
            regions = metis_sender(txs, 2, seed=7)
        finally:
            if previous is None:
                del sys.modules["pymetis"]
            else:
                sys.modules["pymetis"] = previous
        self.assertEqual(regions, [0, 0, 1, 1])
        self.assertEqual(
            sender_locality_violations(txs, regions)["split_senders"], 0)

    def test_interaction_cut_excludes_unplaced_receivers(self):
        txs = [
            tx(0, 0, 99, 1, 0, 0, (1, 0)),
            tx(1, 1, 0, 1, 0, 0, (1, 0)),
            tx(2, 0, 1, 1, 1, 0, (1, 0)),
        ]
        result = cross_region_ratio(txs, [0, 1, 0])
        self.assertAlmostEqual(result["interaction_coverage"], 2 / 3)
        self.assertEqual(result["cross_region_frac"], 1.0)

    def test_distinct_assets_do_not_share_a_balance(self):
        txs = [
            tx(0, 0, 1, 80, 0, 0, (1, 0, 0.2)),
            tx(1, 0, 2, 80, 1, 1, (0, 1, 0.2)),
        ]
        result = run_parallel(txs, uniform_state(txs, 100), [0, 1], 2,
                              0, 10, "split-assets", "regional")
        self.assertEqual(result.double_spend_accounts, 0)
        self.assertEqual(
            resource_locality_violations(txs, [0, 1])["split_resources"], 0)

    def test_same_asset_split_can_overdraw(self):
        txs = [
            tx(0, 0, 1, 80, 0, 0, (1, 0.2)),
            tx(1, 0, 2, 80, 1, 0, (1, 0.8)),
        ]
        result = run_parallel(txs, uniform_state(txs, 100), [0, 1], 2,
                              0, 10, "split-key", "regional")
        self.assertEqual(result.double_spend_accounts, 1)
        self.assertEqual(
            resource_locality_violations(txs, [0, 1])["split_resources"], 1)

    def test_future_nonce_is_retried(self):
        txs = [
            tx(0, 0, 1, 1, 0, 0, (1, 0.2)),
            tx(1, 0, 2, 1, 1, 0, (1, 0.8)),
        ]
        result = run_parallel(txs, fully_funded_state(txs), [1, 0], 2,
                              0, 10, "nonce-split", "global")
        self.assertEqual(result.committed, 2)
        self.assertEqual(result.pending, 0)
        self.assertGreater(result.deferred_attempts, 0)

    def test_alps_hard_constrains_sender_locality(self):
        txs = []
        for sender in range(12):
            for nonce in range(4):
                category = sender % 3
                feature = np.eye(3)[category].tolist() + [nonce / 4]
                txs.append(tx(len(txs), sender, (sender + 1) % 12, 1,
                              nonce, category, feature))
        n_users = 12
        sf = np.zeros((n_users, 4)); count = np.zeros(n_users)
        for t in txs:
            sf[t.sender] += np.asarray(t.features); count[t.sender] += 1
        sf /= count[:, None]
        regions, diagnostics = alps(
            txs, 3, sender_features=sf, over=1.15, seed=0,
            return_diagnostics=True)
        self.assertEqual(
            sender_locality_violations(txs, regions)["split_senders"], 0)
        self.assertLessEqual(load_balance(regions, 3)["max_over_mean"], 1.15)
        self.assertFalse(diagnostics["capacity_relaxed"])
        trace = diagnostics["objective_trace"]
        self.assertTrue(all(after <= before + 1e-10
                            for before, after in zip(trace, trace[1:])))

    def test_temporal_cold_sender_is_placed_once(self):
        txs = [
            tx(0, 0, 1, 1, 0, 0, (1, 0)),
            tx(1, 1, 0, 1, 0, 1, (0, 1)),
            tx(2, 0, 1, 1, 1, 0, (1, 0)),
            tx(3, 1, 0, 1, 1, 1, (0, 1)),
            tx(4, 2, 0, 1, 0, 0, (1, 0)),
            tx(5, 2, 1, 1, 1, 1, (0, 1)),
            tx(6, 3, 0, 1, 0, 1, (0, 1)),
            tx(7, 3, 1, 1, 1, 0, (1, 0)),
        ]
        result = run_temporal_allocator(txs, 2, "ALPS", frac=0.5, seed=0)
        self.assertEqual(result["cold_senders"], 2)
        self.assertEqual(result["split_senders"], 0)


if __name__ == "__main__":
    unittest.main()
