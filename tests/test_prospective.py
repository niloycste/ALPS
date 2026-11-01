"""Leakage/cold-start and receipt-classification checks for the new scripts."""
import copy
from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_prospective_retrieval import PrefixEncoder, frozen_plans, query_benchmark, route_future
from tools.audit_receipts import TRANSFER_TOPIC, RPC, NotAttempted, classify, validate_transaction


def row(sender="0x" + "1" * 40, recipient="0x" + "2" * 40, token="0x" + "3" * 40, value="10", nonce="0"):
    return dict(block="100", tx_index="2", token=token, symbol="T",
                **{"from": sender, "to": recipient}, value=value, nonce=nonce)


class ProspectiveTests(unittest.TestCase):
    def test_suffix_outliers_do_not_refit_scaling_or_expand_features(self):
        encoder = PrefixEncoder([row(value="10"), row(value="30", nonce="1")])
        original = copy.deepcopy((encoder.medians, encoder.tokens, encoder.dim, encoder.fallback))
        encoder.encode(row(token="0x" + "4" * 40, value=str(10**30)), 2)
        self.assertEqual(original, (encoder.medians, encoder.tokens, encoder.dim, encoder.fallback))

    def test_plan_frozen_and_unseen_token_queries_scan_all_regions(self):
        encoder = PrefixEncoder([row()])
        known = encoder.encode(row(), 0)
        unseen = encoder.encode(row(token="0x" + "4" * 40, nonce="1"), 1)
        plans = frozen_plans({known.category: np.array([10, 0])}, [.9], 2)
        original = plans[.9][known.category].copy()
        measurements = query_benchmark([known, unseen], [1, 0], plans, 2, 2)
        by_category = {m["category"]: m for m in measurements}
        self.assertEqual(by_category[known.category]["result_recall"], 0)
        self.assertEqual(by_category[unseen.category]["result_recall"], 1)
        self.assertEqual(by_category[unseen.category]["contacted_regions"], 2)
        np.testing.assert_array_equal(original, plans[.9][known.category])

    def test_new_sender_is_pinned_despite_changed_payload(self):
        encoder = PrefixEncoder([row()])
        encoder.encode(row(), 0)
        new = "0x" + "5" * 40
        future_rows = [row(sender=new), row(sender=new, value="2000", nonce="1")]
        _, regions, metrics = route_future(future_rows, encoder, {},
                                            np.zeros((2, encoder.dim)), 2, "ALPS", 2.)
        self.assertEqual(regions[0], regions[1])
        self.assertEqual(metrics["split_senders"], 0)

    def test_graph_weight_can_change_first_placement_toward_receiver(self):
        def place(weight):
            encoder = PrefixEncoder([row()])
            tx = encoder.encode(row(), 0)
            prototypes = np.array([tx.features, tx.features], dtype=float)
            prototypes[1, -1] += 1
            assigned = {tx.receiver: 1}
            _, regions, _ = route_future([row()], encoder, assigned, prototypes,
                                         2, "ALPS", 10., weight)
            return regions[0]
        self.assertEqual(place(.15), 0)
        self.assertEqual(place(2.4), 1)


class ReceiptTests(unittest.TestCase):
    def test_offline_cache_preserves_response_and_marks_unattempted_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "responses.jsonl"
            key = json.dumps(["https://example.invalid", "test", [1]], separators=(",", ":"))
            cache.write_text(json.dumps(dict(key=key, result={"status": "0x1"})) + "\n")
            rpc = RPC("https://example.invalid", cache, 1, offline=True)
            self.assertEqual(rpc.get("test", [1]), {"status": "0x1"})
            with self.assertRaises(NotAttempted):
                rpc.get("test", [2])
            self.assertEqual(rpc.network_requests, 0)

    def test_recovered_calldata_must_match_record(self):
        r = row()
        tx = dict(blockNumber="0x64", transactionIndex="0x2", nonce="0x0",
                  to=r["token"], input="0xa9059cbb" + "0" * 24 + r["to"][2:] + f'{10:064x}',
                  **{"from": r["from"]})
        self.assertTrue(validate_transaction(tx, r))
        tx["input"] = tx["input"][:-1] + "b"
        self.assertFalse(validate_transaction(tx, r))

    def test_receipt_status_and_exact_event_are_distinct(self):
        r = row()
        receipt = dict(transactionHash="0xabc", blockNumber="0x64", transactionIndex="0x2",
            status="0x1", logs=[dict(address=r["token"], topics=[TRANSFER_TOPIC,
                "0x" + "0" * 24 + r["from"][2:], "0x" + "0" * 24 + r["to"][2:]],
                data="0x" + f'{10:064x}')])
        self.assertEqual(classify(receipt, r, "0xabc")[0], "success_matching_transfer")
        receipt["logs"][0]["data"] = "0x" + f'{11:064x}'
        self.assertEqual(classify(receipt, r, "0xabc")[0], "success_no_exact_transfer")
        receipt["status"] = "0x0"
        self.assertEqual(classify(receipt, r, "0xabc")[0], "reverted")
        self.assertEqual(classify(receipt, r, "0xdef")[0], "record_mismatch")


if __name__ == "__main__":
    unittest.main()
