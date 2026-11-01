"""Bounded historical RPC retry; every selected row gets independent block/hash checks.

Uses the original seed-0 sample, two attempts per RPC, and no circuit breaker.
Unavailable responses never imply a reverted transaction. Run with --endpoint.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
from collections import Counter

from audit_receipts import ROOT, TRACES, RPC, classify, validate_transaction


def audit_trace(trace, endpoint, timeout):
    path = ROOT / "data" / TRACES[trace][0]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = sorted(random.Random(0).sample(range(len(rows)), 100))
    rpc = RPC(endpoint, ROOT / "results" / "receipt_cache" / (trace + ".jsonl"),
              timeout, continue_on_error=True)
    results = []
    for index in selected:
        row = rows[index]
        tx_hash = row.get("tx_hash", "")
        result = dict(trace=trace, csv_row=index, block=row["block"], tx_index=row["tx_index"],
                      tx_hash=tx_hash, block_check="unavailable", receipt_check="unavailable",
                      classification="unavailable", block_error="", receipt_error="")
        try:
            tx = rpc.get("eth_getTransactionByBlockNumberAndIndex",
                         [hex(int(row["block"])), hex(int(row["tx_index"]))])
            matches = validate_transaction(tx, row) and (
                not tx_hash or tx["hash"].lower() == tx_hash.lower())
            result["block_check"] = "verified" if matches else "record_mismatch"
            if matches and not tx_hash:
                tx_hash = tx["hash"]
                result["tx_hash"] = tx_hash
        except Exception as exc:
            result["block_error"] = str(exc)
        if tx_hash:
            try:
                receipt = rpc.get("eth_getTransactionReceipt", [tx_hash])
                result["receipt_check"] = classify(receipt, row, tx_hash)[0]
            except Exception as exc:
                result["receipt_error"] = str(exc)
        else:
            result["receipt_error"] = "hash recovery unavailable; no receipt query possible"
        if result["block_check"] == "record_mismatch":
            result["classification"] = "record_mismatch"
        elif result["block_check"] == "verified":
            result["classification"] = result["receipt_check"]
        results.append(result)
        if len(results) % 10 == 0:
            print(f"{trace}: {len(results)}/100 attempted", flush=True)
    summary = dict(trace=trace, selected=100, seed=0, rpc_endpoint=endpoint,
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   classifications=dict(Counter(r["classification"] for r in results)),
                   block_checks=dict(Counter(r["block_check"] for r in results)),
                   receipt_checks=dict(Counter(r["receipt_check"] for r in results)),
                   network_requests=rpc.network_requests)
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--output", default="receipt_ethereum_retry")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda trace: audit_trace(trace, args.endpoint, args.timeout),
                               ["eth_aug14", "eth_aug13", "eth_aug07"]))
    rows = [row for records, _ in outputs for row in records]
    with (ROOT / "results" / (args.output + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(traces=[s for _, s in outputs], attempts_per_rpc=2,
                   interpretation="unavailable is unresolved, never inferred reverted")
    (ROOT / "results" / (args.output + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
