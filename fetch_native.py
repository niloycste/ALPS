"""
Fetch real Ethereum transactions with their *actual* initiator and nonce.

Transfer logs identify the token holder being debited, but the Ethereum nonce
belongs to the externally-owned account that signed the transaction, and the two
need not coincide. To keep the state owner, the signer and the nonce owner the
same entity, we fetch whole blocks and keep only direct ERC-20 ``transfer``
calls, where the signer is the token holder being debited.

For each kept transaction we record:
  from   the signing EOA -- the account debited, and the nonce owner
  nonce  the real Ethereum account nonce
  to     the token contract, which serves as the payload category
  recip  the token recipient, decoded from the call data
  value  the token amount, decoded from the call data

    python fetch_native.py --blocks 400 --out data/native.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import time
import urllib.request

# keccak("transfer(address,uint256)")[:4].  The selector, the calldata layout
# and the nonce semantics are identical on every EVM chain, so only the RPC
# endpoint and the token table change between chains.
TRANSFER_SELECTOR = "0xa9059cbb"

CHAINS = {
    "ethereum": {
        "name": "Ethereum mainnet",
        "rpc": "https://ethereum-rpc.publicnode.com",
        "tokens": {
            "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
            "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
            "0x514910771af9ca656af840dff83e8264ecf986ca": "LINK",
            "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
            "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": "SHIB",
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "UNI",
        },
    },
    # A second EVM chain with a disjoint token set, a different dominant
    # asset and a shorter block time, used to test whether the invariant and
    # the allocator transfer without re-tuning.
    "bsc": {
        "name": "BNB Smart Chain",
        "rpc": "https://bsc-rpc.publicnode.com",
        "tokens": {
            "0x55d398326f99059ff775485246999027b3197955": "USDT",
            "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d": "USDC",
            "0xe9e7cea3dedca5984780bafc599bd69add087d56": "BUSD",
            "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "WBNB",
            "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82": "CAKE",
            "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c": "BTCB",
            "0x2170ed0880ac9a755fd29b2688956bd959f933f8": "ETH",
            "0x1d2f0da169ceb9fc7b3144628db156f3f6c60dbe": "XRP",
        },
    },
}

# Rebound in main() once the chain is selected.
RPC = CHAINS["ethereum"]["rpc"]
TOKENS = CHAINS["ethereum"]["tokens"]
CHAIN_NAME = CHAINS["ethereum"]["name"]

FIELDS = [
    "block", "block_timestamp", "tx_index", "tx_hash", "token", "symbol",
    "from", "nonce", "to", "value",
]


def rpc(method, params, retries=5):
    body = json.dumps({"jsonrpc": "2.0", "method": method,
                       "params": params, "id": 1}).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                RPC, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": "research-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            if "error" in out:
                raise RuntimeError(out["error"])
            return out["result"]
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"{method} failed: {last}")


def decode_transfer(data: str):
    """Decode transfer(address,uint256) call data -> (recipient, amount)."""
    if not data or not data.startswith(TRANSFER_SELECTOR):
        return None
    body = data[10:]
    if len(body) < 128:
        return None
    try:
        recip = "0x" + body[24:64]
        amount = int(body[64:128], 16)
    except ValueError:
        return None
    if amount == 0:
        return None
    return recip.lower(), amount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--back", type=int, default=64)
    ap.add_argument("--start", type=int,
                    help="fixed first block (requires --end; overrides --blocks/--back)")
    ap.add_argument("--end", type=int,
                    help="fixed last block, inclusive (requires --start)")
    ap.add_argument("--out", default="data/native.csv")
    ap.add_argument("--manifest",
                    help="JSON provenance file (default: <out>.manifest.json)")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--chain", choices=sorted(CHAINS), default="ethereum",
                    help="EVM chain to collect from (selects RPC and tokens)")
    a = ap.parse_args()

    global RPC, TOKENS, CHAIN_NAME
    cfg = CHAINS[a.chain]
    RPC, TOKENS, CHAIN_NAME = cfg["rpc"], cfg["tokens"], cfg["name"]
    print(f"chain: {CHAIN_NAME} via {RPC}", flush=True)

    head = int(rpc("eth_blockNumber", []), 16)
    if (a.start is None) != (a.end is None):
        ap.error("--start and --end must be supplied together")
    if a.start is not None:
        start, end = a.start, a.end
        if start > end:
            ap.error("--start must not exceed --end")
        if end > head:
            ap.error(f"--end {end} exceeds current head {head}")
    else:
        end = head - a.back
        start = end - a.blocks + 1
    requested_blocks = end - start + 1
    print(f"head {head}; scanning {start}..{end} ({requested_blocks} blocks)",
          flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    f.flush()

    total = 0
    scanned = 0
    first_timestamp = None
    last_timestamp = None
    for bn in range(start, end + 1):
        try:
            blk = rpc("eth_getBlockByNumber", [hex(bn), True])
        except Exception as e:
            f.close()
            raise RuntimeError(
                f"block {bn} could not be fetched; refusing an incomplete trace"
            ) from e
        if not blk:
            f.close()
            raise RuntimeError(
                f"block {bn} was missing; refusing an incomplete trace")

        block_timestamp = int(blk["timestamp"], 16)
        first_timestamp = (block_timestamp if first_timestamp is None
                           else first_timestamp)
        last_timestamp = block_timestamp
        scanned += 1

        rows = []
        for i, t in enumerate(blk.get("transactions", [])):
            to = (t.get("to") or "").lower()
            if to not in TOKENS:
                continue
            dec = decode_transfer(t.get("input", ""))
            if dec is None:
                continue
            recip, amount = dec
            sender = t["from"].lower()
            if recip == sender:
                continue
            rows.append({
                "block": bn,
                "block_timestamp": block_timestamp,
                "tx_index": i,
                "tx_hash": t["hash"].lower(),
                "token": to,
                "symbol": TOKENS[to],
                "from": sender,
                "nonce": int(t["nonce"], 16),
                "to": recip,
                "value": amount,
            })

        if rows:
            w.writerows(rows)
            f.flush()
            total += len(rows)
        if (bn - start) % 25 == 0:
            print(f"  block {bn} ({bn-start+1}/{requested_blocks}): "
                  f"total {total}", flush=True)
        time.sleep(a.sleep)

    f.close()
    with open(a.out, "rb") as data_f:
        sha256 = hashlib.sha256(data_f.read()).hexdigest()
    manifest_path = a.manifest or f"{a.out}.manifest.json"
    manifest = {
        "chain": CHAIN_NAME,
        "chain_id": int(rpc("eth_chainId", []), 16),
        "rpc": RPC,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_first_block": start,
        "requested_last_block": end,
        "requested_blocks": requested_blocks,
        "scanned_blocks": scanned,
        "first_block_utc": dt.datetime.fromtimestamp(
            first_timestamp, dt.timezone.utc).isoformat(),
        "last_block_utc": dt.datetime.fromtimestamp(
            last_timestamp, dt.timezone.utc).isoformat(),
        "selection": "nonzero direct ERC-20 transfer(address,uint256) calls; "
                     "self-transfers excluded",
        "token_contracts": TOKENS,
        "rows": total,
        "csv_sha256": sha256,
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        json.dump(manifest, manifest_f, indent=2, sort_keys=True)
        manifest_f.write("\n")
    print(f"\nwrote {a.out}: {total} direct ERC-20 transfers with real nonces")
    print(f"wrote {manifest_path}: sha256={sha256}")


if __name__ == "__main__":
    main()
