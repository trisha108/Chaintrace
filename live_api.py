"""
Background workers: Bitcoin block feed and RSS intel (blockchain.info + feeds).

All HTTP calls use ``timeout=5`` and run in daemon threads; the UI drains
``tx_queue`` and ``intel_queue`` with ``get_nowait`` so it never blocks on I/O.

``tx_queue`` carries unified tx rows and optional ``{"type": "cio", ...}`` items
for common-input ownership from the block parser.
"""

from __future__ import annotations

import queue
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout

import rss_monitor
from data_loader import UNIFIED_COLUMNS, _ensure_unified

tx_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
intel_queue: "queue.Queue[list]" = queue.Queue(maxsize=4)

def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _cio_payload_from_tx(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Common-input ownership: all distinct input addresses for this tx (real blockchain data).
    """
    input_addrs = [
        inp["prev_out"]["addr"]
        for inp in tx.get("inputs", [])
        if isinstance(inp, dict)
        and "prev_out" in inp
        and isinstance(inp.get("prev_out"), dict)
        and "addr" in inp["prev_out"]
    ]
    seen = set()
    uniq = []
    for a in input_addrs:
        sa = str(a).strip()
        if sa and sa not in seen:
            seen.add(sa)
            uniq.append(sa)
    if len(uniq) < 2:
        return None
    h = tx.get("hash")
    if not h:
        return None
    return {
        "type": "cio",
        "tx_id": str(h),
        "addresses": uniq,
    }


def parse_bitcoin_tx(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one blockchain.info raw tx into unified schema; None if unusable."""
    tx_hash = str(tx.get("hash", "unknown"))
    outs = tx.get("out") or []
    ins = tx.get("inputs") or []
    if not outs:
        return None
    to_addr = outs[0].get("addr") or "unknown_out"
    fr = "unknown_in"
    if ins and isinstance(ins[0], dict):
        po = ins[0].get("prev_out") or {}
        if isinstance(po, dict):
            fr = po.get("addr") or "unknown_in"
    try:
        val = float(outs[0].get("value", 0)) / 1e8
    except (TypeError, ValueError):
        val = 0.0
    ts = float(tx.get("time", _now_ts()))
    return {
        "tx_id": f"live_btc_{tx_hash[:16]}",
        "from_address": str(fr).strip(),
        "to_address": str(to_addr).strip(),
        "amount": val,
        "timestamp": ts,
        "chain": "bitcoin",
        "label": "unknown",
        "time_step": int(ts // 3600.0),
    }


def live_feed_worker() -> None:
    circuit_failures = 0
    while True:
        try:
            if circuit_failures >= 3:
                time.sleep(300)
                circuit_failures = 0
                continue

            response = requests.get(
                "https://blockchain.info/latestblock",
                timeout=5,
            )
            response.raise_for_status()
            block_hash = response.json()["hash"]

            block_response = requests.get(
                f"https://blockchain.info/rawblock/{block_hash}",
                timeout=5,
            )
            block_response.raise_for_status()
            txs = block_response.json().get("tx", [])

            for tx in txs[:50]:
                parsed = parse_bitcoin_tx(tx)
                if parsed:
                    tx_queue.put(parsed)
                cio_item = _cio_payload_from_tx(tx)
                if cio_item:
                    tx_queue.put(cio_item)

            circuit_failures = 0
            time.sleep(60)

        except (Timeout, ConnectionError):
            circuit_failures += 1
            time.sleep(10)
        except Exception:
            circuit_failures += 1
            time.sleep(10)


INTEL_POLL_SEC = 90


def intel_feed_worker() -> None:
    """Poll RSS intel feeds in background; UI drains ``intel_queue`` (non-blocking)."""
    while True:
        try:
            items = rss_monitor.fetch_feed_items(8, timeout=5)
        except Exception:
            items = []
        try:
            while True:
                try:
                    intel_queue.get_nowait()
                except queue.Empty:
                    break
            intel_queue.put_nowait(items)
        except queue.Full:
            pass
        time.sleep(INTEL_POLL_SEC)


def fetch_bitcoin_block_height() -> Optional[int]:
    try:
        r = requests.get("https://blockchain.info/latestblock", timeout=5)
        r.raise_for_status()
        return int(r.json().get("height", 0))
    except Exception:
        return None


def seconds_until_next_poll(last_ts: float, interval: int = 60) -> int:
    elapsed = time.time() - last_ts
    return max(0, int(interval - elapsed))


def poll_all_live(_etherscan_api_key: Optional[str] = None) -> pd.DataFrame:
    """Deprecated for thread feed; returns empty unified frame."""
    return _ensure_unified(pd.DataFrame(columns=UNIFIED_COLUMNS))
