"""
Load and normalize multiple AML-related datasets into one unified transaction schema.

Each output row contains:
  tx_id, from_address, to_address, amount, timestamp, chain, label

Missing files and failed downloads are skipped with warnings (non-fatal).
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

# Columns for every normalized dataframe we produce.
UNIFIED_COLUMNS = [
    "tx_id",
    "from_address",
    "to_address",
    "amount",
    "timestamp",
    "chain",
    "label",
    "time_step",
]

# Demo cap when loading Elliptic from disk (matches dashboard default).
ELLIPTIC_DEMO_ROWS = 3000

OFAC_SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"

# Regex helpers for OFAC / free-text address extraction.
_RE_ETH = re.compile(r"\b(0x[a-fA-F0-9]{40})\b")
_RE_BTC_LEGACY = re.compile(r"\b([13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
_RE_BTC_BECH32 = re.compile(r"\b(bc1[a-z0-9]{39,59})\b")


def _sim_btc_addr(seed: str) -> str:
    """Deterministic pseudo Bitcoin-style address for synthetic / Elliptic-derived rows."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    return "bc1sim" + h[:38]


def _empty_unified() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def _ensure_unified(df: pd.DataFrame) -> pd.DataFrame:
    for c in UNIFIED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[UNIFIED_COLUMNS]


def extract_sanctioned_addresses_from_text(text: str) -> Set[str]:
    """Pull likely crypto addresses from a free-text OFAC / sanctions field."""
    if not isinstance(text, str) or not text.strip():
        return set()
    found: Set[str] = set()
    found.update(_RE_ETH.findall(text))
    found.update(_RE_BTC_LEGACY.findall(text))
    found.update(_RE_BTC_BECH32.findall(text))
    return found


def load_ofac_sdn_addresses(
    cache_path: Optional[str] = None,
    timeout: int = 5,
) -> Tuple[Set[str], List[str]]:
    """
    Download (or load from cache) OFAC SDN list and extract crypto-like addresses.

    Returns (address_set, warning_messages).
    """
    msgs: List[str] = []
    addresses: Set[str] = set()
    if cache_path and os.path.isfile(cache_path):
        try:
            raw = open(cache_path, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            msgs.append(f"Could not read OFAC cache {cache_path}: {e}")
            return addresses, msgs
    else:
        try:
            r = requests.get(OFAC_SDN_URL, timeout=timeout)
            r.raise_for_status()
            raw = r.text
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(raw)
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001
            msgs.append(f"OFAC SDN download failed (skipping): {e}")
            return addresses, msgs

    # SDN is CSV; fields vary — scan whole file as text for address tokens.
    for line in raw.splitlines():
        addresses.update(extract_sanctioned_addresses_from_text(line))
    return addresses, msgs


def load_known_bad_csv(path: str) -> Set[str]:
    """Load extra sanctioned / test addresses from a one-column or labeled CSV."""
    if not os.path.isfile(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "address" in df.columns:
        return set(df["address"].astype(str).str.strip())
    return set(df.iloc[:, 0].astype(str).str.strip())


def load_elliptic_unified(
    data_dir: str,
    max_feature_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Map Elliptic transaction graph edges into pseudo wallet-to-wallet transfers.

    Elliptic nodes are transactions; we synthesize stable ``bc1sim…`` addresses from
    integer tx ids so the rest of the pipeline can stay wallet-centric.
    """
    msgs: List[str] = []
    base = os.path.abspath(data_dir)
    fp = os.path.join(base, "elliptic_sample_features.csv")
    ep = os.path.join(base, "elliptic_sample_edgelist.csv")
    cp = os.path.join(base, "elliptic_sample_classes.csv")
    if not all(os.path.isfile(p) for p in (fp, ep, cp)):
        msgs.append("Elliptic CSV trio not found; skipping Elliptic.")
        return _empty_unified(), msgs

    try:
        feat = pd.read_csv(fp, header=None, nrows=max_feature_rows)
        feat = feat.rename(columns={0: "txId", 1: "time_step"})
        id_class = {}
        cls_df = pd.read_csv(cp)
        cls_df.columns = [str(c).strip() for c in cls_df.columns]
        for _, row in cls_df.iterrows():
            tid = int(row["txId"])
            raw = str(row["class"]).strip()
            if raw == "1":
                id_class[tid] = "illicit"
            elif raw == "2":
                id_class[tid] = "licit"
            else:
                id_class[tid] = "unknown"

        valid = set(feat["txId"].astype(int))
        edges = pd.read_csv(ep)
        edges.columns = [str(c).strip() for c in edges.columns]
        low = {c.lower(): c for c in edges.columns}
        c1 = low.get("txid1", edges.columns[0])
        c2 = low.get("txid2", edges.columns[1])
        edges = edges.rename(columns={c1: "txId1", c2: "txId2"})
        edges = edges[
            edges["txId1"].isin(valid) & edges["txId2"].isin(valid)
        ]

        tid_step: Dict[int, int] = {}
        for _, fr in feat.iterrows():
            tid = int(fr["txId"])
            tid_step[tid] = int(fr["time_step"])

        rows = []
        for i, e in edges.iterrows():
            u, v = int(e["txId1"]), int(e["txId2"])
            ts = float(u % 49 + 1)  # coarse period proxy when per-tx time not joined
            step_u = tid_step.get(u, int(u % 49))
            step_v = tid_step.get(v, int(v % 49))
            row_time_step = int((step_u + step_v) // 2)
            lbl_u = id_class.get(u, "unknown")
            lbl_v = id_class.get(v, "unknown")
            if lbl_u == "illicit" or lbl_v == "illicit":
                lab = "illicit"
            elif lbl_u == "licit" and lbl_v == "licit":
                lab = "licit"
            else:
                lab = "unknown"
            amt = 1.0 + (i % 1000) / 1000.0
            rows.append(
                {
                    "tx_id": f"elliptic_{u}_{v}_{i}",
                    "from_address": _sim_btc_addr(f"e{u}"),
                    "to_address": _sim_btc_addr(f"e{v}"),
                    "amount": amt,
                    "timestamp": float(ts),
                    "chain": "bitcoin",
                    "label": lab,
                    "time_step": row_time_step,
                }
            )
        return _ensure_unified(pd.DataFrame(rows)), msgs
    except Exception as e:  # noqa: BLE001
        msgs.append(f"Elliptic load failed: {e}")
        return _empty_unified(), msgs


def build_builtin_demo_transactions() -> pd.DataFrame:
    """
    Pure synthetic graph for first-run demo (no CSV download required).

    Includes fan-out, fan-in, and peel-friendly structure for pattern testing.
    """
    rng = np.random.default_rng(42)
    hubs = ["bc1demohub_fanout", "bc1demohub_fanin", "bc1demopeel"]
    leafs = [_sim_btc_addr(f"leaf{i}") for i in range(24)]
    rows = []
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()

    # Fan-out from hub A -> 12 wallets same window
    for i in range(12):
        ts = t0 + 60 * i
        rows.append(
            {
                "tx_id": f"demo_fo_{i}",
                "from_address": hubs[0],
                "to_address": leafs[i],
                "amount": float(5000 + i),
                "timestamp": ts,
                "chain": "bitcoin",
                "label": "unknown",
                "time_step": int((ts - t0) // 3600),
            }
        )
    # Fan-in into hub B
    for i in range(12):
        ts = t0 + 120 + 60 * i
        rows.append(
            {
                "tx_id": f"demo_fi_{i}",
                "from_address": leafs[i + 12],
                "to_address": hubs[1],
                "amount": float(4800 + rng.integers(0, 40)),
                "timestamp": ts,
                "chain": "bitcoin",
                "label": "unknown",
                "time_step": int((ts - t0) // 3600),
            }
        )
    # Peel chain
    prev = hubs[2]
    for step in range(8):
        nxt = _sim_btc_addr(f"peel_{step}")
        ts = t0 + 300 + step * 120
        rows.append(
            {
                "tx_id": f"demo_peel_{step}",
                "from_address": prev,
                "to_address": nxt,
                "amount": float(100_000 - step * 11_000),
                "timestamp": ts,
                "chain": "bitcoin",
                "label": "unknown",
                "time_step": int((ts - t0) // 3600),
            }
        )
        prev = nxt
    return _ensure_unified(pd.DataFrame(rows))


def load_all_datasets(
    data_dir: str,
    demo_mode: bool = True,
    elliptic_max_rows: Optional[int] = None,
    ofac_cache_path: Optional[str] = None,
    known_bad_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Set[str], List[str]]:
    """
    Load Elliptic (primary) and OFAC sanctions addresses only.

    When ``demo_mode`` is True and Elliptic files are missing, uses only
    ``build_builtin_demo_transactions()``. When Elliptic exists, loads a capped
    row count for speed.

    Returns:
      transactions_df, sanctioned_addresses_set, all_warning_messages
    """
    all_msgs: List[str] = []
    parts: List[pd.DataFrame] = []

    elliptic_rows = ELLIPTIC_DEMO_ROWS if demo_mode else None
    if not demo_mode:
        elliptic_rows = elliptic_max_rows  # None = full file

    elliptic_fp = os.path.join(os.path.abspath(data_dir), "elliptic_sample_features.csv")
    if demo_mode and not os.path.isfile(elliptic_fp):
        parts.append(build_builtin_demo_transactions())
        all_msgs.append("Demo mode: using built-in synthetic Bitcoin demo (no CSV).")
    else:
        e_df, e_msg = load_elliptic_unified(data_dir, max_feature_rows=elliptic_rows)
        all_msgs.extend(e_msg)
        if not e_df.empty:
            parts.append(e_df)

    if not parts:
        parts.append(build_builtin_demo_transactions())
        all_msgs.append("Fell back to built-in demo (no Elliptic rows).")

    tx_df = _ensure_unified(pd.concat(parts, ignore_index=True))

    cache = ofac_cache_path or os.path.join(
        os.path.abspath(data_dir), "ofac_sdn_download.csv"
    )
    ofac_addrs, ofac_msgs = load_ofac_sdn_addresses(cache_path=cache, timeout=45)
    all_msgs.extend(ofac_msgs)

    kb_path = known_bad_path or os.path.join(
        os.path.abspath(data_dir), "..", "known_bad_addresses.csv"
    )
    kb_path = os.path.normpath(kb_path)
    known = load_known_bad_csv(kb_path)

    sanctioned = set(ofac_addrs) | known
    return tx_df, sanctioned, all_msgs
