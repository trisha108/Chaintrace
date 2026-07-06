"""
Build a directed wallet graph from normalized transaction rows.

Nodes are addresses; directed edges are individual transfers (may be multi-edge).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

# Align with pattern_detector hourly windows for discrete time steps.
_TIME_BUCKET_SEC = 3600.0


def _row_time_step(row: pd.Series) -> int:
    raw = row.get("time_step")
    if raw is not None and not pd.isna(raw):
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    ts = float(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0.0
    return int(ts // _TIME_BUCKET_SEC)


def _new_node_attrs_from_row(row: pd.Series) -> Dict[str, object]:
    ts = _row_time_step(row)
    return {
        "time_step": ts,
        "pattern": str(row.get("pattern", "unknown") or "unknown"),
        "score": float(row.get("score", 0.0) or 0.0),
        "out_degree": 0,
        "in_degree": 0,
        "avg_amount": float(row.get("avg_amount", 0.0) or 0.0),
        "tx_frequency": float(row.get("tx_frequency", 0.0) or 0.0),
        "chains": "",
        "total_volume": 0.0,
        "tx_count": 0,
        "risk_score": float(row.get("score", 0.0) or 0.0),
        "active_time_steps": [],
    }


def _ensure_node(G: nx.DiGraph, wallet_id: str, row: pd.Series) -> None:
    if G.has_node(wallet_id):
        return
    a = _new_node_attrs_from_row(row)
    G.add_node(wallet_id, **a)


def recompute_node_metrics(G: nx.DiGraph, nodes: Optional[Iterable[str]] = None) -> None:
    """
    Refresh volume, chain string, tx counts, avg_amount, tx_frequency,
    active_time_steps, time_step, and directed degrees for all or selected nodes.
    """
    target = list(nodes) if nodes is not None else list(G.nodes)
    for n in target:
        if not G.has_node(n):
            continue
        vol = 0.0
        ch: set = set()
        cnt = 0
        amounts: list = []
        steps: set = set()

        for _, _, data in G.in_edges(n, data=True):
            vol += float(data.get("amount", 0))
            ch.add(str(data.get("chain", "")))
            cnt += 1
            amounts.append(float(data.get("amount", 0)))
            steps.add(int(float(data.get("timestamp", 0)) // _TIME_BUCKET_SEC))

        for _, _, data in G.out_edges(n, data=True):
            vol += float(data.get("amount", 0))
            ch.add(str(data.get("chain", "")))
            cnt += 1
            amounts.append(float(data.get("amount", 0)))
            steps.add(int(float(data.get("timestamp", 0)) // _TIME_BUCKET_SEC))

        G.nodes[n]["total_volume"] = vol
        G.nodes[n]["tx_count"] = cnt
        G.nodes[n]["chains"] = ",".join(sorted(x for x in ch if x))
        G.nodes[n]["avg_amount"] = float(np.mean(amounts)) if amounts else 0.0
        G.nodes[n]["tx_frequency"] = float(cnt)
        sorted_steps = sorted(steps)
        G.nodes[n]["active_time_steps"] = sorted_steps
        G.nodes[n]["time_step"] = sorted_steps[0] if sorted_steps else 0

        G.nodes[n]["out_degree"] = int(G.out_degree(n))
        G.nodes[n]["in_degree"] = int(G.in_degree(n))


def build_wallet_graph(tx_df: pd.DataFrame) -> nx.DiGraph:
    """
    Construct ``DiGraph`` where nodes are ``from_address`` / ``to_address`` strings.

    Node attributes include time_step, pattern, score, degrees, avg_amount,
    tx_frequency (plus chains, total_volume, tx_count for existing UI).
    """
    G = nx.DiGraph()
    if tx_df is None or tx_df.empty:
        return G

    for _, row in tx_df.iterrows():
        u = str(row["from_address"]).strip()
        v = str(row["to_address"]).strip()
        if not u or not v:
            continue
        amt = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
        ts = float(row["timestamp"]) if pd.notna(row["timestamp"]) else 0.0
        chain = str(row["chain"]) if pd.notna(row["chain"]) else "unknown"
        tx_id = str(row["tx_id"])

        _ensure_node(G, u, row)
        _ensure_node(G, v, row)

        G.add_edge(u, v, amount=amt, timestamp=ts, chain=chain, tx_id=tx_id)

    recompute_node_metrics(G)
    return G


def merge_live_rows(G: nx.DiGraph, new_rows: pd.DataFrame) -> nx.DiGraph:
    """Append live API rows into an existing graph (mutates and returns G)."""
    if new_rows is None or new_rows.empty:
        return G
    H = build_wallet_graph(new_rows)
    G = nx.compose(G, H)
    recompute_node_metrics(G)
    return G


def graph_to_feature_row(G: nx.DiGraph, address: str) -> Dict[str, float]:
    """Scalar features for ML / rules (degrees, volume, time span)."""
    if address not in G:
        return {
            "degree": 0,
            "in_degree": 0,
            "out_degree": 0,
            "volume": 0.0,
            "time_span": 0.0,
        }
    ts_list = []
    for _, _, d in G.in_edges(address, data=True):
        ts_list.append(float(d.get("timestamp", 0)))
    for _, _, d in G.out_edges(address, data=True):
        ts_list.append(float(d.get("timestamp", 0)))
    span = max(ts_list) - min(ts_list) if len(ts_list) > 1 else 0.0
    return {
        "degree": int(G.degree(address)),
        "in_degree": int(G.in_degree(address)),
        "out_degree": int(G.out_degree(address)),
        "volume": float(G.nodes[address].get("total_volume", 0)),
        "time_span": float(span),
    }
