"""
Network membership confidence: behavioral similarity, common-input ownership, temporal correlation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

CONFIDENCE_THRESHOLD = 0.35


def _node_attrs(G: nx.DiGraph, w: str) -> Dict[str, Any]:
    if not G.has_node(w):
        return {}
    return dict(G.nodes[w])


def behavioral_similarity(
    w1: str, w2: str, G: nx.DiGraph
) -> Tuple[float, List[str]]:
    """
    Signal 1: Do these two wallets behave identically?
    Bots controlling multiple wallets run the same code
    so their wallets exhibit near-identical behavior.
    Returns similarity score 0.0 to 1.0
    """
    w1_data = _node_attrs(G, w1)
    w2_data = _node_attrs(G, w2)

    if not w1_data or not w2_data:
        return 0.0, []

    similarity = 0.0
    evidence: List[str] = []

    p1 = str(w1_data.get("pattern", "") or "")
    p2 = str(w2_data.get("pattern", "") or "")
    if p1 and p2 and p1 == p2 and p1 != "unknown":
        similarity += 0.30
        evidence.append(f"Same pattern: {p1}")

    d1_out = int(w1_data.get("out_degree", G.out_degree(w1)))
    d2_out = int(w2_data.get("out_degree", G.out_degree(w2)))
    if d1_out > 0 and d2_out > 0:
        ratio = min(d1_out, d2_out) / max(d1_out, d2_out)
        if ratio >= 0.8:
            similarity += 0.20
            evidence.append(f"Similar fan-out: {d1_out} vs {d2_out}")
    elif d1_out == 0 and d2_out == 0:
        similarity += 0.10

    d1_in = int(w1_data.get("in_degree", G.in_degree(w1)))
    d2_in = int(w2_data.get("in_degree", G.in_degree(w2)))
    if d1_in > 0 and d2_in > 0:
        ratio = min(d1_in, d2_in) / max(d1_in, d2_in)
        if ratio >= 0.8:
            similarity += 0.15
            evidence.append(f"Similar fan-in: {d1_in} vs {d2_in}")

    s1 = float(w1_data.get("score", 0))
    s2 = float(w2_data.get("score", 0))
    if s1 > 0 and s2 > 0:
        score_diff = abs(s1 - s2)
        if score_diff < 0.5:
            similarity += 0.20
            evidence.append(f"Similar risk score: {s1:.1f} vs {s2:.1f}")
        elif score_diff < 1.5:
            similarity += 0.10

    f1 = float(w1_data.get("tx_frequency", 0))
    f2 = float(w2_data.get("tx_frequency", 0))
    if f1 > 0 and f2 > 0:
        ratio = min(f1, f2) / max(f1, f2)
        if ratio >= 0.75:
            similarity += 0.15
            evidence.append(f"Similar tx frequency: {f1:.2f} vs {f2:.2f}")

    return min(similarity, 1.0), evidence


def get_active_timesteps(G: nx.DiGraph, wallet: str) -> Set[int]:
    """
    Returns the set of time steps when this wallet
    was active. Bot networks activate and deactivate
    their wallets in perfect synchrony.
    """
    node_data = _node_attrs(G, wallet)

    stored = node_data.get("active_time_steps")
    if stored is not None:
        try:
            return {int(x) for x in stored}
        except (TypeError, ValueError):
            pass

    timestep = node_data.get("time_step")
    if timestep is not None:
        try:
            return {int(timestep)}
        except (ValueError, TypeError):
            return set()

    active_steps: Set[int] = set()
    for neighbor in list(G.predecessors(wallet)) + list(G.successors(wallet)):
        nd = _node_attrs(G, neighbor)
        ts = nd.get("time_step")
        if ts is not None:
            try:
                active_steps.add(int(ts))
            except (ValueError, TypeError):
                continue
    return active_steps


def temporal_correlation(
    w1: str, w2: str, G: nx.DiGraph
) -> Tuple[float, List[str]]:
    """
    Signal 3: Do these wallets activate together?
    Bot-controlled wallets switch on and off in
    perfect synchrony because the same script
    controls them. Jaccard similarity of their
    active time step sets.
    Returns correlation score 0.0 to 1.0
    """
    steps1 = get_active_timesteps(G, w1)
    steps2 = get_active_timesteps(G, w2)

    if not steps1 or not steps2:
        return 0.0, []

    overlap = steps1.intersection(steps2)
    union = steps1.union(steps2)

    if not union:
        return 0.0, []

    score = len(overlap) / len(union)

    evidence: List[str] = []
    if score >= 0.8:
        evidence.append(
            f"Highly synchronized: active same {len(overlap)} time steps"
        )
    elif score >= 0.5:
        evidence.append(
            f"Partially synchronized: {len(overlap)} shared time steps"
        )

    return score, evidence


def load_synthetic_cio(data_dir: str) -> Dict[str, List[str]]:
    """
    Load the synthetic common input ownership dataset.
    In demo mode this simulates real Bitcoin co-signing
    data. In live mode this is replaced by real
    blockchain API data.
    """
    cio_path = os.path.join(data_dir, "synthetic_cio.csv")

    if not os.path.exists(cio_path):
        return {}

    try:
        df = pd.read_csv(cio_path)
        clusters: Dict[str, List[str]] = {}

        for _, row in df.iterrows():
            addrs: List[str] = []
            for col in ("input_addr_1", "input_addr_2", "input_addr_3"):
                if col in row.index and pd.notna(row[col]):
                    s = str(row[col]).strip()
                    if s:
                        addrs.append(s)

            if len(addrs) >= 2:
                tx_id = str(row.get("tx_id", "") or "")
                clusters[tx_id or f"row_{len(clusters)}"] = addrs

        return clusters

    except Exception as e:  # noqa: BLE001
        print(f"CIO load warning: {e}")
        return {}


def check_common_input_ownership(
    addr1: str, addr2: str, cio_clusters: Dict[str, List[str]]
) -> Tuple[bool, Optional[str]]:
    """
    Check if two wallets co-signed any transaction.
    If yes they are mathematically proven to be
    controlled by the same entity in Bitcoin.
    This is the strongest possible network membership
    signal — near 100% certainty.
    """
    if not cio_clusters:
        return False, None

    for tx_id, addrs in cio_clusters.items():
        if addr1 in addrs and addr2 in addrs:
            return True, tx_id

    return False, None


def network_membership_confidence(
    w1: str,
    w2: str,
    G: nx.DiGraph,
    cio_clusters: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Calculate confidence that two flagged wallets
    belong to the same criminal network.

    Three signals:
    Signal 1 - Behavioral similarity  (max +0.30)
    Signal 2 - Common input ownership (max +0.50)
    Signal 3 - Temporal correlation   (max +0.20)

    Total max score: 1.00
    Threshold for same network: 0.35
    """
    total_score = 0.0
    all_evidence: List[str] = []
    breakdown: Dict[str, float] = {}

    cio_score = 0.0
    cio_evidence: List[str] = []
    if cio_clusters:
        shared, tx_id = check_common_input_ownership(w1, w2, cio_clusters)
        if shared and tx_id is not None:
            cio_score = 0.50
            cio_evidence.append(
                f"Co-signed transaction {tx_id} "
                f"— mathematically same entity"
            )
    breakdown["signal_2_cio"] = cio_score
    all_evidence.extend(cio_evidence)
    total_score += cio_score

    sim_result = behavioral_similarity(w1, w2, G)
    if isinstance(sim_result, tuple):
        sim_score, sim_evidence = sim_result
    else:
        sim_score = float(sim_result)
        sim_evidence = []

    weighted_sim = sim_score * 0.30
    breakdown["signal_1_behavioral"] = weighted_sim
    all_evidence.extend(sim_evidence)
    total_score += weighted_sim

    temp_result = temporal_correlation(w1, w2, G)
    if isinstance(temp_result, tuple):
        temp_score, temp_evidence = temp_result
    else:
        temp_score = float(temp_result)
        temp_evidence = []

    weighted_temp = temp_score * 0.20
    breakdown["signal_3_temporal"] = weighted_temp
    all_evidence.extend(temp_evidence)
    total_score += weighted_temp

    final_score = min(total_score, 1.0)

    if final_score >= 0.70:
        strength = "Strong network evidence"
    elif final_score >= 0.40:
        strength = "Probable network"
    elif final_score >= 0.25:
        strength = "Weak association — monitor"
    else:
        strength = "Insufficient evidence"

    return {
        "confidence": round(final_score, 3),
        "strength": strength,
        "breakdown": breakdown,
        "evidence": all_evidence,
    }


def cluster_into_networks(
    flagged_wallets: Set[str],
    G: nx.DiGraph,
    cio_clusters: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Group flagged wallets into criminal networks
    using confidence-based clustering.
    Only wallets with confidence >= threshold
    get grouped together.
    """
    clusters: List[Dict[str, Any]] = []
    assigned: Set[str] = set()

    flagged_list = list(flagged_wallets)

    for i, w1 in enumerate(flagged_list):
        if w1 in assigned:
            continue

        cluster: Set[str] = {w1}
        cluster_evidence: Dict[Tuple[str, str], Dict[str, Any]] = {}
        assigned.add(w1)

        for w2 in flagged_list[i + 1 :]:
            if w2 in assigned:
                continue

            try:
                connected = nx.has_path(G.to_undirected(), w1, w2)
            except nx.NetworkXError:
                connected = False

            if not connected:
                continue

            result = network_membership_confidence(w1, w2, G, cio_clusters)

            conf = float(result["confidence"])

            if conf >= CONFIDENCE_THRESHOLD:
                cluster.add(w2)
                assigned.add(w2)
                cluster_evidence[(w1, w2)] = result

        clusters.append(
            {
                "wallets": cluster,
                "evidence": cluster_evidence,
            }
        )

    return clusters
