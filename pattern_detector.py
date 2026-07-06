"""
Wallet-level AML pattern rules on the unified transaction graph.

Outputs structured flags, 1–10 scores, and risk tiers (high / amber / low).
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

import data_loader
import graph_builder
import network as membership
import numpy as np
import pandas as pd

# Time window for "same window" fan in/out (seconds).
WINDOW_SEC = 3600.0
# Pass-through: receive then send within this many seconds.
PASS_THROUGH_SEC = 600.0
# Structuring: coefficient of variation threshold on outgoing amounts in a window.
STRUCTURING_CV_MAX = 0.08
# Peeling: detect descending large->smaller chain on out-edges in time order.
PEEL_MIN_STEPS = 3
# Mixer: unique in/out counterparties threshold
MIXER_IN_MIN = 4
MIXER_OUT_MIN = 4


@dataclass
class PatternHit:
    code: str
    description: str


@dataclass
class WalletAssessment:
    address: str
    score: int
    risk: str  # high | amber | low
    patterns: List[PatternHit] = field(default_factory=list)
    ofac_match: bool = False


def _risk_tier(score: int) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "amber"
    return "low"


def _score_from_patterns(patterns: List[PatternHit], ofac: bool) -> int:
    """Map qualitative hits to 1–10 scale.

    Weights:
      high pattern  → +6  (one high alone reaches 7 = high tier)
      medium pattern → +3  (one medium alone reaches 4 = amber tier)
      other          → +1
    """
    base = 1
    high_w = ("layering", "round-trip", "ofac", "mixer")
    med_w = (
        "fan-out",
        "fan-in",
        "pass-through",
        "dormancy",
        "structuring",
        "peeling",
    )
    for p in patterns:
        key = p.code.lower()
        if any(h in key for h in high_w):
            base += 6
        elif any(m in key for m in med_w):
            base += 3
        else:
            base += 1
    if ofac:
        base = max(base, 9)
    return int(max(1, min(10, base)))


def _edges_by_time(G: nx.DiGraph, n: str) -> List[Tuple[str, str, float, float, str]]:
    """List (u,v,ts,amt,chain) for edges incident to n."""
    out = []
    for _, v, d in G.out_edges(n, data=True):
        out.append(
            (
                n,
                v,
                float(d.get("timestamp", 0)),
                float(d.get("amount", 0)),
                str(d.get("chain", "")),
            )
        )
    for u, _, d in G.in_edges(n, data=True):
        out.append(
            (
                u,
                n,
                float(d.get("timestamp", 0)),
                float(d.get("amount", 0)),
                str(d.get("chain", "")),
            )
        )
    out.sort(key=lambda x: x[2])
    return out


def detect_fan_out(G: nx.DiGraph, n: str) -> bool:
    by_win: Dict[int, Set[str]] = defaultdict(set)
    for _, v, d in G.out_edges(n, data=True):
        ts = float(d.get("timestamp", 0))
        wk = int(ts // WINDOW_SEC)
        by_win[wk].add(v)
    return any(len(s) >= 5 for s in by_win.values())


def detect_fan_in(G: nx.DiGraph, n: str) -> bool:
    by_win: Dict[int, Set[str]] = defaultdict(set)
    for u, _, d in G.in_edges(n, data=True):
        ts = float(d.get("timestamp", 0))
        wk = int(ts // WINDOW_SEC)
        by_win[wk].add(u)
    return any(len(s) >= 5 for s in by_win.values())


def detect_pass_through(G: nx.DiGraph, n: str) -> bool:
    times_in = sorted(float(d["timestamp"]) for _, _, d in G.in_edges(n, data=True))
    times_out = sorted(float(d["timestamp"]) for _, _, d in G.out_edges(n, data=True))
    if not times_in or not times_out:
        return False
    for ti in times_in:
        for to in times_out:
            if 0 < to - ti < PASS_THROUGH_SEC and int(G.out_degree(n)) <= 3:
                return True
    return False


def detect_layering(G: nx.DiGraph, n: str) -> bool:
    """Four or more hops along strictly time-increasing edges."""
    best = 0
    stack: List[Tuple[str, int, float]] = [(n, 0, -1e30)]
    while stack:
        node, depth, last_t = stack.pop()
        best = max(best, depth)
        if depth > 24:
            continue
        for _, v, d in G.out_edges(node, data=True):
            t = float(d.get("timestamp", 0))
            if t > last_t:
                stack.append((v, depth + 1, t))
    return best >= 4


def detect_dormancy_burst(G: nx.DiGraph, n: str) -> bool:
    events = _edges_by_time(G, n)
    if len(events) < 3:
        return False
    times = sorted({e[2] for e in events})
    gaps = [b - a for a, b in zip(times, times[1:])]
    if not gaps:
        return False
    if max(times) - min(times) < WINDOW_SEC * 5:
        # use gap between first cluster and burst
        if gaps and max(gaps) >= WINDOW_SEC * 5:
            return True
    return False


def detect_round_trip(G: nx.DiGraph, n: str, max_depth: int = 8) -> bool:
    """Detect if n can reach a predecessor (cycle) via BFS."""
    roots = {u for u, _ in G.in_edges(n)}
    if not roots:
        return False
    seen = {n}
    stack = list(G.successors(n))
    depth_map = {s: 1 for s in stack}
    while stack:
        cur = stack.pop()
        dpt = depth_map.get(cur, 0)
        if cur in roots and dpt >= 2:
            return True
        if dpt >= max_depth:
            continue
        for w in G.successors(cur):
            if w not in seen:
                seen.add(w)
                depth_map[w] = dpt + 1
                stack.append(w)
    return False


def detect_structuring(G: nx.DiGraph, n: str) -> bool:
    by_win: Dict[int, List[float]] = defaultdict(list)
    for _, v, d in G.out_edges(n, data=True):
        ts = float(d["timestamp"])
        wk = int(ts // WINDOW_SEC)
        by_win[wk].append(float(d["amount"]))
    for amts in by_win.values():
        if len(amts) < 5:
            continue
        arr = np.array(amts, dtype=np.float64)
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        if mu > 0 and sd / mu < STRUCTURING_CV_MAX:
            return True
    return False


def detect_peeling_chain(G: nx.DiGraph, n: str) -> bool:
    """Sequential spends with decreasing amounts (same chain of outflows)."""
    outs = sorted(
        (
            (float(d["timestamp"]), float(d["amount"]), v)
            for _, v, d in G.out_edges(n, data=True)
        ),
        key=lambda x: x[0],
    )
    if len(outs) < PEEL_MIN_STEPS:
        return False
    dec = 0
    for i in range(1, len(outs)):
        if outs[i][1] < outs[i - 1][1] * 0.85:
            dec += 1
    return dec >= PEEL_MIN_STEPS - 1


def detect_mixer_hub(G: nx.DiGraph, n: str) -> bool:
    ins = {u for u, _ in G.in_edges(n)}
    outs = set(G.successors(n))
    if len(ins) < MIXER_IN_MIN or len(outs) < MIXER_OUT_MIN:
        return False
    amts_in = [float(d["amount"]) for _, _, d in G.in_edges(n, data=True)]
    amts_out = [float(d["amount"]) for _, _, d in G.out_edges(n, data=True)]
    if not amts_in or not amts_out:
        return False
    cv_in = float(np.std(amts_in)) / (float(np.mean(amts_in)) + 1e-9)
    cv_out = float(np.std(amts_out)) / (float(np.mean(amts_out)) + 1e-9)
    return cv_in > 0.4 and cv_out > 0.4


def assess_wallet(
    G: nx.DiGraph,
    address: str,
    sanctioned: Set[str],
) -> WalletAssessment:
    """Run all detectors for one address."""
    hits: List[PatternHit] = []
    ofac = address in sanctioned or address.lower() in {s.lower() for s in sanctioned}

    if ofac:
        hits.append(
            PatternHit(
                "OFAC match",
                "Address string matches a token from the OFAC SDN list or known bad list.",
            )
        )
    if G.has_node(address):
        if detect_fan_out(G, address):
            hits.append(
                PatternHit(
                    "Fan-out",
                    "Ten or more distinct counterparties received funds within the same hourly window.",
                )
            )
        if detect_fan_in(G, address):
            hits.append(
                PatternHit(
                    "Fan-in",
                    "Ten or more distinct senders paid this wallet within the same hourly window.",
                )
            )
        if detect_pass_through(G, address):
            hits.append(
                PatternHit(
                    "Pass-through",
                    "Funds arrived and left again within minutes, consistent with rapid pass-through.",
                )
            )
        if detect_layering(G, address):
            hits.append(
                PatternHit(
                    "Layering chain",
                    "Five or more sequential hops with strictly increasing timestamps downstream.",
                )
            )
        if detect_dormancy_burst(G, address):
            hits.append(
                PatternHit(
                    "Dormancy burst",
                    "Long quiet period followed by a burst of new activity.",
                )
            )
        if detect_round_trip(G, address):
            hits.append(
                PatternHit(
                    "Round-trip",
                    "Funds appear to cycle back toward prior senders through intermediaries.",
                )
            )
        if detect_structuring(G, address):
            hits.append(
                PatternHit(
                    "Structuring",
                    "Multiple near-identical outgoing amounts in the same window (smurfing pattern).",
                )
            )
        if detect_peeling_chain(G, address):
            hits.append(
                PatternHit(
                    "Peeling chain",
                    "Repeated spends with stepped-down amounts across sequential hops.",
                )
            )
        if detect_mixer_hub(G, address):
            hits.append(
                PatternHit(
                    "Mixer hub",
                    "Many unrelated sources and sinks with highly variable amounts (mixer-like hub).",
                )
            )

    score = _score_from_patterns(hits, ofac)
    return WalletAssessment(
        address=address,
        score=score,
        risk=_risk_tier(score),
        patterns=hits,
        ofac_match=ofac,
    )


def detect_on_new_tx(
    G: nx.DiGraph,
    tx_row: Dict[str, Any],
    sanctioned: Set[str],
    already_flagged: Set[str],
) -> List[WalletAssessment]:
    """
    Merge one transaction into G and return assessments for endpoints that are
    newly flagged (not in already_flagged). Mutates G and already_flagged.
    """
    df = data_loader._ensure_unified(pd.DataFrame([tx_row]))
    graph_builder.merge_live_rows(G, df)
    out: List[WalletAssessment] = []
    for addr in (
        str(tx_row.get("from_address", "")).strip(),
        str(tx_row.get("to_address", "")).strip(),
    ):
        if not addr or addr.startswith("unknown"):
            continue
        if not G.has_node(addr):
            continue
        if int(G.degree(addr)) < 1:
            continue
        a = assess_wallet(G, addr, sanctioned)
        if (a.patterns or a.ofac_match) and addr not in already_flagged:
            out.append(a)
            already_flagged.add(addr)
    return out


def run_detection(
    G: nx.DiGraph,
    sanctioned: Set[str],
    min_degree: int = 1,
) -> Dict[str, WalletAssessment]:
    """Assess every node that has at least one edge."""
    out: Dict[str, WalletAssessment] = {}
    for n in G.nodes:
        if int(G.degree(n)) < min_degree:
            continue
        a = assess_wallet(G, n, sanctioned)
        if a.patterns or a.ofac_match:
            out[n] = a
    return out


def address_lat_lon(address: str) -> Tuple[float, float]:
    """Stable pseudo coordinates for any address string (for globe arcs)."""
    h = hashlib.md5(address.encode()).hexdigest()
    lat = int(h[:4], 16) / 0xFFFF * 160 - 80
    lon = int(h[4:8], 16) / 0xFFFF * 360 - 180
    return float(lat), float(lon)


def apply_assessments_to_graph(
    G: nx.DiGraph, flagged: Dict[str, WalletAssessment]
) -> None:
    """Copy dominant pattern and risk score onto graph nodes for membership signals."""
    for addr, a in flagged.items():
        if not G.has_node(addr):
            continue
        dom = a.patterns[0].code if a.patterns else "unknown"
        G.nodes[addr]["pattern"] = str(dom)
        G.nodes[addr]["score"] = float(a.score)
        G.nodes[addr]["risk_score"] = float(a.score)
    graph_builder.recompute_node_metrics(G)


def _cluster_networks_legacy(
    G: nx.DiGraph,
    flagged: Dict[str, WalletAssessment],
) -> Dict[int, Dict[str, object]]:
    """Graph-edge components among flagged wallets (fallback)."""
    U = nx.Graph()
    for a in flagged:
        U.add_node(a)
    for u, v in G.edges():
        if u in flagged and v in flagged:
            U.add_edge(u, v)

    nets: Dict[int, Dict[str, object]] = {}
    for i, comp in enumerate(nx.connected_components(U), start=1):
        members = set(comp)
        scores = [flagged[m].score for m in members if m in flagged]
        best = max(scores) if scores else 1
        pats = []
        for m in members:
            pats.extend(h.code for h in flagged[m].patterns)
        dom = Counter(pats).most_common(1)
        pattern = dom[0][0] if dom else "Unknown"
        val = sum(float(G.nodes[m].get("total_volume", 0)) for m in members)
        h = hashlib.md5(f"net{i}".encode()).hexdigest()
        lat = int(h[:4], 16) / 0xFFFF * 160 - 80
        lon = int(h[4:8], 16) / 0xFFFF * 360 - 180
        nets[i] = {
            "members": members,
            "score": float(best),
            "pattern": pattern,
            "value": val,
            "lat": lat,
            "lon": lon,
            "city": f"Grid sector {i:03d}",
            "pair_evidence": {},
        }
    return nets


def cluster_networks(
    G: nx.DiGraph,
    flagged: Dict[str, WalletAssessment],
    data_dir: Optional[str] = None,
    live_cio_clusters: Optional[Dict[str, List[str]]] = None,
) -> Dict[int, Dict[str, object]]:
    """
    Confidence-based groups among flagged wallets (path + membership score),
    else legacy edge-components if no multi-wallet confidence cluster exists.

    If ``live_cio_clusters`` is not None (live mode), Signal 2 uses only that
    dict (blockchain CIO). Otherwise synthetic CIO is loaded from ``data_dir``.

    Returns:
      network_id -> {members, score, pattern, value, lat, lon, city, pair_evidence}
    """
    if not flagged:
        return {}

    apply_assessments_to_graph(G, flagged)

    cio: Dict[str, List[str]] = {}
    if live_cio_clusters is not None:
        cio = {str(k): list(v) for k, v in live_cio_clusters.items()}
    elif data_dir:
        cio = membership.load_synthetic_cio(os.path.abspath(data_dir))

    raw_clusters = membership.cluster_into_networks(
        set(flagged.keys()), G, cio_clusters=cio or None
    )
    multi = [c for c in raw_clusters if len(c["wallets"]) >= 2]

    if not multi:
        return _cluster_networks_legacy(G, flagged)

    nets: Dict[int, Dict[str, object]] = {}
    for i, c in enumerate(multi, start=1):
        members: Set[str] = set(c["wallets"])
        scores = [flagged[m].score for m in members if m in flagged]
        best = max(scores) if scores else 1
        pats: List[str] = []
        for m in members:
            pats.extend(h.code for h in flagged[m].patterns)
        dom = Counter(pats).most_common(1)
        pattern = dom[0][0] if dom else "Unknown"
        val = sum(float(G.nodes[m].get("total_volume", 0)) for m in members)
        h = hashlib.md5(f"net{i}".encode()).hexdigest()
        lat = int(h[:4], 16) / 0xFFFF * 160 - 80
        lon = int(h[4:8], 16) / 0xFFFF * 360 - 180

        pair_evidence: Dict[str, object] = {}
        for (wa, wb), res in c["evidence"].items():
            pair_evidence[f"{wa}||{wb}"] = res

        nets[i] = {
            "members": members,
            "score": float(best),
            "pattern": pattern,
            "value": val,
            "lat": lat,
            "lon": lon,
            "city": f"Grid sector {i:03d}",
            "pair_evidence": pair_evidence,
        }
    return nets
