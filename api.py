from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import threading

import data_loader
import graph_builder
import pattern_detector
import rss_monitor
import network as membership

app = FastAPI(title="FIU Detection Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

GLOBAL_STATE = {}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
KNOWN_BAD_CSV = os.path.join(ROOT_DIR, "known_bad_addresses.csv")


def load_data_background():
    tx_full, sanctioned, load_msgs = data_loader.load_all_datasets(
        DATA_DIR,
        demo_mode=True,
        known_bad_path=KNOWN_BAD_CSV
    )
    G = graph_builder.build_wallet_graph(tx_full)
    flagged = pattern_detector.run_detection(G, sanctioned)
    nets = pattern_detector.cluster_networks(
        G, flagged, data_dir=DATA_DIR)
    GLOBAL_STATE['tx_full'] = tx_full
    GLOBAL_STATE['sanctioned'] = sanctioned
    GLOBAL_STATE['G'] = G
    GLOBAL_STATE['flagged'] = flagged
    GLOBAL_STATE['nets'] = nets
    GLOBAL_STATE['ready'] = True


def get_state():
    if not GLOBAL_STATE.get('ready'):
        if not GLOBAL_STATE.get('loading'):
            GLOBAL_STATE['loading'] = True
            thread = threading.Thread(
                target=load_data_background,
                daemon=True
            )
            thread.start()
    return GLOBAL_STATE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def get_stats():
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    tx_full = GLOBAL_STATE['tx_full']
    flagged = GLOBAL_STATE['flagged']
    nets = GLOBAL_STATE['nets']
    G = GLOBAL_STATE['G']
    value = sum(
        float(G.nodes[w].get("total_volume", 0))
        for w in flagged
        if w in G.nodes
    )
    return {
        "transactions": len(tx_full),
        "flagged": len(flagged),
        "networks": len(nets),
        "value": value
    }


@app.get("/api/networks")
def get_networks(
    pattern: str = Query("all"),
    risk: str = Query("all"),
    mode: str = Query("demo")
):
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    nets = GLOBAL_STATE['nets']
    risk_f = risk.lower() if risk.lower() != "all" else None
    pat_f = pattern if pattern.lower() != "all" else "All"
    results = []
    for nid, inf in nets.items():
        score = float(inf.get("score", 0))
        nrisk = "high" if score >= 7 else "amber" if score >= 4 else "low"
        nrisk_mapped = "medium" if nrisk == "amber" else nrisk
        if risk_f is not None and nrisk_mapped != risk_f:
            continue
        pat_str = str(inf.get("pattern", "Unknown"))
        if pat_f != "All" and pat_f not in pat_str:
            continue
        results.append({
            "net_id": int(nid),
            "score": score,
            "risk": nrisk_mapped,
            "pattern": pat_str,
            "wallet_count": len(inf.get("members", [])),
            "lat": float(inf.get("lat", 0)),
            "lon": float(inf.get("lon", 0))
        })
    results.sort(key=lambda x: -x["score"])
    return results


@app.get("/api/network/{net_id}")
def get_network_detail(net_id: int):
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    nets = GLOBAL_STATE['nets']
    G = GLOBAL_STATE['G']
    if net_id not in nets:
        raise HTTPException(status_code=404, detail="Network not found")
    inf = nets[net_id]
    members = inf["members"]
    entry_ext = set()
    exit_ext = set()
    for m in members:
        for u, _, _ in G.in_edges(m, data=True):
            if u not in members:
                entry_ext.add(u)
        for _, v, _ in G.out_edges(m, data=True):
            if v not in members:
                exit_ext.add(v)
    score = float(inf.get("score", 0))
    nrisk = "high" if score >= 7 else "medium" if score >= 4 else "low"
    ofac_matches = []
    for m in members:
        if m in GLOBAL_STATE['flagged'] and \
                GLOBAL_STATE['flagged'][m].ofac_match:
            ofac_matches.append(m)
    return {
        "net_id": net_id,
        "score": score,
        "risk": nrisk,
        "pattern": str(inf.get("pattern", "Unknown")),
        "wallet_count": len(members),
        "value": float(inf.get("value", 0)),
        "confidence": min(1.0, score / 10.0 + 0.1),
        "entry_points": len(entry_ext),
        "exit_points": len(exit_ext),
        "first_seen": "Unknown",
        "wallets": list(members),
        "ofac_matches": ofac_matches,
        "signal_breakdown": {
            "cio": 0.4 * score,
            "behavioral": 0.5 * score,
            "temporal": 0.1 * score
        }
    }


@app.get("/api/globe-points")
def get_globe_points():
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    flagged = GLOBAL_STATE['flagged']
    nets = GLOBAL_STATE['nets']
    addr_to_net = {}
    for nid, inf in nets.items():
        for m in inf.get("members", []):
            if isinstance(m, str):
                addr_to_net[m] = int(nid)
    results = []
    for addr, fa in flagged.items():
        lat, lon = pattern_detector.address_lat_lon(addr)
        sc = float(fa.score)
        color = "#34d399"
        if fa.ofac_match:
            color = "#378ADD"
        elif sc >= 7:
            color = "#f87171"
        elif sc >= 4:
            color = "#fbbf24"
        base_s = 60000 if sc >= 7 else 35000 if sc >= 4 else 20000
        results.append({
            "address": addr,
            "lat": lat,
            "lon": lon,
            "color": color,
            "size": base_s,
            "score": sc,
            "pattern": fa.patterns[0].code if fa.patterns else "?",
            "net_id": addr_to_net.get(addr, -1)
        })
    return results


@app.get("/api/globe-edges/{net_id}")
def get_globe_edges(net_id: int):
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    nets = GLOBAL_STATE['nets']
    if net_id not in nets:
        raise HTTPException(status_code=404, detail="Network not found")
    inf = nets[net_id]
    members = sorted(
        m for m in inf["members"] if isinstance(m, str))[:40]
    coords = {}
    for m in members:
        coords[m] = pattern_detector.address_lat_lon(m)
    edges = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            lat1, lon1 = coords[members[i]]
            lat2, lon2 = coords[members[j]]
            edges.append({
                "start_lat": lat1,
                "start_lon": lon1,
                "end_lat": lat2,
                "end_lon": lon2
            })
    return edges


@app.get("/api/threat-intel")
def get_threat_intel():
    get_state()
    if not GLOBAL_STATE.get('ready'):
        return {"status": "loading",
                "message": "Data still loading, try again in 30 seconds"}
    items = rss_monitor.fetch_feed_items(5, timeout=5)
    results = []
    for it in items:
        results.append({
            "source": it.get("source", ""),
            "headline": it.get("title", ""),
            "time_ago": it.get("age", "")
        })
    return results