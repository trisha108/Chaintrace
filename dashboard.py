"""
FIU ChainTrace — single-page investigation dashboard (Streamlit + Plotly globe).

Layout: one viewport (no page scroll), globe center, fixed ticker.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="FIU AML Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THE LOCKDOWN CSS ---
st.markdown("""
    <style>
    /* Kill the global browser scrollbar entirely */
    html, body, [data-testid="stAppViewContainer"], .stApp, .main {
        overflow: hidden !important;
        height: 100vh !important;
        background: transparent !important;
    }

    /* Remove the 'Giant Gap' at the top of Streamlit */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: -2rem !important; /* Pulls the header and globe up */
        height: 100vh !important;
    }

    /* Target the main vertical block to remove internal spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    /* Ensure the sidebar doesn't trigger a page scroll */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1rem !important;
    }

    /* 1. FORCE THE SIDEBAR TO THE FRONT */
    body:has([data-testid="stSidebar"]) [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        z-index: 999999 !important;
        position: fixed !important;
        visibility: visible !important;
        background-color: #0d1117 !important; /* Ensure it's solid so stars don't bleed through text */
    }

    /* 2. KILL THE "LATE-LOAD" DIMMING OVERLAY */
    /* This targets the Streamlit 'fade-in' containers that cause the dimming after 2 seconds */
    .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
        background: transparent !important;
        background-color: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        opacity: 1 !important;
    }

    /* 3. ENSURE THE STARFIELD REMAINS IN THE BASEMENT */
    /* Force the Starfield component to the absolute back */
    iframe[title="streamlit_components.v1.html"],
    [data-testid="element-container"]:has(iframe[title="streamlit_components.v1.html"]) {
        z-index: -1 !important;
        position: fixed !important;
        pointer-events: none !important;
    }

    /* Bring right panel forward so it's not blocked by globe/starfield */
    [data-testid="column"]:nth-of-type(2) {
        z-index: 100 !important;
        position: relative !important;
    }
    
    /* Hide the footer and top decoration bar */
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    div[data-testid="stToolbar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


def _import_app_module(name: str):
    """Load a repo-local module; retry on KeyError.

    Streamlit's file watcher can delete ``sys.modules`` entries for watched
    files while an import is still finishing, which makes importlib's internal
    ``sys.modules.pop(name)`` raise KeyError. Retrying the import is safe.
    """
    last: Optional[BaseException] = None
    for _ in range(8):
        try:
            return importlib.import_module(name)
        except KeyError as e:
            last = e
            continue
    if last is not None:
        raise last
    raise RuntimeError(f"failed to import {name!r}")


data_loader = _import_app_module("data_loader")
graph_builder = _import_app_module("graph_builder")
live_api = _import_app_module("live_api")
pdet = _import_app_module("pattern_detector")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
KNOWN_BAD_CSV = os.path.join(ROOT_DIR, "known_bad_addresses.csv")

GLOBE_FIG_HEIGHT = 1080

def _starfield_canvas_html() -> str:
    """Canvas starfield for ``components.html`` (script runs in iframe only)."""
    return """<!DOCTYPE html><html><head><style>
html,body{margin:0;padding:0;width:100%;height:100%;background:transparent;overflow:hidden;}
</style></head><body>
<canvas id="fiu-star-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;display:block;"></canvas>
<script>
(function() {
  var canvas = document.getElementById('fiu-star-canvas');
  var ctx = canvas.getContext('2d');
  function paint() {
    var w = window.innerWidth || 800;
    var h = window.innerHeight || 800;
    canvas.width = w;
    canvas.height = h;
    ctx.fillStyle = '#0d1f35';
    ctx.fillRect(0, 0, w, h);
    for (var i = 0; i < 300; i++) {
      var x = Math.random() * w;
      var y = Math.random() * h;
      var radius = Math.random() * 2 + 0.5;
      var opacity = Math.random() * 0.9 + 0.3;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,' + opacity + ')';
      ctx.fill();
    }
  }
  paint();
  window.addEventListener('resize', paint);
})();
</script>
</body></html>"""

GLOBE_GLOW_HTML = """
<div style="
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(30, 80, 160, 0.15) 0%,
        rgba(10, 30, 80, 0.08) 50%,
        transparent 70%
    );
    pointer-events: none;
    z-index: 2;
">
</div>
"""

# Left panel pattern pills: query-string value -> PATTERN_OPTIONS value
LEFT_PILL_Q_TO_OPTION: Dict[str, str] = {
    "All": "All",
    "Fan-out": "Fan-out",
    "Fan-in": "Fan-in",
    "Layering": "Layering chain",
    "Pass-thru": "Pass-through",
    "Mixer": "Mixer hub",
}

PATTERN_OPTIONS = [
    "All",
    "Fan-out",
    "Fan-in",
    "Pass-through",
    "Layering chain",
    "Dormancy burst",
    "Round-trip",
    "Structuring",
    "Peeling chain",
    "Mixer hub",
]

PATTERN_BLURBS: Dict[str, str] = {
    "All": "Show every flagged wallet regardless of heuristic hit.",
    "Fan-out": "Many distinct recipients funded in the same time window — rapid distribution.",
    "Fan-in": "Many distinct senders paying one wallet in the same window — consolidation.",
    "Pass-through": "Funds in and out within minutes with few hops — laundering relay.",
    "Layering chain": "Long sequence of strictly time-ordered onward transfers.",
    "Dormancy burst": "Quiet period followed by a burst of new activity.",
    "Round-trip": "Paths that cycle funds back toward earlier senders.",
    "Structuring": "Many similar-sized outflows in the same window — smurfing pattern.",
    "Peeling chain": "Sequential spends with stepped-down amounts along a chain.",
    "Mixer hub": "High in/out fan with highly variable amounts — mixer-like hub.",
}

# Component classes (sidebar styled in main() CSS).
LAYOUT_CSS = """
<style>
.fiu-flag-card {
  margin-bottom: 8px; padding: 10px 10px 10px 12px; border-radius: 8px;
  background: rgba(15,25,35,0.85); border: 0.5px solid #1e2d3d;
  border-left: 3px solid var(--fiu-b, #34d399);
  transition: background 0.15s ease, border-color 0.15s ease;
}
.fiu-flag-card:hover { background: rgba(25,38,52,0.95); border-color: #2d4a6b; }
.fiu-flag-card-inner { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.fiu-flag-addr { font-size: 10px; font-family: ui-monospace, monospace; color: #d0e8ff;
  word-break: break-all; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.fiu-flag-badge { font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 4px; flex-shrink: 0; }
.fiu-flag-meta { font-size: 10px; color: #64748b; margin-top: 6px; }
.fiu-new {
  display: inline-block; font-size: 8px; background: #ffffff; color: #000510;
  padding: 1px 4px; margin-left: 4px; border-radius: 2px; animation: flash 1s infinite;
}
@keyframes flash { 50% { opacity: 0.35; } }
.fiu-profile-card {
  background: rgba(15, 25, 35, 0.6); border: 0.5px solid #1e2d3d; border-radius: 10px;
  padding: 14px; margin-bottom: 12px;
}
.fiu-stat-row { display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 0.5px solid rgba(30, 45, 61, 0.5); font-size: 12px; }
.fiu-stat-row:last-child { border-bottom: none; }
.fiu-stat-lbl { color: #6a9abf; }
.fiu-stat-val { color: #85C7FF; font-weight: 600; }
.fiu-nav-logo { color: #ffffff !important; font-size: 15px; font-weight: 800; letter-spacing: 0.12em; }
.fiu-nav-btn-wrap { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; justify-content: center; }
.fiu-nav-counters {
  font-size: 11px; color: #6a9abf; text-align: right; line-height: 1.5;
}
.fiu-nav-counters strong { color: #85C7FF; }
.fiu-badge-demo { color: #94a3b8; font-size: 10px; padding: 3px 8px; border: 0.5px solid #1e2d3d; border-radius: 6px; }
.fiu-badge-live { color: #34d399; font-size: 10px; padding: 3px 8px; border: 0.5px solid #1e2d3d; border-radius: 6px;
  animation: fiuPulse 2s infinite; }
@keyframes fiuPulse { 50% { opacity: 0.55; } }
</style>
"""


def add_simulation_steps(tx_df: pd.DataFrame, max_steps: int = 5) -> pd.DataFrame:
    if tx_df.empty:
        tx_df = tx_df.copy()
        tx_df["_step"] = []
        return tx_df
    df = tx_df.copy().reset_index(drop=True)
    n = len(df)
    df = df.sort_values("timestamp")
    df["_step"] = np.clip((np.arange(n) * max_steps // max(n, 1)) + 1, 1, max_steps)
    return df.reset_index(drop=True)


def cached_base_transactions(demo_mode: bool, data_dir: str) -> Tuple[pd.DataFrame, Set[str], List[str]]:
    """One load per Streamlit session per (demo_mode, data_dir).

    Intentionally not ``@st.cache_data``: Streamlit tokenizes ``inspect.getsource``
    of cached functions; a stray or mismatched triple-quoted string elsewhere in
    this large module can surface as ``TokenError: EOF in multi-line string`` at
    an unrelated line. Session cache avoids that machinery.
    """
    key = (bool(demo_mode), str(data_dir))
    slot = st.session_state.get("_fiu_base_tx_cache")
    if isinstance(slot, dict) and slot.get("key") == key:
        out = slot.get("result")
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2]
    with st.spinner("Loading datasets and sanctions data (cached)…"):
        res = data_loader.load_all_datasets(
            data_dir,
            demo_mode=demo_mode,
            known_bad_path=KNOWN_BAD_CSV,
        )
    st.session_state["_fiu_base_tx_cache"] = {"key": key, "result": res}
    return res


def slice_by_sim_step(tx_full: pd.DataFrame, sim_step: int) -> pd.DataFrame:
    if "_step" not in tx_full.columns:
        return tx_full
    return tx_full[tx_full["_step"] <= sim_step].copy()


def _empty_tx_df() -> pd.DataFrame:
    return pd.DataFrame(columns=data_loader.UNIFIED_COLUMNS)


def network_entry_exit_counts(G: nx.DiGraph, members: Set[str]) -> Tuple[int, int]:
    """External wallets with edges into / out of the flagged cluster."""
    mem = set(members)
    entry_ext: Set[str] = set()
    exit_ext: Set[str] = set()
    for m in mem:
        for u, _, _ in G.in_edges(m, data=True):
            if u not in mem:
                entry_ext.add(u)
        for _, v, _ in G.out_edges(m, data=True):
            if v not in mem:
                exit_ext.add(v)
    return len(entry_ext), len(exit_ext)


def network_growth_status(
    nid: int,
    member_count: int,
    new_networks: Set[int],
    prev_sizes: Dict[int, int],
) -> str:
    if nid in new_networks:
        return "Newly detected"
    prev = prev_sizes.get(nid)
    if prev is not None and member_count > prev:
        return "Growing"
    if prev is not None and member_count < prev:
        return "Contracting"
    return "Stable"


def network_risk_label(score: float) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "amber"
    return "low"


def _network_notification_chrome(
    net_id: int,
    score: float,
    wallet_count: int,
    pattern: str,
) -> None:
    esc_pat = (
        pattern.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f"""
<div style="
  position:fixed; bottom:50px; right:20px;
  background:#0f1923; border:0.5px solid #f87171;
  border-left:3px solid #f87171; border-radius:8px;
  padding:10px 14px; color:#e2e8f0; font-size:12px; z-index:9999;
  min-width:220px;
  animation: fiuSlideIn 0.3s ease, fiuFadeOut 0.5s ease 3.5s forwards;
">
  <div style="color:#f87171;font-weight:500;margin-bottom:4px">New network detected</div>
  <div>Network #{net_id:03d} · Score {score:.1f}</div>
  <div style="color:#4a6b8a;font-size:10px;margin-top:3px">
    {wallet_count} wallets · {esc_pat}
  </div>
</div>
<style>
@keyframes fiuSlideIn {{
  from {{ transform: translateX(100px); opacity:0; }}
  to {{ transform: translateX(0); opacity:1; }}
}}
@keyframes fiuFadeOut {{
  to {{ opacity: 0; transform: translateX(100px); }}
}}
</style>
<script>
(function(){{
const ctx = new (window.AudioContext || window.webkitAudioContext)();
function beep(freq, duration, vol) {{
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.connect(g);
  g.connect(ctx.destination);
  o.frequency.value = freq;
  g.gain.setValueAtTime(vol, ctx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
  o.start(ctx.currentTime);
  o.stop(ctx.currentTime + duration);
}}
beep(440, 0.15, 0.3);
setTimeout(function() {{ beep(660, 0.2, 0.3); }}, 150);
setTimeout(function() {{ beep(880, 0.3, 0.3); }}, 300);
}})();
</script>
"""
    components.html(html, height=1, width=1, scrolling=False)


MAJOR_CITIES: Dict[str, Tuple[float, float]] = {
    "New York": (40.7, -74.0),
    "London": (51.5, -0.1),
    "Tokyo": (35.6, 139.6),
    "Moscow": (55.7, 37.6),
    "Dubai": (25.2, 55.3),
    "Singapore": (1.3, 103.8),
    "Lagos": (6.5, 3.4),
    "São Paulo": (-23.5, -46.6),
    "Shanghai": (31.2, 121.5),
    "Mumbai": (19.1, 72.9),
}


def _wcoords(wid: str) -> Tuple[float, float]:
    h = hashlib.md5(str(wid).encode()).hexdigest()
    lat = (int(h[0:4], 16) / 65535) * 160 - 80
    lon = (int(h[4:8], 16) / 65535) * 360 - 180
    return lat, lon


def _fiu_qp_first(key: str) -> Optional[str]:
    if key not in st.query_params:
        return None
    v = st.query_params[key]
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else None
    return str(v) if v is not None else None


def main() -> None:
    print("SEL:", st.session_state.get("selected_network"))
    st.markdown(
        """
<style>
/* Keep the dark theme but remove the aggressive structural overrides */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background: #0d1f35 !important;
    color: #d0e8ff !important;
}
[data-testid="stSidebar"] {
    background: rgba(13, 31, 53, 0.96) !important;
    border-right: 0.5px solid #2d5a8a !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: rgba(13, 31, 53, 0.96) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(15,25,35,0.85) !important;
    border: 0.5px solid #1e2d3d !important;
    border-radius: 8px !important;
    color: #d0e8ff !important;
    text-align: left !important;
    font-size: 11px !important;
    padding: 8px 10px 8px 12px !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    height: auto !important;
    line-height: 1.4 !important;
    white-space: pre-line !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(25,38,52,0.95) !important;
    border-color: #2d4a6b !important;
}
[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
    border-color: #2d5a8a !important;
    box-shadow: none !important;
}
iframe[title*="autorefresh" i], iframe[title*="streamlit_autorefresh" i] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(LAYOUT_CSS, unsafe_allow_html=True)

    if "background_threads_started" not in st.session_state:
        threading.Thread(target=live_api.live_feed_worker, daemon=True).start()
        threading.Thread(target=live_api.intel_feed_worker, daemon=True).start()
        st.session_state.background_threads_started = True

    defaults = {
        "demo_mode": True,
        "live_mode": False,
        "_inited_scan": False,
        "current_step": 1,
        "max_demo_steps": 5,
        "demo_auto_play": True,
        "selected_wallet": None,
        "selected_network": None,  # full dict once a network is chosen
        "_sel_net_id": None,       # int ID used for nets lookup
        "auto_advance": True,
        "new_wallets": set(),
        "new_networks": set(),
        "tx_accum": None,
        "new_detections": [],
        "prev_net_ids": set(),
        "prev_flagged": set(),
        "intel_items": [],
        "net_sizes": {},
        "known_networks": set(),
        "networks_sound_seeded": False,
        "globe_selected_net": None,
        "live_cio_clusters": {},
        "menu_risk": "All",
        "custom_panel_pattern": "All",
        "fiu_sidebar_pattern": "All",
        "wallet_coords": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _url_pat = _fiu_qp_first("fiu_pat")
    if _url_pat:
        _opt = LEFT_PILL_Q_TO_OPTION.get(_url_pat)
        if _opt in PATTERN_OPTIONS:
            st.session_state.custom_panel_pattern = _opt
            st.session_state.fiu_sidebar_pattern = _opt

    while not live_api.intel_queue.empty():
        try:
            st.session_state.intel_items = live_api.intel_queue.get_nowait()
        except queue.Empty:
            break

    prev_net_sizes: Dict[int, int] = dict(st.session_state.get("net_sizes", {}))

    if st.session_state.tx_accum is None:
        st.session_state.tx_accum = _empty_tx_df()

    if "sim_step" in st.session_state and "current_step_migrated" not in st.session_state:
        st.session_state.current_step = int(st.session_state.sim_step)
        st.session_state.current_step_migrated = True

    effective_demo = bool(st.session_state.demo_mode) and not bool(
        st.session_state.live_mode
    )

    try:
        tx_full, sanctioned, load_msgs = cached_base_transactions(
            effective_demo, DATA_DIR
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Load error: {e}")
        st.stop()

    tx_full = add_simulation_steps(tx_full, int(st.session_state.max_demo_steps))
    max_s = int(st.session_state.max_demo_steps)
    step = int(st.session_state.current_step) if effective_demo else max_s
    step = max(1, min(max_s, step))
    st.session_state.current_step = step
    tx_base = slice_by_sim_step(tx_full, step)

    tx_accum: pd.DataFrame = st.session_state.tx_accum
    if tx_accum is None or tx_accum.empty:
        tx_live_part = _empty_tx_df()
    else:
        tx_live_part = tx_accum

    G_pre = graph_builder.build_wallet_graph(
        pd.concat([tx_base, tx_live_part], ignore_index=True)
        if not tx_live_part.empty
        else tx_base
    )

    new_flags_batch: List[Any] = []
    drained: List[Dict[str, Any]] = []
    if st.session_state.live_mode and not effective_demo:
        already = set(pdet.run_detection(G_pre, sanctioned).keys())
        G_work = G_pre.copy()
        lcio: Dict[str, Any] = st.session_state.live_cio_clusters
        if not isinstance(lcio, dict):
            lcio = {}
            st.session_state.live_cio_clusters = lcio
        while not live_api.tx_queue.empty():
            try:
                item = live_api.tx_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("type") == "cio":
                tx_id = str(item.get("tx_id", "") or "")
                addrs = item.get("addresses") or []
                if tx_id and isinstance(addrs, list):
                    clean = [str(a).strip() for a in addrs if str(a).strip()]
                    if len(clean) >= 2:
                        if tx_id not in lcio:
                            lcio[tx_id] = clean
                        else:
                            ex = set(lcio[tx_id])
                            for a in clean:
                                if a not in ex:
                                    ex.add(a)
                                    lcio[tx_id].append(a)
                continue
            drained.append(item)
            nf = pdet.detect_on_new_tx(G_work, item, sanctioned, already)
            new_flags_batch.extend(nf)
        if drained:
            st.session_state.tx_accum = pd.concat(
                [tx_live_part, pd.DataFrame(drained)], ignore_index=True
            )
            tx_live_part = st.session_state.tx_accum
        G = G_work if drained else G_pre
    else:
        G = G_pre

    if drained and new_flags_batch:
        st.session_state.new_detections.extend(new_flags_batch)
        for fa in new_flags_batch:
            st.session_state.new_wallets.add(fa.address)

    tx = (
        pd.concat([tx_base, tx_live_part], ignore_index=True)
        if not tx_live_part.empty
        else tx_base
    )
    if st.session_state.live_mode and not effective_demo:
        G = graph_builder.build_wallet_graph(tx)

    # Cache key: detection result doesn't depend on which network is selected,
    # only on the underlying data. Skip expensive recompute on selection reruns.
    _det_key = (
        int(step),
        bool(effective_demo),
        len(tx),
        len(st.session_state.get("live_cio_clusters") or {}),
    )
    _det_cache = st.session_state.get("_fiu_det_cache")
    if isinstance(_det_cache, dict) and _det_cache.get("key") == _det_key:
        flagged = _det_cache["flagged"]
        nets = _det_cache["nets"]
    else:
        flagged = pdet.run_detection(G, sanctioned)
        live_cio_arg: Optional[Dict[str, List[str]]] = None
        if st.session_state.live_mode and not effective_demo:
            raw_cio = st.session_state.get("live_cio_clusters") or {}
            live_cio_arg = {
                str(k): list(v) for k, v in raw_cio.items() if isinstance(v, list)
            }
        nets = pdet.cluster_networks(
            G,
            flagged,
            data_dir=DATA_DIR,
            live_cio_clusters=live_cio_arg,
        )
        st.session_state["_fiu_det_cache"] = {
            "key": _det_key,
            "flagged": flagged,
            "nets": nets,
        }
    addr_to_net: Dict[str, int] = {}
    for nid, inf in nets.items():
        for m in inf.get("members", []):
            if isinstance(m, str):
                addr_to_net[m] = int(nid)

    _url_w = _fiu_qp_first("fiu_w")
    if _url_w and _url_w in flagged:
        st.session_state.selected_wallet = _url_w
        _nid_u = addr_to_net.get(_url_w)
        if _nid_u is not None:
            st.session_state.globe_selected_net = int(_nid_u)
            st.session_state.selected_network = int(_nid_u)

    _url_net = _fiu_qp_first("fiu_net")
    if _url_net:
        try:
            _nid_url = int(_url_net)
            if _nid_url in nets:
                st.session_state.selected_network = _nid_url
                st.session_state.globe_selected_net = _nid_url
                st.session_state.selected_wallet = None
        except (ValueError, TypeError):
            pass

    st.session_state.net_sizes = {
        int(nid): len(inf["members"]) for nid, inf in nets.items()
    }

    cur_net_ids = set(nets.keys())
    cur_flagged = set(flagged.keys())
    if not st.session_state._inited_scan:
        st.session_state.prev_net_ids = cur_net_ids.copy()
        st.session_state.prev_flagged = cur_flagged.copy()
        st.session_state._inited_scan = True
        added_nets = set()
        added_wallets = set()
    else:
        added_nets = cur_net_ids - st.session_state.prev_net_ids
        added_wallets = cur_flagged - st.session_state.prev_flagged
    st.session_state.new_networks |= added_nets
    st.session_state.new_wallets |= added_wallets
    st.session_state.prev_net_ids = cur_net_ids.copy()
    st.session_state.prev_flagged = cur_flagged.copy()

    if nets:
        if not st.session_state.networks_sound_seeded:
            st.session_state.known_networks = set(nets.keys())
            st.session_state.networks_sound_seeded = True
        else:
            for nid in list(nets.keys()):
                if nid not in st.session_state.known_networks:
                    st.session_state.known_networks.add(nid)
                    inf = nets[nid]
                    sc = float(inf.get("score", 0))
                    wc = len(inf.get("members", set()))
                    pat = str(inf.get("pattern", "—"))
                    _network_notification_chrome(int(nid), sc, wc, pat)

    demo_mode = effective_demo

    risk_map = {"All": None, "High": "high", "Medium": "amber", "Low": "low"}
    menu_risk = str(st.session_state.get("menu_risk") or "All")
    risk_f = risk_map.get(menu_risk)

    suspicious_val = sum(
        float(G.nodes[w].get("total_volume", 0))
        for w in flagged
        if w in G.nodes
    )
    mode_badge = (
        '<span class="fiu-badge-demo">DEMO</span>'
        if demo_mode
        else '<span class="fiu-badge-live">LIVE</span>'
    )

    _live = bool(st.session_state.live_mode)
    (
        c_logo,
        c_demo,
        c_live,
        c_ra,
        c_rh,
        c_rm,
        c_rl,
        c_ct,
    ) = st.columns([1.25, 0.42, 0.42, 0.32, 0.32, 0.52, 0.32, 1.85])
    with c_logo:
        st.markdown(
            '<span class="fiu-nav-logo">CHAINTRACE</span>',
            unsafe_allow_html=True,
        )
    with c_demo:
        if st.button(
            "DEMO",
            key="fiu_top_demo",
            use_container_width=True,
            type="primary" if not _live else "secondary",
        ):
            st.session_state.demo_mode = True
            st.session_state.live_mode = False
            st.rerun()
    with c_live:
        if st.button(
            "LIVE",
            key="fiu_top_live",
            use_container_width=True,
            type="primary" if _live else "secondary",
        ):
            st.session_state.demo_mode = False
            st.session_state.live_mode = True
            st.rerun()
    with c_ra:
        if st.button(
            "All",
            key="fiu_risk_all",
            use_container_width=True,
            type="primary" if menu_risk == "All" else "secondary",
        ):
            st.session_state.menu_risk = "All"
            st.rerun()
    with c_rh:
        if st.button(
            "High",
            key="fiu_risk_hi",
            use_container_width=True,
            type="primary" if menu_risk == "High" else "secondary",
        ):
            st.session_state.menu_risk = "High"
            st.rerun()
    with c_rm:
        if st.button(
            "Medium",
            key="fiu_risk_med",
            use_container_width=True,
            type="primary" if menu_risk == "Medium" else "secondary",
        ):
            st.session_state.menu_risk = "Medium"
            st.rerun()
    with c_rl:
        if st.button(
            "Low",
            key="fiu_risk_lo",
            use_container_width=True,
            type="primary" if menu_risk == "Low" else "secondary",
        ):
            st.session_state.menu_risk = "Low"
            st.rerun()
    with c_ct:
        st.markdown(
            f'<div class="fiu-nav-counters">'
            f"Transactions: <strong>{len(tx)}</strong> &nbsp;|&nbsp; "
            f"Flagged: <strong>{len(flagged)}</strong> &nbsp;|&nbsp; "
            f"Networks: <strong>{len(nets)}</strong> &nbsp;|&nbsp; "
            f"Value: <strong>${suspicious_val:,.0f}</strong><br/>{mode_badge}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div style="
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 130px;
    font-weight: 900;
    color: rgba(15, 40, 70, 0.22);
    letter-spacing: 0.2em;
    pointer-events: none;
    z-index: 2;
    white-space: nowrap;
    user-select: none;
    font-family: system-ui, sans-serif;
">CHAINTRACE</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div style="position:fixed;bottom:40px;left:198px;display:flex;gap:6px;z-index:8;pointer-events:none;">
  <div style="background:rgba(13,31,53,0.96);border:0.5px solid #2d5a8a;border-radius:10px;
    padding:12px 14px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);min-width:158px;">
    <div style="font-size:10px;color:#6a9abf;font-weight:500;letter-spacing:0.08em;margin-bottom:10px;">DOT LEGEND</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#f87171;box-shadow:0 0 6px #f8717166;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">High risk</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#fbbf24;box-shadow:0 0 6px #fbbf2466;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Medium risk</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d39966;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Low risk</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#378ADD;box-shadow:0 0 6px #378ADD66;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">OFAC sanctioned</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#ffffff;box-shadow:0 0 6px #ffffff66;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">New detection</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:9px;height:9px;border-radius:50%;background:transparent;border:2px solid #ffffff;box-shadow:0 0 6px #ffffff66;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Selected network</span>
    </div>
  </div>
  <div style="background:rgba(13,31,53,0.96);border:0.5px solid #2d5a8a;border-radius:10px;
    padding:12px 14px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);min-width:190px;">
    <div style="font-size:10px;color:#6a9abf;font-weight:500;letter-spacing:0.08em;margin-bottom:10px;">EDGE LEGEND</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:28px;height:2px;background:#f87171;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Fund flow · high risk</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:28px;height:2px;background:#fbbf24;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Fund flow · medium</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:28px;height:0;border-top:2px dashed #f87171;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">Weak association</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <div style="width:28px;height:2px;background:#378ADD;flex-shrink:0;"></div>
      <span style="font-size:11px;color:#d0e8ff;">OFAC linked flow</span>
    </div>
    <div style="font-size:10px;color:#64748b;">Line width = transaction volume</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _wallet_search = st.text_input(
            "wallet_search",
            placeholder="Enter wallet address...",
            label_visibility="collapsed",
            key="fiu_wallet_search",
        )
        if _wallet_search and _wallet_search.strip() in flagged:
            _ws = _wallet_search.strip()
            st.session_state.selected_wallet = _ws
            _nid_search = addr_to_net.get(_ws)
            if _nid_search is not None:
                st.session_state.globe_selected_net = int(_nid_search)
                st.session_state.selected_network = int(_nid_search)
        st.caption("Pattern filter")
        _pat_choice = st.selectbox(
            "pattern_filter",
            PATTERN_OPTIONS,
            key="fiu_sidebar_pattern",
            label_visibility="collapsed",
        )
        st.session_state.custom_panel_pattern = _pat_choice

    pat_f = str(st.session_state.get("custom_panel_pattern") or "All")

    # Build sorted network list for sidebar
    sel_nid_sidebar = st.session_state.get("_sel_net_id")
    net_rows: List[Tuple[int, float, int, str, str]] = []
    for _nid_s, _inf_s in sorted(nets.items(), key=lambda x: -float(x[1].get("score", 0))):
        _nscore_s = float(_inf_s.get("score", 0))
        _nrisk_s = network_risk_label(_nscore_s)
        if risk_f is not None and _nrisk_s != risk_f:
            continue
        _pat_str_s = str(_inf_s.get("pattern", "Unknown"))
        if pat_f != "All" and pat_f not in _pat_str_s:
            continue
        net_rows.append((int(_nid_s), _nscore_s, len(_inf_s.get("members", [])), _pat_str_s, _nrisk_s))

    with st.sidebar:
        st.divider()
        st.header(f"📍 Detected Networks ({len(net_rows)})")
        networks = []
        for _nid_r, _nscore_r, _wcount_r, _pat_r, _nrisk_r in net_rows[:40]:
            _is_sel_r = sel_nid_sidebar is not None and int(sel_nid_sidebar) == _nid_r
            _risk_tag = "HIGH" if _nrisk_r == "high" else "MED" if _nrisk_r == "amber" else "LOW"
            _btn_label = (
                f"Net #{_nid_r:03d}   Score {_nscore_r:.1f}   {_risk_tag}\n"
                f"{_wcount_r} wallets · {_pat_r[:22]}"
            )
            networks.append({
                "net_id": _nid_r,
                "label": _btn_label,
                "is_sel": _is_sel_r
            })

        for net in networks:
            if st.sidebar.button(
                net.get("label"),
                key=f"net_{net.get('net_id')}",
                use_container_width=True,
                type="primary" if net.get("is_sel") else "secondary",
            ):
                st.session_state.selected_network = net
                st.session_state._sel_net_id = net.get("net_id")
                st.session_state.globe_selected_net = net.get("net_id")
                st.session_state.selected_wallet = None
                st.session_state.auto_advance = False
                st.rerun()

    st.session_state.flagged = [
        {
            "address": addr,
            "score": float(fa.score),
            "pattern": fa.patterns[0].code if fa.patterns else "?",
            "net_id": addr_to_net.get(addr),
            "ofac": bool(fa.ofac_match),
        }
        for addr, fa in sorted(flagged.items(), key=lambda x: -x[1].score)
    ]

    fw = list(st.session_state.get("flagged", []))
    nw_set = st.session_state.new_wallets
    sel_net = st.session_state.globe_selected_net
    if sel_net is None and st.session_state.get("_sel_net_id") is not None:
        sel_net = int(st.session_state._sel_net_id)

    _wcache = st.session_state.wallet_coords
    nodes_data = []
    if fw:
        for w in fw[:150]:
            wid = str(w.get("address", w.get("txId", str(w))))
            if wid not in _wcache:
                _wcache[wid] = _wcoords(wid)
            la, lo = _wcache[wid]
            sc = float(w.get("score", 5))
            nid = w.get("net_id")
            ofac = bool(w.get("ofac"))
            
            if ofac:
                c = [55, 138, 221, 255]
            elif wid in nw_set:
                c = [255, 255, 255, 255]
            elif sc >= 7:
                c = [248, 113, 113, 255]
            elif sc >= 4:
                c = [251, 191, 36, 255]
            else:
                c = [52, 211, 153, 255]
                
            # Base radius in meters for Pydeck
            base_s = 60000 if sc >= 7 else 35000 if sc >= 4 else 20000
            # Highlight if selected
            _in_sel = sel_net is not None and nid is not None and int(nid) == int(sel_net)
            if _in_sel:
                base_s *= 1.5
                
            net_hint = f"\nNetwork #{int(nid):03d}" if nid is not None else ""
            _htxt = f"{wid}\nScore:{sc:.1f} · {w.get('pattern', '?')}{net_hint}"
            
            nodes_data.append({
                'id': wid,
                'lat': la,
                'lng': lo,
                'color': c,
                'radius': base_s,
                'network_id': int(nid) if nid is not None else -1,
                'tooltip': _htxt
            })
    
    # Add major cities as faint dots
    for city, coords in MAJOR_CITIES.items():
        nodes_data.append({
            'id': city,
            'lat': coords[0],
            'lng': coords[1],
            'color': [226, 232, 240, 100],
            'radius': 15000,
            'network_id': -1,
            'tooltip': city
        })
        
    nodes_df = pd.DataFrame(nodes_data)

    # Build lines dataframe ONLY for selected network to prevent performance issues
    lines_data = []
    if sel_net is not None and int(sel_net) in nets:
        _jnid = int(sel_net)
        _jinf = nets[_jnid]
        _jscore = float(_jinf.get("score", 5))
        # Line color (RGBA)
        _l_color = (
            [248, 113, 113, 200] if _jscore >= 7
            else [251, 191, 36, 200] if _jscore >= 4
            else [52, 211, 153, 200]
        )
        
        _jmem = sorted(m for m in _jinf["members"] if isinstance(m, str))[:40]  # Cap to 40 to prevent O(N^2) explosion
        _wc2 = st.session_state.wallet_coords
        for _m in _jmem:
            if _m not in _wc2:
                _wc2[_m] = _wcoords(_m)
                
        for _i in range(len(_jmem)):
            for _j in range(_i + 1, len(_jmem)):
                _lat1, _lon1 = _wc2[_jmem[_i]]
                _lat2, _lon2 = _wcache[_jmem[_j]]
                lines_data.append({
                    'network_id': _jnid,
                    'src_lat': _lat1,
                    'src_lng': _lon1,
                    'dst_lat': _lat2,
                    'dst_lng': _lon2,
                    'color': _l_color
                })
                
    lines_df = pd.DataFrame(lines_data)

    # --- THE GLOBE.GL COMPONENT ---
    def render_globe():
        fig = go.Figure()
        
        lats = [float(w.get('lat', 0)) for w in nodes_data]
        lons = [float(w.get('lng', 0)) for w in nodes_data]
        
        def _to_rgba(c):
            if isinstance(c, list) and len(c) == 4:
                return f"rgba({c[0]},{c[1]},{c[2]},{c[3]/255.0})"
            return '#f87171'
            
        colors = [_to_rgba(w.get('color')) for w in nodes_data]
        texts = [f"{str(w.get('id', ''))[:16]}<br>{str(w.get('tooltip', '')).replace(chr(10), '<br>')}" for w in nodes_data]
        sizes = [max(2, float(w.get('radius', 20000))/10000.0) for w in nodes_data]
        
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(size=sizes, color=colors),
            text=texts,
            hoverinfo='text',
            showlegend=False
        ))
        
        network_lats = []
        network_lons = []
        if sel_net is not None and int(sel_net) in nets:
            _jinf = nets[int(sel_net)]
            _jmem = sorted(m for m in _jinf["members"] if isinstance(m, str))[:40]
            _wc2 = st.session_state.wallet_coords
            for _m in _jmem:
                if _m in _wc2:
                    _lat, _lon = _wc2[_m]
                    network_lats.append(_lat)
                    network_lons.append(_lon)
                    
        if network_lats:
            arc_lats = []
            arc_lons = []
            for i in range(len(network_lats)):
                for j in range(i+1, len(network_lats)):
                    arc_lats.extend([network_lats[i], network_lats[j], None])
                    arc_lons.extend([network_lons[i], network_lons[j], None])
            
            fig.add_trace(go.Scattergeo(
                lat=arc_lats,
                lon=arc_lons,
                mode='lines',
                line=dict(
                    width=1.5,
                    color='rgba(248,113,113,0.6)'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_geos(
            projection_type="orthographic",
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.2)",
            showland=True, landcolor="rgba(13, 31, 53, 0.96)",
            showocean=True, oceancolor="rgba(0, 0, 0, 0)",
            showlakes=False,
            bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True, height=650)


    # --- COMPACT HEADER ---
    t1, t2, t3 = st.columns([1, 3, 1])
    with t2:
        st.markdown("<h3 style='text-align: center; color: #58a6ff; margin-bottom: 0;'>FINHACK: GLOBAL AML MONITOR</h3>", unsafe_allow_html=True)

    # 4. Main Dashboard Layout (Locked to 80% of screen height)
    with st.container(height=680, border=False):
        col_graph, col_details = st.columns([3, 1])
        
        with col_graph:
            # Legend Overlay
            st.markdown('<div class="legend-box"><p style="color:#8b949e;font-size:12px;margin:0;">INTEL LEGEND</p>' \
                        '<span style="color:#f85149;">●</span> High Risk Cluster<br>' \
                        '<span style="color:#f0883e;">●</span> Medium Risk<br>' \
                        '<span style="color:#3fb950;">●</span> Verified Entity</div>', unsafe_allow_html=True)

            components.html(
                _starfield_canvas_html(),
                height=0,
                scrolling=False,
            )
            st.markdown(GLOBE_GLOW_HTML, unsafe_allow_html=True)
            with st.container(height=650, border=False):
                render_globe()

    sw = st.session_state.selected_wallet
    active_nid: Optional[int] = None
    sn = st.session_state.get("_sel_net_id")
    if sn is not None and int(sn) in nets:
        active_nid = int(sn)
    elif sw and sw in flagged:
        for nn, inf in nets.items():
            if sw in inf["members"]:
                active_nid = int(nn)
                st.session_state._sel_net_id = active_nid
                st.session_state.auto_advance = False
                break

    # ── Build net_detail for JS panel ────────────────────────────────────────
    _rp_ofac = sanctioned if isinstance(sanctioned, set) else set()
    if active_nid is not None:
        _rp_inf = nets[active_nid]
        _rp_members = _rp_inf["members"]
        _rp_ent, _rp_ex = network_entry_exit_counts(G, _rp_members)
        _rp_score = float(_rp_inf["score"])
        _rp_pat = str(_rp_inf.get("pattern", "Unknown"))
        _rp_rc = "#f87171" if _rp_score >= 7 else "#fbbf24" if _rp_score >= 4 else "#34d399"
        _rp_value = sum(float(G.nodes[m].get("total_volume", 0)) for m in _rp_members if m in G.nodes)
        _rp_chains: Set[str] = set()
        for _m in list(_rp_members)[:10]:
            if _m in G.nodes:
                _ch = G.nodes[_m].get("chains", "")
                if _ch:
                    _rp_chains.add(str(_ch))
        _rp_chain_str = ", ".join(sorted(_rp_chains)) or "—"
        _rp_first = "—"
        if not tx.empty:
            _rp_mset = {str(m) for m in _rp_members if isinstance(m, str)}
            _rp_tscol = "timestamp" if "timestamp" in tx.columns else None
            if _rp_tscol:
                _rp_mask = pd.Series([False] * len(tx), index=tx.index)
                for _c in ("from_address", "sender", "from", "source"):
                    if _c in tx.columns:
                        _rp_mask |= tx[_c].astype(str).isin(_rp_mset)
                for _c in ("to_address", "receiver", "to", "destination"):
                    if _c in tx.columns:
                        _rp_mask |= tx[_c].astype(str).isin(_rp_mset)
                _rp_tx = tx[_rp_mask]
                if not _rp_tx.empty:
                    _rp_ts_val = _rp_tx[_rp_tscol].min()
                    if pd.notnull(_rp_ts_val):
                        try:
                            _rp_first = str(pd.Timestamp(_rp_ts_val).strftime("%Y-%m-%d"))
                        except Exception:
                            _rp_first = str(_rp_ts_val)[:10]
        _rp_pe = _rp_inf.get("pair_evidence") or {}
        if isinstance(_rp_pe, dict) and _rp_pe:
            _rp_confs = [float(r.get("confidence", 0)) for r in _rp_pe.values() if isinstance(r, dict)]
            _rp_avg = sum(_rp_confs) / len(_rp_confs) if _rp_confs else 0
            _rp_cio = round(min(0.99, _rp_avg * 0.60), 2)
            _rp_beh = round(min(0.99, _rp_avg * 0.25), 2)
            _rp_tmp = round(min(0.99, _rp_avg * 0.15), 2)
        else:
            _rp_cio = round(min(0.99, _rp_score * 0.055), 2)
            _rp_beh = round(min(0.99, _rp_score * 0.023), 2)
            _rp_tmp = round(min(0.99, _rp_score * 0.016), 2)
        _rp_conf_total = round(_rp_cio + _rp_beh + _rp_tmp, 2)
        _rp_wallet_list = sorted(str(m) for m in _rp_members if isinstance(m, str))[:100]
        _net_detail: Optional[Dict[str, Any]] = {
            "net_id": active_nid,
            "score": _rp_score,
            "risk_color": _rp_rc,
            "wallet_count": len(_rp_members),
            "pattern": _rp_pat,
            "confidence": _rp_conf_total,
            "entry_points": _rp_ent,
            "exit_points": _rp_ex,
            "chain": _rp_chain_str,
            "first_seen": _rp_first,
            "value": _rp_value,
            "cio": _rp_cio,
            "behavioral": _rp_beh,
            "temporal": _rp_tmp,
            "wallets": _rp_wallet_list,
            "ofac_wallets": [w for w in _rp_wallet_list if w in _rp_ofac],
        }
    else:
        _net_detail = None

    # Persist built dict so right panel survives reruns without re-lookup
    print(f"[DBG] _sel_net_id={st.session_state.get('_sel_net_id')}  active_nid={active_nid}  _net_detail={'SET' if _net_detail else 'NONE'}")
    if _net_detail is not None:
        st.session_state.selected_network = _net_detail
    else:
        st.session_state.selected_network = None
    print(f"[DBG] selected_network in ss = {'SET' if st.session_state.get('selected_network') else 'NONE'}")

    # Pause step-advance when a network is selected
    if st.session_state.get("selected_network"):
        st.session_state.auto_advance = False

    pass

    # ── Right panel — Details Column ──
    with col_details:
        st.markdown("### 🔍 Intel Report")
        with st.container(height=550):
            st.markdown('<div class="scrollable-panel">', unsafe_allow_html=True)
            net = st.session_state.get("selected_network")
            if net is not None:
                st.markdown(f"### 🔍 Network Details")
                st.write(f"**Network ID:** `#{net['net_id']:03d}`")
                
                _score = float(net.get("score", 0))
                if _score >= 7:
                    st.error("Risk Level: HIGH")
                elif _score >= 4:
                    st.warning("Risk Level: MEDIUM")
                else:
                    st.success("Risk Level: LOW")
                
                with st.expander("Risk Heuristics", expanded=True):
                    pat = str(net.get("pattern", "?"))
                    st.write(f"- [X] {pat} Detected")
                    ofac_matches = len(net.get("ofac_wallets", []))
                    if ofac_matches > 0:
                        st.write(f"- [X] OFAC Sanctions Match ({ofac_matches})")
                    else:
                        st.write("- [ ] OFAC Sanctions Match")
                
                st.metric(label="Total Value", value=f"${float(net.get('value', 0)):,.0f}")
                st.metric(label="Fraud Probability", value=f"{float(net.get('confidence', 0))*100:.0f}%")
                st.metric("Entry points", int(net.get("entry_points", 0)))
                st.metric("Exit points", int(net.get("exit_points", 0)))
                
                with st.expander(f"Show wallets ({net.get('wallet_count', 0)})"):
                    _ofac_set_rp = set(net.get("ofac_wallets", []))
                    for _wm in list(net.get("wallets", []))[:20]:
                        _wc_tag = " · OFAC" if _wm in _ofac_set_rp else ""
                        st.code(f"{str(_wm)[:28]}{_wc_tag}", language=None)
                        
                if st.button("Clear View"):
                    st.session_state.selected_network = None
                    st.session_state._sel_net_id = None
                    st.session_state.globe_selected_net = None
                    st.rerun()
            else:
                st.info("Select a network from the left panel or graph to view detailed intelligence.")
            st.markdown('</div>', unsafe_allow_html=True)

    tick_items = list(st.session_state.intel_items or [])
    parts_html = []
    for it in tick_items:
        esc = (
            (it.get("title") or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        parts_html.append(
            f'<span style="color:#94a3b8;font-size:11px;">{it.get("source", "")}: {esc[:120]}</span>'
        )
    rss_repeat = "".join(
        f'<span style="display:inline-block;padding-right:40px;">{p}</span>'
        for p in (parts_html * 3 if parts_html else ['<span style="color:#64748b;">Intel feeds loading…</span>'])
    )
    st.markdown(
        f"""
<div style="
  position:fixed;
  bottom:0;left:0;right:0;
  height:36px;
  background:#0d1f35;
  border-top:0.5px solid #1e2d3d;
  overflow:hidden;
  display:flex;
  align-items:center;
  z-index:9998;
">
  <span style="color:#4a6b8a;font-size:11px;
    padding:0 12px;white-space:nowrap;
    border-right:0.5px solid #1e2d3d;">
    THREAT INTEL
  </span>
  <div style="overflow:hidden;flex:1">
    <div id="ticker" style="
      display:flex;gap:40px;
      animation:ticker 30s linear infinite;
      white-space:nowrap;
    ">
      {rss_repeat}
    </div>
  </div>
</div>
<style>
@keyframes ticker {{
  0% {{ transform: translateX(100%); }}
  100% {{ transform: translateX(-100%); }}
}}
</style>
""",
        unsafe_allow_html=True,
    )

    for m in load_msgs[:3]:
        st.caption(f"⚠ {m}")

    if effective_demo and st.session_state.demo_auto_play:
        tick = st_autorefresh(
            interval=3000,
            debounce=True,
            key="fiu_demo_autorefresh",
        )
        prev = st.session_state.get("_fiu_demo_ar_prev")
        if prev is None:
            st.session_state._fiu_demo_ar_prev = tick
        elif tick > prev:
            st.session_state._fiu_demo_ar_prev = tick
            if st.session_state.get("auto_advance", True):
                cur = int(st.session_state.current_step)
                st.session_state.current_step = (cur + 1) if cur < max_s else 1
                st.rerun()
    else:
        st.session_state.pop("_fiu_demo_ar_prev", None)

    st.markdown(
        """
<div style="position:fixed;bottom:44px;left:50%;transform:translateX(-50%);
  font-size:10px;color:#3a5a7a;z-index:100;pointer-events:none;
  white-space:nowrap;">
  Double-click globe to resume rotation
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
