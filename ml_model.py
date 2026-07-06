"""
Optional XGBoost risk scorer trained on labeled rows (primarily Elliptic-derived).

Default off in the UI; when enabled, outputs a fraud probability in [0, 1].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graph_builder import graph_to_feature_row

# Substrings matched against joined pattern titles from the rule engine.
PATTERN_ALIASES = [
    ("fan_out", ("fan-out", "fanout")),
    ("fan_in", ("fan-in", "fanin")),
    ("pass_through", ("pass-through", "pass through")),
    ("layering", ("layering",)),
    ("dormancy", ("dormancy",)),
    ("round_trip", ("round-trip", "round trip")),
    ("structuring", ("structuring",)),
    ("peeling", ("peeling",)),
    ("mixer", ("mixer",)),
    ("ofac", ("ofac",)),
]

PATTERN_KEYS = [k for k, _ in PATTERN_ALIASES]


def _pattern_vector(pattern_codes: List[str]) -> List[float]:
    blob = " ".join(pattern_codes).lower()
    vec = []
    for _name, aliases in PATTERN_ALIASES:
        vec.append(1.0 if any(a in blob for a in aliases) else 0.0)
    return vec


def build_feature_matrix(
    G: Any,
    addresses: List[str],
    pattern_map: Dict[str, List[str]],
) -> np.ndarray:
    """Stack [degree features | pattern bits] for each address."""
    rows = []
    for a in addresses:
        base = graph_to_feature_row(G, a)
        vec = [
            base["degree"],
            base["in_degree"],
            base["out_degree"],
            base["volume"],
            base["time_span"],
        ]
        vec.extend(_pattern_vector(pattern_map.get(a, [])))
        rows.append(vec)
    return np.array(rows, dtype=np.float64)


def train_xgb_if_possible(
    tx_df: pd.DataFrame,
    G: Any,
    pattern_codes_by_addr: Dict[str, List[str]],
) -> Tuple[Optional[Any], Optional[List[str]], str]:
    """
    Train XGBoost on rows with label illicit (1) or licit (0).

    Returns (model, feature_names, status_message).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None, "XGBoost not installed."

    labeled = tx_df[tx_df["label"].isin(["illicit", "licit"])].copy()
    if labeled.empty or len(labeled) < 200:
        return None, None, "Not enough labeled transactions to train."

    # Wallet = destination of transfer for supervision signal.
    y_map: Dict[str, int] = {}
    for _, r in labeled.iterrows():
        to = str(r["to_address"]).strip()
        y_map[to] = 1 if r["label"] == "illicit" else 0

    addrs = [a for a in y_map if a in G.nodes]
    if len(addrs) < 200:
        return None, None, "Not enough labeled wallets in graph."

    X = build_feature_matrix(G, addrs, pattern_codes_by_addr)
    y = np.array([y_map[a] for a in addrs], dtype=np.int32)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return None, None, "Label imbalance too extreme."

    feat_names = [
        "degree",
        "in_degree",
        "out_degree",
        "volume",
        "time_span",
    ] + PATTERN_KEYS

    clf = XGBClassifier(
        n_estimators=80,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X, y)
    return clf, feat_names, f"Trained on {len(addrs)} labeled wallets."


def predict_proba_for_address(
    model: Any,
    G: Any,
    address: str,
    pattern_codes: List[str],
) -> float:
    """Return illicit probability for one wallet."""
    X = build_feature_matrix(G, [address], {address: pattern_codes})
    prob = float(model.predict_proba(X)[0, 1])
    return prob
