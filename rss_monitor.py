"""
RSS threat-intelligence feeds (FinCEN, FATF, OFAC recent actions).

Parsed with ``feedparser``; safe to call on a timer in the dashboard.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import feedparser
import requests

FEEDS = [
    ("FinCEN", "https://www.fincen.gov/rss.xml"),
    ("FATF", "https://www.fatf-gafi.org/en/publications.rss"),
    ("OFAC", "https://home.treasury.gov/policy-issues/financial-sanctions/recent-actions/rss"),
]


def _age(entry: Any) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            t = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - t
            h = int(delta.total_seconds() // 3600)
            if h < 1:
                return "just now"
            if h < 48:
                return f"{h}h ago"
            return f"{h // 24}d ago"
        except Exception:
            pass
    return "recent"


def fetch_feed_items(max_per_feed: int = 5, timeout: int = 5) -> List[Dict[str, str]]:
    """Return flattened recent items across all configured feeds (HTTP with timeout)."""
    out: List[Dict[str, str]] = []
    for source, url in FEEDS:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
            for ent in parsed.entries[:max_per_feed]:
                out.append(
                    {
                        "source": source,
                        "title": getattr(ent, "title", "Untitled")[:200],
                        "link": getattr(ent, "link", ""),
                        "age": _age(ent),
                    }
                )
        except Exception:
            continue
    return out


def ticker_lines(items: List[Dict[str, str]], limit: int = 12) -> str:
    """Single string for bottom ticker (sentence case fragments)."""
    parts = []
    for it in items[:limit]:
        parts.append(f"{it['source']}: {it['title']}")
    return "   •   ".join(parts) if parts else "Intelligence feeds loading…"


def cache_key() -> int:
    """Bucket time for 5-minute cache windows."""
    return int(time.time() // 300)
