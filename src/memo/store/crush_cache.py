"""Reversible compression cache for SmartCrusher results.

Stores original JSON arrays before crushing, allows LLM to retrieve on demand.
TTL-based eviction via memo maintain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


class CrushCache:
    """Local cache for crushed JSON originals (TTL 30d default)."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.cache_dir = state_dir / "crush_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def cache(self, hash_val: str, content: str) -> None:
        """Store original JSON content under hash key.

        Args:
            hash_val: SHA256 hash (used as filename)
            content: Original JSON string
        """
        path = self.cache_dir / f"{hash_val}.json"
        metadata = {
            "ts": datetime.now(UTC).isoformat(),
            "content": content,
        }
        path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    def retrieve(self, hash_val: str, ttl_days: int = 30) -> str | None:
        """Retrieve cached JSON by hash.

        Returns None if not found or expired.
        """
        path = self.cache_dir / f"{hash_val}.json"
        if not path.is_file():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            ts_str = doc.get("ts")
            if not ts_str:
                return None
            ts = datetime.fromisoformat(ts_str)
            if (datetime.now(UTC) - ts) > timedelta(days=ttl_days):
                return None  # Expired
            return doc.get("content")
        except (json.JSONDecodeError, OSError):
            return None

    def evict_expired(self, ttl_days: int = 30) -> int:
        """Remove expired cache entries. Returns count evicted."""
        if not self.cache_dir.is_dir():
            return 0
        evicted = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                ts_str = doc.get("ts")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    if (datetime.now(UTC) - ts) > timedelta(days=ttl_days):
                        path.unlink()
                        evicted += 1
            except (json.JSONDecodeError, OSError):
                # Corrupt or inaccessible; skip
                pass
        return evicted


def crush_marker(dropped_count: int, hash_val: str) -> dict:
    """Sentinel object to mark dropped rows in JSON array.

    Format: {"_compressed": "N rows offloaded — ask `memo retrieve <<memo-crush:HASH>>` for full"}
    """
    return {
        "_compressed": f"{dropped_count} rows offloaded — ask `memo retrieve <<memo-crush:{hash_val}>>` for full"
    }
