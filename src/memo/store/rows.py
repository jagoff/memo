from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from typing import Any


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = {k: row[k] for k in row.keys() if k != "distance"}  # noqa: SIM118
    if "tags" in d and isinstance(d["tags"], str):
        try:
            d["tags"] = json.loads(d["tags"])
        except (ValueError, TypeError):
            d["tags"] = []
    if d.get("extra_json"):
        try:
            d["extra"] = json.loads(d["extra_json"])
        except (ValueError, TypeError):
            d["extra"] = {}
        d.pop("extra_json", None)
    elif "extra_json" in d:
        d.pop("extra_json", None)
        d["extra"] = {}
    return d


def _fts_match_expr(query: str) -> str:
    if not query or not query.strip():
        return ""
    tokens = [t for t in re.findall(r"\w+", query, flags=re.UNICODE) if t]
    return " ".join(f'"{t}"' for t in tokens)


def _repo_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    raw = dict(zip(row.keys(), row, strict=True))
    return {k: v for k, v in raw.items() if k not in {"distance", "bm25_score"}}


def _repo_bm25_row_to_dict(row: sqlite3.Row, match_type: str) -> dict[str, Any]:
    d = _repo_row_to_dict(row)
    bm = float(row["bm25_score"])
    d["score"] = 1.0 - 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
    d["match_type"] = match_type
    return d


def _batches(items: list[str], size: int = 500) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
