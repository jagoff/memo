"""Source dedup: collapse (§N/M) chunk siblings of the same doc into one row."""

from __future__ import annotations

import re
from typing import Any

CHUNK_MARKER = re.compile(r"\s*\(§\d+\s*/\s*\d+\)\s*$")
CHUNK_NUM = re.compile(r"\(§(\d+)\s*/\s*(\d+)\)\s*$")
_MERGED_SNIPPET_MAX = 6000
_SCORE_FIELDS = ("rerank_score", "normalized_score", "score")


def normalize_title(title: str | None) -> str:
    return CHUNK_MARKER.sub("", (title or "").strip()).lower()


def _path_root(s: dict[str, Any]) -> str:
    raw = str(s.get("path") or s.get("locator") or "")
    return CHUNK_MARKER.sub("", raw.split("@", 1)[0]).strip().lower()


def dedup_key(s: dict[str, Any]) -> tuple[str, str, str]:
    return (str(s.get("source") or ""), normalize_title(s.get("title")), _path_root(s))


def score_of(s: dict[str, Any]) -> float:
    for field in _SCORE_FIELDS:
        value = s.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _merge_chunk_snippets(members: list[dict[str, Any]]) -> str | None:
    chunks: list[tuple[int, str]] = []
    for m in members:
        match = CHUNK_NUM.search(str(m.get("title") or ""))
        snippet = str(m.get("snippet") or "").strip()
        if match and snippet:
            chunks.append((int(match.group(1)), snippet))
    if len(chunks) < 2:
        return None
    chunks.sort(key=lambda t: t[0])
    seen: set[str] = set()
    parts: list[str] = []
    for _, text in chunks:
        if text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)[:_MERGED_SNIPPET_MAX]


def collapse_near_duplicates(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[object, list[dict[str, Any]]] = {}
    order: list[object] = []
    for idx, s in enumerate(sources):
        key: object = dedup_key(s) if str(s.get("title") or "").strip() else ("__untitled__", idx)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(s)
    out: list[dict[str, Any]] = []
    for key in order:
        members = grouped[key]
        if len(members) == 1:
            out.append(members[0])
            continue
        best = max(members, key=lambda m: (score_of(m), len(str(m.get("snippet") or ""))))
        survivor: dict[str, Any] = dict(best)
        merged = _merge_chunk_snippets(members)
        if merged:
            survivor["snippet"] = merged
        survivor["collapsed_variants"] = len(members) - 1
        out.append(survivor)
    out.sort(key=score_of, reverse=True)
    return out
