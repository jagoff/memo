"""Declarative extraction helpers for temporal fact edges.

The first temporal-fact ingestion lane is intentionally deterministic:
records can declare fact edges in frontmatter/``extra`` and reindex can rebuild
the sidecar from markdown without calling an LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FACT_EDGES_KEY = "fact_edges"
FACT_ASSERTION_PREDICATE = "asserts"


def _coerce_fact_items(raw: Any) -> Iterable[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw if raw is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, value))


def _coerce_supersedes(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def fact_edges_from_metadata(
    *,
    record_id: str,
    title: str,
    type_: str,
    created: str,
    updated: str,
    extra: dict[str, Any] | None,
    top_level: Any = None,
) -> list[dict[str, Any]]:
    """Return normalized fact-edge kwargs declared by one memory record.

    Supported forms:

    ``extra={"fact_edges": [{subject, predicate, object, ...}]}``
    ``fact_edges:`` as top-level frontmatter when reindexing hand-edited files.

    A durable ``type: fact`` memory without explicit edges is also represented
    as ``memory asserts <title>``. This gives the fact graph useful coverage
    before LLM extraction exists, while preserving the original memory body as
    source of truth.
    """
    items: list[dict[str, Any]] = []
    if isinstance(extra, dict):
        items.extend(_coerce_fact_items(extra.get(FACT_EDGES_KEY)))
    items.extend(_coerce_fact_items(top_level))

    out: list[dict[str, Any]] = []
    for item in items:
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        object_ = str(item.get("object") or item.get("object_") or "").strip()
        if not subject or not predicate or not object_:
            continue
        provenance = item.get("provenance")
        metadata = item.get("metadata")
        out.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "source_record_id": str(item.get("source_record_id") or record_id),
                "valid_at": item.get("valid_at") or created,
                "invalid_at": item.get("invalid_at"),
                "expired_at": item.get("expired_at"),
                "confidence": _coerce_confidence(item.get("confidence")),
                "provenance": provenance if isinstance(provenance, dict) else {},
                "metadata": metadata if isinstance(metadata, dict) else {},
                "supersedes": _coerce_supersedes(item.get("supersedes")),
            }
        )

    if not out and type_ == "fact":
        normalized_title = (title or "").strip()
        if normalized_title:
            out.append(
                {
                    "subject": "memory",
                    "predicate": FACT_ASSERTION_PREDICATE,
                    "object": normalized_title,
                    "source_record_id": record_id,
                    "valid_at": created,
                    "invalid_at": None,
                    "expired_at": None,
                    "confidence": 1.0,
                    "provenance": {"extractor": "memo.fact_extraction", "mode": "type-fact"},
                    "metadata": {"updated": updated},
                    "supersedes": [],
                }
            )
    return out


def upsert_declared_fact_edges(
    fact_store: Any,
    *,
    record_id: str,
    title: str,
    type_: str,
    created: str,
    updated: str,
    extra: dict[str, Any] | None,
    top_level: Any = None,
) -> int:
    """Upsert fact edges declared by one record; return count written."""
    count = 0
    for edge in fact_edges_from_metadata(
        record_id=record_id,
        title=title,
        type_=type_,
        created=created,
        updated=updated,
        extra=extra,
        top_level=top_level,
    ):
        fact_store.upsert_fact(**edge)
        count += 1
    return count


def assertion_fact_edge(
    *,
    title: str,
    extractor: str,
    mode: str,
    confidence: float = 1.0,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a coarse temporal assertion edge for an atomic memory."""
    normalized_title = title.strip()
    if not normalized_title:
        return None
    return {
        "subject": "memory",
        "predicate": FACT_ASSERTION_PREDICATE,
        "object": normalized_title,
        "confidence": _coerce_confidence(confidence),
        "provenance": {
            "extractor": extractor,
            "mode": mode,
            **(provenance or {}),
        },
        "metadata": metadata or {},
    }


def candidate_fact_edges(
    *,
    title: str,
    raw_edges: Any,
    extractor: str,
    mode: str,
    confidence: float = 1.0,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize candidate-declared fact edges, falling back to assertion."""
    base_provenance = {
        "extractor": extractor,
        "mode": mode,
        **(provenance or {}),
    }
    base_metadata = metadata or {}
    out: list[dict[str, Any]] = []
    for item in _coerce_fact_items(raw_edges):
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        object_ = str(item.get("object") or item.get("object_") or "").strip()
        if not subject or not predicate or not object_:
            continue
        item_provenance = item.get("provenance")
        item_metadata = item.get("metadata")
        edge: dict[str, Any] = {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "confidence": _coerce_confidence(item.get("confidence", confidence)),
            "provenance": {
                **base_provenance,
                **(item_provenance if isinstance(item_provenance, dict) else {}),
            },
            "metadata": {
                **base_metadata,
                **(item_metadata if isinstance(item_metadata, dict) else {}),
            },
        }
        for key in ("valid_at", "invalid_at", "expired_at", "supersedes"):
            if key in item:
                edge[key] = item[key]
        out.append(edge)
    if out:
        return out
    fallback = assertion_fact_edge(
        title=title,
        extractor=extractor,
        mode=mode,
        confidence=confidence,
        provenance=provenance,
        metadata=metadata,
    )
    return [fallback] if fallback is not None else []


__all__ = [
    "FACT_EDGES_KEY",
    "assertion_fact_edge",
    "candidate_fact_edges",
    "fact_edges_from_metadata",
    "upsert_declared_fact_edges",
]
