"""MCP tools — temporal-reasoning domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_fact_edges(
        subject: Annotated[
            str | None,
            Field(
                description="Exact-match filter on the fact's subject; None matches any subject."
            ),
        ] = None,
        predicate: Annotated[
            str | None,
            Field(
                description="Exact-match filter on the fact's predicate; "
                "None matches any predicate."
            ),
        ] = None,
        object: Annotated[
            str | None,
            Field(description="Exact-match filter on the fact's object; None matches any object."),
        ] = None,
        source_record_id: Annotated[
            str | None,
            Field(
                description="Exact-match filter on the memo record id the fact was derived from; "
                "None matches any source."
            ),
        ] = None,
        as_of: Annotated[
            str | None,
            Field(
                description="ISO-8601 timestamp to evaluate validity at (naive times are treated "
                "as UTC); defaults to now. An edge is live when valid_at <= as_of and neither "
                "invalid_at nor expired_at has passed."
            ),
        ] = None,
        include_inactive: Annotated[
            bool,
            Field(
                description="When true, skip the validity-window filter and also return "
                "invalidated/expired edges."
            ),
        ] = False,
        limit: Annotated[
            int,
            Field(
                description="Maximum edges to return (floored to 1, no upper clamp). Results are "
                "ordered by valid_at, then confidence, then updated_at, descending."
            ),
        ] = 50,
    ) -> list[dict[str, Any]]:
        """List temporal fact edges, optionally as-of a timestamp.

        Fact edges are explicit subject/predicate/object facts with
        ``valid_at``/``invalid_at``/``expired_at`` windows. This is a lower-level
        temporal substrate than memo_search: use it when the question is about
        fact validity over time, not general semantic recall.
        """
        return memory.fact_edges.query(
            subject=subject,
            predicate=predicate,
            object=object,
            source_record_id=source_record_id,
            as_of=as_of,
            include_inactive=include_inactive,
            limit=limit,
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_fact_edge_save(
        subject: Annotated[
            str,
            Field(
                description="Fact subject (whitespace-stripped; must be non-empty). "
                "Part of the deterministic fact id."
            ),
        ],
        predicate: Annotated[
            str,
            Field(
                description="Fact predicate/relation (whitespace-stripped; must be non-empty). "
                "Part of the deterministic fact id."
            ),
        ],
        object: Annotated[
            str,
            Field(
                description="Fact object/value (whitespace-stripped; must be non-empty). "
                "Part of the deterministic fact id."
            ),
        ],
        source_record_id: Annotated[
            str | None,
            Field(
                description="Optional memo record id this fact was derived from. "
                "Part of the deterministic fact id."
            ),
        ] = None,
        valid_at: Annotated[
            str | None,
            Field(
                description="ISO-8601 timestamp when the fact became valid (naive times are "
                "treated as UTC); defaults to now. Part of the deterministic fact id."
            ),
        ] = None,
        invalid_at: Annotated[
            str | None,
            Field(
                description="Optional ISO-8601 timestamp when the fact stopped being valid "
                "(naive times are treated as UTC)."
            ),
        ] = None,
        expired_at: Annotated[
            str | None,
            Field(
                description="Optional ISO-8601 timestamp that ends the validity window "
                "independently of invalid_at; the edge is inactive once either has passed."
            ),
        ] = None,
        confidence: Annotated[
            float,
            Field(
                description="Confidence stored on the edge (default 1.0, not clamped); used as a "
                "descending tiebreaker when querying."
            ),
        ] = 1.0,
        provenance: Annotated[
            dict[str, Any] | None,
            Field(
                description="Optional JSON object recording where the fact came from; "
                "stored as-is, None becomes {}."
            ),
        ] = None,
        metadata: Annotated[
            dict[str, Any] | None,
            Field(
                description="Optional JSON object of arbitrary extra fields; "
                "stored as-is, None becomes {}."
            ),
        ] = None,
        supersedes: Annotated[
            list[str] | None,
            Field(
                description="Ids of older fact edges to invalidate at this fact's valid_at. Only "
                "edges without an existing invalid_at are stamped; unknown ids are ignored."
            ),
        ] = None,
    ) -> dict[str, Any] | None:
        """Save one temporal fact edge.

        Re-saving the same subject/predicate/object/source/valid_at combination
        is idempotent. Passing ``supersedes`` invalidates older fact ids at this
        fact's ``valid_at`` timestamp.
        """
        fact_id = memory.fact_edges.upsert_fact(
            subject=subject,
            predicate=predicate,
            object=object,
            source_record_id=source_record_id,
            valid_at=valid_at,
            invalid_at=invalid_at,
            expired_at=expired_at,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata,
            supersedes=supersedes,
        )
        return memory.fact_edges.get(fact_id)

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_fact_edge_invalidate(
        id: Annotated[
            str,
            Field(
                description="Fact edge id (32-char hash) as returned by "
                "memo_fact_edge_save or memo_fact_edges."
            ),
        ],
        invalid_at: Annotated[
            str | None,
            Field(
                description="ISO-8601 timestamp to record as invalid_at (naive times are treated "
                "as UTC); defaults to now. Overwrites any existing invalid_at."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Invalidate one temporal fact edge without deleting its provenance.

        Sets the edge's ``invalid_at`` (overwriting any prior value); re-invoking
        just re-stamps the timestamp. Returns ``invalidated: false`` when the id
        does not exist.
        """
        return {"id": id, "invalidated": memory.fact_edges.invalidate(id, invalid_at=invalid_at)}

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_contradictions(
        entity: str,
        entity_type: str | None = None,
        confidence_threshold: float = 0.7,
        max_pairs: int = 20,
    ) -> list[dict[str, Any]]:
        """Detect contradictions among memories mentioning a specific entity.

        Uses the helper LLM to classify pairs of memories as contradiction,
        evolution, consistent, or unrelated. Returns only contradictions and
        evolutions above the confidence threshold.

        Args:
            entity: The entity name to analyze (e.g. "ollama", "mlx").
            entity_type: Optional entity type filter from graph.
            confidence_threshold: Minimum confidence (0-1). Default 0.7.
            max_pairs: Maximum number of pairs to analyze (LLM is expensive).
        """
        contradictions = memory.temporal.detect_entity_contradictions(
            entity_name=entity,
            entity_type=entity_type,
            confidence_threshold=confidence_threshold,
            max_pairs=max_pairs,
        )
        return [c.__dict__ for c in contradictions]

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_timeline(
        entity: str,
        entity_type: str | None = None,
        limit: Annotated[
            int,
            Field(
                description="Maximum events to return (floored to 0). The most "
                "RECENT `limit` events are kept, still in chronological order. "
                "`event_count` is always the true total and `truncated` says "
                "whether any were dropped; `first_seen`/`last_seen` span the "
                "whole timeline either way."
            ),
        ] = 30,
    ) -> dict[str, Any] | None:
        """Build a chronological timeline of memories mentioning an entity.

        Returns a timeline with events ordered by date, including first/last
        seen timestamps. Useful for tracking evolution of decisions or
        opinions over time.

        Bounded: `build_entity_timeline` emits ONE event per mention, each
        carrying a 200-char snippet, so a hub entity's timeline tracks the
        corpus rather than the request. The conformance gate measured 110,326
        tokens for an entity with 700 mentions, against a 10,000-token MCP
        response cap. The library and CLI paths still build the whole
        timeline; only the MCP surface, the one with a token budget, is
        trimmed — and the trim is reported, never silent.

        Args:
            entity: The entity name to analyze.
            entity_type: Optional entity type filter from graph.
            limit: Maximum events to return, most recent kept.
        """
        timeline = memory.temporal.build_entity_timeline(
            entity_name=entity,
            entity_type=entity_type,
        )
        if timeline is None:
            return None
        events = [e.__dict__ for e in timeline.events]
        cap = max(0, limit)
        # `events` is already sorted ascending by date, so the tail is the
        # newest slice. Spelled out rather than `events[-cap:]` because that
        # returns the WHOLE list when cap is 0.
        kept = events[len(events) - cap :] if cap else []
        return {
            "entity_name": timeline.entity_name,
            "entity_type": timeline.entity_type,
            "first_seen": timeline.first_seen,
            "last_seen": timeline.last_seen,
            "events": kept,
            "event_count": len(events),
            "truncated": len(kept) < len(events),
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_stale(
        days_threshold: int = 180,
        min_access_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Find memories that may be stale based on age and lack of access.

        Returns potentially stale memories with metadata including days since
        update and access count. Useful for identifying outdated information
        that may need review.

        Args:
            days_threshold: Days since last update to consider stale.
            min_access_count: Minimum access count to exclude (frequently-accessed
                old memories may still be relevant).
        """
        return memory.temporal.detect_stale_memories(
            days_threshold=days_threshold,
            min_access_count=min_access_count,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_patterns() -> dict[str, Any]:
        """Analyze high-level temporal patterns across the entire corpus.

        Returns metrics including:
        - memories_per_month: histogram of creation activity
        - type_distribution_over_time: how memory types change over time
        - most_active_entities: entities with most temporal churn
        """
        return memory.temporal.detect_temporal_patterns()
