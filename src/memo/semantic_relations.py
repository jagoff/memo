"""Knowledge graph semantic relations — infrastructure for future enhancements.

Currently a stub. When enabled (MEMO_GRAPH_SEMANTIC_RELATIONS=1), the graph
will track relation types ('causes', 'contradicts', 'extends', 'depends_on')
in addition to co-mention edges. This improves:

1. **Semantic retrieval** — find memories related via 'causes', not just
   shared entities.
2. **Contradiction detection** — 'contradicts' edges flag potential conflicts.
3. **Dependency tracking** — 'depends_on' identifies critical prerequisites.

## Schema (future)

The `graph` table will store:

    CREATE TABLE relations (
        id INTEGER PRIMARY KEY,
        source_id TEXT NOT NULL,        -- memory.id
        target_id TEXT NOT NULL,
        relation_type TEXT NOT NULL,    -- 'causes', 'contradicts', 'extends', 'depends_on'
        confidence REAL,                -- 0.0-1.0 LLM verdict confidence
        created_at TEXT,
        updated_at TEXT,
        UNIQUE(source_id, target_id, relation_type)
    );

## Extraction (future)

A semantic_relations_extractor module will:

1. Take a pair of memories (source, target).
2. Call an LLM to classify the relationship type + confidence.
3. Write to the `relations` table.

Triggered by:
- `memo dream` (nightly synthesis pass).
- `MEMO_GRAPH_SEMANTIC_EXTRACT_ON_SAVE=1` (every save, ~1 LLM call).
- Manual: `memo graph extract-relations <query>`.

## Scoring integration (future)

`search_scoring_ops._fetch_graph_candidates()` will:

1. Extract entities from query.
2. Find memories via entity overlap (current).
3. ALSO find memories via semantic relations:
   - If query entity A 'causes' entity B, boost memories about B.
   - If A 'contradicts' B, penalize B or flag for contradiction scan.

## Current state

This file documents the design. Implementation deferred to v2.13+.
Gated by MEMO_GRAPH_SEMANTIC_RELATIONS (default=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SemanticRelation:
    """A semantic relation between two memories."""

    source_id: str
    target_id: str
    relation_type: Literal["causes", "contradicts", "extends", "depends_on"]
    confidence: float  # 0.0-1.0, LLM verdict


def extract_relations_batch(
    memory_pairs: list[tuple[str, str]], model: str = "auto"
) -> list[SemanticRelation]:
    """Extract semantic relations from a batch of memory pairs.

    Args:
        memory_pairs: List of (source_id, target_id) tuples.
        model: LLM model to use ('auto' = use MEMO_MODEL_PROFILE).

    Returns:
        List of SemanticRelation objects with confidence scores.

    Status:
        Stub. When implemented, calls LLM to classify relation types
        and populate the results. Batching amortizes token cost.
    """
    # TODO: implement LLM-based relation classification
    return []
