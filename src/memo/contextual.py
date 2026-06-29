"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Contextual recall enhancement — conversation history + user preferences.

Extends the basic recall-hook with:
- Conversation context tracking (last N prompts)
- User preference learning (what memory types/entities they prefer)
- Contextual re-ranking based on current conversation
- Personalized similarity thresholds

## Context Tracking

Maintains a sliding window of recent prompts (default: last 10) to provide
context for recall. The context is used to:
- Boost memories that reference entities mentioned in recent context
- Penalize memories that are irrelevant to the current topic
- Detect topic shifts and adjust recall accordingly

## Preference Learning

Tracks which memories the user actually clicks/views to learn preferences:
- Preferred memory types (decision vs fact vs note)
- Preferred entities (projects, technologies)
- Time-of-day patterns
- Recency preference

## Contextual Re-ranking

Re-ranks search results based on:
- Entity overlap with recent context
- Type preference alignment
- Temporal relevance (recent context prefers recent memories)
- User feedback (if available)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """A single prompt in the conversation history."""

    timestamp: str
    prompt: str
    recalled_memorias: list[str]  # IDs of memories recalled for this prompt


@dataclass
class UserPreferences:
    """Learned user preferences for memory recall."""

    preferred_types: dict[str, float] = field(default_factory=dict)  # type -> score
    preferred_entities: dict[str, float] = field(default_factory=dict)  # entity -> score
    recency_weight: float = 0.5  # 0-1, how much to prefer recent memories
    diversity_weight: float = 0.3  # 0-1, how much to prefer diverse results
    last_updated: str = ""


@dataclass
class ContextualSearchResult:
    """A search result with contextual scoring."""

    memory_id: str
    title: str
    original_score: float | None
    contextual_score: float
    boost_factors: dict[str, float]  # factor -> contribution
    snippet: str


class ContextStore:
    """Stores conversation context and user preferences.

    Args:
        state_dir: Directory to store context state files.
        max_context_length: Maximum number of prompts to keep in history.
    """

    def __init__(self, state_dir: Path, max_context_length: int = 10) -> None:
        self.state_dir = state_dir
        self.max_context_length = max_context_length
        self.context_file = state_dir / "context_history.json"
        self.preferences_file = state_dir / "user_preferences.json"

        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._context: deque[PromptContext] = deque(maxlen=max_context_length)
        self._preferences = UserPreferences()

        self._load()

    def _load(self) -> None:
        """Load context and preferences from disk."""
        if self.context_file.is_file():
            try:
                data = json.loads(self.context_file.read_text(encoding="utf-8"))
                self._context = deque(
                    (PromptContext(**item) for item in data),
                    maxlen=self.max_context_length,
                )
            except Exception as exc:
                _log.debug("contextual: failed to load context from %s: %s", self.context_file, exc)
                self._context = deque(maxlen=self.max_context_length)

        if self.preferences_file.is_file():
            try:
                data = json.loads(self.preferences_file.read_text(encoding="utf-8"))
                self._preferences = UserPreferences(**data)
            except Exception as exc:
                _log.debug(
                    "contextual: failed to load preferences from %s: %s",
                    self.preferences_file,
                    exc,
                )
                self._preferences = UserPreferences()

    def _sync_context_maxlen(self) -> None:
        if self._context.maxlen != self.max_context_length:
            self._context = deque(
                list(self._context)[-self.max_context_length :],
                maxlen=self.max_context_length,
            )

    def _save(self) -> None:
        """Save context and preferences to disk."""
        try:
            self.context_file.write_text(
                json.dumps([c.__dict__ for c in self._context], indent=2),
                encoding="utf-8",
            )
            self.preferences_file.write_text(
                json.dumps(self._preferences.__dict__, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _log.debug("contextual preferences save failed: %s", exc)

    def add_prompt(self, prompt: str, recalled_memorias: list[str]) -> None:
        """Add a prompt to the conversation history."""
        self._sync_context_maxlen()
        context = PromptContext(
            timestamp=datetime.now(UTC).isoformat(),
            prompt=prompt,
            recalled_memorias=recalled_memorias,
        )
        self._context.append(context)
        self._save()

    def get_recent_context(self, n: int = 5) -> list[PromptContext]:
        """Get the N most recent prompts."""
        self._sync_context_maxlen()
        return list(self._context)[-n:]

    def record_feedback(self, memory_id: str, memoria_type: str, entities: list[str]) -> None:
        """Record user feedback (e.g., they clicked/viewed a memory)."""
        # Don't let the bulk `reference` tier teach a type preference. It
        # dominates the corpus, so learning "prefer reference" would amplify
        # the very noise the recall tiering exists to suppress (this is the
        # bug that produced `preferred_types: {note: 0.6}` pre-tiering). See
        # `memo.tiers`.
        from memo.tiers import REFERENCE_TYPES

        if memoria_type not in REFERENCE_TYPES:
            # Boost the type
            self._preferences.preferred_types[memoria_type] = (
                self._preferences.preferred_types.get(memoria_type, 0.5) + 0.1
            )
            # Cap at 1.0
            if self._preferences.preferred_types[memoria_type] > 1.0:
                self._preferences.preferred_types[memoria_type] = 1.0

        # Boost the entities
        for entity in entities:
            self._preferences.preferred_entities[entity] = (
                self._preferences.preferred_entities.get(entity, 0.5) + 0.1
            )
            if self._preferences.preferred_entities[entity] > 1.0:
                self._preferences.preferred_entities[entity] = 1.0

        self._preferences.last_updated = datetime.now(UTC).isoformat()
        self._save()

    def get_preferences(self) -> UserPreferences:
        """Get current user preferences."""
        return self._preferences


class ContextualRecall:
    """Contextual recall with conversation history and preference learning.

    Args:
        memory: The Memory instance to search.
        context_store: The ContextStore for history and preferences.
    """

    def __init__(self, memory: Any, context_store: ContextStore) -> None:
        self.memory = memory
        self.context = context_store

    def search_with_context(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[ContextualSearchResult]:
        """Search with contextual re-ranking.

        Args:
            query: Search query.
            limit: Max results.
            mode: Search mode (vec, bm25, hybrid).

        Returns:
            List of ContextualSearchResult with contextual scores.
        """
        # Get base search results
        hits = self.memory.search(query, limit=limit * 2, mode=mode)  # Fetch more for re-ranking

        # Get recent context
        recent_context = self.context.get_recent_context(n=3)
        context_entities = self._extract_entities_from_context(recent_context)

        # Get user preferences
        prefs = self.context.get_preferences()

        # Re-rank with contextual factors
        contextual_results = []
        for hit in hits:
            contextual_score = hit.score or 0.0
            boost_factors = {}

            # Entity overlap boost
            memoria_entities = {
                e["name"]
                for e in self.memory.graph.memoria_entities(hit.id)
                if isinstance(e, dict) and e.get("name")
            }
            entity_overlap = len(memoria_entities & context_entities)
            if entity_overlap > 0:
                entity_boost = 0.1 * entity_overlap
                contextual_score += entity_boost
                boost_factors["entity_overlap"] = entity_boost

            # Type preference boost
            type_boost = prefs.preferred_types.get(hit.type, 0.0) * 0.05
            contextual_score += type_boost
            boost_factors["type_preference"] = type_boost

            # Entity preference boost
            entity_pref_boost = sum(
                prefs.preferred_entities.get(e, 0.0) * 0.03 for e in memoria_entities
            )
            contextual_score += entity_pref_boost
            boost_factors["entity_preference"] = entity_pref_boost

            # Recency boost (if user prefers recent)
            try:
                updated_dt = datetime.fromisoformat(hit.updated.replace("Z", "+00:00"))
                days_old = (datetime.now(UTC) - updated_dt).days
                if days_old < 30:
                    recency_boost = (1 - days_old / 30) * prefs.recency_weight * 0.1
                    contextual_score += recency_boost
                    boost_factors["recency"] = recency_boost
            except (ValueError, TypeError) as exc:
                _log.debug(
                    "contextual: bad updated timestamp %r, skipping recency boost: %s",
                    hit.updated,
                    exc,
                )

            contextual_results.append(
                ContextualSearchResult(
                    memory_id=hit.id,
                    title=hit.title,
                    original_score=hit.score,
                    contextual_score=contextual_score,
                    boost_factors=boost_factors,
                    snippet=(hit.body or "")[:200],
                )
            )

        # Sort by contextual score and return top N
        contextual_results.sort(key=lambda r: r.contextual_score, reverse=True)
        return contextual_results[:limit]

    def _extract_entities_from_context(self, context: list[PromptContext]) -> set[str]:
        """Extract entities mentioned in recent conversation context."""
        entities = set()
        for ctx in context:
            for token in ctx.prompt.lower().replace("-", " ").split():
                token = token.strip(".,:;!?()[]{}\"'")
                if len(token) >= 3:
                    entities.add(token)
        return entities

    def record_search(self, query: str, recalled_ids: list[str]) -> None:
        """Record a search in the context history."""
        self.context.add_prompt(query, recalled_ids)

    def record_click(self, memory_id: str) -> None:
        """Record that the user clicked/viewed a memory."""
        rec = self.memory.get(memory_id)
        if rec:
            entities = [
                e["name"]
                for e in self.memory.graph.memoria_entities(memory_id)
                if isinstance(e, dict) and e.get("name")
            ]
            self.context.record_feedback(memory_id, rec.type, entities)
            # Closed-loop "used" signal: a fetched memory is one acted on, not
            # just shown. Cross-referenced against recall.log by
            # `memo usefulness` → referenced_rate. Best-effort, off hot path.
            try:
                from memo.dashboard import append_usage_log

                append_usage_log(self.context.context_file.parent, memory_id)
            except Exception as exc:
                _log.debug("contextual: failed to append usage log for %s: %s", memory_id, exc)


__all__ = [
    "ContextStore",
    "ContextualRecall",
    "ContextualSearchResult",
    "PromptContext",
    "UserPreferences",
]
