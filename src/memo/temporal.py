"""Temporal reasoning & contradiction detection for memory corpus.

Analyzes the temporal dimension of memories to detect:
- Contradictions between facts over time (e.g. "used Ollama" → "migrated to MLX")
- Evolution of decisions/opinions
- Stale/outdated information
- Temporal patterns in the user's knowledge

## Schema

Uses existing `meta.created` / `meta.updated` timestamps. No new tables —
all analysis is computed on-the-fly from the corpus + history.

## Detection Strategy

1. **Semantic contradiction detection**: For a given entity/topic, search
   for memories that express conflicting facts. Uses the helper LLM to
   classify pairs as {contradiction, evolution, unrelated, consistent}.

2. **Temporal evolution tracking**: For entities with multiple memories over
   time, build a timeline and detect phase changes (e.g. opinion shifts).

3. **Staleness detection**: Mark memories that haven't been accessed in
   N months and reference technologies/tools that may have evolved.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from memo.llm import MLXChat

_log = logging.getLogger(__name__)

# Cap on rows pulled into memory for whole-corpus temporal analysis. A corpus
# larger than this is silently truncated — we log a warning when the cap bites.
_ANALYSIS_ROW_CAP = 10_000
def _pair_classify_timeout() -> float:
    """Get timeout from flag, with hard fallback for early boot."""
    try:
        from memo.flags import flag_float
        return flag_float("MEMO_CONTRADICTION_TIMEOUT") or 30.0
    except Exception:
        return 30.0

_CONTRADICTION_SYSTEM_PROMPT = """You analyze two memory notes from a personal archive to detect temporal contradictions.

You receive two memories with timestamps. Output a JSON object:

{
  "relationship": "contradiction" | "evolution" | "consistent" | "unrelated",
  "rationale": "1-2 sentence explanation",
  "confidence": 0.0-1.0
}

Definitions:
- "contradiction": the two notes state mutually exclusive facts (e.g. "I use Ollama" vs "I migrated to MLX")
- "evolution": the later note supersedes or refines the earlier one (e.g. "I prefer X" → "I switched to Y because Z")
- "consistent": both notes agree, possibly adding detail
- "unrelated": the notes discuss different topics; temporal comparison is meaningless

Output ONLY the JSON, no markdown fences, no commentary."""


@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction between two memories."""

    memoria_id_a: str
    memoria_id_b: str
    title_a: str
    title_b: str
    date_a: str
    date_b: str
    relationship: str
    rationale: str
    confidence: float


@dataclass(frozen=True)
class TimelineEvent:
    """One event in an entity's temporal timeline."""

    memoria_id: str
    title: str
    date: str
    type: str
    snippet: str


@dataclass(frozen=True)
class EntityTimeline:
    """Timeline of all memories mentioning a specific entity."""

    entity_name: str
    entity_type: str
    events: list[TimelineEvent]
    first_seen: str
    last_seen: str


class TemporalAnalyzer:
    """Analyzes temporal patterns and contradictions in the memory corpus.

    Args:
        memory: The Memory instance to query against.
        chat: Optional MLXChat instance for LLM-based classification.
            If None, a new one is created on first use.
    """

    def __init__(self, memory: Any, chat: MLXChat | None = None) -> None:
        self.memory = memory
        self._chat = chat
        self._chat_lock = threading.Lock()

    def _ensure_chat(self) -> MLXChat:
        if self._chat is None:
            with self._chat_lock:
                if self._chat is None:
                    self._chat = MLXChat()
        return self._chat

    def detect_entity_contradictions(
        self,
        entity_name: str,
        entity_type: str | None = None,
        confidence_threshold: float = 0.7,
        max_pairs: int = 20,
    ) -> list[Contradiction]:
        """Find contradictions among memories mentioning a specific entity.

        Args:
            entity_name: The entity to analyze (e.g. "ollama", "mlx").
            entity_type: Optional entity type filter from graph.
            confidence_threshold: Minimum confidence to include a contradiction.
            max_pairs: Maximum number of pairs to analyze (LLM is expensive).

        Returns:
            List of detected contradictions, sorted by confidence descending.
        """
        # Get memories mentioning this entity
        memoria_ids = self.memory.graph.entity_memorias(entity_name, entity_type)
        if len(memoria_ids) < 2:
            return []

        # Fetch the actual records
        records = []
        for mid in memoria_ids[:50]:  # Cap to avoid explosion
            rec = self.memory.get(mid)
            if rec:
                records.append(rec)

        if len(records) < 2:
            return []

        # Sort by date for chronological pairing
        records.sort(key=lambda r: r.updated)

        contradictions: list[Contradiction] = []
        pair_count = 0

        # Compare each pair (limit to max_pairs)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if pair_count >= max_pairs:
                    break
                pair_count += 1

                r1, r2 = records[i], records[j]
                # Only compare if they're temporally separated (same-day edits are likely revisions)
                try:
                    d1 = datetime.fromisoformat(r1.updated.replace("Z", "+00:00"))
                    d2 = datetime.fromisoformat(r2.updated.replace("Z", "+00:00"))
                    if abs((d2 - d1).days) < 1:
                        continue
                except (ValueError, TypeError, AttributeError):
                    _log.debug("temporal: skip pair date parse error")

                contr = self._classify_pair(r1, r2)
                if contr and contr.confidence >= confidence_threshold:
                    contradictions.append(contr)

        contradictions.sort(key=lambda c: c.confidence, reverse=True)
        return contradictions

    def _classify_pair(self, r1: Any, r2: Any) -> Contradiction | None:
        """Use LLM to classify the relationship between two memories."""
        chat = self._ensure_chat()

        prompt = f"""Memory A (date: {r1.updated}):
Title: {r1.title}
Type: {r1.type}
Body: {(r1.body or "")[:1000]}

Memory B (date: {r2.updated}):
Title: {r2.title}
Type: {r2.type}
Body: {(r2.body or "")[:1000]}

        Analyze the temporal relationship between these two notes."""

        try:
            from memo.memory.record import chat_with_timeout

            out = chat_with_timeout(
                chat,
                timeout=_pair_classify_timeout(),
                model=self.memory.cfg.helper_model,
                messages=[
                    {"role": "system", "content": _CONTRADICTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "thinking": False,
                },
            )
            if out is None:
                return None
            raw = (out.get("message") or {}).get("content") or ""
        except Exception:
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)

        try:
            import json

            data = json.loads(raw)
        except (ValueError, TypeError):
            return None

        if not isinstance(data, dict):
            return None

        relationship = data.get("relationship", "unrelated")
        if relationship not in ("contradiction", "evolution", "consistent", "unrelated"):
            return None

        # Only flag contradictions and evolutions
        if relationship not in ("contradiction", "evolution"):
            return None

        return Contradiction(
            memoria_id_a=r1.id,
            memoria_id_b=r2.id,
            title_a=r1.title,
            title_b=r2.title,
            date_a=r1.updated,
            date_b=r2.updated,
            relationship=relationship,
            rationale=data.get("rationale", "")[:200],
            confidence=float(data.get("confidence", 0.0)),
        )

    def build_entity_timeline(
        self,
        entity_name: str,
        entity_type: str | None = None,
    ) -> EntityTimeline | None:
        """Build a chronological timeline of all memories mentioning an entity."""
        memoria_ids = self.memory.graph.entity_memorias(entity_name, entity_type)
        if not memoria_ids:
            return None

        events: list[TimelineEvent] = []
        first_seen = None
        last_seen = None

        for mid in memoria_ids:
            rec = self.memory.get(mid)
            if not rec:
                continue

            snippet = (rec.body or "")[:200]
            events.append(
                TimelineEvent(
                    memoria_id=rec.id,
                    title=rec.title,
                    date=rec.updated,
                    type=rec.type,
                    snippet=snippet,
                )
            )

            if not first_seen or rec.updated < first_seen:
                first_seen = rec.updated
            if not last_seen or rec.updated > last_seen:
                last_seen = rec.updated

        if not events:
            return None

        events.sort(key=lambda e: e.date)

        return EntityTimeline(
            entity_name=entity_name,
            entity_type=entity_type or "unknown",
            events=events,
            first_seen=first_seen or "",
            last_seen=last_seen or "",
        )

    def detect_stale_memorias(
        self,
        days_threshold: int = 180,
        min_access_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Find memories that may be stale based on age and lack of access.

        Args:
            days_threshold: Days since last update to consider stale.
            min_access_count: Minimum access count to exclude (frequently-accessed
                old memories may still be relevant).

        Returns:
            List of potentially stale memories with metadata.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=days_threshold)).isoformat()

        # Get all memories, filter by date
        all_records = self.memory.list(limit=_ANALYSIS_ROW_CAP)
        if len(all_records) >= _ANALYSIS_ROW_CAP:
            _log.warning(
                "detect_stale_memorias: corpus hit the %d-row cap; older "
                "memories were not scanned.",
                _ANALYSIS_ROW_CAP,
            )
        stale = []

        for rec in all_records:
            if rec.updated < cutoff:
                # Check access count from history if available
                access_count = 0
                try:
                    history_events = self.memory.history.list_recent(record_id=rec.id, limit=100)
                    # Count only non-save events (saves are initial creation)
                    access_count = sum(1 for e in history_events if e.get("op") != "save")
                except Exception as exc:
                    _log.debug(
                        "temporal: history fetch failed for %s, treating as zero access: %s",
                        rec.id,
                        exc,
                    )

                if access_count < min_access_count:
                    stale.append(
                        {
                            "id": rec.id,
                            "title": rec.title,
                            "type": rec.type,
                            "updated": rec.updated,
                            "days_since_update": (
                                datetime.now(UTC)
                                - datetime.fromisoformat(rec.updated.replace("Z", "+00:00"))
                            ).days,
                            "access_count": access_count,
                        }
                    )

        stale.sort(key=lambda x: x["days_since_update"], reverse=True)
        return stale

    def detect_temporal_patterns(self) -> dict[str, Any]:
        """Analyze high-level temporal patterns across the entire corpus.

        Returns:
            Dict with metrics like:
            - memorias_per_month: histogram of creation activity
            - type_distribution_over_time: how memory types change over time
            - most_active_entities: entities with most temporal churn
        """
        all_records = self.memory.list(limit=_ANALYSIS_ROW_CAP)
        if len(all_records) >= _ANALYSIS_ROW_CAP:
            _log.warning(
                "detect_temporal_patterns: corpus hit the %d-row cap; older "
                "memories were not included.",
                _ANALYSIS_ROW_CAP,
            )

        # Memories per month
        monthly: defaultdict[str, int] = defaultdict(int)
        for rec in all_records:
            try:
                dt = datetime.fromisoformat(rec.created.replace("Z", "+00:00"))
                key = f"{dt.year}-{dt.month:02d}"
                monthly[key] += 1
            except (ValueError, TypeError, AttributeError):
                _log.debug("temporal: skip record with unparseable created date")

        # Type distribution over time
        type_over_time: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for rec in all_records:
            try:
                dt = datetime.fromisoformat(rec.created.replace("Z", "+00:00"))
                key = f"{dt.year}-{dt.month:02d}"
                type_over_time[key][rec.type] += 1
            except (ValueError, TypeError, AttributeError):
                _log.debug("temporal: skip type-over-time record with unparseable created date")

        # Most active entities (by number of memories)
        entity_counts: defaultdict[str, int] = defaultdict(int)
        top_entities = self.memory.graph.top_entities(limit=100)
        for ent in top_entities:
            entity_counts[ent["name"]] = ent.get("mention_count", 0)

        return {
            "memorias_per_month": dict(sorted(monthly.items())),
            "type_distribution_over_time": {k: dict(v) for k, v in sorted(type_over_time.items())},
            "most_active_entities": dict(
                sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
        }


__all__ = [
    "Contradiction",
    "EntityTimeline",
    "TemporalAnalyzer",
    "TimelineEvent",
]
