"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Proactive memory suggestions — detect patterns and suggest saving.

Extends the save-side capture with proactive suggestions:
- Pattern detection in ongoing conversations
- Real-time suggestions to save important insights
- Learning which suggestions the user accepts
- Confidence scoring for suggestions

## Pattern Detection

Monitors conversation patterns to identify:
- Repeated themes/topics that aren't yet memorized
- Decision points (e.g. "I'll use X instead of Y")
- Technical discoveries or learnings
- Bug fixes or workarounds

## Suggestion Engine

Uses the helper LLM to analyze the current conversation context and
suggest potential memories to save. Suggestions include:
- Title suggestion
- Type suggestion
- Tags suggestion
- Confidence score

## Learning

Tracks which suggestions the user accepts to improve future suggestions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.llm import MLXChat

_SUGGESTION_SYSTEM_PROMPT = """You analyze a conversation to suggest potential memories to save.

You receive the recent conversation context (user + assistant turns).
Output a JSON object:

{
  "suggestions": [
    {
      "title": "concise title",
      "type": "decision|fact|bug|preference|note",
      "tags": ["tag1", "tag2"],
      "body_snippet": "key insight to save",
      "confidence": 0.0-1.0,
      "rationale": "why this is worth saving"
    }
  ]
}

Rules:
- Only suggest 1-3 high-quality insights
- Confidence > 0.7 for strong suggestions
- Focus on decisions, learnings, bug fixes
- Skip trivial chatter or confirmations
- Output ONLY the JSON, no markdown fences, no commentary."""


@dataclass
class Suggestion:
    """A proactive memory suggestion."""
    title: str
    type: str
    tags: list[str]
    body_snippet: str
    confidence: float
    rationale: str
    suggested_at: str


@dataclass
class SuggestionFeedback:
    """Feedback on a suggestion."""
    suggestion_id: str
    accepted: bool
    timestamp: str


class ProactiveSuggester:
    """Proactively suggests memories to save based on conversation patterns.

    Args:
        memory: The Memory instance to search against.
        chat: Optional MLXChat instance for LLM-based suggestions.
    """

    def __init__(self, memory: Any, chat: MLXChat | None = None) -> None:
        self.memory = memory
        self._chat = chat
        self._feedback_log: list[SuggestionFeedback] = []

    def _ensure_chat(self) -> MLXChat:
        if self._chat is None:
            self._chat = MLXChat()
        return self._chat

    def analyze_conversation(
        self,
        recent_turns: list[dict[str, str]],
        limit: int = 3,
    ) -> list[Suggestion]:
        """Analyze recent conversation turns and suggest memories.

        Args:
            recent_turns: List of {"user": "...", "assistant": "..."} turns.
            limit: Maximum suggestions to return.

        Returns:
            List of Suggestion objects, sorted by confidence descending.
        """
        if not recent_turns:
            return []

        # Build context string
        context = "Recent conversation:\n\n"
        for _i, turn in enumerate(recent_turns[-5:], 1):  # Last 5 turns
            context += f"User: {turn.get('user', '')}\n"
            context += f"Assistant: {turn.get('assistant', '')}\n\n"

        # Use LLM to generate suggestions
        chat = self._ensure_chat()

        try:
            out = chat.chat(
                model=self.memory.cfg.helper_model,
                messages=[
                    {"role": "system", "content": _SUGGESTION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.0, "max_tokens": 512},
            )
            raw = (out.get("message") or {}).get("content") or ""
        except Exception:
            return []

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)

        try:
            data = json.loads(raw)
        except Exception:
            return []

        suggestions = []
        for item in data.get("suggestions", [])[:limit]:
            if item.get("confidence", 0) >= 0.6:  # Filter low confidence
                suggestions.append(
                    Suggestion(
                        title=item.get("title", ""),
                        type=item.get("type", "note"),
                        tags=item.get("tags", []),
                        body_snippet=item.get("body_snippet", ""),
                        confidence=float(item.get("confidence", 0.0)),
                        rationale=item.get("rationale", ""),
                        suggested_at=datetime.now(UTC).isoformat(),
                    )
                )

        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    def record_feedback(self, suggestion: Suggestion, accepted: bool) -> None:
        """Record user feedback on a suggestion.

        Args:
            suggestion: The suggestion that was shown to the user.
            accepted: Whether the user accepted (saved) it.
        """
        # Generate a simple ID from title + timestamp
        suggestion_id = f"{suggestion.title[:20]}_{suggestion.suggested_at[:10]}".replace(" ", "_")

        feedback = SuggestionFeedback(
            suggestion_id=suggestion_id,
            accepted=accepted,
            timestamp=datetime.now(UTC).isoformat(),
        )

        self._feedback_log.append(feedback)

        # In a full implementation, would use this to adjust suggestion thresholds
        # For now, just log it

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get statistics on suggestion feedback.

        Returns:
            Dict with acceptance rate, total suggestions, etc.
        """
        if not self._feedback_log:
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "acceptance_rate": 0.0,
            }

        total = len(self._feedback_log)
        accepted = sum(1 for f in self._feedback_log if f.accepted)
        rejected = total - accepted

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
        }

    def detect_patterns(
        self,
        transcript_path: Path,
    ) -> dict[str, Any]:
        """Detect patterns in a full transcript file.

        Args:
            transcript_path: Path to the Claude Code JSONL transcript.

        Returns:
            Dict with pattern statistics.
        """
        # This would analyze the full transcript for recurring themes
        # For now, return a placeholder
        return {
            "recurring_themes": [],
            "decision_points": 0,
            "technical_discoveries": 0,
            "total_turns": 0,
        }


__all__ = [
    "ProactiveSuggester",
    "Suggestion",
    "SuggestionFeedback",
]

