"""Real-time Belief Network & Automated Invalidation Engine."""

from __future__ import annotations

import logging
import re

from memo.kernel.world_model import WorldModel

_logger = logging.getLogger(__name__)

# Keywords indicating explicit contradiction/replacement
_REPLACEMENT_MARKERS = re.compile(
    r"\b(instead of|deprecated|replaced|no longer|migrated from|don't use|do not use)\b",
    re.IGNORECASE,
)


class BeliefNetwork:
    """Belief graph network for real-time invalidation and contradiction detection."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def auto_invalidate_conflicts(self, new_statement: str, new_topic: str) -> list[str]:
        """Detect and invalidate conflicting beliefs in real time."""
        invalidated: list[str] = []

        if not _REPLACEMENT_MARKERS.search(new_statement):
            return invalidated

        # Find existing active beliefs under the same topic
        active_beliefs = self.world_model.get_active_beliefs(topic=new_topic)
        for belief in active_beliefs:
            # Check overlap in key terms
            b_words = {w.lower() for w in re.findall(r"\w+", belief.statement) if len(w) > 3}
            n_words = {w.lower() for w in re.findall(r"\w+", new_statement) if len(w) > 3}

            intersection = b_words & n_words
            if len(intersection) >= 2:
                # Invalidate stale belief
                self.world_model.invalidate_belief(
                    belief.id, reason=f"Superseded by: {new_statement[:60]}"
                )
                invalidated.append(belief.id)

        return invalidated
