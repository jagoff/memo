"""Registry of dark (default-OFF) flags the graduation controller may prove and
flip. A candidate is measurable when its effect shows up in the offline recall
eval (precision@K / noise@K); such candidates set ``auto_flip=True``. Flags whose
effect is not offline-measurable stay ``auto_flip=False`` (report-only)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    flag: str                          # the MEMO_* flag to graduate
    on_flags: dict[str, str]           # flag_overrides expressing the ON state
    epsilon: float = 0.0               # min precision gain to count a winning night
    k: int = 5                         # consecutive winning nights required to flip
    auto_flip: bool = True             # False => report-only, never writes overlay


def default_candidates() -> list[Candidate]:
    """Seed set. Phase 1 expands this with the wider tuner knobs; Phase 3 adds
    report-only proactive flags. The seed is a genuinely OFF, offline-measurable
    retrieval lever whose ON/OFF delta the eval corpus can attribute."""
    return [
        Candidate(
            flag="MEMO_GRAPH_SIGNAL_ENABLED",
            on_flags={
                "MEMO_GRAPH_SIGNAL_ENABLED": "1",
                "MEMO_GRAPH_REASON_ENABLED": "1",
                "MEMO_GRAPH_HUB_SUPPRESSION": "1",
            },
            epsilon=0.0,
            k=5,
            auto_flip=True,
        ),
    ]
