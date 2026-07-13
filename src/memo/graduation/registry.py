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


@dataclass(frozen=True)
class NumericCandidate:
    """A numeric ranking knob the controller can tune + graduate. Distinct from
    the boolean ``Candidate``: OFF is the flag's CURRENT DEFAULT (``off_value``),
    never a blind "0" — so the ON/OFF precision delta is attributable to the knob
    alone. Expressed through ``eval_recall.Cfg.knob_overrides`` (the
    recall-faithful seam ``dream_tune.measure_rank_knob`` uses), pinning
    ``field`` on ``RankKnobs``."""

    flag: str                 # MEMO_* flag to graduate
    field: str                # RankKnobs field it pins (e.g. "mmr_lambda")
    off_value: float          # current default = the ON/OFF baseline
    on_value: float           # best-of-grid value to prove (the graduation target)
    grid: tuple[float, ...] = ()   # optional line-search grid; () = single A/B
    epsilon: float = 0.0
    k: int = 5
    auto_flip: bool = True     # False => report-only, never writes overlay


def numeric_candidates() -> list[NumericCandidate]:
    """Offline-measurable numeric ranking knobs (RankKnobs fields reachable via
    Cfg.knob_overrides). project_boost is report-only: its delta is vacuous on a
    corpus whose labels carry no project context (see plan Task 2 note)."""
    return [
        NumericCandidate(
            flag="MEMO_RECALL_MMR_LAMBDA",
            field="mmr_lambda",
            off_value=0.0,
            on_value=0.3,
            grid=(0.0, 0.3, 0.5, 0.7),
        ),
        NumericCandidate(
            flag="MEMO_RECALL_SYNTHESIS_BOOST",
            field="synthesis_boost",
            off_value=0.0,
            on_value=0.05,
            grid=(0.0, 0.05, 0.10),
        ),
        NumericCandidate(
            flag="MEMO_RECALL_GLOBAL_BOOST",
            field="global_boost",
            off_value=0.10,
            on_value=0.20,
            grid=(0.10, 0.20, 0.30),
        ),
        NumericCandidate(
            flag="MEMO_RECALL_PROJECT_BOOST",
            field="project_boost",
            off_value=0.25,
            on_value=0.35,
            grid=(0.25, 0.35, 0.50),
            auto_flip=False,  # not offline-measurable w/o project-tagged labels
        ),
    ]


def report_only_candidates() -> list[Candidate]:
    """Dark flags whose effect the SHARED offline eval cannot attribute, so they
    are shadow-counted only (auto_flip=False), never flipped. rerank-pool: the
    eval harness disables the reranker. Quarantine promotion: needs a grounding-
    replay evaluator (deferred). ``on_flags`` is inert here (never evaluated for
    a flip); it documents the ON intent."""
    return [
        Candidate(
            flag="MEMO_RECALL_RERANK_INPUT_K",
            on_flags={"MEMO_RECALL_RERANK_INPUT_K": "20"},
            auto_flip=False,
        ),
        Candidate(
            flag="MEMO_DREAM_GRADUATION_ENABLED",
            on_flags={"MEMO_DREAM_GRADUATION_ENABLED": "1"},
            auto_flip=False,
        ),
    ]


def all_candidates() -> list[Candidate | NumericCandidate]:
    """The full graduation set: boolean seed + numeric knobs + report-only."""
    return [*default_candidates(), *numeric_candidates(), *report_only_candidates()]
