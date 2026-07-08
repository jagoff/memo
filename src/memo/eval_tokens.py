"""`memo eval tokens` — token-economy measurement gate.

Mirrors the `memo eval recall --gate` retrieval-regression discipline for the
token economy. Measures, per lever, whether it saves tokens WITHOUT degrading
recall, over a committed corpus. Two planes memo controls directly (no LLM):

  P1 recall-output block size — quality guard = precision over ids that survive
     into the injected block.
  P2 capture size — quality guard = a labeled must-keep row surviving the crush.

Design note: we measure the transformation FUNCTION at unit level; a lever need
not be wired into the live pipeline to be measured. The data decides wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

_CHARS_PER_TOKEN = 4  # keep in lockstep with token_meter._CHARS_PER_TOKEN

MIN_SAVING_FRAC = 0.05  # a lever must cut >=5% tokens on its plane to count
QUALITY_EPS = 0.0  # precision must not drop; must-keep row must survive


def count_tokens(text: str) -> int:
    """Estimated tokens for `text` via the chars/4 heuristic (ceil)."""
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def surviving_ids(block_text: str, candidate_ids: list[str]) -> set[str]:
    """Which candidate memory ids survived into a rendered recall block.

    The renderer prints each hit as `**[<id[:8]>] title**`, so an id counts as
    surviving iff its 8-char short prefix appears anywhere in the block.
    """
    return {cid for cid in candidate_ids if cid[:8] in block_text}


@dataclass
class LeverRow:
    lever: str
    plane: str  # "recall_output" | "capture"
    tokens_off: int
    tokens_on: int
    quality_off: float
    quality_on: float

    @property
    def saved_frac(self) -> float:
        return (self.tokens_off - self.tokens_on) / self.tokens_off if self.tokens_off else 0.0

    @property
    def quality_delta(self) -> float:
        return self.quality_on - self.quality_off

    @property
    def passed(self) -> bool:
        return self.saved_frac >= MIN_SAVING_FRAC and self.quality_delta >= -QUALITY_EPS


@dataclass
class P1Sample:
    tokens_off: int
    tokens_on: int
    prec_off: float
    prec_on: float


def _precision(block: str, expect_ids: list[str]) -> float:
    if not expect_ids:
        return 1.0
    return len(surviving_ids(block, expect_ids)) / len(expect_ids)


def measure_recall_sample(block_off: str, block_on: str, expect_ids: list[str]) -> P1Sample:
    """One prompt: token + precision measurement of the OFF vs ON recall block."""
    return P1Sample(
        tokens_off=count_tokens(block_off),
        tokens_on=count_tokens(block_on),
        prec_off=_precision(block_off, expect_ids),
        prec_on=_precision(block_on, expect_ids),
    )


def aggregate_recall(lever: str, samples: list[P1Sample]) -> LeverRow:
    """Fold per-prompt P1 samples into one LeverRow (sum tokens, mean precision)."""
    n = len(samples) or 1
    return LeverRow(
        lever=lever,
        plane="recall_output",
        tokens_off=sum(s.tokens_off for s in samples),
        tokens_on=sum(s.tokens_on for s in samples),
        quality_off=sum(s.prec_off for s in samples) / n,
        quality_on=sum(s.prec_on for s in samples) / n,
    )
