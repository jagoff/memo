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

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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


@contextmanager
def env_pins(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set MEMO_* env vars, restoring prior state on exit."""
    prior: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def render_block(
    hits: list[Any], env: dict[str, str], *, body_chars: int, token_budget: int
) -> str:
    """Render the recall block for `hits` under a lever's env pins.

    Applies L4 verbosity steering when MEMO_RECALL_VERBOSITY_LEVEL > 0, so the
    lever's true effect on the injected block size is measured.
    """
    from memo.cli_recall_hook import maybe_inject_verbosity_steering
    from memo.flags_recall import flag_recall_verbosity_level
    from memo.recall_logic import render_recall_context

    with env_pins(env):
        block = render_recall_context(
            hits, [], turn=2, body_chars=body_chars, token_budget=token_budget
        )
        level = flag_recall_verbosity_level()
        if level > 0:
            block = maybe_inject_verbosity_steering(block, level)
        return block


# The P1 levers to measure. Env-expressible knobs that transform the injected
# recall block. Baseline (OFF) is the empty env; each lever is its ON delta.
RECALL_LEVERS: list[dict[str, Any]] = [
    {"name": "recall_format_compact", "env": {"MEMO_RECALL_FORMAT": "compact"}},
    {"name": "verbosity_steer_L2", "env": {"MEMO_RECALL_VERBOSITY_LEVEL": "2"}},
]
