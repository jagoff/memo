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
