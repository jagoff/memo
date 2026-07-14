"""Source->claim entailment scorer for the write path (grounding-judge) and the
ask path (ask-strict-threshold).

Pure: delegates the LLM call to a chat object PASSED IN by the caller (which
already owns the deferred MLX import). This module never imports mlx. Runs only
off the 5s recall hook (capture / ask), never on it.
"""

from __future__ import annotations

import re
from typing import Any

_SYSTEM_PROMPT = (
    "You are a strict grounding judge. You are given a SOURCE (raw text the "
    "claim was drawn from) and a CLAIM. Rate from 0 to 100 how fully the SOURCE "
    "supports the CLAIM: 100 = fully entailed by the source; 0 = unsupported, "
    "contradicted, or fabricated. Reply with ONLY the integer, nothing else."
)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(raw: str) -> float | None:
    """First number in `raw`, read as a 0-100 score, returned as 0.0-1.0.
    None when there is no number. Values are clamped to [0, 100]."""
    if not raw:
        return None
    m = _NUM_RE.search(raw)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    val = max(0.0, min(100.0, val))
    return round(val / 100.0, 4)


def score_grounding(chat: Any, model: str, *, source: str, claim: str) -> float | None:
    """Score how well `source` entails `claim` (0.0-1.0); None if unjudgeable.

    Fail-open by design: any chat error or unparseable reply returns None so the
    caller does NOT quarantine on a judge failure (never punish a claim because
    the judge was unavailable)."""
    try:
        resp = chat.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"SOURCE:\n{source[:6000]}\n\nCLAIM:\n{claim[:1500]}\n\nScore (0-100):",
                },
            ],
            options={"temperature": 0.0, "num_predict": 8},
        )
    except Exception:
        return None
    raw = ((resp.get("message") or {}).get("content") or "") if isinstance(resp, dict) else ""
    return parse_score(str(raw).strip())
