"""Next-turn verdict classifier — did the USER's next message accept, reject,
or correct the previous answer?

grounding.log proves an answer USED a recalled memory; it says nothing about
whether the user then replied "no, eso está mal". This module classifies the
user's next turn (heuristic regex, ES+EN; optional MLX pass — see
``record_verdicts``) and attributes the verdict to the PRIOR turn's recalled
ids: implicit ``source_feedback`` rows plus negative labels for the tuner
(``eval_recall.harvest_negative_labels``).

Runs from the Stop hook (`memo capture-stop`) — NEVER from the 5s recall hook.
Foundation-leaf style (like grounding.py): stdlib at module level;
memo.memory / memo.llm imports stay deferred inside functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# Only the head of the message carries the reaction; a long follow-up prompt
# may mention "gracias" or "error" deep inside for unrelated reasons.
_HEAD_CHARS = 200

# A verdict applies to a recall at most this many turns back — beyond that the
# reaction is about something else, not the recalled memories.
_MAX_TURN_GAP = 2


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# Order of evaluation: negative → correction → positive. "no funciona" must be
# negative (outcome), "no, eso está mal" correction (content), and a leading
# negation must never be swallowed by a trailing "gracias".
_NEGATIVE_RE = _compile(
    (
        r"\bno (funciona|anda|sirve|sirvió|compila)\b",
        r"\bsigue (fallando|roto|rota|igual|sin funcionar)\b",
        r"\bsigue el (error|problema)\b",
        r"\b(doesn'?t|didn'?t|does not|did not) work\b",
        r"\bstill (failing|broken|fails|not working)\b",
        r"\b(same|mismo) error\b",
        r"\bempeoró\b",
        r"\bworse than before\b",
    )
)
_CORRECTION_RE = _compile(
    (
        r"^\s*(no|nope)\b[\s,.:;—-]",
        r"\beso (no es|está mal|es incorrecto)\b",
        r"\bno es así\b",
        r"\bno era eso\b",
        r"\bte equivocas(te)?\b",
        r"\bestás? equivocad[oa]\b",
        r"\bthat'?s (wrong|not right|incorrect|not it)\b",
        r"\bthat is (wrong|not right|incorrect)\b",
        r"\bnot what i (asked|meant|said)\b",
        r"\ben realidad no\b",
    )
)
_POSITIVE_RE = _compile(
    (
        # bare exclamations only count when they OPEN the message
        r"^\s*(perfecto|genial|excelente|buenísimo|joya|perfect|great|awesome|nice)\b",
        r"\bgracias\b",
        r"\bthanks?\b",
        r"\bthank you\b",
        r"\b(ya|ahora) (funciona|anda|compila)\b",
        r"\bfuncionó\b",
        r"\banduvo\b",
        r"\b(that|it) worked\b",
        r"\bworks now\b",
        r"\bworking now\b",
    )
)


def score_next_turn(
    state_dir: Any,
    payload: dict[str, Any],
    *,
    llm_classify: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    """Classify the latest user turn as a reaction to the PRIOR turn's
    recalled memories; persist to verdict.log. Returns the record or None
    (nothing to score / no verdict / already scored). Never raises."""
    try:
        from memo.dashboard import (
            append_verdict_log,
            read_recall_hook_log,
            read_verdict_log,
        )

        session_id = (payload.get("session_id") or "").strip()
        if not session_id:
            return None
        rows = [
            r
            for r in read_recall_hook_log(state_dir, limit=2000)
            if r.get("session_id") == session_id and isinstance(r.get("turn"), int)
        ]
        if len(rows) < 2:
            return None
        latest = rows[-1]
        turn = int(latest["turn"])
        reaction = str(latest.get("prompt") or "")
        if not reaction:
            return None
        # Idempotent across repeated Stop events for the same turn.
        if any(
            v.get("session_id") == session_id and v.get("turn") == turn
            for v in read_verdict_log(state_dir, limit=200)
        ):
            return None
        verdict = classify_reaction(reaction)
        method = "heuristic"
        if verdict is None and llm_classify is not None:
            verdict = llm_classify(reaction)
            method = "llm"
        if verdict not in ("positive", "negative", "correction"):
            return None
        prior = None
        for r in reversed(rows[:-1]):
            if int(r["turn"]) < turn and (r.get("hits") or []):
                prior = r
                break
        if prior is None or (turn - int(prior["turn"])) > _MAX_TURN_GAP:
            return None
        recall_ids = [h.get("id") for h in (prior.get("hits") or []) if h.get("id")]
        if not recall_ids:
            return None
        record: dict[str, Any] = {
            "session_id": session_id,
            "turn": turn,
            "prior_turn": int(prior["turn"]),
            "verdict": verdict,
            "prompt": str(prior.get("prompt") or ""),
            "reaction": reaction,
            "recall_ids": recall_ids,
            "method": method,
        }
        append_verdict_log(state_dir, **record)
        return record
    except Exception:
        return None


def classify_reaction(text: str) -> str | None:
    """Classify a user message as a reaction to the previous answer.

    Returns ``"positive"`` / ``"negative"`` / ``"correction"``, or ``None``
    when the head of the message carries no reaction signal. Conservative by
    design: prefer None over a wrong verdict."""
    head = (text or "").strip()[:_HEAD_CHARS]
    if not head:
        return None
    if any(r.search(head) for r in _NEGATIVE_RE):
        return "negative"
    if any(r.search(head) for r in _CORRECTION_RE):
        return "correction"
    if any(r.search(head) for r in _POSITIVE_RE):
        return "positive"
    return None
