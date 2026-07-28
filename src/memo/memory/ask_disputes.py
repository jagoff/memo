"""Dispute-awareness helpers for the ask/chat path (MEMO_ASK_DISPUTES).

Pure, fail-open helpers: the batched contradiction lookup mirrors the recall
hook's trust dossier (cli_recall_hook.py) and the gate is deterministic set
logic over the answer's [id8] citations (grounding.cited_ids) — no LLM calls.
See docs/SPECS/2026-07-28-ask-dispute-aware-design.md.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from memo.grounding import cited_ids, match_cited

_log = logging.getLogger(__name__)

# Appended to the ask system prompt at the call site (never baked into the
# user-replaceable prompt file) only when at least one source is disputed.
DISPUTE_PROMPT_SUFFIX = (
    "Snippets marked ⚔ disputed-by are contested by another memory. "
    "Present contested facts as contested and cite the disputing id."
)

# Cap disputing ids rendered per memory in headers/caveats.
_MAX_RENDERED = 3


def dispute_map(mem: Any, ids: Sequence[str]) -> dict[str, list[str]]:
    """Batched open+competing pair lookup, folded both directions.

    Fail-open: any storage error returns {} — ask must never break because
    the contradictions backend is absent or corrupt (dev_audit contract).
    """
    if not ids:
        return {}
    out: dict[str, list[str]] = {}
    try:
        store = mem.contradict_store
        pairs = store.pairs_for_ids(list(ids), status="open") + store.pairs_for_ids(
            list(ids), status="competing"
        )
        for p in pairs:
            out.setdefault(p.memory_id_a, []).append(p.memory_id_b)
            out.setdefault(p.memory_id_b, []).append(p.memory_id_a)
    except Exception:
        _log.debug("ask dispute lookup failed; continuing without", exc_info=True)
        return {}
    return out


def dispute_header_segment(mem_id: str, disputed: Mapping[str, list[str]]) -> str:
    """Snippet-header decoration, same style as graph/facts segments."""
    others = disputed.get(mem_id) or []
    if not others:
        return ""
    rendered = ", ".join(f"[{o[:8]}]" for o in others[:_MAX_RENDERED])
    return f"  |  ⚔ disputed-by: {rendered}"


def _memory_sources(sources: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [s for s in sources if s.get("source") == "memory"]


def _disputed_sources(sources: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        str(s["id"]): list(s["disputed_by"])
        for s in _memory_sources(sources)
        if s.get("disputed_by")
    }


def _contested_message(disputed: Mapping[str, list[str]], ids: Sequence[str]) -> str:
    first = sorted(ids)[0]
    other = (disputed.get(first) or ["?"])[0]
    extra = len(ids) - 1
    more = f" (+{extra} more disputed)" if extra > 0 else ""
    return (
        f"I couldn't find an undisputed answer: [{first[:8]}] and [{other[:8]}] "
        f"record conflicting facts{more}. Resolve with `memo contradict` or ask "
        "about one side explicitly."
    )


def contested_or_none(answer: str, sources: Sequence[Mapping[str, Any]]) -> str | None:
    """Contested-abstention message when the answer rests only on disputed
    evidence; None otherwise. Deterministic; fail-open to None."""
    try:
        disputed = _disputed_sources(sources)
        if not disputed:
            return None
        mem_ids = [str(s["id"]) for s in _memory_sources(sources)]
        cited = match_cited(cited_ids(answer), mem_ids)
        if cited:
            if cited <= set(disputed):
                return _contested_message(disputed, sorted(cited))
            return None
        if len(disputed) == len(mem_ids):
            return _contested_message(disputed, sorted(disputed))
        return None
    except Exception:
        _log.debug("dispute gate failed; leaving answer untouched", exc_info=True)
        return None


def append_dispute_caveat(answer: str, sources: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic caveat for cited-but-disputed evidence the LLM did not
    flag. Idempotent for a given (answer, sources); fail-open to `answer`."""
    try:
        disputed = _disputed_sources(sources)
        if not disputed or not answer:
            return answer
        mem_ids = [str(s["id"]) for s in _memory_sources(sources)]
        cited = match_cited(cited_ids(answer), mem_ids)
        lower = answer.lower()
        lines: list[str] = []
        for sid in sorted(cited & set(disputed)):
            for other in disputed[sid][:_MAX_RENDERED]:
                if other[:8].lower() in lower:
                    break  # the model already surfaced this dispute
            else:
                others = ", ".join(f"[{o[:8]}]" for o in disputed[sid][:_MAX_RENDERED])
                lines.append(f"⚠ Disputed evidence: [{sid[:8]}] is contested by {others}.")
        if not lines:
            return answer
        return answer + "\n\n" + "\n".join(lines)
    except Exception:
        _log.debug("dispute caveat failed; leaving answer untouched", exc_info=True)
        return answer
