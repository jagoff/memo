"""Module-level data, constants, and pure helpers for `memo.memory`.

Split out of the former `memory.py` god-file. Holds the `MemoryRecord`
dataclass, the recency/conversation intent heuristics, and the pure
module-level helper functions used across the `Memory` mixins. System
prompts live in `memo.memory.prompts` and are re-exported here so
existing import paths remain stable. The leaf of the `memory/` package —
imports only foundation modules, never the mixins or facade.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES
from memo.memory.prompts import (  # re-export — callers import from here
    _ASK_SYSTEM_PROMPT,
    _CONSOLIDATE_SYSTEM_PROMPT,
    _DERIVE_SYSTEM_PROMPT,
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,
    _REFLECT_SYSTEM_PROMPT,
    _SYNTHESIS_SYSTEM_PROMPT,
)

_log = logging.getLogger(__name__)

# Durable tiers + the bulk `reference` tier. The split (which types the recall
# hook / briefing surface automatically vs. on-demand-only) lives in
# `memo.tiers`; this set is just "every type a memoria may legally carry".
_VALID_TYPES = DURABLE_TYPES | REFERENCE_TYPES

SYNAPSE_BACKEND_NATIVE_SCHEMA = "synapse.backend_native.v1"
NATIVE_BACKEND_PROTOCOL_VERSION = "backend_native.v1"
MEMO_BACKEND_NAME = "memo"

# Provenance keys carried in `extra` and persisted to both `meta.extra_json`
# and `history.events.delta_json`. Set by callers that operate as part of a
# Synapse-orchestrated write (route_intent → remember). Each key is optional;
# memo never invents values. Listed here so `Memory.provenance()` and
# `MemoSynapseBackend` know exactly which fields are "provenance" vs "user
# extra metadata" without hardcoding the prefix string in multiple places.
_PROVENANCE_KEYS: frozenset[str] = frozenset({
    "synapse_trace_id",
    "synapse_route_reason",
    "synapse_write_policy_schema",
    "synapse_write_target",
    "synapse_agent_id",
    "synapse_agent_signature",
})


def _extract_provenance(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the provenance subset of an extra bag (or {})."""
    if not extra:
        return {}
    return {k: extra[k] for k in _PROVENANCE_KEYS if k in extra}


def _norm_dedup_path(path: str | None) -> str:
    """Normalise a vault/repo path for cross-source dedup in ask context.

    Strips leading ./ and / segments, lowercases, and removes any
    `#chunk-N` suffix so multi-chunk memorias deduplicate back to their
    parent path. Conservative: same string after normalisation means
    same file for the purposes of source merging.
    """
    if not path:
        return ""
    # lstrip("./") already strips any leading '.' and '/' chars, so a second
    # lstrip("/") would be a no-op.
    normalised = path.strip().lstrip("./").lower()
    chunk_idx = normalised.find("#chunk-")
    if chunk_idx != -1:
        normalised = normalised[:chunk_idx]
    return normalised


def _vault_dedup_keys(rec: MemoryRecord) -> set[str]:
    """Signals used to detect that a repo hit covers the same file as a
    vault memoria. Vault ingestion may slugify the on-disk path, so we
    cross-reference title, `extra.abs_path`, and basename.
    """
    keys: set[str] = set()
    for raw in (rec.path, rec.title, (rec.extra or {}).get("abs_path")):
        norm = _norm_dedup_path(raw)
        if norm:
            keys.add(norm)
            # Also key by basename for cases where one side carries the
            # full path and the other only the filename.
            base = norm.rsplit("/", 1)[-1]
            if base:
                keys.add(base)
    return keys


# Recency-intent handling for ask()/chat_ask(). "Qué fue lo último que dijo X",
# "last message", "more recent" are temporal questions: the user wants the most
# recent content, not the highest semantic match. Pure cosine ranks a same-named
# contact/profile card (title == the person's name) above the dated transcript.
# We detect the intent cheaply and, when present, re-order the retrieved hits by
# the most recent calendar date they mention (WhatsApp transcripts carry
# `## YYYY-MM-DD` day headers + dated chunk titles), floating conversation
# sources up when the question itself names the channel ("por whatsapp").
_RECENCY_TOKENS = (
    "ultimo", "ultima", "ultimos", "ultimas",
    "último", "última", "últimos", "últimas",
    "reciente", "recientes", "recientemente",
    "last", "latest", "most recent", "more recent", "recent",
    "qué dijo", "que dijo", "qué escribió", "que escribio",
    "lo que dijo", "what did", "said last",
)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# WhatsApp transcript lines carry per-message clock times: "**yo** (21:06): …".
# Used to break same-day ties between sub-chunks of one long day (see _recency_key).
_CLOCK_TIME_RE = re.compile(r"\((\d{2}:\d{2})\)")


def _is_recency_query(q: str) -> bool:
    """True when the question asks for the *most recent* content (ES/EN)."""
    ql = (q or "").lower()
    return any(tok in ql for tok in _RECENCY_TOKENS)


# Conversation intent ("mostrame el chat con X", "qué me escribió X") is the
# same failure class as recency without the recency word: a person-scoped
# message query whose dated transcript loses to the same-named contact card on
# pure cosine. We float WhatsApp transcripts for these too — see _build_ask_context.
_CONVERSATION_TOKENS = (
    "mensaje", "mensajes", "chat", "conversación", "conversacion",
    "escribió", "escribio", "escribiste", "hablé", "hable", "hablamos",
    "whatsapp", "wpp", "message", "messages", "texted", "wrote",
    "conversation", "chatted", "talked",
)


def _is_conversation_query(q: str) -> bool:
    """True when the user asks about a conversation/messages (channel-scoped),
    even without an explicit recency word. Used to float WhatsApp transcripts
    above a same-named contact/profile card."""
    ql = (q or "").lower()
    return any(tok in ql for tok in _CONVERSATION_TOKENS)


def _is_whatsapp_hit(rec: MemoryRecord) -> bool:
    """True when a hit is an actual WhatsApp *transcript* — keyed on the
    `WhatsApp · …` title prefix or `source: whatsapp` frontmatter, NOT the
    generic `whatsapp` tag (meta-notes *about* whatsapp carry that tag too and
    must not be mistaken for transcripts)."""
    if (rec.title or "").startswith("WhatsApp ·"):
        return True
    return (rec.extra or {}).get("source") == "whatsapp"


def _is_group_chat(rec: MemoryRecord) -> bool:
    """True when a WhatsApp transcript hit is a *group* chat (vs a 1:1).
    Group transcript titles carry a `group`/`grupo` marker; a question about a
    person ("lo último que dijo Grecia") wants her 1:1 chat, not a same-named
    group she belongs to."""
    tl = (rec.title or "").lower()
    return "group" in tl or "grupo" in tl


def _recency_key(rec: MemoryRecord) -> str:
    """Sortable recency signal: the most recent ISO date the hit mentions in
    its title or body, falling back to the record's updated/created stamp.
    ISO `YYYY-MM-DD` strings sort lexicographically, so `max()` == newest.

    For WhatsApp transcripts a long day is split into several sub-chunks that
    all share the same `## YYYY-MM-DD` header, so a date-only key ties them and
    the last message of the day (often the answer to "lo último") never floats
    to the top. Append the latest clock time found in the body so same-day
    sub-chunks order by their tail message: "2026-06-04 21:06" > "2026-06-04 19:52".
    """
    dates = _ISO_DATE_RE.findall(rec.title or "")
    dates += _ISO_DATE_RE.findall(rec.body or "")
    if dates:
        day = max(dates)
        times = _CLOCK_TIME_RE.findall(rec.body or "")
        return f"{day} {max(times)}" if times else day
    return (rec.updated or rec.created or "")[:10]


# -- LLM-call helpers (shared by the maintain/consolidate/synthesize loops) ----

def chat_with_timeout(chat: Any, *, timeout: float, **kwargs: Any) -> dict[str, Any] | None:
    """Run ``chat.chat(**kwargs)`` with a hard wall-clock timeout.

    Returns the result dict, or ``None`` if it exceeds ``timeout``. The worker
    thread is abandoned (``shutdown(wait=False)``) — an MLX forward pass can't be
    interrupted mid-flight, so the timeout only bounds how long *we* wait. Errors
    raised by ``chat.chat`` propagate (caller's try/except handles them).
    """
    import concurrent.futures

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(chat.chat, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return None
    finally:
        ex.shutdown(wait=False)


def strip_llm_output(text: str) -> str:
    """Strip Qwen3 ``<think>…</think>`` traces and a wrapping markdown code
    fence from an LLM response, leaving the bare payload (often JSON)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    return text


# Domain error hierarchy lives in memo.errors; re-exported here so existing
# `from memo.memory import AmbiguousIdError / WriteRefused / MemoError` imports
# keep working. New code may import from either module.
from memo.errors import (  # noqa: E402, F401
    AmbiguousIdError,
    FederationError,
    MemoError,
    NotFoundError,
    StorageError,
    ValidationError,
    WriteRefused,
)


@dataclass(frozen=True)
class MemoryRecord:
    """Public, immutable view of one memory entry.

    Internally the store + on-disk file may diverge briefly during a
    `save()` (vec insert happens after the file write). Callers always
    receive a `MemoryRecord` only after both writes have committed.
    """

    id: str
    path: str  # vault-relative
    title: str
    type: str
    tags: list[str]
    created: str  # ISO8601
    updated: str
    body: str
    extra: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # populated by `search()`; None for direct fetches.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "type": self.type,
            "tags": list(self.tags),
            "created": self.created,
            "updated": self.updated,
            "body": self.body,
            "extra": dict(self.extra),
            "score": self.score,
        }


def _now_iso() -> str:
    # Millisecond precision so within-second event ordering survives
    # — required for time-machine reconstruction to distinguish
    # save/update/delete pairs that happen rapidly. Tooling that
    # parsed second-truncated strings still parses these.
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="milliseconds")


_SLUG_NON_WORD = re.compile(r"[^\w\s-]+")
_SLUG_WS = re.compile(r"[\s_-]+")


@lru_cache(maxsize=4096)
def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_NON_WORD.sub("", s)
    s = _SLUG_WS.sub("-", s)
    return s.strip("-")


def _derive_title(content: str) -> str:
    # First non-empty line, stripped of leading markdown markers.
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[#*\-\"`>\s]+", "", line).rstrip(" .:;")
        if line:
            return line[:80]
    return ""


def _rrf_fuse(
    vec_hits: list[dict[str, Any]],
    bm_hits: list[dict[str, Any]],
    *,
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion. Each hit in each list contributes
    `1 / (k + rank)` to its id's combined score, with `k=60` per the
    Cormack et al. paper. Records that appear in both lists naturally
    get a higher fused score.

    Returns the top-`limit` hits by fused score, hydrated with the
    metadata from whichever source carried the canonical fields
    (vec wins ties — its hit dict is identical to bm25's).
    """
    fused: dict[str, float] = {}
    canon: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vec_hits):
        rid = hit["id"]
        fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
        canon.setdefault(rid, hit)
    for rank, hit in enumerate(bm_hits):
        rid = hit["id"]
        fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
        canon.setdefault(rid, hit)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for rid, score in ranked:
        d = dict(canon[rid])
        d["score"] = score
        out.append(d)
    return out


def _compose_for_embed(title: str, body: str) -> str:
    """Combine title + body into the string passed to the embedder.

    Title-first because: (a) titles carry the highest-density retrieval
    signal in this corpus (memos with terse titles + long bodies dominate),
    (b) head-truncation guarantees the title survives even when body is
    long, (c) avoiding double-prefix when title already appears as an H1
    in the body — we do NOT dedup, the redundancy doesn't hurt the
    embedder and the simpler code is worth the few wasted tokens.
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        return body
    if not body:
        return title
    return f"{title}\n\n{body}"


# Default recency halflife (days) applied when a consumer path requests
# `search(recency=True)` without an explicit MEMO_SEARCH_DECAY_HALFLIFE. ~6
# months: a fact stays at full weight for weeks, then gently yields to fresher
# memorias rather than being crowded out forever.
_RECALL_DECAY_HALFLIFE_DEFAULT = 180.0


def _apply_decay(
    records: list[MemoryRecord],
    *,
    halflife_days: float,
    alpha: float,
) -> list[MemoryRecord]:
    """Blend a freshness bonus into search scores using exponential decay.

    For each record: `decay = exp(-days_since_updated / halflife_days)`.
    Final score: `(1 - alpha) * original_score + alpha * decay`.

    A halflife of 30 days means a 30-day-old memory retains 50% of the
    freshness bonus, a 90-day-old retains ~5%. Results are re-sorted by
    final score so the caller always gets a monotonically ranked list.
    """
    import math

    now = datetime.now(tz=UTC)
    out: list[MemoryRecord] = []
    for r in records:
        if r.score is None:
            out.append(r)
            continue
        try:
            updated = datetime.fromisoformat(r.updated)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            days = max(0.0, (now - updated).total_seconds() / 86400)
        except Exception:
            out.append(r)
            continue
        decay = math.exp(-days / halflife_days)
        final = (1.0 - alpha) * r.score + alpha * decay
        out.append(replace(r, score=round(final, 6)))
    out.sort(key=lambda r: r.score or 0.0, reverse=True)
    return out


def _normalise_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = (t or "").strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _build_freeze_query(
    *,
    title: str | None,
    content: str,
    tags: list[str] | None,
) -> str:
    """Compose a synapse `conflicts` query from the most signal-dense
    fields of a pending write.

    Priority: explicit title → first non-empty tag → first non-empty
    content line (truncated). Synapse semantic search is keyword-tier
    today, so a 4-8 word query is the sweet spot.
    """
    if title and title.strip():
        return title.strip()[:120]
    for tag in tags or []:
        tag = (tag or "").strip()
        if tag and not tag.startswith("project:"):
            return tag[:120]
    for line in (content or "").splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return ""
