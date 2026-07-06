"""Module-level data, constants, and pure helpers for `memo.memory`.

Split out of the former `memory.py` god-file. Holds the `MemoryRecord`
dataclass, the recency/conversation intent heuristics, and the pure
module-level helper functions used across the `Memory` mixins. System
prompts live in `memo.memory.prompts` and are re-exported here so
existing import paths remain stable. The leaf of the `memory/` package —
imports only foundation modules, never the mixins or facade.
"""

from __future__ import annotations

import concurrent.futures as _futures
import contextlib
import logging
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

# Re-export the system prompts — callers import them from here (ask_ops,
# write_ops, maintain_ops, cli_capture, memory/__init__). Unused in this module
# itself, hence the per-symbol noqa.
from memo.memory.prompts import (
    _ASK_SYSTEM_PROMPT,  # noqa: F401
    _CONSOLIDATE_SYSTEM_PROMPT,  # noqa: F401
    _DERIVE_SYSTEM_PROMPT,  # noqa: F401
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,  # noqa: F401
    _REFLECT_SYSTEM_PROMPT,  # noqa: F401
    _SYNTHESIS_SYSTEM_PROMPT,  # noqa: F401
)
from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES, VerificationState

_log = logging.getLogger(__name__)

# Derived-save scope: dream/consolidation passes save memories that are expected
# to be near-duplicates of existing ones (the same run's consolidate pass merges
# them). The interactive "consider `memo update` instead" dedup nag is noise in
# that context, so callers wrap batch/derived saves in `derived_save_scope()` and
# the dedup check (write_ops) demotes its warning to debug while it's active. A
# ContextVar (not a module global) keeps the flag thread- and task-local so a
# concurrent interactive save on another thread still gets the nudge.
_derived_save: ContextVar[bool] = ContextVar("memo_derived_save", default=False)


@contextlib.contextmanager
def derived_save_scope() -> Any:
    """Mark saves within this scope as derived/batch (suppress the dedup nag)."""
    token = _derived_save.set(True)
    try:
        yield
    finally:
        _derived_save.reset(token)


def in_derived_save_scope() -> bool:
    """True when the current thread/task is inside `derived_save_scope()`."""
    return _derived_save.get()


# Durable tiers + the bulk `reference` tier. The split (which types the recall
# hook / briefing surface automatically vs. on-demand-only) lives in
# `memo.tiers`; this set is just "every type a memory may legally carry".
_VALID_TYPES = DURABLE_TYPES | REFERENCE_TYPES

# Bulk `reference` chunks shorter than this with no heading and no link/URL
# carry almost no semantic signal (stray punctuation, empty list items,
# frontmatter fragments) and only add noise + embedding cost to the index. This
# mirrors `repo_index.MIN_CHUNK_CHARS` / `_is_noise_chunk`, kept here in the
# pure-helpers module so the write path can gate without importing repo_index
# (which pulls in the MLX runtime at module load). Applies to the REFERENCE tier
# only — short durable facts/preferences ("User prefers dark mode") are kept.
MIN_REFERENCE_CHARS = 60
_REFERENCE_LINK_RE = re.compile(r"\[\[.+?\]\]|\[.+?\]\(.+?\)|https?://\S+")


def is_reference_noise(body: str) -> bool:
    """True for a near-empty reference chunk with no heading and no link/URL."""
    stripped = body.strip()
    if len(stripped) >= MIN_REFERENCE_CHARS:
        return False
    if "#" in stripped and re.search(r"^#{1,6}\s+\S", stripped, re.MULTILINE):
        return False  # markdown heading — keep
    return not _REFERENCE_LINK_RE.search(stripped)  # link/URL → keep; else noise


SYNAPSE_BACKEND_NATIVE_SCHEMA = "synapse.backend_native.v1"
NATIVE_BACKEND_PROTOCOL_VERSION = "backend_native.v1"
MEMO_BACKEND_NAME = "memo"

# Provenance keys carried in `extra` and persisted to both `meta.extra_json`
# and `history.events.delta_json`. Set by callers that operate as part of a
# Synapse-orchestrated write (route_intent → remember). Each key is optional;
# memo never invents values. Sourced from `consciousness_contracts` so the
# trinity shares ONE definition of "which fields are provenance"; the literal
# below is only a fallback for CI / clean installs without the package, and is
# guarded against drift by `tests/test_synapse_backend.py`.
try:
    from consciousness_contracts import PROVENANCE_KEYS as _PROVENANCE_KEYS
except ImportError:  # pragma: no cover - optional dep, absent in CI/clean installs
    _PROVENANCE_KEYS: frozenset[str] = frozenset(  # type: ignore[no-redef]
        {
            "synapse_trace_id",
            "synapse_route_reason",
            "synapse_write_policy_schema",
            "synapse_write_target",
            "synapse_agent_id",
            "synapse_agent_signature",
        }
    )


def _extract_provenance(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the provenance subset of an extra bag (or {})."""
    if not extra:
        return {}
    return {k: extra[k] for k in _PROVENANCE_KEYS if k in extra}


def _norm_dedup_path(path: str | None) -> str:
    """Normalise a vault/repo path for cross-source dedup in ask context.

    Strips leading ./ and / segments, lowercases, and removes any
    `#chunk-N` suffix so multi-chunk memories deduplicate back to their
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
    vault memory. Vault ingestion may slugify the on-disk path, so we
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
    "ultimo",
    "ultima",
    "ultimos",
    "ultimas",
    "último",
    "última",
    "últimos",
    "últimas",
    "reciente",
    "recientes",
    "recientemente",
    "last",
    "latest",
    "most recent",
    "more recent",
    "recent",
    "qué dijo",
    "que dijo",
    "qué escribió",
    "que escribio",
    "lo que dijo",
    "what did",
    "said last",
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
    "mensaje",
    "mensajes",
    "chat",
    "conversación",
    "conversacion",
    "escribió",
    "escribio",
    "escribiste",
    "hablé",
    "hable",
    "hablamos",
    "whatsapp",
    "wpp",
    "message",
    "messages",
    "texted",
    "wrote",
    "conversation",
    "chatted",
    "talked",
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

_CHAT_EXECUTOR: _futures.ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _get_chat_executor() -> _futures.ThreadPoolExecutor:
    global _CHAT_EXECUTOR
    with _EXECUTOR_LOCK:
        if _CHAT_EXECUTOR is None:
            _CHAT_EXECUTOR = _futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chat-timeout"
            )
        return _CHAT_EXECUTOR


def chat_with_timeout(chat: Any, *, timeout: float, **kwargs: Any) -> dict[str, Any] | None:
    """Run ``chat.chat(**kwargs)`` with a hard wall-clock timeout.

    Returns the result dict, or ``None`` if it exceeds ``timeout``. On timeout
    the executor is shut down (``shutdown(wait=False)``) and replaced with a
    fresh one — an MLX forward pass can't be interrupted mid-flight, but the
    next caller gets a full timeout budget instead of queueing behind the
    abandoned thread.  The submitted thread sets ``_gpu_tl.timeout`` so
    ``gpu_guard()`` imposes a deadline on the GPU lock rather than blocking
    forever.

    Errors raised by ``chat.chat`` propagate (caller's try/except handles them).
    """

    _timeout = timeout

    def _run() -> dict[str, Any]:
        # Set thread-local GPU timeout so gpu_guard() in this thread uses a
        # matching deadline instead of blocking indefinitely.
        try:
            from memo.mlx_gpu import _gpu_tl

            _gpu_tl.timeout = _timeout
        except Exception:  # noqa: S110
            pass
        return chat.chat(**kwargs)

    ex = _get_chat_executor()
    if getattr(ex, "_shutdown", False):
        with _EXECUTOR_LOCK:
            _CHAT_EXECUTOR = None
            _CHAT_EXECUTOR = _futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chat-timeout"
            )
            ex = _CHAT_EXECUTOR
    fut = ex.submit(_run)
    try:
        return fut.result(timeout=timeout)
    except _futures.TimeoutError:
        # Shutdown the executor so the next caller doesn't queue behind
        # the abandoned MLX thread.  An MLX forward pass can't be
        # interrupted in-flight, so the old thread runs to completion in
        # the background, but the next call gets a fresh executor+worker
        # and its full timeout budget instead of cascading.
        with _EXECUTOR_LOCK:
            ex.shutdown(wait=False)
            _CHAT_EXECUTOR = None
        return None


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
    verification_state: VerificationState = VerificationState.UNVERIFIED
    verified_at: int | None = None  # Unix timestamp

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
            "verification_state": self.verification_state.value,
            "verified_at": self.verified_at,
        }


def record_from_row(row: dict[str, Any], *, body: str = "") -> MemoryRecord:
    """Build a MemoryRecord from a store row dict (the `_row_to_dict` shape)."""
    # Extract verification state, defaulting to UNVERIFIED if not present (backward compatible)
    ver_state_str = row.get("verification_state", "unverified")
    try:
        verification_state = VerificationState(ver_state_str)
    except (ValueError, KeyError):
        verification_state = VerificationState.UNVERIFIED

    # Extract verified_at timestamp (can be None)
    verified_at = row.get("verified_at")
    if verified_at is not None and not isinstance(verified_at, int):
        try:
            verified_at = int(verified_at)
        except (ValueError, TypeError):
            verified_at = None

    return MemoryRecord(
        id=row["id"], path=row["path"], title=row["title"], type=row["type"],
        tags=row["tags"], created=row["created"], updated=row["updated"],
        body=body, extra=row.get("extra") or {},
        verification_state=verification_state,
        verified_at=verified_at,
    )


@dataclass(frozen=True)
class RRFConfidenceDecision:
    skip: bool
    top_id: str | None
    ratio: float
    gap: float


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
    *lists: list[dict[str, Any]],
    limit: int,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion. Each hit in each list contributes
    `w * (1 / (k + rank))` to its id's combined score, with `k=60` per the
    Cormack et al. paper. Records that appear in multiple lists naturally
    get a higher fused score.

    `weights` must be the same length as `lists` when provided. Each weight
    scales the RRF contribution of the corresponding list. When omitted (or
    None), all lists are weighted equally at 1.0 (standard unweighted RRF).

    Returns the top-`limit` hits by fused score, hydrated with the
    metadata from whichever source carried the canonical fields.
    """
    fused: dict[str, float] = {}
    canon: dict[str, dict[str, Any]] = {}
    for i, lst in enumerate(lists):
        if not lst:
            continue
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        for rank, hit in enumerate(lst):
            rid = hit["id"]
            fused[rid] = fused.get(rid, 0.0) + w / (k + rank + 1)
            canon.setdefault(rid, hit)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for rid, score in ranked:
        d = dict(canon[rid])
        d["score"] = score
        out.append(d)
    return out


def _rrf_confident_top(
    rows: list[dict[str, Any]] | list[MemoryRecord],
    *,
    min_ratio: float,
    min_gap: float,
) -> RRFConfidenceDecision:
    """Return whether the top fused result is far enough ahead to skip rerank.

    The cross-encoder is valuable when RRF leaves a close pack. When the first
    hit is already separated by both a ratio and absolute gap, loading a large
    reranker is usually wasted latency.
    """
    if not rows:
        return RRFConfidenceDecision(skip=False, top_id=None, ratio=0.0, gap=0.0)

    top = rows[0]
    top_score = float((top.get("score") if isinstance(top, dict) else top.score) or 0.0)
    top_id = str(top.get("id") if isinstance(top, dict) else top.id)
    if len(rows) < 2:
        return RRFConfidenceDecision(skip=True, top_id=top_id, ratio=float("inf"), gap=top_score)

    second = rows[1]
    second_score = float((second.get("score") if isinstance(second, dict) else second.score) or 0.0)
    gap = top_score - second_score
    ratio = float("inf") if second_score <= 0 else top_score / second_score
    skip = top_score > 0 and gap >= min_gap and ratio >= min_ratio
    return RRFConfidenceDecision(skip=skip, top_id=top_id, ratio=ratio, gap=gap)


def _adaptive_rrf_k(lists: list[list[dict[str, Any]]], *, base_k: int) -> int:
    """Density-adaptive RRF `k`.

    The fixed Cormack k=60 is a one-size constant. When the ranked lists
    strongly agree (an id appears across multiple lists), shrink k to
    sharpen fusion toward that consensus; when they barely overlap, grow k
    to soften single-list rank dominance. Bounded to `[base_k/2, base_k*2]`
    so it can never run away. Returns `base_k` unchanged when fewer than two
    non-empty lists exist (overlap is undefined).

    Opt-in via `MEMO_RRF_ADAPTIVE` — the default path keeps `base_k` so the
    eval baseline stays comparable.
    """
    from collections import Counter

    nonempty = [lst for lst in lists if lst]
    if len(nonempty) < 2:
        return base_k
    counts: Counter[str] = Counter()
    for lst in nonempty:
        for hit in lst:
            counts[hit["id"]] += 1
    total = len(counts)
    if total == 0:
        return base_k
    shared = sum(1 for c in counts.values() if c >= 2)
    overlap = shared / total  # 0 (disjoint) .. 1 (identical)
    # overlap 0 → factor 1.5 (grow); overlap 1 → factor 0.5 (shrink).
    k = round(base_k * (1.5 - overlap))
    return max(base_k // 2, min(base_k * 2, k))


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


# Default recency half-life (days) applied when a consumer path requests
# `search(recency=True)` without an explicit MEMO_SEARCH_DECAY_HALFLIFE. At
# one half-life a memory's freshness term is exactly 0.5; ~90 days keeps a
# fact at full weight for weeks, then lets it yield to fresher memories
# rather than being crowded out forever.
_RECALL_DECAY_HALFLIFE_DEFAULT = 90.0


# Per-type decay half-life flag names. Values are the MEMO_* env var names
# that override the global MEMO_SEARCH_DECAY_HALFLIFE for a given memory type.
# Unknown types fall back to the global halflife. No-decay behavior (e.g. for
# 'reference') is achieved by registering a flag whose default is None in
# flags.py — flag_float() returns None, which _halflife_for_type maps to 0.0.
_PER_TYPE_HALFLIFE_FLAGS: dict[str, str] = {
    "decision": "MEMO_DECAY_HALFLIFE_DECISION",
    "feedback": "MEMO_DECAY_HALFLIFE_FEEDBACK",
    "note": "MEMO_DECAY_HALFLIFE_NOTE",
    "fact": "MEMO_DECAY_HALFLIFE_FACT",
    "reference": "MEMO_DECAY_HALFLIFE_REFERENCE",
}


def _halflife_for_type(memory_type: str | None, global_halflife: float) -> float:
    """Return the effective half-life (days) for *memory_type*.

    Resolution order:
    1. Per-type flag (e.g. MEMO_DECAY_HALFLIFE_DECISION) — when set in env,
       this overrides the global for that type.
    2. Per-type default (the registered default in flags.py) — applied when
       the env var is NOT set, overriding the global default.
    3. Global `global_halflife` — for types not in `_PER_TYPE_HALFLIFE_FLAGS`.

    Special case: `reference` has a registered default of None, meaning
    references do not decay unless MEMO_DECAY_HALFLIFE_REFERENCE is set.
    Returns 0.0 (no decay) in that case.
    """
    if not memory_type:
        return global_halflife
    flag_name = _PER_TYPE_HALFLIFE_FLAGS.get(memory_type)
    if flag_name is None:
        # Type not in per-type registry; use global halflife.
        return global_halflife

    # Defer flags import here — record.py must not import from memo.flags at
    # module level (circular risk; also flags depends on nothing, but keep
    # the dependency direction clean).
    from memo.flags import flag_float

    v = flag_float(flag_name)
    # v is None when the flag is not registered (shouldn't happen given our
    # registry entries), or when the env var is absent AND the registered
    # default is None (e.g. MEMO_DECAY_HALFLIFE_REFERENCE default=None).
    if v is None:
        return 0.0  # None default means "no decay"
    return v


def _apply_decay(
    records: list[MemoryRecord],
    *,
    halflife_days: float,
    alpha: float,
) -> list[MemoryRecord]:
    """Blend a freshness bonus into search scores using half-life decay.

    For each record: `decay = 0.5 ** (days_since_updated / halflife_days)`,
    a true half-life — at exactly `halflife_days` the freshness term is 0.5,
    at 2× half-life it is 0.25, and so on.
    Final score: `(1 - alpha) * original_score + alpha * decay`.

    A half-life of 90 days means a 90-day-old memory retains 50% of the
    freshness bonus and a 180-day-old retains 25%. Results are re-sorted by
    final score so the caller always gets a monotonically ranked list.

    Per-type half-life: when per-type MEMO_DECAY_HALFLIFE_<TYPE> flags are
    set (or carry non-None defaults), they override `halflife_days` for that
    record's type. `halflife_days` acts as the global fallback for types not
    covered by per-type flags. When a per-type effective half-life resolves
    to 0, that individual record is not decayed (score passed through as-is).
    """
    now = datetime.now(tz=UTC)
    out: list[MemoryRecord] = []
    for r in records:
        if r.score is None:
            out.append(r)
            continue
        # Resolve the effective half-life for this record's type.
        eff_halflife = _halflife_for_type(r.type, global_halflife=halflife_days)
        if eff_halflife <= 0:
            # No decay for this type (e.g. reference tier or explicitly zero).
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
        decay = 0.5 ** (days / eff_halflife)
        final = (1.0 - alpha) * r.score + alpha * decay
        out.append(replace(r, score=round(final, 6)))
    out.sort(key=lambda r: r.score or 0.0, reverse=True)
    return out


def support_lift() -> float:
    """Resolved MEMO_SUPPORT_CONFIDENCE_LIFT (0.0 = counting only).

    Shared by the corroboration bump call sites (write_ops, consolidation)
    so the flag is read in exactly one place."""
    from memo.flags import flag_float

    v = flag_float("MEMO_SUPPORT_CONFIDENCE_LIFT")
    return 0.0 if v is None else v


def bump_support_if_enabled(store: Any, ids: list[str]) -> None:
    """Corroboration (C1): count a re-assertion of the given memories.

    No-op when MEMO_SUPPORT_COUNT is off; best-effort (never raises). Reads
    the flag + lift here so the save and consolidation call sites stay one
    line and the corroboration policy lives in exactly one place."""
    from memo.flags import flag_bool

    if not ids or not flag_bool("MEMO_SUPPORT_COUNT"):
        return
    with contextlib.suppress(Exception):
        store.bump_support_batch(ids, lift=support_lift())


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
