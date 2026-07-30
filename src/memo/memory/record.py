"""Module-level data, constants, and pure helpers for `memo.memory`.

Split out of the former `memory.py` god-file. Holds the `MemoryRecord`
dataclass, the recency/conversation intent heuristics, and the pure
module-level helper functions used across the `Memory` mixins. System
prompts live in `memo.memory.prompts` and are re-exported here so
existing import paths remain stable. The leaf of the `memory/` package —
imports only foundation modules, never the mixins or facade.
"""

from __future__ import annotations

import base64
import concurrent.futures as _futures
import contextlib
import hashlib
import logging
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from memo.contracts import MEMO_BACKEND_SCHEMA, PROVENANCE_KEYS, normalize_provenance

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

_CANONICAL_MEMORY_ID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
_MEMORY_ID_PREFIX_RE = re.compile(r"^[0-9a-f]{1,32}$", re.IGNORECASE)
_DERIVED_CHUNK_ID_RE = re.compile(r"^[0-9a-f]{32}_chunk_[0-9]+$", re.IGNORECASE)


def is_canonical_memory_id(value: object) -> bool:
    """Return whether ``value`` is memo's canonical 32-hex record id."""
    return isinstance(value, str) and _CANONICAL_MEMORY_ID_RE.fullmatch(value) is not None


def is_memory_id_prefix(value: object) -> bool:
    """Return whether ``value`` is a safe canonical-id prefix."""
    return isinstance(value, str) and _MEMORY_ID_PREFIX_RE.fullmatch(value) is not None


def is_derived_chunk_id(value: object) -> bool:
    """Return whether ``value`` is an exact derived reference-chunk id."""
    return isinstance(value, str) and _DERIVED_CHUNK_ID_RE.fullmatch(value) is not None


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


# Durable tiers + the bulk `reference` tier + the lifecycle-managed temporary
# type. `temp` stays outside DURABLE_TYPES because it is explicitly eligible
# for expiry, but it must be writable for LifecycleManager's documented
# type-based policy to be reachable.
_VALID_TYPES = DURABLE_TYPES | REFERENCE_TYPES | {"temp"}

# Bulk `reference` chunks shorter than this with no heading and no link/URL
# carry almost no semantic signal (stray punctuation, empty list items,
# frontmatter fragments) and only add noise + embedding cost to the index. This
# mirrors `repo_index.MIN_CHUNK_CHARS` / `_is_noise_chunk`, kept here in the
# pure-helpers module so the write path can gate without importing repo_index
# (which pulls in the MLX runtime at module load). Applies to the REFERENCE tier
# only — short durable facts/preferences ("User prefers dark mode") are kept.
MIN_REFERENCE_CHARS = 60


def _contains_reference_link(s: str) -> bool:
    """Whether `s` holds a wikilink, a markdown link, or a URL.

    Plain substring scans (linear, C-level, no backtracking) instead of a
    regex: the previous `re` pattern used ambiguous lazy quantifiers that
    CodeQL flagged as a polynomial ReDoS (py/polynomial-redos). Containment
    checks are immune regardless of input length.
    """
    if "http://" in s or "https://" in s:
        return True
    wiki = s.find("[[")
    if wiki != -1 and s.find("]]", wiki + 2) != -1:
        return True
    md = s.find("](")  # markdown link: [text](url)
    return md > 0 and s.find(")", md + 2) != -1


def is_reference_noise(body: str) -> bool:
    """True for a near-empty reference chunk with no heading and no link/URL."""
    stripped = body.strip()
    if len(stripped) >= MIN_REFERENCE_CHARS:
        return False
    if "#" in stripped and re.search(r"^#{1,6}\s+\S", stripped, re.MULTILINE):
        return False  # markdown heading — keep
    return not _contains_reference_link(stripped)  # link/URL → keep; else noise


def is_verified_offload_content(
    body: str,
    *,
    type_: str,
    tags: list[str],
    extra: dict[str, Any],
) -> bool:
    """Whether body is a content-addressed offload payload with a valid SHA."""
    expected_sha = extra.get("offload_sha256")
    return (
        type_ in REFERENCE_TYPES
        and "offload" in {str(tag).strip().lower() for tag in tags}
        and isinstance(expected_sha, str)
        and hashlib.sha256(body.encode("utf-8")).hexdigest() == expected_sha
    )


def markdown_body(post: Any) -> str:
    """Decode a lossless offload payload, or return the normal markdown body."""
    parsed_body = str(post.content or "")
    metadata = post.metadata or {}
    raw_tags = metadata.get("tags") or []
    tags = [raw_tags] if isinstance(raw_tags, str) else list(raw_tags)
    extra = metadata.get("extra") or {}
    if not isinstance(extra, dict) or extra.get("offload_payload_encoding") != "base64:utf-8:v1":
        return parsed_body
    encoded = extra.get("offload_payload_b64")
    if not isinstance(encoded, str):
        return parsed_body
    try:
        raw_body = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return parsed_body
    if is_verified_offload_content(
        raw_body,
        type_=str(metadata.get("type") or ""),
        tags=tags,
        extra=extra,
    ):
        return raw_body
    return parsed_body


MEMO_BACKEND_NATIVE_SCHEMA = MEMO_BACKEND_SCHEMA
NATIVE_BACKEND_PROTOCOL_VERSION = MEMO_BACKEND_SCHEMA
MEMO_BACKEND_NAME = "memo"

_PROVENANCE_KEYS = PROVENANCE_KEYS


def _extract_provenance(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the provenance subset of an extra bag (or {})."""
    return normalize_provenance(extra)


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
        # Only clock times inside the max-day's `## <day>` section count: a
        # short (unsplit) multi-day chunk (e.g. `## 2026-06-03 … ## 2026-06-04`)
        # would otherwise pair `max(day)` with a `max(time)` scanned from a
        # DIFFERENT day and fabricate a timestamp. Anchor on the last heading
        # for that day and read times after it; fall back to day-only when the
        # day has no associated `## ` heading (e.g. the date came from the title).
        body = rec.body or ""
        marker = body.rfind(f"## {day}")
        if marker != -1:
            times = _CLOCK_TIME_RE.findall(body[marker:])
            if times:
                return f"{day} {max(times)}"
        return day
    return (rec.updated or rec.created or "")[:10]


# -- LLM-call helpers (shared by the maintain/consolidate/synthesize loops) ----

_CHAT_WORKER_CONDITION = threading.Condition()
_CHAT_ACTIVE: _futures.Future[dict[str, Any]] | None = None
_CHAT_ACTIVE_TIMED_OUT = False


def chat_with_timeout(chat: Any, *, timeout: float, **kwargs: Any) -> dict[str, Any] | None:
    """Run ``chat.chat(**kwargs)`` with a hard wall-clock timeout.

    Returns the result dict, or ``None`` if it exceeds ``timeout``. MLX work
    cannot be interrupted mid-flight, so a timed-out worker continues in one
    daemon thread. Calls made while that worker is still alive fail closed
    immediately instead of starting competing model loads. A daemon is used
    deliberately: unlike ``ThreadPoolExecutor`` workers, it does not get joined
    by ``concurrent.futures`` during interpreter shutdown.

    Errors raised by ``chat.chat`` propagate (caller's try/except handles them).
    """
    global _CHAT_ACTIVE, _CHAT_ACTIVE_TIMED_OUT

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

    deadline = time.monotonic() + timeout
    import contextvars

    with _CHAT_WORKER_CONDITION:
        while _CHAT_ACTIVE is not None:
            if _CHAT_ACTIVE_TIMED_OUT:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            _CHAT_WORKER_CONDITION.wait(timeout=remaining)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None

        fut: _futures.Future[dict[str, Any]] = _futures.Future()
        _CHAT_ACTIVE = fut
        _CHAT_ACTIVE_TIMED_OUT = False
        context = contextvars.copy_context()

        def _worker() -> None:
            global _CHAT_ACTIVE, _CHAT_ACTIVE_TIMED_OUT
            if not fut.set_running_or_notify_cancel():
                return
            try:
                result = context.run(_run)
            except BaseException as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
            finally:
                with _CHAT_WORKER_CONDITION:
                    if _CHAT_ACTIVE is fut:
                        _CHAT_ACTIVE = None
                        _CHAT_ACTIVE_TIMED_OUT = False
                    _CHAT_WORKER_CONDITION.notify_all()

        threading.Thread(target=_worker, name="chat-timeout", daemon=True).start()

    try:
        return fut.result(timeout=remaining)
    except _futures.TimeoutError:
        with _CHAT_WORKER_CONDITION:
            if _CHAT_ACTIVE is fut:
                _CHAT_ACTIVE_TIMED_OUT = True
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
    IdentityConflictError,
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
    review_after: str | None = None
    # Record-level bi-temporal validity (distinct from `created`/`updated`,
    # which are learned/transaction time). `valid_at` = world-validity start;
    # `invalid_at` = world-validity end (None = interval still open). Both ISO8601.
    valid_at: str | None = None
    invalid_at: str | None = None
    # Set only on the immediate result of save(). Reads/lists keep these
    # ephemeral outcome fields absent from their serialized representation.
    action: str | None = None
    index_pending: bool = False
    relation_candidates: list[dict[str, Any]] | None = None
    relation_detection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
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
            "review_after": self.review_after,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
        }
        if self.action is not None:
            result["action"] = self.action
            result["index_pending"] = self.index_pending
            if self.relation_detection is not None:
                result["relation_detection"] = self.relation_detection
                result["relation_candidates"] = list(self.relation_candidates or [])
        return result


def _state_decay_factor(memory_record: MemoryRecord) -> float:
    """Score multiplier for a memory's verification state + age.

    VERIFIED memories rank at (or near) full weight; STALE and UNVERIFIED are
    penalized so freshly-verified facts surface first. Returns a float in
    (0.7, 1.0]:

    - VERIFIED & fresh (< 7 days): 1.0 (no penalty)
    - VERIFIED & old (7+ days): 0.95 (5% penalty)
    - STALE: 0.7 (30% penalty)
    - UNVERIFIED (or no verified_at): 0.8 (20% penalty)

    Consumed by the recall penalty (`_apply_verification_decay`) — gated by
    MEMO_VERIFICATION_STATE_TRACKING — so it is a no-op multiplier for a corpus
    that never marks anything VERIFIED (all-UNVERIFIED → uniform 0.8, ordering
    unchanged).
    """
    if not memory_record.verified_at:
        return 0.8  # UNVERIFIED: 20% penalty

    days_since_verified = (int(time.time()) - memory_record.verified_at) / 86400.0

    if memory_record.verification_state == VerificationState.VERIFIED:
        return 1.0 if days_since_verified < 7 else 0.95
    elif memory_record.verification_state == VerificationState.STALE:
        return 0.7  # STALE: 30% penalty
    else:  # UNVERIFIED
        return 0.8


def record_from_row(row: dict[str, Any], *, body: str = "") -> MemoryRecord:
    """Build a MemoryRecord from a store row dict (the `_row_to_dict` shape)."""
    extra = dict(row.get("extra") or {})
    if extra.get("offload_payload_encoding") == "base64:utf-8:v1":
        extra.pop("offload_payload_encoding", None)
        extra.pop("offload_payload_b64", None)

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
        id=row["id"],
        path=row["path"],
        title=row["title"],
        type=row["type"],
        tags=row["tags"],
        created=row["created"],
        updated=row["updated"],
        body=body,
        extra=extra,
        score=row.get("score"),
        verification_state=verification_state,
        verified_at=verified_at,
        review_after=row.get("review_after"),
        valid_at=row.get("valid_at"),
        invalid_at=row.get("invalid_at"),
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
