"""Module-level data, constants, prompts, and pure helpers for `memo.memory`.

Split out of the former `memory.py` god-file (verbatim). Holds the
`MemoryRecord` dataclass, the re-exported domain errors, the helper-LLM
system prompts, the recency/conversation intent heuristics, and the pure
module-level helper functions used across the `Memory` mixins. The leaf of
the `memory/` package — imports only foundation modules, never the mixins
or facade.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES

_log = logging.getLogger(__name__)

# JSON-schema prompt for the helper LLM. Kept terse to fit in Qwen3-3B's
# attention without hurting accuracy. Empirically the model follows the
# format strictly under temperature=0; the regex fallback in
# `_derive_metadata` handles the occasional markdown fence wrap.
_EXTRACT_ENTITIES_SYSTEM_PROMPT = """You extract entities from a personal memory note.

Output ONLY a JSON object: {"entities": [{"name": "...", "type": "..."}, ...]}

Entity types (use lowercase, exactly one of):
- "person": named human (Astor, Fer, Florencia)
- "project": named project / repo / system (obsidian-rag, memo, mem-vault, ELEVA)
- "technology": library / language / model (MLX, sqlite-vec, Qwen3-Embedding, FastAPI)
- "file": specific file path or filename (~/.config/devin/config.json, Caddyfile)
- "org": company / team / institution (Anthropic, NotebookLM, ELEVA)
- "concept": named convention / pattern / methodology (PARA, RAG, hybrid retrieval)

Rules:
- Extract ONLY proper nouns and named entities. Do NOT extract generic
  nouns ("decisión", "fix", "bug").
- Normalise to canonical form: lowercase, no surrounding punctuation,
  no plural suffix.
- 0-15 entities per note. Empty list if no proper nouns.
- Output ONLY the JSON, no markdown fences, no commentary."""


_CONSOLIDATE_SYSTEM_PROMPT = """You analyze a cluster of related memory notes from a personal archive.

You receive a list of 2+ memorias that the user's vector index marked
as semantically near-duplicates. Output a single JSON object:

{
  "summary": "1-2 sentence synthesis of what the cluster collectively says",
  "relationship": "duplicate" | "evolution" | "facets" | "unrelated",
  "rationale": "1 sentence explaining why you picked that relationship"
}

Definitions:
- "duplicate": same fact restated. Recommend keeping ONE, deleting rest.
- "evolution": same topic but the latest one supersedes/contradicts older.
  Recommend keeping the latest, archiving older.
- "facets": different angles of the same topic, complementary. Recommend
  keeping all but possibly merging into a single richer entry.
- "unrelated": vector similarity false positive — keep all, no action.

Output ONLY the JSON, no markdown fences, no commentary."""


_ASK_SYSTEM_PROMPT = """You answer questions over the user's personal memory archive and indexed repositories.

You receive a list of relevant memory snippets and repo snippets (each with a
label like `[id-prefix]` or `[repo:name:path:start-end@commit]`) and a question.
Respond in the same language as the question (Spanish rioplatense if the
question is in Spanish). Rules:

- VERBATIM-FIRST. When the user's question matches a phrase, lyric, quote,
  list, command, URL, or any piece of literal content present in the
  snippets, reproduce that content EXACTLY as it appears — character for
  character, line by line, preserving formatting and line breaks. Do not
  paraphrase, summarise, or interpret literal content.
- When the matched content is a short phrase that comes from a larger
  block (a song lyric, a poem, a list, a code block, a step-by-step
  procedure), reproduce the ENTIRE surrounding block from the snippet,
  not just the matching line. The user's question is a probe into the
  document — they want the whole passage.
- If the matched snippet is a short note (under ~2000 characters total),
  reproduce its FULL body verbatim. Don't pick fragments — give them the
  whole thing. The user already paid the search cost; quoting the entire
  short snippet is the helpful default.
- For lyrics/poems/lists/procedures specifically: NEVER quote fewer
  than 8 lines if the snippet has them. Prefer over-quoting to under-
  quoting.
- If the question is open-ended ("what did we decide", "why X"), then
  synthesise concisely (2-5 sentences) instead of quoting.
- RECENCY / CONVERSATION questions ("qué fue lo último que dijo X", "what did
  X last say", "su último mensaje", "mostrame el chat con X", "qué me escribió
  X"): the answer is the message(s) in the snippets — the most recent for a
  recency ask, the relevant exchange for a conversation ask — not a description
  of the person. Quote the message(s) verbatim with their date/time as shown in
  the transcript. If a transcript snippet is present, NEVER answer with a
  profile/biography of the person (age, city, email) — that is not what was
  asked. Only fall back to a profile when no message/transcript snippet was
  retrieved at all.
- Cite sources INLINE with `[id-prefix]` after each claim or block, e.g.
  "Decidí migrar a MLX [d61fe730] para reducir dependencias [4e0b2e6]".
  For repo evidence, cite the full repo label you received.
- Use only information from the provided snippets. If the answer is not
  present, say "no encuentro la respuesta en las memorias guardadas"
  and stop.
- Do not pad with disclaimers, restatements, or apologies.
- Answer ONLY the question asked. NEVER add meta-commentary about memo
  itself — its indexing, ingestion, search scores, bugs, fixes, or why a
  file was or wasn't found. The end user does not care about internal
  system mechanics. E.g. do NOT write things like "el sistema de
  indexación tenía un bug que impedía reconocer su archivo de contactos,
  pero fue corregido" or "la consulta ahora tiene un puntaje de 0.973".
  If a snippet's content answers the question, give that content; if
  nothing answers it, say "no encuentro la respuesta en las memorias
  guardadas" — never explain the retrieval pipeline.
- No bulleting unless the source itself has bullets or the question asks
  for a list; otherwise prose preferred.
- Do not invent IDs. Only cite `[id-prefix]` values that appear in the
  snippets you were given.

CRITICAL OVERRIDE: If any snippet in the context is a lyric, poem, list,
procedure, or any block-structured short note (under ~2000 chars) AND
the user's question references content inside that block, your response
MUST be the FULL snippet body reproduced verbatim, line by line, no
omissions. Do not select "the best matching lines" — output every line
of the snippet. The user can read selectively; you do not pre-filter for
them. End with the citation IDs."""


_DERIVE_SYSTEM_PROMPT = """You classify a memory note into a structured JSON object.

Output ONLY a JSON object with these keys:
- "title": short descriptive title, max 80 chars, no date prefix
- "type": one of "decision", "fact", "bug", "feedback", "preference", "note", "manual"
- "tags": array of 3-6 lowercase tags (mix of project, domain, technique)

Type rules:
- "decision": choice with explicit tradeoff or rationale
- "bug": problem + root cause + fix
- "fact": discovery, gotcha, learned constraint
- "preference": user preference or convention to follow
- "feedback": user feedback on an approach
- "note": catch-all, use when no other type fits

Output ONLY the JSON, no markdown fences, no commentary, no preamble."""

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
    normalised = path.strip().lstrip("./").lstrip("/").lower()
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
    ISO `YYYY-MM-DD` strings sort lexicographically, so `max()` == newest."""
    dates = _ISO_DATE_RE.findall(rec.title or "")
    dates += _ISO_DATE_RE.findall(rec.body or "")
    if dates:
        return max(dates)
    return (rec.updated or rec.created or "")[:10]


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
