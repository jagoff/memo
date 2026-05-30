"""High-level `Memory` API — saves to vault + indexes to sqlite-vec.

This is the layer that callers (CLI, MCP server, library users)
interact with. Wraps `MLXEmbedder` + `VecStore` + frontmatter writer
into a coherent interface mirroring `mem-vault.Memory`:

- `save(content, ...)` → write `.md` file under
  `vault/memory_subdir/<slug>.md`, embed, index. Returns
  `MemoryRecord`.
- `search(query, limit)` → embed the query, top-k vec search, hydrate
  each hit with metadata + on-disk content snippet.
- `list(type_, limit)` → recent entries by `updated` desc.
- `get(id_)` → full record + body.
- `update(id_, ...)` → patch one or more fields, re-embed if content
  changed (body_hash check).
- `delete(id_)` → remove from vec + meta + delete `.md` file.

The `.md` storage of record uses Obsidian-friendly frontmatter so the
user can edit memories from Obsidian and the next index pass picks
them up via `body_hash` mismatch.

## Frontmatter schema

```yaml
---
id: <uuid4 hex>
title: Short descriptive title
type: decision | fact | bug | feedback | preference | note
tags: [tag1, tag2, tag3]
created: 2026-05-06T19:30:00-03:00
updated: 2026-05-06T19:30:00-03:00
---

(body — markdown, free-form)
```

## Slugify

`<YYYY-MM-DD>-<slug-of-title>.md` — e.g.
`2026-05-06-decision-tema.md`. Same convention as obsidian-rag's
conversation writer.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

from memo.analytics import AnalyticsEngine, Dashboard
from memo.collaborative import CollaborativeFilter, CollaborativeGraph, CollaborativeManager
from memo.config import Config
from memo.consolidation import AdvancedConsolidator
from memo.contextual import ContextStore, ContextualRecall
from memo.contextual_retrieval import get_or_generate_context, prepend_context
from memo.contradict import ContradictionScanner, ContradictionStore
from memo.crossref import CrossReferenceIndex, LinkSuggester
from memo.embedder import MLXEmbedder, assert_valid_embedding
from memo.encryption import EncryptionManager, Encryptor, KeyManager
from memo.federation import FederationConfig, FederationSearcher
from memo.graph import GraphStore
from memo.import_export import ImportExportManager
from memo.lifecycle import (
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
    LifecycleManager,
)
from memo.llm import MLXChat
from memo.multimodal import CrossModalSearch, MultiModalManager, MultiModalStore, UniversalEmbedder
from memo.navigation import GraphNavigator
from memo.proactive import ProactiveSuggester
from memo.queries import QueryComposer, QueryStore
from memo.sharing import ShareManager, ShareStore
from memo.store import VecStore
from memo.sync import BackupManager, SyncManager
from memo.temporal import TemporalAnalyzer
from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES
from memo.util import sha256_short as _sha256_short
from memo.util import stable_hash as _stable_content_hash
from memo.util import utc_now_iso as _utc_now_iso
from memo.versioning import VersionManager

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


class Memory:
    """High-level memory API. Construct once per process; methods are
    thread-safe (delegate to store/embedder which both serialise their
    critical sections).

    Example:

        cfg = Config.from_env()
        cfg.ensure_dirs()
        mem = Memory(cfg)

        rec = mem.save(
            content="**What**: Migré obsidian-rag a MLX. ...",
            title="MLX migration cierre formal",
            type_="decision",
            tags=["mlx", "obsidian-rag", "migration"],
        )

        hits = mem.search("cómo migré a mlx", limit=5)
        for h in hits:
            print(f"{h.score:.3f} · {h.title}")
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        # MEMO_EMBEDDER_VIA_DAEMON=1: embed via the recall daemon socket
        # instead of loading a second in-process copy of the embedder. Used by
        # the warm memo-mcp chat daemon so its resident footprint is just the
        # synthesis model — the recall daemon already holds the embedder warm.
        # Falls back in-process automatically if the socket is down.
        if os.environ.get("MEMO_EMBEDDER_VIA_DAEMON", "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            from memo.embedder_client import SocketEmbedder
            self.embedder: Any = SocketEmbedder(
                cfg.embedder_dims, state_dir=cfg.state_dir,
            )
        else:
            self.embedder = MLXEmbedder(
                model_path=cfg.embedder_model,
                expected_dims=cfg.embedder_dims,
            )
        self.store = VecStore(cfg.db_path, dims=cfg.embedder_dims)
        # Lazy: opened on first log call. Audit failures must never
        # propagate to the caller, so HistoryStore swallows its own
        # exceptions internally.
        from memo.history import HistoryStore as _HS
        self.history = _HS(cfg.history_db)
        # Helper LLM is lazy — only constructed when `auto_derive=True`
        # is requested. Cold load of Qwen2.5-3B is ~2-3s; users who
        # don't opt in pay nothing.
        self._chat: MLXChat | None = None
        # Knowledge-graph store. Cheap to open (just sqlite); creating
        # eagerly so graph queries never lazy-stall a CLI command.
        self.graph = GraphStore(cfg.graph_db)
        # Reranker is lazy — first hybrid `search()` triggers the load
        # if `cfg.reranker_enabled`. Cold load of Qwen3-Reranker-0.6B
        # is ~1-2s; users who disable it (CI, vec-only mode) pay zero.
        self._reranker: Any | None = None
        # Self-heal probe: warn (don't crash) if the store has paths
        # that don't resolve in the current `memory_dir` layout. Common
        # after upgrading from a legacy install without running
        # `memo migrate-vault` or `memo reindex`.
        self._maybe_warn_legacy_paths()
        # Temporal analyzer for contradiction detection and timeline analysis
        self._temporal: TemporalAnalyzer | None = None
        # Persistent contradiction sidecar — opened lazily so callers
        # that never scan don't pay for the extra sqlite handle.
        self._contradict_store: ContradictionStore | None = None

    def _ensure_chat(self) -> MLXChat:
        """Construct the chat wrapper without loading model weights yet."""
        if self._chat is None:
            self._chat = MLXChat()
        return self._chat

    def _generate_contextual_summary(self, prompt: str) -> str:
        """Generate the short indexing-only context for contextual retrieval."""
        out = self._ensure_chat().chat(
            model=self.cfg.helper_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "max_tokens": 96},
        )
        return str((out.get("message") or {}).get("content") or "")

    def _compose_for_embed(self, title: str, body: str) -> str:
        base = _compose_for_embed(title, body)
        context = get_or_generate_context(
            title=title,
            body=body,
            state_dir=self.cfg.state_dir,
            generate=self._generate_contextual_summary,
        )
        return prepend_context(base, context)

    @property
    def temporal(self) -> TemporalAnalyzer:
        """Lazy accessor for TemporalAnalyzer."""
        if self._temporal is None:
            chat = self._ensure_chat()
            self._temporal = TemporalAnalyzer(self, chat)
        return self._temporal

    @property
    def consolidator(self) -> AdvancedConsolidator:
        """Lazy accessor for AdvancedConsolidator."""
        return AdvancedConsolidator(self, self._chat)

    @property
    def contradict_store(self) -> ContradictionStore:
        """Lazy accessor for the persistent contradictions sidecar."""
        if self._contradict_store is None:
            try:
                self._contradict_store = ContradictionStore(self.cfg.contradictions_db)
            except Exception as exc:
                _log.warning("contradict_store init failed: %s", exc)
                # Return a fresh instance that will fail gracefully on use
                self._contradict_store = ContradictionStore(self.cfg.contradictions_db)
        return self._contradict_store

    @property
    def contradict_scanner(self) -> ContradictionScanner:
        """Lazy accessor for the corpus-wide contradiction scanner."""
        return ContradictionScanner(self, self.contradict_store, self.temporal)

    @property
    def navigator(self) -> GraphNavigator:
        """Lazy accessor for GraphNavigator."""
        return GraphNavigator(self.graph)

    @property
    def contextual(self) -> ContextualRecall:
        """Lazy accessor for ContextualRecall."""
        context_store = ContextStore(self.cfg.state_dir)
        return ContextualRecall(self, context_store)

    @property
    def crossref(self) -> CrossReferenceIndex:
        """Lazy accessor for CrossReferenceIndex."""
        return CrossReferenceIndex(self.cfg.crossref_db)

    @property
    def link_suggester(self) -> LinkSuggester:
        """Lazy accessor for LinkSuggester."""
        return LinkSuggester(self, self.crossref)

    @property
    def lifecycle(self) -> LifecycleManager:
        """Lazy accessor for LifecycleManager."""
        return LifecycleManager(self)

    @property
    def proactive(self) -> ProactiveSuggester:
        """Lazy accessor for ProactiveSuggester."""
        return ProactiveSuggester(self, self._chat)

    @property
    def versioning(self) -> VersionManager:
        """Lazy accessor for VersionManager."""
        return VersionManager(self)

    @property
    def query_composer(self) -> QueryComposer:
        """Lazy accessor for QueryComposer."""
        return QueryComposer(self, QueryStore(self.cfg.state_dir))

    @property
    def federation(self) -> FederationSearcher:
        """Lazy accessor for FederationSearcher."""
        return FederationSearcher(FederationConfig(self.cfg.state_dir / "federation.json"))

    @property
    def backup(self) -> BackupManager:
        """Lazy accessor for BackupManager."""
        return BackupManager(
            memory_dir=self.cfg.memory_dir,
            db_dir=self.cfg.state_dir,
            backup_dir=self.cfg.state_dir / "backups",
        )

    @property
    def sync(self) -> SyncManager:
        """Lazy accessor for SyncManager."""
        return SyncManager(self)

    @property
    def encryption(self) -> EncryptionManager:
        """Lazy accessor for EncryptionManager."""
        key_manager = KeyManager(self.cfg.state_dir)
        encryptor = Encryptor(key_manager)
        return EncryptionManager(key_manager, encryptor)

    @property
    def sharing(self) -> ShareManager:
        """Lazy accessor for ShareManager."""
        return ShareManager(ShareStore(self.cfg.state_dir))

    @property
    def analytics(self) -> AnalyticsEngine:
        """Lazy accessor for AnalyticsEngine."""
        return AnalyticsEngine(self)

    @property
    def dashboard(self) -> Dashboard:
        """Lazy accessor for Dashboard."""
        return Dashboard(self.analytics)

    @property
    def import_export(self) -> ImportExportManager:
        """Lazy accessor for ImportExportManager."""
        return ImportExportManager(self)

    @property
    def multimodal(self) -> MultiModalManager:
        """Lazy accessor for MultiModalManager."""
        store = MultiModalStore(self.cfg.state_dir)
        embedder = UniversalEmbedder()
        search = CrossModalSearch(store, embedder)
        return MultiModalManager(store, embedder, search)

    @property
    def collaborative(self) -> CollaborativeManager:
        """Lazy accessor for CollaborativeManager."""
        graph = CollaborativeGraph(self.cfg.state_dir)
        filter = CollaborativeFilter(graph)
        return CollaborativeManager(graph, filter)

    def _maybe_warn_legacy_paths(self) -> None:
        """Stderr warning when stored `meta.path` rows don't resolve.

        Skipped after `user_version >= 1` (set by `reindex`) so a clean
        install doesn't pay the probe cost on every startup. Probe is
        bounded to 5 sample rows.
        """
        try:
            if self.store.get_user_version() >= 1:
                return
            sample = self.store.list_recent(limit=5)
            if not sample:
                return
            missing = sum(
                1 for r in sample
                if not self._resolve_existing(r["path"]).is_file()
            )
            if missing == len(sample):
                import os as _os
                import sys
                # Suppressible — TUI sets MEMO_SUPPRESS_LEGACY_WARN=1 because
                # the message is moot while in alt-screen mode.
                if _os.environ.get("MEMO_SUPPRESS_LEGACY_WARN") == "1":
                    return
                print(
                    f"[memo] heads-up: tu índice apunta a paths antiguos "
                    f"(data_dir={self.cfg.data_dir}). Corré `memo reindex` "
                    f"para re-embeder, o `memo migrate-vault <nuevo-dir>` si "
                    f"moviste el vault. Esto NO es un error — sólo un aviso "
                    f"de inconsistencia entre disco e índice.",
                    file=sys.stderr,
                )
        except Exception:
            # Probe failures must never block startup — they're advisory.
            pass

    # -- save ---------------------------------------------------------------

    def _derive_metadata(self, content: str) -> dict[str, Any]:
        """Use the helper LLM (Qwen2.5-3B-Instruct-4bit) to derive
        {title, type, tags} from raw content. Returns a dict with
        whatever keys the model produced (any can be None on parse
        failure). Caller decides whether to fill missing fields.

        Failure modes are absorbed: a bad LLM response yields an empty
        dict and the caller falls back to heuristics. We never propagate
        an LLM error up to a save() call — the save must succeed even
        if the helper is broken.
        """
        if self._chat is None:
            self._chat = MLXChat()
        try:
            out = self._chat.chat(
                model=self.cfg.helper_model,
                messages=[
                    {"role": "system", "content": _DERIVE_SYSTEM_PROMPT},
                    # Cap input to keep the prompt cheap. The helper only
                    # needs the gist, not the full body.
                    {"role": "user", "content": content[:2000]},
                ],
                options={"temperature": 0.0, "max_tokens": 256},
            )
            text = (out.get("message") or {}).get("content") or ""
        except Exception as exc:
            _log.warning("_derive_metadata LLM call failed: %s", exc)
            return {}
        # Tolerate markdown code fences even though the prompt forbids them.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        try:
            data = json.loads(text)
        except Exception as exc:
            _log.warning("_derive_metadata JSON parse failed (%r…): %s", text[:80], exc)
            return {}
        if not isinstance(data, dict):
            return {}
        derived: dict[str, Any] = {}
        t_title = (data.get("title") or "")
        if isinstance(t_title, str) and t_title.strip():
            derived["title"] = t_title.strip()[:80]
        t_type = data.get("type")
        if isinstance(t_type, str) and t_type in _VALID_TYPES:
            derived["type"] = t_type
        t_tags = data.get("tags") or []
        if isinstance(t_tags, list):
            derived["tags"] = _normalise_tags([t for t in t_tags if isinstance(t, str)])
        return derived

    def save(
        self,
        *,
        content: str,
        title: str | None = None,
        type_: str = "note",
        type: str | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        auto_derive: bool = False,
        auto_project: bool = True,
        cwd: str | None = None,
        defer_embed: bool = False,
        respect_synapse_freeze: bool | None = None,
        skip_memflow_receipt: bool = False,
    ) -> MemoryRecord:
        """Persist a memory to disk + index.

        - `content`: free-form markdown body (no frontmatter; we add it).
        - `title`: optional. If omitted, derived from the first line of
          content (truncated, slug-safe).
        - `type_`: must be in `_VALID_TYPES`. `type` is accepted as a
          compatibility alias. `note` is the default neutral value.
        - `tags`: optional list. Lower-cased + de-duplicated.
        - `extra`: arbitrary JSON-serialisable metadata bag.
        - `auto_derive`: when True, calls the helper LLM
          (`Qwen2.5-3B-Instruct-4bit`) to fill any missing field
          (title is None, type_ is "note" with no tags). Adds ~1-2s
          latency on first call (cold model load) plus ~0.5-1s per save.
          Use for callers (eg. another agent) that don't carry context
          to derive metadata themselves.
        - `defer_embed`: when True, write markdown + metadata + BM25
          index only. Semantic search won't see the record until
          `memo reindex` runs with the embedder available.
        - `respect_synapse_freeze`: when True, query synapse's
          `RealityConflict` ledger before commit and raise
          `WriteRefused` if a blocking freeze-write covers this
          memoria's topic. Defaults to the env knob
          `MEMO_RESPECT_SYNAPSE_FREEZE=1` (opt-in). Only fires when
          `extra` carries a `synapse_trace_id` — anonymous saves
          bypass the check.
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")

        # Auto-attach SYNAPSE_TRACE_ID from env when the caller did not
        # carry an explicit trace_id in `extra`. Lets provenance walks
        # link memo writes back to the synapse session that spawned the
        # subprocess, even for direct `memo save` CLI invocations.
        env_trace = os.environ.get("SYNAPSE_TRACE_ID", "").strip()
        if env_trace and (extra is None or not extra.get("synapse_trace_id")):
            extra = dict(extra or {})
            extra["synapse_trace_id"] = env_trace

        # Freeze-write protocol: opt-in pre-write check against synapse.
        # Only fires when (a) the caller asked (kwarg or env), (b) the
        # save carries provenance (otherwise we have no agent context
        # to reason about), and (c) synapse is on PATH.
        if respect_synapse_freeze is None:
            respect_synapse_freeze = (
                os.environ.get("MEMO_RESPECT_SYNAPSE_FREEZE") == "1"
            )
        if respect_synapse_freeze and extra and extra.get("synapse_trace_id"):
            self._enforce_synapse_freeze(
                title=title, content=content, tags=tags,
                trace_id=str(extra.get("synapse_trace_id") or ""),
            )
        if type is not None:
            if type_ != "note" and type_ != type:
                raise ValueError("Pass either `type_` or `type`, not conflicting values")
            type_ = type
        if type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        if auto_derive:
            # Only fire the LLM if at least one field looks "default-y".
            # User-provided values always win.
            wants_title = title is None
            wants_type = type_ == "note"
            wants_tags = not tags
            if wants_title or wants_type or wants_tags:
                derived = self._derive_metadata(content)
                if wants_title and derived.get("title"):
                    title = derived["title"]
                if wants_type and derived.get("type"):
                    type_ = derived["type"]
                if wants_tags and derived.get("tags"):
                    tags = derived["tags"]

        title = (title or _derive_title(content)).strip()
        if not title:
            title = "untitled"

        norm_tags = _normalise_tags(tags or [])

        # Auto-tag with the caller's project (git toplevel basename or
        # MEMO_PROJECT_TAG) so per-repo recall can boost the right
        # memorias. Skipped when the caller already passed any
        # `project:` tag — explicit always wins.
        if auto_project and os.environ.get("MEMO_AUTO_PROJECT_TAG", "1") == "1":
            try:
                from memo.project import current_project_tag, has_project_tag
                if not has_project_tag(norm_tags):
                    pt = current_project_tag(cwd)
                    if pt:
                        norm_tags = _normalise_tags([*norm_tags, pt])
            except Exception as exc:
                _log.warning("auto-project tag failed (cwd=%s): %s", cwd, exc)

        now_iso = _now_iso()
        # Truncate content for embedding (vec store doesn't truncate;
        # disk file keeps full content). 64KB is the default cap.
        if len(content) > self.cfg.max_content_chars:
            _log.warning(
                "save: content truncated from %d to %d chars (title=%r)",
                len(content), self.cfg.max_content_chars, title,
            )
        content = content[: self.cfg.max_content_chars]

        record_id = uuid.uuid4().hex
        rel_path = self._build_rel_path(title, now_iso)
        body_hash = _sha256_short(content)

        # Write `.md` first — if anything fails after this, the user
        # can recover by re-indexing. Conversely if we write the index
        # first and the disk write fails, the index points to a
        # non-existent file.
        abs_path = self.cfg.memory_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        extra_for_store = dict(extra or {})
        if defer_embed:
            extra_for_store["_memo_embed_pending"] = True

        post = frontmatter.Post(
            content,
            id=record_id,
            title=title,
            type=type_,
            tags=norm_tags,
            created=now_iso,
            updated=now_iso,
        )
        if extra_for_store:
            post["extra"] = extra_for_store
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        if defer_embed:
            self.store.upsert_text_only(
                id_=record_id,
                path=rel_path,
                title=title,
                type_=type_,
                tags=norm_tags,
                created=now_iso,
                updated=now_iso,
                body_hash=body_hash,
                extra=extra_for_store,
                body_text=content,
            )
            self.history.log_save(
                ts=now_iso, record_id=record_id, title=title, type_=type_,
                provenance=_extract_provenance(extra_for_store),
            )
            deferred_rec = MemoryRecord(
                id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
                created=now_iso, updated=now_iso, body=content, extra=extra_for_store,
            )
            self._emit_save_receipt(
                deferred_rec, deferred=True, disabled=skip_memflow_receipt,
            )
            return deferred_rec

        # Embed `title + body`: the title carries the highest-density
        # signal for retrieval ("Astor — Informe TO" is a much better
        # match for a query like "informe terapia ocupacional astor"
        # than the body's clinical paragraphs alone). Prepending also
        # protects the title from head-truncation when the body is
        # long — see embedder.py for the truncation rationale.
        embedding = self.embedder.embed([self._compose_for_embed(title, content)])[0]
        assert_valid_embedding(
            embedding, self.cfg.embedder_dims, context=f"save id={record_id[:8]}",
        )

        self.store.upsert(
            id_=record_id,
            path=rel_path,
            title=title,
            type_=type_,
            tags=norm_tags,
            created=now_iso,
            updated=now_iso,
            body_hash=body_hash,
            embedding=embedding,
            extra=extra,
            body_text=content,
        )

        self.history.log_save(
            ts=now_iso, record_id=record_id, title=title, type_=type_,
            provenance=_extract_provenance(extra),
        )

        rec = MemoryRecord(
            id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
            created=now_iso, updated=now_iso, body=content, extra=extra or {},
        )
        self._emit_save_receipt(rec, deferred=False, disabled=skip_memflow_receipt)
        return rec

    def _emit_save_receipt(
        self, rec: MemoryRecord, *, deferred: bool, disabled: bool,
    ) -> None:
        """Fire-and-forget memflow receipt for a successful save.

        No-op unless `MEMO_EMIT_RECEIPTS=1`; never raises. `disabled`
        is the synapse-originated opt-out (synapse keeps its own ledger).
        """
        from memo.receipts import emit_receipt

        prov = _extract_provenance(rec.extra or {})
        emit_receipt(
            "save",
            text=f"Memo saved memoria {rec.id[:8]} ({rec.type}): {rec.title}",
            meta={
                "id": rec.id,
                "type": rec.type,
                "tags": ",".join(rec.tags),
                "path": rec.path,
                "deferred": deferred,
                "synapse_trace_id": prov.get("synapse_trace_id", ""),
                "synapse_route_reason": prov.get("synapse_route_reason", ""),
                "synapse_agent_id": prov.get("synapse_agent_id", ""),
            },
            disabled=disabled,
        )
        # M2b: also emit to the unified trinity ledger (best-effort,
        # independent of the memflow receipt path).
        self._emit_ledger("save", rec, prov, deferred=deferred)

    def _emit_ledger(
        self,
        op: str,
        rec: MemoryRecord,
        prov: dict[str, Any] | None = None,
        *,
        deferred: bool = False,
    ) -> None:
        """Fire-and-forget ConsciousnessEvent for the unified ledger (M2)."""
        from memo.consciousness_ledger import emit_event

        emit_event(
            op,
            subject_uri=f"memo://memoria/{rec.id}",
            trace_id=(prov or {}).get("synapse_trace_id", "") or "",
            actor=(prov or {}).get("synapse_agent_id", "") or "memo",
            payload={
                "id": rec.id,
                "type": rec.type,
                "title": rec.title or "",
                "tags": list(rec.tags or []),
                "deferred": deferred,
            },
        )

    # -- search -------------------------------------------------------------

    def search(
        self, query: str, *, limit: int | None = None, type_: str | None = None,
        mode: str = "hybrid", load_bodies: bool = True, disable_reranker: bool = False,
        recency: bool = False, exclude_types: set[str] | None = None,
        include_forgotten: bool = False,
    ) -> list[MemoryRecord]:
        """Top-k search. Three modes:

        - `vec` (semantic only): query embedded via `embed_query`,
          ranked by cosine.
        - `bm25` (keyword only): FTS5 over title+tags+body.
        - `hybrid` (default): reciprocal rank fusion of vec + bm25
          candidates. Picks up both diffuse semantic matches AND
          precise keyword matches (tag names, code snippets, file
          paths) that the small embedder model misses on its own.

        Each result has `.score` populated. For hybrid, `.score` is the
        fused RRF score (not directly comparable to single-mode scores
        but monotonic for ranking).

        Args:
            load_bodies: If False, bodies are not loaded from disk (lazy).
                Useful for reranking/filtering before final formatting.
                Caller must call `_load_body(record.path)` when needed.
            disable_reranker: If True, skip cross-encoder reranking even
                when enabled in config. Useful for chat synthesis where
                RRF is sufficient and reranker adds latency.
            recency: If True, blend a freshness bonus into the final score
                (newer memorias rank higher) even when MEMO_SEARCH_DECAY_HALFLIFE
                is unset. The consumer-facing paths (recall hook, ask/chat) pass
                this so stale facts don't crowd out recent ones; the eval
                harness leaves it False to keep a raw, comparable baseline.
            exclude_types: Drop hits whose `type` is in this set, pushed into
                SQL so the candidate pool isn't spent on rows the caller will
                discard. The recall hook + briefing pass `REFERENCE_TYPES`
                (see `memo.tiers`) so bulk vault material stays searchable on
                demand but never drowns durable knowledge in the prompt.
        """
        if not query or not query.strip():
            return []
        limit = limit or self.cfg.search_default_limit

        if mode == "bm25":
            rows = self.store.search_bm25(query, limit=limit, type_=type_, exclude_types=exclude_types)
        elif mode == "vec":
            # Asymmetric retrieval: queries are embedded WITH the
            # instruction prefix; documents are embedded RAW (in
            # `save()` / `update()`). See `_QUERY_INSTRUCTION_PREFIX`
            # in `embedder.py` for the why.
            emb = self.embedder.embed_query(query)
            rows = self.store.search(emb, limit=limit, type_=type_, exclude_types=exclude_types)
        else:
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = self.cfg.rerank_input_k if self.cfg.reranker_enabled else limit
            k_each = max(input_k * 2, 30)
            emb = self.embedder.embed_query(query)
            vec_hits = self.store.search(emb, limit=k_each, type_=type_, exclude_types=exclude_types)
            bm_hits = self.store.search_bm25(query, limit=k_each, type_=type_, exclude_types=exclude_types)
            rows = _rrf_fuse(vec_hits, bm_hits, limit=input_k)
        out: list[MemoryRecord] = []
        for r in rows:
            body = self._read_body(r["path"]) if load_bodies else ""
            out.append(
                MemoryRecord(
                    id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                    tags=r["tags"], created=r["created"], updated=r["updated"],
                    body=body, extra=r.get("extra") or {}, score=r.get("score"),
                ),
            )
        # Drop soft-forgotten memorias (forget_after TTL elapsed, see
        # lifecycle.py) before feedback/rerank so they never reach the
        # consumer — recall, ask, chat all route through here. Reversible
        # via `unforget`; pass include_forgotten=True to surface them.
        if out and not include_forgotten:
            out = [r for r in out if not (r.extra or {}).get(IS_FORGOTTEN_KEY)]
        # Source-level feedback (👍 / 👎) — applied AFTER RRF/vec retrieval
        # but BEFORE cross-encoder rerank so the reranker doesn't waste
        # cycles on hits the user already vetoed. Embeds the query once
        # (reusing the vec-mode embedding when available) and consults
        # the `source_feedback` table for each hit. Disabled when
        # MEMO_FEEDBACK_DISABLED=1.
        if out and os.environ.get("MEMO_FEEDBACK_DISABLED") != "1":
            try:
                fb_emb = locals().get("emb")
                if fb_emb is None:
                    fb_emb = self.embedder.embed_query(query)
                out = self._apply_source_feedback(out, fb_emb)
            except Exception as exc:
                _log.debug("source feedback skipped: %s", exc)

        # Cross-encoder rerank on hybrid mode only. Skipped for vec/bm25
        # since those callers explicitly opted out of fusion entirely;
        # adding rerank to single-mode searches would surprise users
        # benchmarking the raw bi-encoder or BM25 surfaces.
        # Also skipped when disable_reranker=True (e.g., chat synthesis).
        if mode == "hybrid" and self.cfg.reranker_enabled and not disable_reranker and out:
            out = self._rerank(query, out, top_n=limit)
        # Recency decay: blend a freshness bonus into the score so older
        # memories don't crowd out recent ones. MEMO_SEARCH_DECAY_HALFLIFE
        # (days) sets the halflife explicitly; if unset, the consumer paths
        # (recall/ask/chat) still get a sensible default when they pass
        # `recency=True`, while raw `search()` callers (e.g. the eval harness)
        # stay decay-free for a comparable baseline.
        halflife_days = float(os.environ.get("MEMO_SEARCH_DECAY_HALFLIFE", "0") or 0)
        if halflife_days <= 0 and recency:
            halflife_days = _RECALL_DECAY_HALFLIFE_DEFAULT
        if halflife_days > 0 and out:
            alpha = min(max(float(os.environ.get("MEMO_SEARCH_DECAY_ALPHA", "0.15")), 0.0), 1.0)
            out = _apply_decay(out, halflife_days=halflife_days, alpha=alpha)
        return out

    # -- repo corpus -------------------------------------------------------

    def _repo_corpus(self):
        from memo.repo_index import RepoCorpus

        return RepoCorpus(self.cfg, store=self.store, embedder=self.embedder)

    def repo_index(
        self,
        url: str,
        *,
        name: str | None = None,
        ref: str | None = None,
        force: bool = False,
        with_embeddings: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_file_bytes: int | None = None,
        progress=None,
    ) -> dict[str, Any]:
        return self._repo_corpus().index(
            url,
            name=name,
            ref=ref,
            force=force,
            with_embeddings=with_embeddings,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
            progress=progress,
        )

    def repo_embed(self, repo: str, *, force: bool = False, progress=None) -> dict[str, Any]:
        return self._repo_corpus().embed(repo, force=force, progress=progress)

    def repo_status(self, repo: str) -> dict[str, Any] | None:
        return self._repo_corpus().status(repo)

    def repo_search(
        self,
        query: str,
        *,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
    ):
        return self._repo_corpus().search(
            query, limit=limit, repo=repo, path=path, mode=mode,
        )

    def repo_get_file(
        self,
        repo: str,
        path: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        return self._repo_corpus().get_file(repo, path, start=start, end=end)

    def repo_list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._repo_corpus().list(limit=limit)

    def repo_delete(self, repo: str, *, remove_clone: bool = True) -> bool:
        return self._repo_corpus().delete(repo, remove_clone=remove_clone)

    # -- source feedback (public) ------------------------------------------

    def feedback_record(
        self, source_id: str, *, query_text: str, rating: str,
    ) -> dict[str, Any]:
        """Public wrapper around store.record_source_feedback.

        Accepts `rating` as "up"/"down" or "+1"/"-1". Resolves a short
        `source_id` prefix to a full meta.id when possible (errors on
        ambiguity). Embeds `query_text` with the asymmetric retrieval
        prefix so future queries — which use the same prefix — can be
        compared on equal footing.
        """
        rating_norm = self._normalize_rating(rating)
        resolved = self._resolve_source_id(source_id)
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        emb = self.embedder.embed_query(query_text)
        fid = self.store.record_source_feedback(
            source_id=resolved,
            query_text=query_text,
            query_emb=list(emb),
            rating=rating_norm,
        )
        return {
            "feedback_id": fid,
            "source_id": resolved,
            "query_text": query_text,
            "rating": "up" if rating_norm > 0 else "down",
        }

    def feedback_list(
        self, *, source_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        if source_id:
            source_id = self._resolve_source_id(source_id)
        return self.store.list_source_feedback(source_id=source_id, limit=limit)

    def feedback_clear(self, source_id: str) -> int:
        resolved = self._resolve_source_id(source_id)
        return self.store.clear_source_feedback(resolved)

    @staticmethod
    def _normalize_rating(rating: str | int) -> int:
        raw = str(rating).strip().lower()
        if raw in {"up", "+1", "1", "thumbs_up", "positive", "pos"}:
            return 1
        if raw in {"down", "-1", "thumbs_down", "negative", "neg"}:
            return -1
        raise ValueError(f"unknown rating {rating!r}; expected up/down")

    def _resolve_source_id(self, source_id: str) -> str:
        sid = (source_id or "").strip()
        if not sid:
            raise ValueError("source_id is required")
        # Already a full id (32 hex chars) — accept as-is.
        if len(sid) >= 32:
            return sid
        # Prefix lookup. Must match exactly one row.
        matches = self.store.find_by_prefix(sid, limit=2)
        if not matches:
            raise ValueError(f"no memoria matches source_id prefix {sid!r}")
        if len(matches) > 1:
            raise AmbiguousIdError(sid, matches)
        return matches[0]

    def _apply_source_feedback(
        self, hits: list[MemoryRecord], query_emb: list[float],
        *, sim_threshold: float = 0.85, boost_per_vote: float = 0.15,
        boost_cap: float = 0.6,
    ) -> list[MemoryRecord]:
        """Filter/boost hits using prior 👍/👎 votes for the user query.

        For each hit, look up `source_feedback` rows on `hit.id` whose
        query embedding is cosine-similar to `query_emb` at >=
        `sim_threshold`. Then:

        - Any negative match → drop the hit (hard exclude). User said
          this source is wrong for this kind of query; trust them.
        - Positive matches → score += `boost_per_vote * n`, capped at
          `boost_cap`. Doesn't replace ranking entirely — just lifts
          well-reviewed sources up the list.
        - No relevant feedback → hit passes through unchanged.

        Tunables (env, optional):
        - `MEMO_FEEDBACK_SIM_THRESHOLD` (default 0.85)
        - `MEMO_FEEDBACK_BOOST_PER_VOTE` (default 0.15)
        - `MEMO_FEEDBACK_BOOST_CAP` (default 0.6)
        """
        sim_threshold = float(os.environ.get("MEMO_FEEDBACK_SIM_THRESHOLD") or sim_threshold)
        boost_per_vote = float(os.environ.get("MEMO_FEEDBACK_BOOST_PER_VOTE") or boost_per_vote)
        boost_cap = float(os.environ.get("MEMO_FEEDBACK_BOOST_CAP") or boost_cap)
        from dataclasses import replace
        out: list[MemoryRecord] = []
        for h in hits:
            try:
                fb = self.store.find_feedback_for_source(
                    h.id, query_emb, threshold=sim_threshold,
                )
            except Exception:
                fb = []
            if not fb:
                out.append(h)
                continue
            if any(r["rating"] < 0 for r in fb):
                # Hard exclude — user vetoed this source for similar queries.
                continue
            pos = sum(1 for r in fb if r["rating"] > 0)
            if pos > 0:
                boost = min(boost_cap, boost_per_vote * pos)
                h = replace(h, score=(h.score or 0.0) + boost)
            out.append(h)
        return out

    def rerank_hits(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        """Score externally-supplied hit dicts with the cross-encoder.

        Mirrors the ``memo rerank`` CLI but reuses THIS instance's cached
        reranker (`self._reranker`), so a long-lived server (memo-mcp HTTP
        daemon) pays the Qwen3-Reranker load only once. This is the warm
        equivalent of the per-process CLI used by Synapse's `memo_ce` rerank.

        Each hit is scored on ``"{title}\\n\\n{snippet|body}"`` (truncated to
        ``body_chars``); returns the list reordered with a ``rerank_score``
        field added per hit, original fields preserved. Pass-through (input
        order, no scores) when reranking is disabled in this install.
        """
        if not query or not hits:
            return list(hits or [])
        if not self.cfg.reranker_enabled:
            return list(hits)
        reranker = self._reranker
        if reranker is None:
            from memo.reranker import MLXReranker
            reranker = MLXReranker(
                model_path=self.cfg.reranker_model,
                revision=self.cfg.reranker_revision,
            )
            self._reranker = reranker
        scored: list[tuple[float, dict[str, Any]]] = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            title = str(h.get("title") or "")
            body_src = str(h.get("snippet") or h.get("body") or "")[: max(0, body_chars)]
            doc = f"{title}\n\n{body_src}" if body_src else title
            try:
                p = float(reranker.score(query, doc))
            except Exception:
                p = 0.0
            new = dict(h)
            new["rerank_score"] = p
            scored.append((p, new))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = [h for _p, h in scored]
        if top_n is not None and top_n > 0:
            out = out[:top_n]
        return out

    def _rerank(
        self, query: str, hits: list[MemoryRecord], *, top_n: int,
    ) -> list[MemoryRecord]:
        """Apply the cross-encoder to `hits`, return top-N reordered.

        Score fusion: the final ranking blends the reranker's `P(yes)`
        with the original RRF position so a single noisy cross-encoder
        score can't promote a candidate the bi-encoder + BM25 fusion
        had ranked far down. Position bonus is `1 - i / N` where `i`
        is the original 0-indexed RRF rank — top-of-RRF gets +1.0,
        bottom gets ~0.

        Lazy-loads the reranker on first call. Failures are absorbed:
        if MLX runs into a Metal hiccup mid-rerank we fall back to the
        original RRF order so search never goes dark on the user.
        """
        reranker = self._reranker
        if reranker is None:
            from memo.reranker import MLXReranker
            reranker = MLXReranker(
                model_path=self.cfg.reranker_model,
                revision=self.cfg.reranker_revision,
            )
            self._reranker = reranker

        # Snapshot original RRF positions BEFORE rerank rewrites the
        # `score` field. Index by id rather than object identity so the
        # reranker can return new replaced records without losing the
        # mapping.
        n = len(hits)
        rrf_pos: dict[str, int] = {h.id: i for i, h in enumerate(hits)}

        try:
            reranked = reranker.rerank(query, hits, top_n=None)
        except Exception as exc:
            _log.error(
                "reranker failed (model=%s, revision=%s): %s",
                self.cfg.reranker_model,
                self.cfg.reranker_revision,
                exc,
            )
            _log.info("falling back to RRF order (no cross-encoder reranking)")
            return hits[:top_n]

        alpha = self.cfg.rerank_fusion_alpha
        fused: list[MemoryRecord] = []
        for h in reranked:
            rerank_score = h.score or 0.0
            pos = rrf_pos.get(h.id, n - 1)
            rrf_bonus = 1.0 - (pos / max(n - 1, 1))
            final = alpha * rerank_score + (1.0 - alpha) * rrf_bonus
            fused.append(replace(h, score=final))
        fused.sort(key=lambda h: h.score or 0.0, reverse=True)
        return fused[:top_n]

    # -- chat ask -----------------------------------------------------------

    def chat_ask(
        self,
        question: str,
        *,
        k: int = 7,
        type_: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat-shaped RAG envelope owned by Memo.

        Synapse may provide federation context and Memflow-backed history, but
        retrieval, citations, and synthesis stay inside Memo.
        """
        started = time.perf_counter()
        clean_history = self._normalize_chat_history(history or [])
        clean_context = context or {}
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        rag = self.ask(retrieval_question, k=k, type_=type_)
        total_ms = int((time.perf_counter() - started) * 1000)
        answer = str(rag.get("answer") or "").strip()
        sources = [item for item in (rag.get("sources") or []) if isinstance(item, dict)]
        synthesis_error = ""
        if not question.strip():
            status = "unavailable"
            synthesis_error = "empty question"
        elif answer.startswith("(error consultando el modelo:"):
            status = "error"
            synthesis_error = answer
        elif not answer:
            status = "error"
            synthesis_error = "empty answer"
        elif not sources:
            status = "unavailable"
            synthesis_error = answer
        else:
            status = "ok"
        context_keys = sorted(str(key) for key in clean_context)
        return {
            "schema": "memo.chat_ask.v2",
            "question": question,
            "answer": answer,
            "sources": sources,
            "citations": self._chat_citations(sources),
            "retrieval_trace": [
                {
                    "stage": "memo.chat_ask",
                    "ms": total_ms,
                    "source_count": len(sources),
                    "history_turns": len(clean_history),
                    "context_keys": context_keys,
                    "retrieval_query_chars": len(retrieval_question),
                }
            ],
            "synthesis_status": status,
            "synthesis_source": f"memo.ask:{self.cfg.llm_model}" if sources else "memo.ask",
            "synthesis_error": synthesis_error,
            "total_ms": total_ms,
            "history_turns_used": len(clean_history),
            "context_keys": context_keys,
        }

    def chat_ask_stream(
        self,
        question: str,
        *,
        k: int = 7,
        type_: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming chat-shaped RAG envelope.

        Wraps `ask_stream`, re-shaping events into the `memo.chat_ask.v2`
        schema and emitting:

          - {"event":"context", "schema":"memo.chat_ask.v2",
             "sources":[...], "citations":[...]}              once
          - {"event":"token",   "delta":"..."}                 N times
          - {"event":"done",    ...full envelope...}           once
          - on synthesis error: final `done` has
             synthesis_status="error", synthesis_error=<exc>,
             answer=<partial accumulator>
        """
        started = time.perf_counter()
        clean_history = self._normalize_chat_history(history or [])
        clean_context = context or {}
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        context_keys = sorted(str(key) for key in clean_context)

        sources: list[dict[str, Any]] = []
        accum_parts: list[str] = []
        synthesis_error = ""
        had_error = False

        if not question.strip():
            total_ms = int((time.perf_counter() - started) * 1000)
            yield {
                "event": "done",
                "schema": "memo.chat_ask.v2",
                "question": question,
                "answer": "",
                "sources": [],
                "citations": [],
                "retrieval_trace": [{
                    "stage": "memo.chat_ask_stream",
                    "ms": total_ms,
                    "source_count": 0,
                    "history_turns": len(clean_history),
                    "context_keys": context_keys,
                    "retrieval_query_chars": len(retrieval_question),
                }],
                "synthesis_status": "unavailable",
                "synthesis_source": "memo.ask",
                "synthesis_error": "empty question",
                "total_ms": total_ms,
                "history_turns_used": len(clean_history),
                "context_keys": context_keys,
            }
            return

        for ev in self.ask_stream(retrieval_question, k=k, type_=type_):
            kind = ev.get("event")
            if kind == "sources":
                sources = list(ev.get("sources") or [])
                yield {
                    "event": "context",
                    "schema": "memo.chat_ask.v2",
                    "sources": sources,
                    "citations": self._chat_citations(sources),
                }
            elif kind == "token":
                delta = str(ev.get("delta") or "")
                if delta:
                    accum_parts.append(delta)
                    yield {"event": "token", "delta": delta}
            elif kind == "error":
                had_error = True
                synthesis_error = str(ev.get("message") or "synthesis error")
                partial = str(ev.get("answer_partial") or "")
                if partial and not accum_parts:
                    accum_parts.append(partial)
                break
            elif kind == "done":
                # ask_stream's done carries the final accumulated answer and
                # the source list; prefer it as the authoritative state.
                done_answer = str(ev.get("answer") or "")
                done_sources = ev.get("sources")
                if done_answer and not accum_parts:
                    accum_parts.append(done_answer)
                if isinstance(done_sources, list) and not sources:
                    sources = done_sources

        total_ms = int((time.perf_counter() - started) * 1000)
        answer = "".join(accum_parts).strip()
        if had_error:
            status = "error"
        elif answer.startswith("(error consultando el modelo:"):
            status = "error"
            synthesis_error = answer
        elif not answer:
            status = "error"
            synthesis_error = "empty answer"
        elif not sources:
            status = "unavailable"
            synthesis_error = answer
        else:
            status = "ok"

        yield {
            "event": "done",
            "schema": "memo.chat_ask.v2",
            "question": question,
            "answer": answer,
            "sources": sources,
            "citations": self._chat_citations(sources),
            "retrieval_trace": [{
                "stage": "memo.chat_ask_stream",
                "ms": total_ms,
                "source_count": len(sources),
                "history_turns": len(clean_history),
                "context_keys": context_keys,
                "retrieval_query_chars": len(retrieval_question),
            }],
            "synthesis_status": status,
            "synthesis_source": (
                f"memo.ask_stream:{self.cfg.llm_model}" if sources else "memo.ask_stream"
            ),
            "synthesis_error": synthesis_error,
            "total_ms": total_ms,
            "history_turns_used": len(clean_history),
            "context_keys": context_keys,
        }

    @staticmethod
    def _normalize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in history[-8:]:
            role = str(item.get("role") or "").strip().lower()
            text = str(item.get("text") or item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not text:
                continue
            normalized.append({"role": role, "text": text[:1200]})
        return normalized

    @staticmethod
    def _chat_retrieval_question(
        question: str,
        *,
        history: list[dict[str, str]],
        context: dict[str, Any],
    ) -> str:
        parts = [question.strip()]
        if history:
            turns = "\n".join(
                f"{turn['role']}: {turn['text']}" for turn in history[-6:]
            )
            parts.append(f"Conversation history:\n{turns}")
        if context:
            compact = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
            parts.append(f"Federation context:\n{compact[:2400]}")
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _chat_citations(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for index, source in enumerate(sources, start=1):
            source_kind = str(source.get("source") or "memo")
            source_id = str(
                source.get("id_short")
                or source.get("locator")
                or source.get("id")
                or index
            )
            metadata: dict[str, Any] = {}
            if source.get("id"):
                metadata["id"] = source.get("id")
            if source.get("path"):
                metadata["path"] = source.get("path")
            if source.get("repo_name"):
                metadata["repo_name"] = source.get("repo_name")
            citations.append(
                {
                    "n": index,
                    "id": source_id,
                    "source": "memo" if source_kind == "memory" else source_kind,
                    "title": str(source.get("title") or source_id),
                    "metadata": metadata,
                }
            )
        return citations

    # -- ask ----------------------------------------------------------------

    def _build_ask_context(
        self, question: str, *, k: int, type_: str | None,
        snippet_chars: int, include_repos: bool, disable_reranker: bool = True,
    ) -> tuple[str, list[dict[str, Any]], str, list[MemoryRecord]]:
        """Retrieval half of ask()/ask_stream().

        Returns (normalized_question, sources, user_msg, hits). When no
        hits found, returns (question, [], "", []) — caller must
        short-circuit. `hits` is the raw `MemoryRecord` list (with `body`
        populated) so callers can run cheap heuristics like verbatim
        short-circuit without re-running search.

        Args:
            disable_reranker: If True (default for chat), skip cross-encoder
                reranking. RRF is sufficient for synthesis and reranker adds
                ~150ms latency.
        """
        if not question or not question.strip():
            return question, [], "", []
        _MAX_QUESTION_CHARS = 4000
        if len(question) > _MAX_QUESTION_CHARS:
            _log.warning(
                "ask: question truncated from %d to %d chars",
                len(question), _MAX_QUESTION_CHARS,
            )
            question = question[:_MAX_QUESTION_CHARS]
        # Lazy-load bodies: defer disk I/O until after reranking
        hits: list[MemoryRecord] = self.search(
            question, limit=k, type_=type_, mode="hybrid", load_bodies=False,
            disable_reranker=disable_reranker,
        )
        repo_hits = []
        if include_repos and self.store.list_repo_sources(limit=1):
            with contextlib.suppress(Exception):
                repo_hits = self.repo_search(question, limit=k, mode="hybrid")
        if not hits and not repo_hits:
            return question, [], "", []

        # Load bodies only for the final hits that will be used.
        # MemoryRecord is frozen, so rebuild rather than mutate in place.
        hits = [
            h if h.body else replace(h, body=self._read_body(h.path))
            for h in hits
        ]

        snippet_lines: list[str] = []
        sources: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for h in hits:
            id_short = h.id[:8]
            snippet = (h.body or "")[:snippet_chars]
            if len(h.body or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            tags = ", ".join(h.tags) or "—"
            snippet_lines.append(
                f"[{id_short}] title: {h.title}  |  type: {h.type}  |  tags: {tags}\n"
                f"{snippet}\n"
            )
            sources.append({
                "source": "memory",
                "id": h.id,
                "id_short": id_short,
                "title": h.title,
                "type": h.type,
                "score": h.score,
                "snippet": snippet,
            })
            seen_paths.update(_vault_dedup_keys(h))
        seen_repo_keys: set[tuple[str, str]] = set()
        for h in repo_hits:
            norm = _norm_dedup_path(h.path)
            base = norm.rsplit("/", 1)[-1] if norm else ""
            # Skip if same file already surfaced as a vault memoria.
            if norm and (norm in seen_paths or base in seen_paths):
                continue
            # Dedup intra-repo: keep only the first (highest-score) chunk per file.
            repo_key = (h.repo_name, norm)
            if repo_key in seen_repo_keys:
                continue
            seen_repo_keys.add(repo_key)
            label = h.locator
            snippet = (h.text or "")[:snippet_chars]
            if len(h.text or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            snippet_lines.append(
                f"[{label}] source: repo  |  path: {h.path}  |  "
                f"lines: {h.line_start}-{h.line_end}  |  match: {h.match_type}\n"
                f"{snippet}\n"
            )
            sources.append({
                "source": "repo",
                "id": h.id,
                "id_short": label,
                "title": h.path,
                "type": "repo",
                "score": h.score,
                "snippet": snippet,
                "repo_name": h.repo_name,
                "path": h.path,
                "line_start": h.line_start,
                "line_end": h.line_end,
                "locator": label,
            })

        user_msg = (
            f"Pregunta del user:\n{question}\n\n"
            f"Contexto relevante ({len(hits)} memorias, {len(repo_hits)} snippets de repo):\n\n"
            + "\n---\n".join(snippet_lines)
        )
        return question, sources, user_msg, hits

    def _verbatim_short_circuit(
        self, question: str, hits: list[MemoryRecord],
    ) -> str | None:
        """If the query is a literal phrase lookup (short, no `?`) and the
        text appears inside the top hit's body, return that body verbatim
        instead of calling the LLM.

        The LLM tends to "helpfully" condense — for `letra`, `comando`,
        `snippet`, `CBU`, `URL` style lookups the user wants the WHOLE
        note dumped, not a 2-sentence summary. Returning early dodges
        token spend and avoids the model second-guessing the user.
        """
        if not hits:
            return None
        q = (question or "").strip()
        if not q:
            return None
        # Question form → defer to synthesis.
        if "?" in q or "¿" in q:
            return None
        tokens = [t for t in q.split() if t]
        if len(tokens) > 12:
            return None
        top = hits[0]
        body = (top.body or "").strip()
        if not body or len(body) <= len(q):
            return None
        if q.lower() not in body.lower():
            return None
        return f"{body}\n\n[{top.id[:8]}]"

    def ask(
        self, question: str, *, k: int = 5, type_: str | None = None,
        snippet_chars: int = 2000, include_repos: bool = True,
    ) -> dict[str, Any]:
        """Synthesised Q&A over the memory archive (RAG).

        Pipeline: hybrid search top-`k` → format snippets with `[id]`
        citations → MLXChat 7B (`MEMO_LLM_MODEL`) generates a prose
        answer with inline citations. Returns:

            {
                "question": str,
                "answer": str,            # may say "no encuentro ..."
                "sources": [               # the snippets the LLM saw
                    {id, title, type, score, snippet}, ...
                ],
            }

        Latency: ~3-8s on a cold 7B load + ~1-2s decode for short
        answers. For token-by-token output use `ask_stream`. Use
        `search` if you only need IDs to scan manually.
        """
        if not question or not question.strip():
            return {"question": question, "answer": "", "sources": []}
        norm_question, sources, user_msg, hits = self._build_ask_context(
            question, k=k, type_=type_,
            snippet_chars=snippet_chars, include_repos=include_repos,
        )
        if not sources:
            return {
                "question": norm_question,
                "answer": "no encuentro la respuesta en las memorias guardadas",
                "sources": [],
            }

        # Verbatim short-circuit: literal phrase lookups bypass the LLM
        # and return the matched note body directly. Avoids the model
        # over-summarising when the user clearly wants the raw content.
        verbatim = self._verbatim_short_circuit(question, hits)
        if verbatim is not None:
            return {
                "question": norm_question,
                "answer": verbatim,
                "sources": sources,
            }

        # Lazy-construct the chat client (same instance used by auto_derive).
        if self._chat is None:
            self._chat = MLXChat()
        try:
            out = self._chat.chat(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _ASK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                # Higher max_tokens than auto_derive — answers can run a
                # paragraph or two.
                options={"temperature": 0.0, "max_tokens": 768},
            )
            answer = ((out.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            answer = f"(error consultando el modelo: {type(exc).__name__})"

        return {
            "question": norm_question,
            "answer": answer,
            "sources": sources,
        }

    def ask_stream(
        self, question: str, *, k: int = 5, type_: str | None = None,
        snippet_chars: int = 2000, include_repos: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Streaming variant of `ask()` — yields token-level events.

        Event protocol (NDJSON-compatible dicts):
          - {"event": "sources", "sources": [...]}            once, after retrieval
          - {"event": "token",   "delta": "<chunk>"}          one per LLM yield
          - {"event": "done",    "answer": "<acc>", "sources": [...]}
          - {"event": "error",   "message": "...", "answer_partial": "..."}

        Empty-question / no-hits paths short-circuit with a single `done`
        carrying the same refusal text as `ask()`.
        """
        if not question or not question.strip():
            yield {"event": "done", "answer": "", "sources": []}
            return
        _, sources, user_msg, hits = self._build_ask_context(
            question, k=k, type_=type_,
            snippet_chars=snippet_chars, include_repos=include_repos,
        )
        if not sources:
            yield {
                "event": "done",
                "answer": "no encuentro la respuesta en las memorias guardadas",
                "sources": [],
            }
            return

        yield {"event": "sources", "sources": sources}

        # Verbatim short-circuit (literal-phrase queries): emit body as a
        # single token-style event so consumers that show progressive
        # output still get something to render, then a terminal `done`.
        verbatim = self._verbatim_short_circuit(question, hits)
        if verbatim is not None:
            yield {"event": "token", "delta": verbatim}
            yield {"event": "done", "answer": verbatim, "sources": sources}
            return

        if self._chat is None:
            self._chat = MLXChat()
        accum_parts: list[str] = []
        try:
            for delta in self._chat.chat_stream(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _ASK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0, "max_tokens": 768},
            ):
                accum_parts.append(delta)
                yield {"event": "token", "delta": delta}
        except Exception as exc:
            yield {
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "answer_partial": "".join(accum_parts).strip(),
            }
            return

        yield {
            "event": "done",
            "answer": "".join(accum_parts).strip(),
            "sources": sources,
        }

    # -- list ---------------------------------------------------------------

    def list(
        self, *, limit: int = 20, type_: str | None = None,
        include_forgotten: bool = False,
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each.

        Soft-forgotten memorias (see `forget`) are excluded unless
        `include_forgotten=True`.
        """
        rows = self.store.list_recent(limit=limit, type_=type_)
        if not include_forgotten:
            rows = [r for r in rows if not (r.get("extra") or {}).get(IS_FORGOTTEN_KEY)]
        return [
            MemoryRecord(
                id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                tags=r["tags"], created=r["created"], updated=r["updated"],
                body=self._read_body(r["path"]), extra=r.get("extra") or {},
            )
            for r in rows
        ]

    # -- get ----------------------------------------------------------------

    def resolve_id(self, id_or_prefix: str) -> str | None:
        """Resolve a full id or a unique prefix.

        Returns the canonical 32-char id if `id_or_prefix` matches exactly
        one record (full or prefix), or None if nothing matches. Raises
        `AmbiguousIdError` when 2+ records share the prefix — the caller
        is expected to surface the candidates so the user can disambiguate.

        Why prefix lookup: pasting a 32-char UUID4 from chat is friction.
        Git-style 7-char prefixes are unique with overwhelming probability
        for the corpus sizes memo targets (~thousands).
        """
        if not id_or_prefix:
            return None
        # Fast path: full hex hit.
        if len(id_or_prefix) == 32 and self.store.get(id_or_prefix) is not None:
            return id_or_prefix
        matches = self.store.find_by_prefix(id_or_prefix.lower())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousIdError(id_or_prefix, matches)
        return None

    def get(self, id_: str) -> MemoryRecord | None:
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if not r:
            return None
        return MemoryRecord(
            id=r["id"], path=r["path"], title=r["title"], type=r["type"],
            tags=r["tags"], created=r["created"], updated=r["updated"],
            body=self._read_body(r["path"]), extra=r.get("extra") or {},
        )

    def backend_native_replay_resolve(
        self,
        uri: str,
        *,
        trace_id: str = "",
        backend_version: str = "",
    ) -> dict[str, Any]:
        """Resolve Synapse backend_native.v1 evidence without mutating Memo."""

        def payload(
            status: str,
            detail: str,
            *,
            content_hash: str = "",
            target: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {
                "schema": SYNAPSE_BACKEND_NATIVE_SCHEMA,
                "protocol_version": NATIVE_BACKEND_PROTOCOL_VERSION,
                "backend": MEMO_BACKEND_NAME,
                "uri": uri,
                "status": status,
                "detail": detail,
                "content_hash": content_hash,
                "observed_at": _utc_now_iso(),
                "backend_version": backend_version,
                "trace_id": trace_id,
                "resolution_mode": "backend_native",
            }
            if target is not None:
                out["target"] = target
            return out

        memoria_prefix = "memo://memoria/"
        repo_index_prefix = "memo://repo-index/"
        repo_prefix = "memo://repo/"

        if uri.startswith(memoria_prefix):
            memoria_id = uri[len(memoria_prefix):].strip()
            if not memoria_id:
                return payload("missing", "memo://memoria URI did not include an id.")
            try:
                rec = self.get(memoria_id)
            except AmbiguousIdError as exc:
                return payload(
                    "error",
                    f"ambiguous memoria id prefix {exc.prefix!r}: {len(exc.matches)} matches",
                )
            if rec is None:
                return payload("missing", "Memo memoria was not found.")
            return payload(
                "found",
                f"resolved memoria: {rec.id}",
                content_hash=_stable_content_hash(rec.to_dict()),
                target={"kind": "memoria", "id": rec.id, "path": rec.path},
            )

        if uri.startswith(repo_index_prefix):
            rest = uri[len(repo_index_prefix):].strip("/")
            if not rest or "/" not in rest:
                return payload(
                    "missing",
                    "memo://repo-index URI must include <repo-name>/<commit-prefix>.",
                )
            repo_name, commit_prefix = rest.split("/", 1)
            source = self.store.get_repo_source(repo_name)
            if source is None:
                return payload("missing", "Memo repo source was not found.")
            commit = str(source.get("commit_sha") or "")
            if commit_prefix and commit_prefix != "unknown" and not commit.startswith(commit_prefix):
                return payload(
                    "missing",
                    "Memo repo source exists but commit did not match the receipt URI.",
                    target={
                        "kind": "repo_index",
                        "repo_id": source.get("id") or "",
                        "name": source.get("name") or repo_name,
                        "commit_sha": commit,
                    },
                )
            resolved = self._repo_replay_payload(source)
            return payload(
                "found",
                f"resolved repo index: {source.get('name')}@{commit[:12]}",
                content_hash=_stable_content_hash(resolved),
                target=resolved,
            )

        if uri.startswith(repo_prefix):
            repo_key = uri[len(repo_prefix):].strip()
            if not repo_key:
                return payload("missing", "memo://repo URI did not include a repo id/name/url.")
            source = self.store.get_repo_source(repo_key)
            if source is None:
                return payload("missing", "Memo repo source was not found.")
            resolved = self._repo_replay_payload(source)
            return payload(
                "found",
                f"resolved repo: {source.get('name')}",
                content_hash=_stable_content_hash(resolved),
                target=resolved,
            )

        return payload(
            "unsupported",
            "Memo backend-native only replays memo://memoria/<id>, "
            "memo://repo/<id|name|url>, and memo://repo-index/<name>/<commit> evidence.",
        )

    def _repo_replay_payload(self, source: dict[str, Any]) -> dict[str, Any]:
        repo_id = str(source.get("id") or "")
        counts = self.store.repo_counts(repo_id) if repo_id else {
            "files": 0,
            "lines": 0,
            "chunks": 0,
            "embedded_chunks": 0,
        }
        pending_chunks = counts["chunks"] - counts["embedded_chunks"]
        return {
            "kind": "repo",
            "id": repo_id,
            "name": source.get("name") or "",
            "url": source.get("url") or "",
            "ref": source.get("ref") or "",
            "commit_sha": source.get("commit_sha") or "",
            "indexed_at": source.get("indexed_at") or "",
            "status": source.get("status") or "",
            "semantic_status": (
                "semantic_ready" if counts["chunks"] and pending_chunks == 0
                else "semantic_pending" if pending_chunks > 0
                else str(source.get("status") or "")
            ),
            "counts": {
                **counts,
                "pending_chunks": pending_chunks,
            },
        }

    # -- update -------------------------------------------------------------

    def update(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Patch one or more fields on an existing record.

        Only the kwargs you pass are touched; everything else stays as-is.
        Re-embed only if `content` changed (body_hash check). The file
        path stays stable — renaming the slug after the fact would break
        wikilinks the user may have created in their vault.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        id_ = resolved
        r = self.store.get(id_)
        if r is None:
            return None
        if type_ is not None and type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        new_title = (title.strip() if title else r["title"]) or r["title"]
        new_type = type_ or r["type"]
        new_tags = _normalise_tags(tags) if tags is not None else r["tags"]
        new_extra = extra if extra is not None else r.get("extra") or {}
        now_iso = _now_iso()

        # Body resolution: provided > on-disk > empty.
        old_body = self._read_body(r["path"])
        new_body = (content if content is not None else old_body)
        new_body = new_body[: self.cfg.max_content_chars]
        new_body_hash = _sha256_short(new_body)
        body_changed = new_body_hash != r["body_hash"]
        title_changed = new_title != r["title"]

        abs_path = self._resolve_existing(r["path"])
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(
            new_body,
            id=id_,
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
        )
        if new_extra:
            post["extra"] = new_extra
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type
        # changes still skip the embedder.
        if body_changed or title_changed:
            embedding = self.embedder.embed([self._compose_for_embed(new_title, new_body)])[0]
            assert_valid_embedding(embedding, self.cfg.embedder_dims, context=f"update id={id_[:8]}")
            self.store.upsert(
                id_=id_, path=r["path"], title=new_title, type_=new_type,
                tags=new_tags, created=r["created"], updated=now_iso,
                body_hash=new_body_hash, embedding=embedding, extra=new_extra,
                body_text=new_body,
            )
        else:
            self.store.update_meta(
                id_=id_, title=new_title, type_=new_type, tags=new_tags,
                updated=now_iso, extra=new_extra,
            )

        # Audit log: build a delta of just the fields that changed.
        delta: dict[str, tuple[Any, Any]] = {}
        if title_changed:
            delta["title"] = (r["title"], new_title)
        if new_type != r["type"]:
            delta["type"] = (r["type"], new_type)
        if new_tags != r["tags"]:
            delta["tags"] = (r["tags"], new_tags)
        if body_changed:
            delta["body_hash"] = (r["body_hash"], new_body_hash)
        # Track provenance churn so a re-route (e.g. Synapse re-issues a
        # different trace_id on the same memoria) shows up in history.
        old_prov = _extract_provenance(r.get("extra") or {})
        new_prov = _extract_provenance(new_extra)
        if old_prov != new_prov:
            delta["_provenance"] = (old_prov, new_prov)
        if delta:
            self.history.log_update(
                ts=now_iso, record_id=id_, title=new_title, type_=new_type,
                delta=delta,
            )

        updated_rec = MemoryRecord(
            id=id_, path=r["path"], title=new_title, type=new_type,
            tags=new_tags, created=r["created"], updated=now_iso,
            body=new_body, extra=new_extra,
        )
        if delta:
            from memo.receipts import emit_receipt

            emit_receipt(
                "update",
                text=f"Memo updated memoria {id_[:8]}: {', '.join(sorted(delta.keys()))}",
                meta={
                    "id": id_,
                    "type": new_type,
                    "title": new_title,
                    "delta_keys": ",".join(sorted(delta.keys())),
                },
            )
            # M2b: also emit to the unified trinity ledger.
            self._emit_ledger("update", updated_rec, new_prov)
        return updated_rec

    # -- forget (soft, reversible) ------------------------------------------

    def forget(self, id_: str, *, reason: str | None = None) -> MemoryRecord | None:
        """Soft-forget a memoria: keep the file + index, but exclude it from
        `search` / recall / `list` by default.

        Distinct from `delete` (which removes file + index) — `forget` is
        reversible via `unforget`. Sets `is_forgotten` (and an optional
        `forget_reason`) in the `extra` bag, merging onto existing metadata so
        provenance and other keys survive. Returns the updated record, or None
        if the id is unknown.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if r is None:
            return None
        merged = dict(r.get("extra") or {})
        merged[IS_FORGOTTEN_KEY] = True
        if reason:
            merged[FORGET_REASON_KEY] = reason
        return self.update(resolved, extra=merged)

    def unforget(self, id_: str) -> MemoryRecord | None:
        """Reverse a `forget`: clear `is_forgotten` so the memoria is searchable
        again. Also clears `forget_after` / `forget_reason` so the next
        maintenance pass doesn't immediately re-forget it. No-op (returns the
        record) if it wasn't forgotten.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if r is None:
            return None
        merged = dict(r.get("extra") or {})
        for key in (IS_FORGOTTEN_KEY, FORGET_AFTER_KEY, FORGET_REASON_KEY):
            merged.pop(key, None)
        return self.update(resolved, extra=merged)

    # -- delete -------------------------------------------------------------

    def delete(self, id_: str) -> bool:
        """Remove from store + disk. Returns True if anything was deleted."""
        resolved = self.resolve_id(id_)
        if resolved is None:
            return False
        id_ = resolved
        r = self.store.get(id_)
        if not r:
            return False
        existed = self.store.delete(id_)
        # File deletion is best-effort; the store is the authoritative delete signal.
        with contextlib.suppress(OSError):
            self._resolve_existing(r["path"]).unlink(missing_ok=True)
        if existed:
            self.history.log_delete(
                ts=_now_iso(), record_id=id_, title=r["title"], type_=r["type"],
            )
            # Drop graph edges for this memoria so entity counts stay
            # honest. Cheap (one DELETE + counter decrement per edge).
            self.graph.drop_for_memoria(id_)
            # Drop dangling contradiction pairs touching this memoria.
            # Only walks if the sidecar was already opened, so callers
            # that never used the radar pay nothing.
            if self._contradict_store is not None:
                with contextlib.suppress(Exception):
                    self._contradict_store.drop_for_memoria(id_)
            from memo.receipts import emit_receipt

            emit_receipt(
                "delete",
                text=f"Memo deleted memoria {id_[:8]} ({r['type']}): {r['title']}",
                meta={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
            # M2b: also emit to the unified trinity ledger.
            from memo.consciousness_ledger import emit_event

            emit_event(
                "delete",
                subject_uri=f"memo://memoria/{id_}",
                trace_id=(_extract_provenance(r.get("extra") or {}) or {}).get("synapse_trace_id", ""),
                actor="memo",
                payload={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
        return existed

    # -- synapse freeze-write protocol -------------------------------------

    def _enforce_synapse_freeze(
        self,
        *,
        title: str | None,
        content: str,
        tags: builtins.list[str] | None,
        trace_id: str,
    ) -> None:
        """Query synapse for blocking RealityConflicts; raise on hit.

        Derives a query from the most signal-dense fields available
        (title, first non-empty tags, first content line). Best-effort:
        if synapse is not on PATH, returns without raising — the
        opt-in nature already implies "best information available".
        """
        # Deferred import: keeps memo's hard deps free of synapse.
        from memo import synapse_client

        if not synapse_client.is_available():
            return
        query = _build_freeze_query(title=title, content=content, tags=tags)
        if not query:
            return
        try:
            conflicts = synapse_client.list_conflicts(
                query, trace_id=trace_id,
            )
        except Exception as exc:  # pragma: no cover - subprocess noise
            _log.debug("synapse freeze-check failed: %s", exc)
            return
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        if blocked and conflict is not None:
            raise WriteRefused(conflict)

    # -- provenance ---------------------------------------------------------

    def provenance(self, id_: str) -> dict[str, Any] | None:
        """Return the full provenance trail for a memoria.

        Combines the current state (provenance subset of `meta.extra_json`)
        with the per-op history (each save/update event carrying its own
        provenance snapshot in `delta_json`). Returns `None` if the id is
        unknown.

        Shape:

            {
              "id": "<full id>",
              "current": {synapse_trace_id, synapse_route_reason, ...},
              "events": [
                {"ts", "op", "title", "type", "provenance": {...}},
                ...
              ]
            }
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        rec = self.store.get(resolved)
        if rec is None:
            return None
        current = _extract_provenance(rec.get("extra") or {})
        events: list[dict[str, Any]] = []
        for raw in self.history.list_recent(limit=10_000, record_id=resolved):
            entry: dict[str, Any] = {
                "ts": raw.get("ts"),
                "op": raw.get("op"),
                "title": raw.get("title"),
                "type": raw.get("type"),
            }
            delta = raw.get("delta") or {}
            if isinstance(delta, dict) and "_provenance" in delta:
                prov = delta["_provenance"]
                # save op stores `{...keys...}`; update op stores
                # `[old_dict, new_dict]` (delta-pair convention). Surface
                # the post-state in both cases.
                if isinstance(prov, list) and len(prov) == 2:
                    entry["provenance"] = prov[1] or {}
                elif isinstance(prov, dict):
                    entry["provenance"] = prov
            events.append(entry)
        events.reverse()  # oldest first
        return {"id": resolved, "current": current, "events": events}

    # -- reindex / gc -------------------------------------------------------

    def reindex(self, *, force: bool = False) -> dict[str, int]:
        """Scan the memory dir, re-embed entries whose on-disk body
        diverged from `body_hash`. Picks up edits the user made in
        Obsidian directly. Also indexes any `.md` with a valid `id` in
        frontmatter that the store doesn't know about (e.g. restored
        from a backup or copied from another machine).

        With `force=True`, re-embeds EVERY indexed entry regardless of
        body_hash match. Use after an embedder model swap, after a
        change to `_compose_for_embed`, or to refresh the index after
        a corruption/incident.

        Returns counts: `{"checked", "reindexed", "added", "skipped"}`.
        """
        memory_root = self.cfg.memory_dir
        checked = reindexed = added = skipped = 0
        if not memory_root.is_dir():
            return {"checked": 0, "reindexed": 0, "added": 0, "skipped": 0}

        for md_path in sorted(memory_root.rglob("*.md")):
            checked += 1
            try:
                post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
            except Exception as exc:
                _log.warning("reindex: skipping %s (parse error): %s", md_path.name, exc)
                skipped += 1
                continue
            meta: dict[str, Any] = post.metadata
            md_id = meta.get("id")
            if not md_id or not isinstance(md_id, str):
                skipped += 1
                continue
            body = post.content or ""
            new_hash = _sha256_short(body)
            existing = self.store.get(md_id)
            # Path relative to memory_dir — paths in the store no longer
            # carry the legacy `<vault>/<memory_subdir>/...` prefix.
            rel = str(md_path.relative_to(self.cfg.memory_dir))

            title = (meta.get("title") or _derive_title(body) or "untitled").strip()
            type_ = meta.get("type") or "note"
            if type_ not in _VALID_TYPES:
                _log.warning("reindex: invalid type %r in %s, coercing to 'note'", type_, md_path.name)
                type_ = "note"
            tags = _normalise_tags(list(meta.get("tags") or []))
            created = meta.get("created") or _now_iso()
            updated = meta.get("updated") or created
            extra = meta.get("extra") or {}
            # Obsidian-friendly: accept `forget_after` / `forget_reason` as
            # TOP-LEVEL frontmatter keys (what a user naturally types in their
            # editor), folding them into the extra bag the lifecycle layer
            # reads. The nested `extra:` form still works and takes precedence.
            for _fk in (FORGET_AFTER_KEY, FORGET_REASON_KEY):
                if _fk in meta and _fk not in extra:
                    extra = {**extra, _fk: meta[_fk]}

            if existing is None:
                # Path-collision guard: an .md may have its frontmatter id
                # regenerated (manual edit, restore-from-backup, or a stale
                # row pointing at a file whose id was rewritten) while the
                # vault-relative path stays the same. The store's
                # UNIQUE(meta.path) constraint blocks a plain INSERT, so we
                # drop the orphan row before re-adding under the new id.
                stale = self.store.get_by_path(rel)
                if stale is not None:
                    _log.warning(
                        "reindex: path %r reused with new id (%s → %s); "
                        "replacing stale row",
                        rel, stale["id"][:8], md_id[:8],
                    )
                    self.store.delete(stale["id"])
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex add {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=created, updated=updated, body_hash=new_hash,
                    embedding=emb, extra=extra if extra else None,
                    body_text=body,
                )
                added += 1
                continue
            missing_vector = not self.store.has_vector(md_id)
            if force or existing["body_hash"] != new_hash or missing_vector:
                if isinstance(extra, dict):
                    extra = dict(extra)
                    extra.pop("_memo_embed_pending", None)
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex update {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=existing["created"], updated=_now_iso(),
                    body_hash=new_hash, embedding=emb,
                    extra=extra if extra else None,
                    body_text=body,
                )
                reindexed += 1
        # Successful reindex: every meta.path now uses the current
        # memory_dir-relative layout, so future startups can skip the
        # legacy-path probe in `_maybe_warn_legacy_paths`.
        self.store.set_user_version(1)
        counts = {"checked": checked, "reindexed": reindexed, "added": added, "skipped": skipped}
        if reindexed or added:
            from memo.receipts import emit_receipt

            emit_receipt(
                "reindex",
                text=(
                    f"Memo reindex: checked={checked} reindexed={reindexed} "
                    f"added={added} skipped={skipped} force={force}"
                ),
                meta={
                    "checked": checked,
                    "reindexed": reindexed,
                    "added": added,
                    "skipped": skipped,
                    "force": force,
                },
            )
        return counts

    def lint(self) -> dict[str, builtins.list[dict[str, Any]]]:
        """Surface memorias with quality issues.

        Categories:
        - `legacy_extra`: has `extra` keys from mem-vault migration
          (`agent_id`, `last_used`, `usage_count`, `user_id`, `description`).
          These don't affect retrieval but bloat the frontmatter — worth
          a manual cleanup pass.
        - `few_tags`: <3 tags. The CLAUDE.md convention is ≥3 (project +
          domain + technique). Few tags hurt discovery via `memo top <tag>`.
        - `body_skinny`: body shorter than 100 chars. May still be useful
          for one-liner facts but worth checking if the user meant to
          write more.
        - `untitled`: title is literally "untitled" or matches the slug.

        Returns a dict of category → list of {id, title, reason} dicts.
        Pure read; never modifies the store.
        """
        legacy_keys = frozenset({
            "agent_id", "last_used", "usage_count", "user_id", "description",
        })
        out: dict[str, list[dict[str, Any]]] = {
            "legacy_extra": [],
            "few_tags": [],
            "body_skinny": [],
            "untitled": [],
        }
        for r in self.store.list_recent(limit=100_000):
            entry = {"id": r["id"], "title": r["title"]}
            extra = r.get("extra") or {}
            if any(k in extra for k in legacy_keys):
                out["legacy_extra"].append(
                    {**entry, "reason": "mem-vault legacy fields in extra: "
                                        + ", ".join(sorted(set(extra) & legacy_keys))},
                )
            if len(r.get("tags") or []) < 3:
                out["few_tags"].append(
                    {**entry, "reason": f"only {len(r.get('tags') or [])} tag(s)"},
                )
            body = self._read_body(r["path"]) or ""
            if len(body.strip()) < 100:
                out["body_skinny"].append(
                    {**entry, "reason": f"body {len(body.strip())} chars"},
                )
            t = (r["title"] or "").strip().lower()
            if t == "untitled" or not t:
                out["untitled"].append({**entry, "reason": "title missing or 'untitled'"})
        return out

    # -- knowledge graph ----------------------------------------------------

    def extract_entities(
        self, *, ids: builtins.list[str] | None = None, all_: bool = False,
        skip_already_indexed: bool = True,
    ) -> dict[str, int]:
        """Extract named entities from memorias and write to the graph.

        Modes:
        - `ids=[...]`: process exactly the listed memoria ids.
        - `all_=True`: process every memoria in the store.

        With `skip_already_indexed=True` (default), memorias that
        already have entries in `entity_memoria` are skipped — useful
        for incremental runs after adding new memorias. Pass False to
        force re-extraction (e.g. after improving the prompt).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memoria with Qwen2.5-3B. 223 memorias ≈ 2-4 min.
        """
        if not all_ and not ids:
            raise ValueError("pass either ids=[...] or all_=True")

        if all_:
            target = [r["id"] for r in self.store.list_recent(limit=100_000)]
        else:
            target = list(ids or [])

        # Pre-filter already-indexed unless --force.
        if skip_already_indexed:
            target = [
                tid for tid in target
                if not self.graph.memoria_entities(tid)
            ]

        counts = {"processed": 0, "entities_extracted": 0,
                  "links_written": 0, "skipped": 0, "errors": 0}

        if not target:
            return counts

        if self._chat is None:
            self._chat = MLXChat()

        for tid in target:
            r = self.store.get(tid)
            if r is None:
                counts["skipped"] += 1
                continue
            body = self._read_body(r["path"])
            if not body.strip():
                counts["skipped"] += 1
                continue
            # Build prompt: title + body excerpt. Cap to ~3000 chars to
            # keep the helper LLM cheap; entities tend to live in the
            # opening paragraphs.
            user_msg = (
                f"Title: {r['title']}\n"
                f"Tags: {', '.join(r['tags']) if r['tags'] else '—'}\n\n"
                f"{body[:3000]}"
            )
            try:
                out = self._chat.chat(
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _EXTRACT_ENTITIES_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384},
                )
                text = ((out.get("message") or {}).get("content") or "").strip()
            except Exception:
                counts["errors"] += 1
                continue
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
            try:
                data = json.loads(text) if text else {}
            except Exception:
                counts["errors"] += 1
                continue
            ents = data.get("entities") if isinstance(data, dict) else None
            if not isinstance(ents, list):
                ents = []
            # Filter to dicts with both name + type fields.
            ents = [
                {"name": e.get("name"), "type": e.get("type")}
                for e in ents
                if isinstance(e, dict) and e.get("name") and e.get("type")
            ]
            n = self.graph.record_extraction(
                memoria_id=tid,
                memoria_date=r["created"][:10] if r.get("created") else _now_iso()[:10],
                entities=ents,
                extracted_at=_now_iso(),
            )
            counts["processed"] += 1
            counts["entities_extracted"] += len(ents)
            counts["links_written"] += n
        return counts

    def consolidate(
        self, *, threshold: float = 0.85, max_clusters: int = 50,
        type_: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Find clusters of near-duplicate memorias and propose actions.

        Algorithm:
        1. Pull all stored embeddings (we have them already; no re-embed).
        2. Greedy single-link clustering by cosine ≥ `threshold`.
           Each memoria joins the first existing cluster it's
           ≥-similar to, or starts a new one.
        3. Drop singletons. The remaining clusters are candidates.
        4. For each cluster, MLXChat 7B reads the bodies and emits a
           JSON `{summary, relationship, rationale}` per
           `_CONSOLIDATE_SYSTEM_PROMPT`.
        5. Return ranked clusters (largest first), capped at
           `max_clusters` to keep the LLM cost finite on big corpora.

        DOES NOT modify anything. The user reviews the output and
        decides via `memo update` / `memo delete`.

        Threshold tuning: 0.85 catches obvious dupes, 0.92+ only catches
        near-identical text. The default 0.85 is conservative for the
        Qwen3-Embedding-0.6B vector space.
        """
        # 1) Pull all (id, embedding, title, type, tags) tuples.
        #    Direct SQL is cheaper than going through the public store
        #    API per-row; we only need 1024 floats x N to fit in RAM,
        #    fine for thousands of entries.
        import struct

        store_conn = self.store._conn
        rows = store_conn.execute(
            "SELECT vec.id AS id, vec.embedding AS emb, "
            "       meta.title, meta.type, meta.tags, meta.path, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            + ("WHERE meta.type = ? " if type_ else "")
            + "ORDER BY meta.updated DESC",
            (type_,) if type_ else (),
        ).fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            blob = r["emb"]
            v = list(struct.unpack(f"{len(blob)//4}f", blob))
            items.append({
                "id": r["id"],
                "title": r["title"],
                "type": r["type"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "path": r["path"],
                "updated": r["updated"],
                "emb": v,
            })

        # 2) Greedy single-link clustering. O(N²) dot product over L2-normalised
        #    vectors (dot == cosine when vectors are unit-length). Fine for
        #    corpora up to ~5K. For larger, swap to a HNSW pass.
        def _dot(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        clusters: list[list[int]] = []  # list of items[] indices
        for i in range(len(items)):
            joined = False
            for cluster in clusters:
                # Check similarity vs the cluster representative (first
                # member). Single-link → if any member is similar enough,
                # add. We use the first member as representative for
                # speed; full single-link would scan all members.
                rep = items[cluster[0]]
                if _dot(items[i]["emb"], rep["emb"]) >= threshold:
                    cluster.append(i)
                    joined = True
                    break
            if not joined:
                clusters.append([i])

        # 3) Drop singletons; rank by size (then by most-recent updated).
        candidate_clusters = [c for c in clusters if len(c) >= 2]
        candidate_clusters.sort(
            key=lambda c: (-len(c), items[c[0]]["updated"]),
        )
        candidate_clusters = candidate_clusters[:max_clusters]

        if not candidate_clusters:
            return []

        # 4) For each cluster, ask MLXChat to summarise + classify.
        if self._chat is None:
            self._chat = MLXChat()

        out: list[dict[str, Any]] = []
        for ci, cluster in enumerate(candidate_clusters):
            members = []
            for idx in cluster:
                it = items[idx]
                body = self._read_body(it["path"])
                members.append({
                    "id": it["id"],
                    "id_short": it["id"][:8],
                    "title": it["title"],
                    "type": it["type"],
                    "tags": it["tags"],
                    "updated": it["updated"],
                    "body_preview": (body[:600] + ("…" if len(body) > 600 else "")),
                })
            # Build LLM prompt with all members.
            prompt = "Cluster:\n\n" + "\n---\n".join(
                f"[{m['id_short']}] title: {m['title']}  |  updated: {m['updated']}\n"
                f"{m['body_preview']}"
                for m in members
            )
            try:
                chat_out = self._chat.chat(
                    model=self.cfg.llm_model,
                    messages=[
                        {"role": "system", "content": _CONSOLIDATE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384},
                )
                text = ((chat_out.get("message") or {}).get("content") or "").strip()
            except Exception:
                text = ""
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
            try:
                data = json.loads(text) if text else {}
            except Exception:
                data = {}
            out.append({
                "cluster_id": ci,
                "size": len(members),
                "members": members,
                "summary": (data.get("summary") or "").strip(),
                "relationship": data.get("relationship") if data.get("relationship") in
                    ("duplicate", "evolution", "facets", "unrelated") else "unrelated",
                "rationale": (data.get("rationale") or "").strip(),
            })
        return out

    def gc(self, *, fix: bool = False) -> dict[str, builtins.list[str]]:
        """Find orphans between the store and the memory dir.

        - `orphan_store`: store rows whose `.md` is missing on disk.
        - `orphan_disk`: `.md` files with an `id` frontmatter that the
          store doesn't know about. (Untagged `.md` files — no `id` —
          are ignored: they're user-authored content, not memories.)

        With `fix=True`, deletes orphan store rows. `.md` files are
        never deleted automatically — that's destructive and the user
        should review them first. Use `memo reindex` to absorb
        orphan disk files into the store.
        """
        orphan_store: list[str] = []
        orphan_disk: list[str] = []

        # Store-side: walk meta, check file existence (with legacy fallback).
        for r in self.store.list_recent(limit=100_000):
            if not self._resolve_existing(r["path"]).is_file():
                orphan_store.append(r["id"])
                if fix:
                    self.store.delete(r["id"])

        # Disk-side: walk memory dir, check ids in store.
        if self.cfg.memory_dir.is_dir():
            for md_path in self.cfg.memory_dir.rglob("*.md"):
                try:
                    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                md_id = post.get("id")
                if not md_id or not isinstance(md_id, str):
                    continue
                if self.store.get(md_id) is None:
                    orphan_disk.append(str(md_path.relative_to(self.cfg.memory_dir)))

        return {"orphan_store": orphan_store, "orphan_disk": orphan_disk}

    # -- internals ----------------------------------------------------------

    def _build_rel_path(self, title: str, now_iso: str) -> str:
        date = now_iso.split("T", 1)[0]
        slug = _slugify(title)[:80] or "untitled"
        # POSIX path joins. Path is relative to `cfg.memory_dir`.
        return f"{date}-{slug}.md"

    def _resolve_existing(self, rel_path: str) -> Path:
        """Resolve a DB-stored path to an absolute `Path`.

        Tries `memory_dir / rel_path` first (current layout). Falls back
        to `vault_path / rel_path` if `vault_path` is set AND the file
        actually exists there (legacy layout: paths in older DB rows
        carry a `<memory_subdir>/...` prefix relative to `vault_path`).

        Returns the new-layout path even when the file doesn't exist on
        either branch — callers that need to CREATE a file always write
        to the new layout.
        """
        new_path = self.cfg.memory_dir / rel_path
        if new_path.is_file():
            return new_path
        if self.cfg.vault_path is not None:
            legacy = self.cfg.vault_path / rel_path
            if legacy.is_file():
                return legacy
        return new_path

    def _read_body(self, rel_path: str) -> str:
        # Fast path: curated memorias live on disk under memory_dir / vault_path.
        abs_path = self._resolve_existing(rel_path)
        if abs_path.is_file():
            try:
                text = abs_path.read_text(encoding="utf-8")
                post = frontmatter.loads(text)
                return post.content
            except Exception:
                pass
        # Fallback: vault-ingest rows (e.g. `notes/01-Projects/Foo.md`,
        # `work/.../bar.md#chunk-3`) don't resolve to disk via `memory_dir`
        # because the label-prefixed path lives outside `data_dir`. The
        # body was written into the FTS table at ingest time — read it
        # from there so retrieval surfaces real snippets instead of "".
        try:
            row = self.store._conn.execute(
                "SELECT body FROM fts WHERE id = "
                "(SELECT id FROM meta WHERE path = ?)",
                (rel_path,),
            ).fetchone()
            if row and row["body"]:
                return str(row["body"])
        except Exception:
            pass
        return ""


# ── Helpers ──────────────────────────────────────────────────────────────


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


__all__ = ["AmbiguousIdError", "Memory", "MemoryRecord", "WriteRefused"]
