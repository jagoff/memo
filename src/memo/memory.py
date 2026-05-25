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
import hashlib
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

from memo.agent import AutonomousAgent
from memo.analytics import AnalyticsEngine, Dashboard
from memo.cognitive import (
    CognitiveManager,
    CognitiveStateTracker,
    ContextAwareRetrieval,
    ProactiveGuidance,
)
from memo.collaborative import CollaborativeFilter, CollaborativeGraph, CollaborativeManager
from memo.config import Config
from memo.consolidation import AdvancedConsolidator
from memo.contextual import ContextStore, ContextualRecall
from memo.contradict import ContradictionScanner, ContradictionStore
from memo.crossref import CrossReferenceIndex, LinkSuggester
from memo.embedder import MLXEmbedder, assert_valid_embedding
from memo.encryption import EncryptionManager, Encryptor, KeyManager
from memo.federation import FederationConfig, FederationSearcher
from memo.graph import GraphStore
from memo.import_export import ImportExportManager
from memo.lifecycle import LifecycleManager
from memo.llm import MLXChat
from memo.multimodal import CrossModalSearch, MultiModalManager, MultiModalStore, UniversalEmbedder
from memo.navigation import GraphNavigator
from memo.proactive import ProactiveSuggester
from memo.queries import QueryComposer, QueryStore
from memo.sharing import ShareManager, ShareStore
from memo.store import VecStore
from memo.sync import BackupManager, SyncManager
from memo.temporal import TemporalAnalyzer
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
Synthesise a concise answer in the same language as the question (Spanish
rioplatense if the question is in Spanish). Rules:

- Cite sources INLINE with `[id-prefix]` after each claim, e.g.
  "Decidí migrar a MLX [d61fe730] para reducir dependencias [4e0b2e6]".
  For repo evidence, cite the full repo label you received.
- Use only information from the provided snippets. If the answer is not
  present, say "no encuentro la respuesta en las memorias guardadas"
  and stop.
- Stay terse. 2-5 sentences unless the question explicitly asks for a
  long form. Do not pad with disclaimers, restatements, or apologies.
- No bulleting unless the question asks for a list; prose preferred.
- Do not invent IDs. Only cite `[id-prefix]` values that appear in the
  snippets you were given."""


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

_VALID_TYPES = frozenset(
    {"decision", "fact", "bug", "feedback", "preference", "note", "manual"}
)

SYNAPSE_BACKEND_NATIVE_SCHEMA = "synapse.backend_native.v1"
NATIVE_BACKEND_PROTOCOL_VERSION = "backend_native.v1"
MEMO_BACKEND_NAME = "memo"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stable_content_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AmbiguousIdError(ValueError):
    """Raised when an id prefix matches more than one record. Carries
    the candidate matches so the caller can surface them in an error."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        super().__init__(
            f"Ambiguous id prefix {prefix!r}: {len(matches)} matches "
            f"({', '.join(m[:8] for m in matches[:5])}...)",
        )
        self.prefix = prefix
        self.matches = matches


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

    @property
    def temporal(self) -> TemporalAnalyzer:
        """Lazy accessor for TemporalAnalyzer."""
        if self._temporal is None:
            self._temporal = TemporalAnalyzer(self, self._chat)
        return self._temporal

    @property
    def consolidator(self) -> AdvancedConsolidator:
        """Lazy accessor for AdvancedConsolidator."""
        return AdvancedConsolidator(self, self._chat)

    @property
    def contradict_store(self) -> ContradictionStore:
        """Lazy accessor for the persistent contradictions sidecar."""
        if self._contradict_store is None:
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
    def agent(self) -> AutonomousAgent:
        """Lazy accessor for AutonomousAgent (THE GAMECHANGER)."""
        return AutonomousAgent(self, self._ensure_chat())

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

    @property
    def cognitive(self) -> CognitiveManager:
        """Lazy accessor for CognitiveManager."""
        tracker = CognitiveStateTracker(self.cfg.state_dir)
        retrieval = ContextAwareRetrieval(tracker)
        guidance = ProactiveGuidance(tracker)
        return CognitiveManager(tracker, retrieval, guidance)

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
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")
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
            **({"extra": extra_for_store} if extra_for_store else {}),
        )
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
            )
            return MemoryRecord(
                id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
                created=now_iso, updated=now_iso, body=content, extra=extra_for_store,
            )

        # Embed `title + body`: the title carries the highest-density
        # signal for retrieval ("Astor — Informe TO" is a much better
        # match for a query like "informe terapia ocupacional astor"
        # than the body's clinical paragraphs alone). Prepending also
        # protects the title from head-truncation when the body is
        # long — see embedder.py for the truncation rationale.
        embedding = self.embedder.embed([_compose_for_embed(title, content)])[0]
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
        )

        return MemoryRecord(
            id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
            created=now_iso, updated=now_iso, body=content, extra=extra or {},
        )

    # -- search -------------------------------------------------------------

    def search(
        self, query: str, *, limit: int | None = None, type_: str | None = None,
        mode: str = "hybrid",
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
        """
        if not query or not query.strip():
            return []
        limit = limit or self.cfg.search_default_limit

        if mode == "bm25":
            rows = self.store.search_bm25(query, limit=limit, type_=type_)
        elif mode == "vec":
            # Asymmetric retrieval: queries are embedded WITH the
            # instruction prefix; documents are embedded RAW (in
            # `save()` / `update()`). See `_QUERY_INSTRUCTION_PREFIX`
            # in `embedder.py` for the why.
            emb = self.embedder.embed_query(query)
            rows = self.store.search(emb, limit=limit, type_=type_)
        else:
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = self.cfg.rerank_input_k if self.cfg.reranker_enabled else limit
            k_each = max(input_k * 2, 30)
            emb = self.embedder.embed_query(query)
            vec_hits = self.store.search(emb, limit=k_each, type_=type_)
            bm_hits = self.store.search_bm25(query, limit=k_each, type_=type_)
            rows = _rrf_fuse(vec_hits, bm_hits, limit=input_k)
        out: list[MemoryRecord] = []
        for r in rows:
            body = self._read_body(r["path"])
            out.append(
                MemoryRecord(
                    id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                    tags=r["tags"], created=r["created"], updated=r["updated"],
                    body=body, extra=r.get("extra") or {}, score=r.get("score"),
                ),
            )
        # Cross-encoder rerank on hybrid mode only. Skipped for vec/bm25
        # since those callers explicitly opted out of fusion entirely;
        # adding rerank to single-mode searches would surprise users
        # benchmarking the raw bi-encoder or BM25 surfaces.
        if mode == "hybrid" and self.cfg.reranker_enabled and out:
            out = self._rerank(query, out, top_n=limit)
        # Optional recency decay: blend a freshness bonus into the score so
        # older memories don't crowd out recent ones. Disabled by default
        # (halflife_days=0). Enable via MEMO_SEARCH_DECAY_HALFLIFE (days).
        halflife_days = float(os.environ.get("MEMO_SEARCH_DECAY_HALFLIFE", "0") or 0)
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
            reranker = MLXReranker(model_path=self.cfg.reranker_model)
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
            _log.warning("reranker failed, falling back to RRF order: %s", exc)
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
          - on synthesis error: a single `done` with
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
        snippet_chars: int, include_repos: bool,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Retrieval half of ask()/ask_stream().

        Returns (normalized_question, sources, user_msg). When no hits
        found, returns (question, [], "") — caller must short-circuit.
        """
        if not question or not question.strip():
            return question, [], ""
        _MAX_QUESTION_CHARS = 4000
        if len(question) > _MAX_QUESTION_CHARS:
            _log.warning(
                "ask: question truncated from %d to %d chars",
                len(question), _MAX_QUESTION_CHARS,
            )
            question = question[:_MAX_QUESTION_CHARS]
        hits = self.search(question, limit=k, type_=type_, mode="hybrid")
        repo_hits = []
        if include_repos and self.store.list_repo_sources(limit=1):
            with contextlib.suppress(Exception):
                repo_hits = self.repo_search(question, limit=k, mode="hybrid")
        if not hits and not repo_hits:
            return question, [], ""

        snippet_lines: list[str] = []
        sources: list[dict[str, Any]] = []
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
        for h in repo_hits:
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
        return question, sources, user_msg

    def ask(
        self, question: str, *, k: int = 5, type_: str | None = None,
        snippet_chars: int = 800, include_repos: bool = True,
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
        norm_question, sources, user_msg = self._build_ask_context(
            question, k=k, type_=type_,
            snippet_chars=snippet_chars, include_repos=include_repos,
        )
        if not sources:
            return {
                "question": norm_question,
                "answer": "no encuentro la respuesta en las memorias guardadas",
                "sources": [],
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
        snippet_chars: int = 800, include_repos: bool = True,
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
        _, sources, user_msg = self._build_ask_context(
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
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each."""
        rows = self.store.list_recent(limit=limit, type_=type_)
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
            **({"extra": new_extra} if new_extra else {}),
        )
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type
        # changes still skip the embedder.
        if body_changed or title_changed:
            embedding = self.embedder.embed([_compose_for_embed(new_title, new_body)])[0]
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
        if delta:
            self.history.log_update(
                ts=now_iso, record_id=id_, title=new_title, type_=new_type,
                delta=delta,
            )

        return MemoryRecord(
            id=id_, path=r["path"], title=new_title, type=new_type,
            tags=new_tags, created=r["created"], updated=now_iso,
            body=new_body, extra=new_extra,
        )

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
        return existed

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
            md_id = post.get("id")
            if not md_id or not isinstance(md_id, str):
                skipped += 1
                continue
            body = post.content or ""
            new_hash = _sha256_short(body)
            existing = self.store.get(md_id)
            # Path relative to memory_dir — paths in the store no longer
            # carry the legacy `<vault>/<memory_subdir>/...` prefix.
            rel = str(md_path.relative_to(self.cfg.memory_dir))

            title = (post.get("title") or _derive_title(body) or "untitled").strip()
            type_ = post.get("type") or "note"
            if type_ not in _VALID_TYPES:
                _log.warning("reindex: invalid type %r in %s, coercing to 'note'", type_, md_path.name)
                type_ = "note"
            tags = _normalise_tags(list(post.get("tags") or []))
            created = post.get("created") or _now_iso()
            updated = post.get("updated") or created
            extra = post.get("extra") or {}

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
                emb = self.embedder.embed([_compose_for_embed(title, body)])[0]
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
                emb = self.embedder.embed([_compose_for_embed(title, body)])[0]
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
        return {"checked": checked, "reindexed": reindexed, "added": added, "skipped": skipped}

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
        abs_path = self._resolve_existing(rel_path)
        if not abs_path.is_file():
            return ""
        try:
            text = abs_path.read_text(encoding="utf-8")
            post = frontmatter.loads(text)
            return post.content
        except Exception:
            return ""


# ── Helpers ──────────────────────────────────────────────────────────────


def _now_iso() -> str:
    # Millisecond precision so within-second event ordering survives
    # — required for time-machine reconstruction to distinguish
    # save/update/delete pairs that happen rapidly. Tooling that
    # parsed second-truncated strings still parses these.
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="milliseconds")


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


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


__all__ = ["AmbiguousIdError", "Memory", "MemoryRecord"]
