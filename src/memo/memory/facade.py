"""The `Memory` facade — composes the operation mixins + lazy managers.

`Memory` multiply-inherits the verbatim operation mixins (write / search /
ask / rerank / repo / maintain) and itself holds the class docstring, the
`__init__`, every lazy `@property` subsystem manager (each cached in a
`_<name>` backing field initialised to None in `__init__`), and the small
methods that don't belong to any single op group. The MRO resolves every
public + private member of the original god-class.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

from memo.analytics import AnalyticsEngine, Dashboard
from memo.collaborative import CollaborativeFilter, CollaborativeGraph, CollaborativeManager
from memo.config import Config
from memo.consolidation import AdvancedConsolidator
from memo.contextual import ContextStore, ContextualRecall
from memo.contextual_retrieval import get_or_generate_context, prepend_context
from memo.contradict import ContradictionScanner, ContradictionStore
from memo.crossref import CrossReferenceIndex, LinkSuggester
from memo.embedder import MLXEmbedder
from memo.encryption import EncryptionManager, Encryptor, KeyManager
from memo.federation import FederationConfig, FederationSearcher
from memo.graph import GraphStore
from memo.import_export import ImportExportManager
from memo.lifecycle import LifecycleManager
from memo.llm import MLXChat
from memo.memory.ask_ops import _AskOpsMixin
from memo.memory.maintain_ops import _MaintainOpsMixin
from memo.memory.record import _compose_for_embed
from memo.memory.repo_ops import _RepoOpsMixin
from memo.memory.rerank_ops import _RerankOpsMixin
from memo.memory.search_ops import _SearchOpsMixin
from memo.memory.write_ops import _WriteOpsMixin
from memo.multimodal import CrossModalSearch, MultiModalManager, MultiModalStore, UniversalEmbedder
from memo.navigation import GraphNavigator
from memo.proactive import ProactiveSuggester
from memo.saved_queries import QueryComposer, QueryStore
from memo.sharing import ShareManager, ShareStore
from memo.store import VecStore
from memo.sync import BackupManager, SyncManager
from memo.temporal import TemporalAnalyzer
from memo.versioning import VersionManager


class Memory(
    _WriteOpsMixin,
    _SearchOpsMixin,
    _AskOpsMixin,
    _RerankOpsMixin,
    _RepoOpsMixin,
    _MaintainOpsMixin,
):
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
        # History store — cheap to open (just sqlite); creating eagerly.
        # Audit failures never propagate to the caller — HistoryStore
        # swallows its own exceptions internally.
        from memo.history import HistoryStore as _HS
        self.history = _HS(cfg.history_db)
        # Helper LLM is lazy — only constructed when `auto_derive=True`
        # is requested. Cold load of Qwen2.5-3B is ~2-3s; users who
        # don't opt in pay nothing.
        self._chat: MLXChat | None = None
        # Guards lazy `_chat` construction — the FastMCP HTTP transport
        # dispatches tool calls on a worker threadpool, so two concurrent
        # requests could otherwise both build a (multi-GB) MLXChat.
        self._chat_lock = threading.Lock()
        # Knowledge-graph store. Cheap to open (just sqlite); creating
        # eagerly so graph queries never lazy-stall a CLI command.
        self.graph = GraphStore(cfg.graph_db)
        # Persistent contradiction sidecar — opened lazily so callers
        # that never scan don't pay for the extra sqlite handle.
        self._contradict_store: ContradictionStore | None = None
        # Cache-tier manager (opt-in via MEMO_CACHE_MODE) — lazy @property
        # `cache`, memoized here. Construction triggers no cold-start and the
        # backend is built lazily on first use (see CacheManager.ensure_backend).
        self._cache: Any | None = None
        # Reranker is lazy — first hybrid `search()` triggers the load
        # if `cfg.reranker_enabled`. Cold load of Qwen3-Reranker-0.6B
        # is ~1-2s; users who disable it (CI, vec-only mode) pay zero.
        self._reranker: Any | None = None
        # Guards lazy reranker construction across the FastMCP threadpool.
        self._reranker_lock = threading.Lock()
        # Self-heal probe: warn (don't crash) if the store has paths
        # that don't resolve in the current `memory_dir` layout. Common
        # after upgrading from a legacy install without running
        # `memo migrate-vault` or `memo reindex`.
        self._maybe_warn_legacy_paths()
        # Temporal analyzer for contradiction detection and timeline analysis
        self._temporal: TemporalAnalyzer | None = None
        # Memoized ContextualRecall (its ContextStore reads disk on init).
        self._contextual: ContextualRecall | None = None
        # Serialises unique-path allocation + .md creation in save() so two
        # concurrent same-title saves can't race the path probe (see write_ops).
        self._save_path_lock = threading.Lock()

    def _ensure_chat(self) -> MLXChat:
        """Construct the chat wrapper without loading model weights yet.

        Thread-safe (double-checked lock) so concurrent MCP requests share one
        wrapper instead of racing two constructions.
        """
        if self._chat is None:
            with self._chat_lock:
                if self._chat is None:
                    self._chat = MLXChat()
        return self._chat

    def _ensure_reranker(self) -> Any:
        """Lazily construct the cross-encoder reranker (thread-safe). Shared by
        both rerank paths so the construction args live in one place."""
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    from memo.reranker import MLXReranker
                    self._reranker = MLXReranker(
                        model_path=self.cfg.reranker_model,
                        revision=self.cfg.reranker_revision,
                    )
        return self._reranker

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
        # _ensure_chat() (not raw self._chat, which is None until first use) so
        # the consolidator never receives a None LLM.
        return AdvancedConsolidator(self, self._ensure_chat())

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
        """Lazy, memoized accessor for ContextualRecall.

        Memoized because ContextStore() reads JSON state from disk on init —
        re-creating it on every access (e.g. twice per recall) is wasted I/O.
        """
        if self._contextual is None:
            self._contextual = ContextualRecall(self, ContextStore(self.cfg.state_dir))
        return self._contextual

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
    def cache(self):
        """Lazy, memoized accessor for the cache-tier manager (CacheManager).

        Opt-in via MEMO_CACHE_MODE; no-ops entirely when off so memo stays a
        durable store. Memoized so the resolved backing-store client persists
        across save/search calls within one Memory instance.
        """
        if self._cache is None:
            from memo.cache import CacheManager
            self._cache = CacheManager(self)
        return self._cache

    @property
    def proactive(self) -> ProactiveSuggester:
        """Lazy accessor for ProactiveSuggester."""
        return ProactiveSuggester(self, self._ensure_chat())

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

    # -- cache tier (opt-in backing-store front) ---------------------------

    def _cache_backend(self):
        """The configured cache backend (or None when cache mode is off).
        Delegates to the manager's lazy, memoized builder."""
        return self.cache.ensure_backend()

    def _mark_dirty(self, id_: str) -> None:
        """Flag a memoria as written-locally-but-not-yet-on-backing-store
        (write-back). Metadata-only update — no re-embed."""
        from memo.cache import CACHE_DIRTY_KEY
        r = self.store.get(id_)
        if r is None:
            return
        merged = dict(r.get("extra") or {})
        merged[CACHE_DIRTY_KEY] = True
        with contextlib.suppress(Exception):
            self.update(id_, extra=merged)

