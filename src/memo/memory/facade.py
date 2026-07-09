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
import logging
import threading
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.consolidation import AdvancedConsolidator
from memo.contextual_retrieval import get_or_generate_context, prepend_context
from memo.contradict import ContradictionScanner, ContradictionStore
from memo.errors import MemoError
from memo.graph import GraphStore
from memo.llm import MLXChat
from memo.memory.ask_ops import _AskOpsMixin
from memo.memory.capabilities import OPTIONAL_CAPABILITIES
from memo.memory.chat_ask_ops import _ChatAskOpsMixin
from memo.memory.consolidate_ops import _ConsolidateOpsMixin
from memo.memory.delete_ops import _DeleteOpsMixin
from memo.memory.maintain_ops import _MaintainOpsMixin
from memo.memory.record import _compose_for_embed
from memo.memory.replay_ops import _ReplayOpsMixin
from memo.memory.repo_ops import _RepoOpsMixin
from memo.memory.rerank_ops import _RerankOpsMixin
from memo.memory.search_ops import _SearchOpsMixin
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.memory.secret_ops import _SecretOpsMixin
from memo.memory.update_ops import _UpdateOpsMixin
from memo.memory.write_ops import _WriteOpsMixin
from memo.store import VecStore
from memo.temporal import TemporalAnalyzer

_log = logging.getLogger(__name__)


class Memory(
    _WriteOpsMixin,
    _UpdateOpsMixin,
    _DeleteOpsMixin,
    _SearchOpsMixin,
    _SearchScoringMixin,
    _AskOpsMixin,
    _ChatAskOpsMixin,
    _RerankOpsMixin,
    _RepoOpsMixin,
    _MaintainOpsMixin,
    _ConsolidateOpsMixin,
    _ReplayOpsMixin,
    _SecretOpsMixin,
):
    """High-level memory API. Construct once per process; methods are
    thread-safe (delegate to store/embedder which both serialise their
    critical sections).

    Example:

        cfg = Config.from_env()
        cfg.ensure_dirs()
        mem = Memory(cfg)

        rec = mem.save(
            content="**What**: Migrated obsidian-rag to MLX. ...",
            title="MLX migration formal wrap-up",
            type_="decision",
            tags=["mlx", "obsidian-rag", "migration"],
        )

        hits = mem.search("how I migrated to mlx", limit=5)
        for h in hits:
            print(f"{h.score:.3f} · {h.title}")
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self._close_lock = threading.Lock()
        self._closed = False
        # Prefer the recall-daemon socket over a second in-process MLX copy.
        # Priority:
        #   1. MEMO_EMBEDDER_VIA_DAEMON=1 → always use socket (explicit opt-in)
        #   2. MEMO_EMBEDDER_VIA_DAEMON=0 → always use in-process (explicit opt-out)
        #   3. unset → auto-detect: use socket if the daemon is already alive,
        #      otherwise load MLX in-process (zero configuration required).
        #
        # Auto-detect is safe for the recall-daemon itself because cleanup() removes
        # the old socket before Memory.__init__ runs, so ping() returns None and
        # the daemon loads MLX directly (as it must).
        # os.environ.get gives the raw value (None if unset) which is exactly the
        # three-way check we need: explicit "1", explicit "0", or auto-detect.
        # flag_bool() can't do this because it collapses unset → False (the default).
        import os as _os

        _via_daemon_raw = _os.environ.get("MEMO_EMBEDDER_VIA_DAEMON")
        if _via_daemon_raw == "1":
            _use_socket = True
        elif _via_daemon_raw == "0":
            _use_socket = False
        else:
            # Auto-detect: probe the socket with a cheap ping (< 1 ms on loopback).
            # Returns None when the daemon is absent — including during the daemon's
            # own startup, where cleanup() has already removed the old socket file.
            from memo.embedder_client import ping as _ping

            _use_socket = _ping(state_dir=cfg.state_dir) is not None

        if _use_socket:
            from memo.embedder_client import SocketEmbedder

            self.embedder: Any = SocketEmbedder(
                cfg.embedder_dims,
                state_dir=cfg.state_dir,
            )
        else:
            # Resolve the query-cache size from the flags registry (default 256)
            # and pass it explicitly — the embedder can't import memo.flags
            # itself, and a raw env read there defaults to 0/off, silently
            # disabling the cache on every Memory-backed path (recall hook, CLI).
            # make_embedder picks MLX (Apple Silicon) or the CPU sentence-
            # transformers backend (Linux/Ubuntu, Intel mac). See embedder_select.
            from memo.embedder_select import make_embedder
            from memo.flags import flag_int as _flag_int

            self.embedder = make_embedder(cfg, cache_size=_flag_int("MEMO_QUERY_CACHE_SIZE"))
        self.store = VecStore(
            cfg.db_path, dims=cfg.embedder_dims, embedder_model=cfg.embedder_model
        )
        # History store — cheap to open (just sqlite); creating eagerly.
        # Audit failures never propagate to the caller — HistoryStore
        # swallows its own exceptions internally.
        from memo.history import HistoryStore as _HS

        self.history = _HS(cfg.history_db, device_id=cfg.device_id)
        # Helper LLM is lazy — only constructed when a helper-backed path
        # is requested. Users who don't opt in pay nothing.
        self._chat: MLXChat | None = None
        # Guards lazy `_chat` construction — the FastMCP HTTP transport
        # dispatches tool calls on a worker threadpool, so two concurrent
        # requests could otherwise both build a (multi-GB) MLXChat.
        self._chat_lock = threading.RLock()
        # Knowledge-graph store. Cheap to open (just sqlite); creating
        # eagerly so graph queries never lazy-stall a CLI command.
        self.graph = GraphStore(cfg.graph_db)
        # Persistent contradiction sidecar — opened lazily so callers
        # that never scan don't pay for the extra sqlite handle.
        self._contradict_store: ContradictionStore | None = None
        self._contradict_scanner: ContradictionScanner | None = None
        self._consolidator: AdvancedConsolidator | None = None
        # Cache-tier manager (opt-in via MEMO_CACHE_MODE) — lazy @property
        # `cache`, memoized here. Construction triggers no cold-start and the
        # backend is built lazily on first use (see CacheManager.ensure_backend).
        self._cache: Any | None = None
        self._capabilities: dict[str, Any] = {}
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
        # Serialises unique-path allocation + .md creation in save() so two
        # concurrent same-title saves can't race the path probe (see write_ops).
        self._save_path_lock = threading.Lock()
        # Write-generation counter; bumped by save/update/delete to bust the
        # RAG cache's corpus-version memo (see _corpus_version in ask_ops).
        self._write_gen = 0

    def _ensure_chat(self) -> MLXChat:
        """Construct the chat wrapper without loading model weights yet.

        Thread-safe (double-checked lock) so concurrent MCP requests share one
        wrapper instead of racing two constructions.
        """
        if self._chat is None:
            from memo.platform_detect import mlx_available

            if not mlx_available():
                raise MemoError(
                    "LLM features (ask / synthesize / dream) require the MLX "
                    "runtime (Apple Silicon). They are unavailable on this host; "
                    "search, recall, and save work without them. See docs/ubuntu.md."
                )
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

                    model_path = str(self.cfg.reranker_model or "").strip()
                    local_candidate = Path(model_path).expanduser()
                    if (
                        model_path
                        and (local_candidate.is_absolute() or model_path.startswith((".", "~")))
                        and not local_candidate.exists()
                    ):
                        raise FileNotFoundError(
                            f"reranker model path does not exist: {local_candidate}. "
                            "Set MEMO_RERANKER_MODEL to an existing local model path "
                            "or a Hugging Face model id."
                        )
                    try:
                        _log.info(
                            "Loading reranker model=%s revision=%s",
                            self.cfg.reranker_model,
                            self.cfg.reranker_revision or "default",
                        )
                        self._reranker = MLXReranker(
                            model_path=self.cfg.reranker_model,
                            revision=self.cfg.reranker_revision,
                        )
                        _log.info("Reranker loaded successfully")
                    except Exception as exc:
                        _log.error(
                            "Failed to load reranker model=%s revision=%s: %s",
                            self.cfg.reranker_model,
                            self.cfg.reranker_revision,
                            exc,
                        )
                        raise
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
            with self._chat_lock:
                if self._temporal is None:
                    chat = self._ensure_chat()
                    self._temporal = TemporalAnalyzer(self, chat)
        return self._temporal

    @property
    def consolidator(self) -> AdvancedConsolidator:
        """Lazy accessor for AdvancedConsolidator."""
        # _ensure_chat() (not raw self._chat, which is None until first use) so
        # the consolidator never receives a None LLM.
        if self._consolidator is None:
            with self._chat_lock:
                if self._consolidator is None:
                    self._consolidator = AdvancedConsolidator(self, self._ensure_chat())
        return self._consolidator

    @property
    def contradict_store(self) -> ContradictionStore:
        """Lazy accessor for the persistent contradictions sidecar."""
        if self._contradict_store is None:
            with self._chat_lock:
                if self._contradict_store is None:
                    try:
                        self._contradict_store = ContradictionStore(self.cfg.contradictions_db)
                    except Exception as exc:
                        _log.warning("contradict_store init failed: %s", exc)
                        raise
        return self._contradict_store

    @property
    def contradict_scanner(self) -> ContradictionScanner:
        """Lazy accessor for the corpus-wide contradiction scanner."""
        if self._contradict_scanner is None:
            with self._chat_lock:
                if self._contradict_scanner is None:
                    self._contradict_scanner = ContradictionScanner(
                        self, self.contradict_store, self.temporal
                    )
        return self._contradict_scanner

    @property
    def navigator(self) -> Any:
        """Lazy accessor for GraphNavigator."""
        from memo.navigation import GraphNavigator

        return GraphNavigator(self.graph)

    @property
    def contextual(self) -> Any:
        """Lazy, memoized accessor for ContextualRecall.

        Memoized because ContextStore() reads JSON state from disk on init —
        re-creating it on every access (e.g. twice per recall) is wasted I/O.
        """
        return self.capability("contextual")

    @property
    def crossref(self) -> Any:
        """Lazy accessor for CrossReferenceIndex."""
        return self.capability("crossref")

    @property
    def link_suggester(self) -> Any:
        """Lazy accessor for LinkSuggester."""
        return self.capability("link_suggester")

    @property
    def lifecycle(self) -> Any:
        """Lazy accessor for LifecycleManager."""
        return self.capability("lifecycle")

    @property
    def cache(self):
        """Lazy, memoized accessor for the cache-tier manager (CacheManager).

        Opt-in via MEMO_CACHE_MODE; no-ops entirely when off so memo stays a
        durable store. Memoized so the resolved backing-store client persists
        across save/search calls within one Memory instance.
        """
        if self._cache is None:
            with self._chat_lock:
                if self._cache is None:
                    from memo.cache import CacheManager

                    self._cache = CacheManager(self)
        return self._cache

    @property
    def versioning(self) -> Any:
        """Lazy accessor for VersionManager."""
        return self.capability("versioning")

    @property
    def query_composer(self) -> Any:
        """Lazy accessor for QueryComposer."""
        return self.capability("query_composer")

    @property
    def backup(self) -> Any:
        """Lazy accessor for BackupManager."""
        return self.capability("backup")

    @property
    def sync(self) -> Any:
        """Lazy accessor for SyncManager."""
        return self.capability("sync")

    @property
    def analytics(self) -> Any:
        """Lazy accessor for AnalyticsEngine."""
        return self.capability("analytics")

    @property
    def dashboard(self) -> Any:
        """Lazy accessor for Dashboard."""
        return self.capability("dashboard")

    @property
    def import_export(self) -> Any:
        """Lazy accessor for ImportExportManager."""
        return self.capability("import_export")

    @property
    def collaborative(self) -> Any:
        """Lazy accessor for CollaborativeManager."""
        return self.capability("collaborative")

    @property
    def project(self) -> str:
        """Detect project from cwd (5-case algorithm)."""
        from memo.server_session_patterns import _project_from_cwd

        return _project_from_cwd()

    def capability(self, name: str) -> Any:
        """Build and memoize an optional subsystem by registry name."""
        if name not in OPTIONAL_CAPABILITIES:
            raise KeyError(f"unknown Memory capability: {name}")
        if self._closed:
            raise RuntimeError(f"Memory is closed — cannot access capability '{name}'")
        if name not in self._capabilities:
            self._capabilities[name] = OPTIONAL_CAPABILITIES[name](self)
        return self._capabilities[name]

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
            missing = sum(1 for r in sample if not self._resolve_existing(r["path"]).is_file())
            if missing == len(sample):
                import sys

                from memo.flags import flag_bool

                # Suppressible — TUI sets MEMO_SUPPRESS_LEGACY_WARN=1 because
                # the message is moot while in alt-screen mode.
                if flag_bool("MEMO_SUPPRESS_LEGACY_WARN"):
                    return
                print(
                    f"[memo] heads-up: your index points to old paths "
                    f"(data_dir={self.cfg.data_dir}). Run `memo reindex` "
                    f"to re-embed, or `memo migrate-vault <new-dir>` if "
                    f"you moved the vault. This is NOT an error — just a "
                    f"notice that disk and index are out of sync.",
                    file=sys.stderr,
                )
        except Exception:  # noqa: S110
            # Probe failures must never block startup — they're advisory.
            pass

    # -- cache tier (opt-in backing-store front) ---------------------------

    def _cache_backend(self):
        """The configured cache backend (or None when cache mode is off).
        Delegates to the manager's lazy, memoized builder."""
        return self.cache.ensure_backend()

    def close(self) -> None:
        """Close all open SQLite connections held by this Memory instance.

        Call in teardown (e.g. pytest fixture finalizers) to release file
        descriptors.  Each sub-store closes only the calling-thread's
        connection; other threads' connections are released when those
        threads end.  Idempotent — safe to call multiple times.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        with contextlib.suppress(Exception):
            self.store.close()
        with contextlib.suppress(Exception):
            self.history.close()
        with contextlib.suppress(Exception):
            self.graph.close()
        if self._contradict_store is not None:
            with contextlib.suppress(Exception):
                self._contradict_store.close()
        for capability in list(self._capabilities.values()):
            with contextlib.suppress(Exception):
                close = getattr(capability, "close", None)
                if close is not None:
                    close()
        with contextlib.suppress(Exception):
            if hasattr(self.embedder, "unload"):
                self.embedder.unload()
        with contextlib.suppress(Exception):
            if self._chat is not None and hasattr(self._chat, "close"):
                self._chat.close()
        with contextlib.suppress(Exception):
            if self._reranker is not None and hasattr(self._reranker, "close"):
                self._reranker.close()
        with contextlib.suppress(Exception):
            if self._temporal is not None and hasattr(self._temporal, "close"):
                self._temporal.close()
        self._capabilities.clear()

    def __del__(self) -> None:  # pragma: no cover - GC safety net
        with contextlib.suppress(Exception):
            self.close()

    def _mark_dirty(self, id_: str) -> None:
        """Flag a memory as written-locally-but-not-yet-on-backing-store
        (write-back). Metadata-only update — no re-embed."""
        from memo.cache import CACHE_DIRTY_KEY

        r = self.store.get(id_)
        if r is None:
            return
        merged = dict(r.get("extra") or {})
        merged[CACHE_DIRTY_KEY] = True
        try:
            self.update(id_, extra=merged)
        except Exception as exc:
            _log.debug("_mark_dirty(%s): failed to set dirty bit — %s", id_[:8], exc)
