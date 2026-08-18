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
from typing import TYPE_CHECKING, Any

from memo.config import Config
from memo.consolidation import AdvancedConsolidator
from memo.contextual_retrieval import get_or_generate_context, prepend_context
from memo.contradict import CanonicalContradictionAdapter, ContradictionScanner
from memo.errors import MemoError
from memo.graph import GraphStore
from memo.graph_projection import GraphProjectionStore
from memo.llm import ChatBackend, MLXChat

if TYPE_CHECKING:
    from memo.sampling import SamplingChat
from memo.memory.ask_ops import _AskOpsMixin
from memo.memory.capabilities import OPTIONAL_CAPABILITIES
from memo.memory.chat_ask_ops import _ChatAskOpsMixin
from memo.memory.consolidate_ops import _ConsolidateOpsMixin
from memo.memory.delete_ops import _DeleteOpsMixin
from memo.memory.evidence_ops import _EvidenceOpsMixin
from memo.memory.graph_ops import _GraphOpsMixin
from memo.memory.lifecycle_ops import _LifecycleOpsMixin
from memo.memory.maintain_ops import _MaintainOpsMixin
from memo.memory.outcome_feedback_ops import _OutcomeFeedbackOpsMixin
from memo.memory.record import _compose_for_embed
from memo.memory.relation_ops import _RelationOpsMixin
from memo.memory.replay_ops import _ReplayOpsMixin
from memo.memory.repo_ops import _RepoOpsMixin
from memo.memory.rerank_ops import _RerankOpsMixin
from memo.memory.search_ops import _SearchOpsMixin
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.memory.secret_ops import _SecretOpsMixin
from memo.memory.update_ops import _UpdateOpsMixin
from memo.memory.write_ops import _WriteOpsMixin
from memo.store import VecStore
from memo.store.fact_edge_store import FactEdgeStore
from memo.temporal import TemporalAnalyzer

_log = logging.getLogger(__name__)


class Memory(
    _WriteOpsMixin,
    _UpdateOpsMixin,
    _RelationOpsMixin,
    _LifecycleOpsMixin,
    _DeleteOpsMixin,
    _GraphOpsMixin,
    _SearchOpsMixin,
    _SearchScoringMixin,
    _EvidenceOpsMixin,
    _OutcomeFeedbackOpsMixin,
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

        from memo.embedder_select import active_embedder_identity

        _expected_embedder_model = active_embedder_identity(cfg)

        _via_daemon_raw = _os.environ.get("MEMO_EMBEDDER_VIA_DAEMON")
        if _via_daemon_raw == "1":
            _use_socket = True
        elif _via_daemon_raw == "0":
            _use_socket = False
        else:
            # Auto-detect: probe the socket with a cheap ping (< 1 ms on loopback).
            # Returns None when the daemon is absent — including during the daemon's
            # own startup, where cleanup() has already removed the old socket file.
            from memo.embedder_client import daemon_is_compatible as _daemon_is_compatible
            from memo.embedder_client import ping as _ping

            _use_socket = _daemon_is_compatible(
                _ping(state_dir=cfg.state_dir),
                expected_model=_expected_embedder_model,
                expected_dims=cfg.embedder_dims,
            )

        if _use_socket:
            from memo.embedder_client import SocketEmbedder

            self.embedder: Any = SocketEmbedder(
                cfg.embedder_dims,
                expected_model=_expected_embedder_model,
                state_dir=cfg.state_dir,
            )
        else:
            # Resolve the query-cache size from the flags registry (default 256)
            # and pass it explicitly — the embedder can't import memo.flags
            # itself, and a raw env read there defaults to 0/off, silently
            # disabling the cache on every Memory-backed path (recall hook, CLI).
            # make_embedder picks MLX (Apple Silicon) or the CPU sentence-
            # transformers backend (Linux/Ubuntu). See embedder_select.
            from memo.embedder_select import make_embedder
            from memo.flags import flag_int as _flag_int

            self.embedder = make_embedder(cfg, cache_size=_flag_int("MEMO_QUERY_CACHE_SIZE"))
        from memo.flags import flag_str as _flag_str

        self.store = VecStore(
            cfg.db_path,
            dims=cfg.embedder_dims,
            embedder_model=_expected_embedder_model,
            vec_quant=_flag_str("MEMO_VEC_QUANTIZE"),
        )
        # History store — cheap to open (just sqlite); creating eagerly.
        # Audit failures never propagate to the caller — HistoryStore
        # swallows its own exceptions internally.
        from memo.history import HistoryStore as _HS

        self.history = _HS(cfg.history_db, device_id=cfg.device_id)
        # Native operational continuity and write policy.  These are lightweight
        # (journal paths + lazy snapshot reads), always available, and replace
        # the historical cross-runtime boundary.
        from memo.operational import OperationalStore as _OperationalStore
        from memo.write_policy import WritePolicyEngine as _WritePolicyEngine

        self.operational = _OperationalStore(cfg.state_dir, device_id=cfg.device_id)
        self.write_policy = _WritePolicyEngine(self.operational)
        # Helper LLM is lazy — only constructed when a helper-backed path
        # is requested. Users who don't opt in pay nothing.
        self._chat: MLXChat | None = None
        # SamplingChat router (MEMO_SAMPLING_SYNTH_ENABLED) — cached like
        # `_chat`; safe because it holds no request state (contextvar does).
        self._sampling_chat: SamplingChat | None = None
        # Guards lazy `_chat` construction — the FastMCP HTTP transport
        # dispatches tool calls on a worker threadpool, so two concurrent
        # requests could otherwise both build a (multi-GB) MLXChat.
        self._chat_lock = threading.RLock()
        # Knowledge-graph store. Cheap to open (just sqlite); creating
        # eagerly so graph queries never lazy-stall a CLI command.
        self.graph = GraphStore(cfg.graph_db, projection_factory=GraphProjectionStore)
        self.fact_edges = FactEdgeStore(cfg.fact_edges_db)
        # Compatibility projection over the canonical relation ledger. The old
        # contradictions.db is opened only by the one-way importer.
        self._contradict_store: CanonicalContradictionAdapter | None = None
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
        # Re-entrant because maintenance operations hold the shared data lock
        # across a corpus transaction and may call helpers that take it again.
        self._save_path_lock = threading.RLock()
        self._data_lock_depth = 0
        # Write-generation counter; bumped by save/update/delete to bust the
        # RAG cache's corpus-version memo (see _corpus_version in ask_ops).
        self._write_gen = 0

    def _build_mlx_chat(self) -> MLXChat:
        """Construct the MLX chat wrapper without loading model weights yet.

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
                    self._chat = MLXChat(
                        model_revisions={
                            self.cfg.llm_model: self.cfg.llm_revision,
                            self.cfg.helper_model: self.cfg.helper_revision,
                        }
                    )
        return self._chat

    def _ensure_chat(self) -> ChatBackend:
        """Resolve the chat backend for synthesis.

        With MEMO_SAMPLING_SYNTH_ENABLED on, returns a cached SamplingChat
        router that consults the per-request sampling contextvar at every
        call (client model inside MCP scope, MLX everywhere else). The MLX
        availability check moves to first actual use in that mode — a host
        without MLX can still answer via a sampling-capable client.
        """
        from memo.flags import flag_bool

        if flag_bool("MEMO_SAMPLING_SYNTH_ENABLED"):
            if self._sampling_chat is None:
                from memo.sampling import SamplingChat

                with self._chat_lock:
                    if self._sampling_chat is None:
                        self._sampling_chat = SamplingChat(self._build_mlx_chat)
            return self._sampling_chat
        return self._build_mlx_chat()

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
    def contradict_store(self) -> CanonicalContradictionAdapter:
        """Old contradiction API projected from canonical relation rows."""
        if self._contradict_store is None:
            with self._chat_lock:
                if self._contradict_store is None:
                    try:
                        self._contradict_store = CanonicalContradictionAdapter(self)
                        self.import_legacy_contradictions()
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
    def federation(self) -> Any:
        """Lazy accessor for signed, ACL-aware federation bundles."""
        return self.capability("federation")

    @property
    def project(self) -> str:
        """Detect project from cwd (5-case algorithm)."""
        from memo.server_session_patterns import _project_from_cwd

        return _project_from_cwd()

    def open_loops(self, limit: int = 5, *, days: int = 7) -> list[tuple[str, str]]:
        """Recently-updated non-reference memories as `(id, title)` pairs.

        Same "open loops" query as `briefing.py`'s "Open loops" section —
        durable memories touched in the last `days` days, most-recent first.
        """
        from datetime import UTC, datetime, timedelta

        cutoff = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
        rows = self.store.list_recent(
            limit=limit,
            exclude_types={"reference", "secret"},
            updated_since=cutoff,
        )
        return [(r.get("id") or "", r.get("title") or "—") for r in rows]

    def superseded_pairs(self, *, limit: int = 50) -> list[tuple[str, str, str]]:
        """Archived memories with a live successor, as `(stale_id, superseding_id, title)`.

        Scans `cfg.memory_dir / "inactive" / *.md` for files stamped by
        `lifecycle.py`'s `archive_memory(superseded_by=...)` — the WINNING
        memory id lives in frontmatter `extra.superseded_by`. Dream-only (a
        disk scan is fine there; never called from the recall path). Guarded
        per-file: a malformed archive entry is skipped, not fatal.
        """
        inactive_dir = self.cfg.memory_dir / "inactive"
        if not inactive_dir.is_dir():
            return []
        import frontmatter

        out: list[tuple[str, str, str]] = []
        for path in sorted(inactive_dir.glob("*.md")):
            if len(out) >= limit:
                break
            try:
                post = frontmatter.loads(path.read_text(encoding="utf-8"))
                extra = post.get("extra")
                superseded_by = extra.get("superseded_by") if isinstance(extra, dict) else None
                if not superseded_by:
                    continue
                stale_id = str(post.get("id") or "")
                if not stale_id:
                    continue
                title = str(post.get("title") or "—")
                out.append((stale_id, str(superseded_by), title))
            except Exception:  # noqa: S112 — a malformed archive entry (bad YAML
                # frontmatter, unreadable file, etc.) must not sink reliability
                # nudges for every OTHER valid pair (I3 review fix: frontmatter.loads
                # can raise yaml.YAMLError, which (OSError, ValueError) missed).
                continue
        return out

    def low_confidence_ids(self, *, threshold: float = 0.4, limit: int = 50) -> list[str]:
        """Memory ids whose `memory_health.confidence` is below `threshold`.

        Thin read over the `memory_health` signal table (`store/schema.py`),
        lowest confidence first. Feeds the proactive health detector's
        "worth reviewing" nudge. Joins `meta` and excludes soft-deleted rows
        (`MEMO_SOFT_DELETE` leaves `memory_health` rows around for up to 90
        days after `meta.deleted_at` is stamped — see `dead_memory_ids` for
        the same deleted-filter idiom) so a deleted memory never gets cited.
        """
        rows = self.store._conn.execute(
            """
            SELECT mh.id
              FROM memory_health mh
              JOIN meta m ON m.id = mh.id
             WHERE mh.confidence < ?
               AND (m.deleted_at IS NULL OR m.deleted_at = '')
             ORDER BY mh.confidence ASC
             LIMIT ?
            """,
            (threshold, limit),
        ).fetchall()
        return [str(r["id"]) for r in rows]

    def dead_memory_ids(self, *, limit: int = 50) -> list[str]:
        """Durable memory ids with zero recorded access (`access` table).

        Never surfaced since creation — candidates the proactive ROI
        detector flags for pruning. Reference/secret types are excluded
        (only `tiers.DURABLE_TYPES` counts as a "dead" memory).
        """
        from memo.tiers import DURABLE_TYPES

        placeholders = ",".join("?" for _ in DURABLE_TYPES)
        rows = self.store._conn.execute(
            f"""
            SELECT m.id
              FROM meta m
              LEFT JOIN access a ON a.id = m.id
             WHERE COALESCE(a.access_count, 0) = 0
               AND m.type IN ({placeholders})
               AND (m.deleted_at IS NULL OR m.deleted_at = '')
             ORDER BY m.created ASC
             LIMIT ?
            """,  # noqa: S608 — placeholders are '?' marks, not interpolated values
            (*DURABLE_TYPES, limit),
        ).fetchall()
        return [str(r["id"]) for r in rows]

    def recurring_pattern_pairs(
        self, *, limit: int = 5, min_count: int = 2
    ) -> list[tuple[str, str]]:
        """Recurring recall-log prompts that already have a matching memory.

        Returns `(memo_id, pattern_text)` pairs for the proactive déjà-vu
        detector. Mines your most-repeated PAST prompts from the recall log
        (`dashboard_logs.read_recall_log`) — true déjà-vu would compare
        against the CURRENT recall's live hits, but that context isn't
        available outside a recall call, so this re-runs `search()` (the
        same retrieval machinery any query uses) to find a real citable
        hit. A pattern with no matching memory is dropped; this never
        fabricates a citation.
        """
        from collections import Counter

        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(self.cfg.state_dir, limit=200)
        counts: Counter[str] = Counter()
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q:
                counts[q] += 1

        # Cap candidate queries examined — sparse matches would otherwise walk
        # the full most_common() list, firing one search() per candidate
        # (dream-only path, but still not unbounded).
        max_candidates = max(limit * 4, 20)
        out: list[tuple[str, str]] = []
        # One pair per memory: the déjà-vu nudge id is derived from the memo id,
        # so two patterns resolving to the same top hit would collide on the
        # candidates primary key.
        seen: set[str] = set()
        for candidates, (query, count) in enumerate(counts.most_common()):
            if len(out) >= limit or candidates >= max_candidates:
                break
            if count < min_count:
                break
            hits = self.search(query, limit=1, disable_reranker=True)
            if hits and hits[0].id not in seen:
                seen.add(hits[0].id)
                out.append((hits[0].id, query))
        return out

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
        with contextlib.suppress(Exception):
            self.fact_edges.close()
        if self._contradict_store is not None:
            with contextlib.suppress(Exception):
                self._contradict_store.close()
        if self._hype_store is not None:
            with contextlib.suppress(Exception):
                self._hype_store.close()
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
