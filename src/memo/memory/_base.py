"""Typed contract base for the `Memory` mixins.

`_MemoryBase` declares every attribute and cross-mixin method that a mixin
references but does not itself define, so each mixin module type-checks in
isolation. The real implementations (instance attributes set in
`Memory.__init__`, the lazy `@property` managers, and the methods owned by
sibling mixins) win at runtime via the `Memory` MRO — these annotations and
ellipsis-body stubs are never executed.
"""

from __future__ import annotations

from typing import Any


class _MemoryBase:
    # -- instance data attributes (set in Memory.__init__) -----------------
    cfg: Any
    embedder: Any
    store: Any
    history: Any
    graph: Any
    _chat: Any
    _cache: Any
    _reranker: Any
    _temporal: Any
    _contradict_store: Any

    # -- lazy @property managers (defined on the facade) -------------------
    temporal: Any
    consolidator: Any
    contradict_store: Any
    contradict_scanner: Any
    navigator: Any
    contextual: Any
    crossref: Any
    link_suggester: Any
    lifecycle: Any
    cache: Any
    proactive: Any
    versioning: Any
    query_composer: Any
    federation: Any
    backup: Any
    sync: Any
    encryption: Any
    sharing: Any
    analytics: Any
    dashboard: Any
    import_export: Any
    multimodal: Any
    collaborative: Any

    # -- methods provided by sibling mixins / the facade -------------------
    # Ellipsis-body stubs so a mixin that calls a method owned by another
    # mixin type-checks standalone; the real impl wins via MRO.
    def save(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def search(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def get(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def update(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def resolve_id(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def ask(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def ask_stream(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def repo_search(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]

    def _apply_source_feedback(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _apply_write_policy(self, *a: Any, **k: Any) -> Any: ...
    def _build_ask_context(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _build_rel_path(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _cache_backend(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _cache_read_through(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _chat_citations(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _chat_retrieval_question(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _compose_for_embed(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _derive_metadata(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _emit_ledger(self, *a: Any, **k: Any) -> Any: ...
    def _emit_save_receipt(self, *a: Any, **k: Any) -> Any: ...
    def _enforce_synapse_freeze(self, *a: Any, **k: Any) -> Any: ...
    def _ensure_chat(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _generate_contextual_summary(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _mark_dirty(self, *a: Any, **k: Any) -> Any: ...
    def _maybe_warn_legacy_paths(self, *a: Any, **k: Any) -> Any: ...
    def _normalize_chat_history(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _normalize_rating(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _read_body(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _record_access(self, *a: Any, **k: Any) -> Any: ...
    def _repo_corpus(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _repo_replay_payload(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _rerank(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _resolve_existing(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _resolve_source_id(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
    def _verbatim_short_circuit(self, *a: Any, **k: Any) -> Any: ...  # type: ignore[empty-body]
