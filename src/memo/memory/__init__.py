"""`memo.memory` — the high-level `Memory` API, split into a package.

This package replaces the former `memory.py` god-file. The public surface is
preserved verbatim: every name that used to be importable from
`memo.memory` is re-exported here, so `from memo.memory import Memory,
MemoryRecord, AmbiguousIdError, _apply_decay, ...` keeps working unchanged.

Layout:
- `record`       — `MemoryRecord`, module-level constants/prompts/helpers,
                   re-exported domain errors (the leaf).
- `_base`        — `_MemoryBase`, the typed contract for the op mixins.
- `write_ops`    — `_WriteOpsMixin` (save / update / forget / delete).
- `search_ops`   — `_SearchOpsMixin` (search / list / get / resolve_id).
- `search_scoring_ops` — `_SearchScoringMixin` (post-fetch graph expansion,
                   penalties, entity/retrieval/health re-scoring, access,
                   cache read-through).
- `ask_ops`      — `_AskOpsMixin` (ask / ask_stream + context building).
- `chat_ask_ops` — `_ChatAskOpsMixin` (chat_ask / chat_ask_stream + history/
                   retrieval-question/citation helpers).
- `rerank_ops`   — `_RerankOpsMixin` (rerank + source feedback).
- `repo_ops`     — `_RepoOpsMixin` (repo corpus).
- `maintain_ops` — `_MaintainOpsMixin` (reindex / lint / gc / entities /
                   consolidate / provenance / freeze / replay).
- `facade`       — `Memory`, composing the mixins + the lazy `@property`
                   managers + `__init__`.
"""

from __future__ import annotations

# -- stdlib names that were re-exportable from the old module ----------------
import uuid  # noqa: F401  (tests monkeypatch `memo.memory.uuid.uuid4`)
from collections.abc import Iterator  # noqa: F401
from dataclasses import dataclass, field, replace  # noqa: F401
from datetime import UTC, datetime  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

# -- foundation manager classes / helpers (re-exported, verbatim surface) ----
from memo.analytics import AnalyticsEngine, Dashboard  # noqa: F401
from memo.collaborative import (  # noqa: F401
    CollaborativeFilter,
    CollaborativeGraph,
    CollaborativeManager,
)
from memo.config import Config  # noqa: F401
from memo.consolidation import AdvancedConsolidator  # noqa: F401
from memo.contextual import ContextStore, ContextualRecall  # noqa: F401
from memo.contextual_retrieval import (  # noqa: F401
    get_or_generate_context,
    prepend_context,
)
from memo.contradict import ContradictionScanner, ContradictionStore  # noqa: F401
from memo.crossref import CrossReferenceIndex, LinkSuggester  # noqa: F401
from memo.embedder import MLXEmbedder, assert_valid_embedding  # noqa: F401
from memo.errors import (  # noqa: F401
    AmbiguousIdError,
    FederationError,
    IdentityConflictError,
    MemoError,
    NotFoundError,
    StorageError,
    ValidationError,
    WriteRefused,
)
from memo.federation import FederationManager  # noqa: F401
from memo.graph import GraphStore  # noqa: F401
from memo.import_export import ImportExportManager  # noqa: F401
from memo.lifecycle import (  # noqa: F401
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
    LifecycleManager,
)
from memo.llm import MLXChat  # noqa: F401

# -- the facade + the full record surface (constants, prompts, helpers) ------
from memo.memory.facade import Memory
from memo.memory.record import (  # noqa: F401
    _ASK_SYSTEM_PROMPT,
    _CONSOLIDATE_SYSTEM_PROMPT,
    _CONVERSATION_TOKENS,
    _DERIVE_SYSTEM_PROMPT,
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,
    _ISO_DATE_RE,
    _PROVENANCE_KEYS,
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    _RECENCY_TOKENS,
    _SLUG_NON_WORD,
    _SLUG_WS,
    _VALID_TYPES,
    MEMO_BACKEND_NAME,
    MEMO_BACKEND_NATIVE_SCHEMA,
    NATIVE_BACKEND_PROTOCOL_VERSION,
    MemoryRecord,
    _apply_decay,
    _compose_for_embed,
    _derive_title,
    _extract_provenance,
    _is_conversation_query,
    _is_group_chat,
    _is_recency_query,
    _is_whatsapp_hit,
    _log,
    _norm_dedup_path,
    _normalise_tags,
    _now_iso,
    _recency_key,
    _rrf_fuse,
    _slugify,
    _vault_dedup_keys,
)
from memo.navigation import GraphNavigator  # noqa: F401
from memo.saved_queries import QueryComposer, QueryStore  # noqa: F401
from memo.store import VecStore  # noqa: F401
from memo.sync import BackupManager, SyncManager  # noqa: F401
from memo.temporal import TemporalAnalyzer  # noqa: F401
from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES  # noqa: F401
from memo.util import sha256_short as _sha256_short  # noqa: F401
from memo.util import stable_hash as _stable_content_hash  # noqa: F401
from memo.util import utc_now_iso as _utc_now_iso  # noqa: F401
from memo.versioning import VersionManager  # noqa: F401

__all__ = [
    "AmbiguousIdError",
    "IdentityConflictError",
    "Memory",
    "MemoryRecord",
    "WriteRefused",
]
