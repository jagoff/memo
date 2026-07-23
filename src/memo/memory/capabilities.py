from __future__ import annotations

from collections.abc import Callable
from typing import Any

CapabilityFactory = Callable[[Any], Any]


def _contextual(memory: Any) -> Any:
    from memo.contextual import ContextStore, ContextualRecall

    return ContextualRecall(memory, ContextStore(memory.cfg.state_dir))


def _crossref(memory: Any) -> Any:
    from memo.crossref import CrossReferenceIndex

    return CrossReferenceIndex(memory.cfg.crossref_db)


def _link_suggester(memory: Any) -> Any:
    from memo.crossref import LinkSuggester

    return LinkSuggester(memory, memory.crossref)


def _lifecycle(memory: Any) -> Any:
    from memo.lifecycle import LifecycleManager

    return LifecycleManager(memory)


def _versioning(memory: Any) -> Any:
    from memo.versioning import VersionManager

    return VersionManager(memory)


def _query_composer(memory: Any) -> Any:
    from memo.saved_queries import QueryComposer, QueryStore

    return QueryComposer(memory, QueryStore(memory.cfg.state_dir))


def _backup(memory: Any) -> Any:
    from memo.sync import BackupManager

    return BackupManager(
        memory_dir=memory.cfg.memory_dir,
        db_dir=memory.cfg.state_dir,
        backup_dir=memory.cfg.state_dir / "backups",
    )


def _sync(memory: Any) -> Any:
    from memo.sync import SyncManager

    return SyncManager(memory)


def _analytics(memory: Any) -> Any:
    from memo.analytics import AnalyticsEngine

    return AnalyticsEngine(memory)


def _dashboard(memory: Any) -> Any:
    from memo.analytics import Dashboard

    return Dashboard(memory.analytics)


def _import_export(memory: Any) -> Any:
    from memo.import_export import ImportExportManager

    return ImportExportManager(memory)


def _collaborative(memory: Any) -> Any:
    from memo.collaborative import CollaborativeFilter, CollaborativeGraph, CollaborativeManager

    graph = CollaborativeGraph(memory.cfg.state_dir)
    filter_ = CollaborativeFilter(graph)
    return CollaborativeManager(graph, filter_)


def _federation(memory: Any) -> Any:
    from memo.federation import FederationManager

    return FederationManager(memory)


OPTIONAL_CAPABILITIES: dict[str, CapabilityFactory] = {
    "analytics": _analytics,
    "backup": _backup,
    "collaborative": _collaborative,
    "contextual": _contextual,
    "crossref": _crossref,
    "dashboard": _dashboard,
    "federation": _federation,
    "import_export": _import_export,
    "lifecycle": _lifecycle,
    "link_suggester": _link_suggester,
    "query_composer": _query_composer,
    "sync": _sync,
    "versioning": _versioning,
}
