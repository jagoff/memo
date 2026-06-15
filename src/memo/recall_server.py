"""Recall daemon facade.

Keeps the historical ``memo.recall_server`` surface stable while the
implementation lives in smaller runtime-oriented modules.
"""

from __future__ import annotations

from memo.daemon_common import is_pid_alive as _is_pid_alive
from memo.daemon_common import serve_until_shutdown as _serve_until_shutdown
from memo.recall_client import _send_request, connect_and_recall, connect_and_send
from memo.recall_logic import (
    RECALL_DIRECTIVE,
    RECALL_FOOTER,
    RECALL_HEADER,
    _apply_preference_boost,
    _apply_project_boost,
    _dedup_key,
    _recall_logic,
    _session_context,
    dedup_hits,
)
from memo.recall_socket import (
    _MAX_LINE_BYTES,
    PriorityLock,
    _cleanup,
    _pid_file,
    _read_pid,
    _RecallHandler,
    _RecallServer,
    _socket_path,
    run_server,
)
from memo.recall_stats import (
    _STATS_DEFAULT_PERSIST_INTERVAL_S,
    _STATS_SAMPLE_CAP,
    _DaemonStats,
    _percentile,
    _stats_file,
    _stats_persister,
)

__all__ = [
    "RECALL_DIRECTIVE",
    "RECALL_FOOTER",
    "RECALL_HEADER",
    "_MAX_LINE_BYTES",
    "_STATS_DEFAULT_PERSIST_INTERVAL_S",
    "_STATS_SAMPLE_CAP",
    "PriorityLock",
    "_DaemonStats",
    "_RecallHandler",
    "_RecallServer",
    "_apply_preference_boost",
    "_apply_project_boost",
    "_cleanup",
    "_dedup_key",
    "_is_pid_alive",
    "_percentile",
    "_pid_file",
    "_read_pid",
    "_recall_logic",
    "_send_request",
    "_serve_until_shutdown",
    "_session_context",
    "_socket_path",
    "_stats_file",
    "_stats_persister",
    "connect_and_recall",
    "connect_and_send",
    "dedup_hits",
    "run_server",
]
