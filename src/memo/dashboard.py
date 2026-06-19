"""Terminal dashboard facade.

Keeps the historical ``memo.dashboard`` import surface stable while the
implementation lives in focused modules.
"""

from __future__ import annotations

from rich.console import Group

from memo.dashboard_logs import (
    append_context_cost_log,
    append_grounding_diag_log,
    append_grounding_log,
    append_recall_log,
    append_usage_log,
    context_cost_log_path,
    grounding_diag_log_path,
    grounding_log_path,
    read_context_cost_log,
    read_grounding_diag_log,
    read_grounding_log,
    read_recall_hook_log,
    read_recall_log,
    read_usage_log,
    recall_hook_log_path,
    recall_log_path,
    usage_log_path,
)
from memo.dashboard_metrics import (
    EXPECTED_CONSUMERS,
    GROUNDED_SCORE,
    STRONG_SCORE,
    _bail_breakdown,
    _jaccard,
    _parse_ts,
    _reask_tokens,
    _row_quality,
    consult_breakdown,
    consumer_label,
    dedup_double_fire,
    grounded_rate,
    grounding_used,
    reask_stats,
    recall_health,
    referenced_rate,
    verdict,
)
from memo.dashboard_panels import _human_age, _human_bytes, sparkline
from memo.dashboard_tui import render, run_tui

__all__ = [
    "EXPECTED_CONSUMERS",
    "GROUNDED_SCORE",
    "STRONG_SCORE",
    "Group",
    "_bail_breakdown",
    "_human_age",
    "_human_bytes",
    "_jaccard",
    "_parse_ts",
    "_reask_tokens",
    "_row_quality",
    "append_context_cost_log",
    "append_grounding_diag_log",
    "append_grounding_log",
    "append_recall_log",
    "append_usage_log",
    "consult_breakdown",
    "consumer_label",
    "context_cost_log_path",
    "dedup_double_fire",
    "grounded_rate",
    "grounding_diag_log_path",
    "grounding_log_path",
    "grounding_used",
    "read_context_cost_log",
    "read_grounding_diag_log",
    "read_grounding_log",
    "read_recall_hook_log",
    "read_recall_log",
    "read_usage_log",
    "reask_stats",
    "recall_health",
    "recall_hook_log_path",
    "recall_log_path",
    "referenced_rate",
    "render",
    "run_tui",
    "sparkline",
    "usage_log_path",
    "verdict",
]
