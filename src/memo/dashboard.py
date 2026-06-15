"""Terminal dashboard facade.

Keeps the historical ``memo.dashboard`` import surface stable while the
implementation lives in focused modules.
"""

from __future__ import annotations

from rich.console import Group

from memo.dashboard_logs import (
    append_grounding_log,
    append_recall_log,
    append_usage_log,
    grounding_log_path,
    read_grounding_log,
    read_recall_log,
    read_usage_log,
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
    "append_grounding_log",
    "append_recall_log",
    "append_usage_log",
    "consult_breakdown",
    "consumer_label",
    "dedup_double_fire",
    "grounded_rate",
    "grounding_log_path",
    "read_grounding_log",
    "read_recall_log",
    "read_usage_log",
    "reask_stats",
    "recall_health",
    "recall_log_path",
    "referenced_rate",
    "render",
    "run_tui",
    "sparkline",
    "usage_log_path",
    "verdict",
]
