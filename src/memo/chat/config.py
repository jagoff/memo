"""Chat knobs — production synapse plist values baked in as code defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ChatConfig:
    base_k: int
    relevance_floor: float
    vote_boost: float
    semantic_threshold: float
    multi_query: bool
    multi_query_n: int
    fulldoc: bool
    answer_max_tokens: int
    synth_head: int
    graph_compact: bool
    graph_compact_min_idf: float
    feedback_dir: Path
    sessions_dir: Path

    @classmethod
    def load(cls, state_dir: Path) -> ChatConfig:
        chat_root = state_dir / "chat"
        return cls(
            base_k=_env_int("MEMO_CHAT_BASE_K", 20),
            relevance_floor=_env_float("MEMO_CHAT_RELEVANCE_FLOOR", 0.25),
            vote_boost=_env_float("MEMO_CHAT_VOTE_BOOST", 1.5),
            semantic_threshold=_env_float("MEMO_CHAT_SEMANTIC_THRESHOLD", 0.75),
            multi_query=_env_bool("MEMO_CHAT_MULTI_QUERY", True),
            multi_query_n=_env_int("MEMO_CHAT_MULTI_QUERY_N", 2),
            fulldoc=_env_bool("MEMO_CHAT_FULLDOC", True),
            answer_max_tokens=_env_int("MEMO_CHAT_ANSWER_MAX_TOKENS", 1200),
            synth_head=_env_int("MEMO_CHAT_SYNTH_HEAD", 8),
            graph_compact=_env_bool("MEMO_CHAT_GRAPH_COMPACT", False),
            graph_compact_min_idf=_env_float("MEMO_CHAT_GRAPH_COMPACT_MIN_IDF", 0.5),
            feedback_dir=chat_root / "feedback",
            sessions_dir=chat_root / "sessions",
        )
