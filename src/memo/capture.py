"""Durable session capture: record turns, ingest into memo, bridge to memflow.

Hook fires on every assistant turn (Stop event), reads the just-finished
exchange from the transcript, asks the configured helper LLM to
extract any actionable insights, dedups against the existing corpus,
and saves the survivors with auto-derived metadata.

## Pipeline

```
Stop event
   │
   ▼  read transcript_path JSONL → last (user, assistant) exchange
   │
   ▼  pre-filter (cheap): skip empty / too-short / pure-tool turns
   │
   ▼  helper LLM extract → JSON [{title, type, body, tags}, ...]
   │
   ▼  dedup: embed each candidate, near-search, drop if max_sim > 0.85
   │
   ▼  save survivors via Memory.save()
```

## State file

`~/.local/share/memo/last-capture.json` tracks the hash of the last
processed assistant message so re-firing on the same turn (e.g. the
user runs `/clear` mid-stream, or two Stop hooks race) doesn't
double-extract.

## Why dedup with the embedder, not the title

Two memories with different titles can describe the same fact. The
embedder is the only signal that catches "same fact, different
phrasing". Threshold 0.85 is empirical: cosine sim between
near-paraphrases is typically 0.85-0.95 with Qwen3-Embedding;
genuinely-distinct memories score below 0.75 even on the same topic.

## Failure modes

All swallowed silently. Capture is opportunistic — a hook that fails
to extract is no worse than the pre-Phase-B world. Exception: if the
user explicitly sets `MEMO_CAPTURE_DEBUG=1`, errors print to stderr.
"""

# Core capture logic
from .capture_core import (
    _EXTRACT_SYSTEM_PROMPT as _EXTRACT_SYSTEM_PROMPT,
    _FILLER_OPENER_RE as _FILLER_OPENER_RE,
    _GENERIC_PREFIXES as _GENERIC_PREFIXES,
    _META_COMMENTARY_RE as _META_COMMENTARY_RE,
    _READ_TOOLS as _READ_TOOLS,
    _SENTENCE_SPLIT_RE as _SENTENCE_SPLIT_RE,
    _TRIGGER_PATTERNS as _TRIGGER_PATTERNS,
    _WRITE_TOOLS as _WRITE_TOOLS,
    _capture_provenance as _capture_provenance,
    _cosine as _cosine,
    _extract_and_save as _extract_and_save,
    _extract_text as _extract_text,
    _hash_assistant as _hash_assistant,
    _jaccard as _jaccard,
    _parse_exchanges as _parse_exchanges,
    _parse_transcript as _parse_transcript,
    _passes_prefilter as _passes_prefilter,
    _passes_quality as _passes_quality,
    _read_last_exchange as _read_last_exchange,
    _read_recent_exchanges as _read_recent_exchanges,
    _strip_private as _strip_private,
    _tool_activity as _tool_activity,
    collect_tool_files as collect_tool_files,
    dedupe_batch as dedupe_batch,
    extract_and_save_text as extract_and_save_text,
    extract_insights as extract_insights,
    find_near_duplicate as find_near_duplicate,
    is_meta_commentary as is_meta_commentary,
    is_near_duplicate as is_near_duplicate,
    reweight_ambiguous_type as reweight_ambiguous_type,
    score_type_confidence as score_type_confidence,
    strip_meta_commentary as strip_meta_commentary,
)

# Hook entry points and watermark management
from .capture_hooks import (
    incremental_tick_due as incremental_tick_due,
    list_sessions_without_watermark as list_sessions_without_watermark,
    run_capture as run_capture,
    run_capture_incremental as run_capture_incremental,
)

__all__ = [
    # Core extraction
    "extract_insights",
    "extract_and_save_text",
    "find_near_duplicate",
    "is_near_duplicate",
    # Hook entry points
    "run_capture",
    "run_capture_incremental",
    # Incremental capture utilities
    "incremental_tick_due",
    "list_sessions_without_watermark",
    # Text processing and hygiene
    "is_meta_commentary",
    "strip_meta_commentary",
    "collect_tool_files",
    # Type classification
    "score_type_confidence",
    "reweight_ambiguous_type",
    # Deduplication
    "dedupe_batch",
    # Internal helpers (re-exported for backward compat)
    "_capture_provenance",
    "_extract_and_save",
    "_extract_text",
    "_hash_assistant",
    "_jaccard",
    "_passes_prefilter",
    "_passes_quality",
    "_read_recent_exchanges",
    "_read_last_exchange",
    "_strip_private",
    "_tool_activity",
    "_parse_transcript",
    "_parse_exchanges",
    "_cosine",
    # Constants
    "_TRIGGER_PATTERNS",
    "_GENERIC_PREFIXES",
    "_EXTRACT_SYSTEM_PROMPT",
    "_READ_TOOLS",
    "_WRITE_TOOLS",
    "_META_COMMENTARY_RE",
    "_FILLER_OPENER_RE",
    "_SENTENCE_SPLIT_RE",
]
