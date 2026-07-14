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
from . import capture_core as _capture_core
from . import capture_hooks as _capture_hooks
from .capture_core import (
    _EXTRACT_SYSTEM_PROMPT as _EXTRACT_SYSTEM_PROMPT,
)
from .capture_core import (
    _FILLER_OPENER_RE as _FILLER_OPENER_RE,
)
from .capture_core import (
    _GENERIC_PREFIXES as _GENERIC_PREFIXES,
)
from .capture_core import (
    _META_COMMENTARY_RE as _META_COMMENTARY_RE,
)
from .capture_core import (
    _READ_TOOLS as _READ_TOOLS,
)
from .capture_core import (
    _SENTENCE_SPLIT_RE as _SENTENCE_SPLIT_RE,
)
from .capture_core import (
    _TRIGGER_PATTERNS as _TRIGGER_PATTERNS,
)
from .capture_core import (
    _WRITE_TOOLS as _WRITE_TOOLS,
)
from .capture_core import (
    _capture_provenance as _capture_provenance,
)
from .capture_core import (
    _cosine as _cosine,
)
from .capture_core import (
    _extract_text as _extract_text,
)
from .capture_core import (
    _hash_assistant as _hash_assistant,
)
from .capture_core import (
    _jaccard as _jaccard,
)
from .capture_core import (
    _parse_exchanges as _parse_exchanges,
)
from .capture_core import (
    _parse_transcript as _parse_transcript,
)
from .capture_core import (
    _passes_prefilter as _passes_prefilter,
)
from .capture_core import (
    _passes_quality as _passes_quality,
)
from .capture_core import (
    _read_last_exchange as _read_last_exchange,
)
from .capture_core import (
    _read_recent_exchanges as _read_recent_exchanges,
)
from .capture_core import (
    _strip_private as _strip_private,
)
from .capture_core import (
    _tool_activity as _tool_activity,
)
from .capture_core import (
    collect_tool_files as collect_tool_files,
)
from .capture_core import (
    dedupe_batch as dedupe_batch,
)
from .capture_core import (
    extract_insights as extract_insights,
)
from .capture_core import (
    find_near_duplicate as find_near_duplicate,
)
from .capture_core import (
    is_meta_commentary as is_meta_commentary,
)
from .capture_core import (
    is_near_duplicate as is_near_duplicate,
)
from .capture_core import (
    reweight_ambiguous_type as reweight_ambiguous_type,
)
from .capture_core import (
    score_type_confidence as score_type_confidence,
)
from .capture_core import (
    strip_meta_commentary as strip_meta_commentary,
)

# Hook entry points and watermark management
from .capture_hooks import (
    incremental_tick_due as incremental_tick_due,
)
from .capture_hooks import (
    list_sessions_without_watermark as list_sessions_without_watermark,
)


def _sync_capture_core_patchables() -> None:
    """Keep legacy monkeypatches on `memo.capture` effective.

    Historically this module owned the capture implementation, so tests and
    downstream users patched `memo.capture.extract_insights` and friends. After
    the implementation moved to `capture_core`, the re-exported function
    objects still looked patchable but `_extract_and_save` resolved globals in
    `capture_core`. Sync the patchable seams just before invoking core helpers.
    """
    _capture_core.extract_insights = extract_insights
    _capture_core._passes_quality = _passes_quality
    _capture_core.find_near_duplicate = find_near_duplicate
    _capture_hooks._extract_and_save = _extract_and_save


def _extract_and_save(*args, **kwargs):
    _sync_capture_core_patchables()
    return _capture_core._extract_and_save(*args, **kwargs)


def run_capture(*args, **kwargs):
    _sync_capture_core_patchables()
    return _capture_hooks.run_capture(*args, **kwargs)


def run_capture_incremental(*args, **kwargs):
    _sync_capture_core_patchables()
    return _capture_hooks.run_capture_incremental(*args, **kwargs)


def extract_and_save_text(*args, **kwargs):
    _sync_capture_core_patchables()
    return _capture_core.extract_and_save_text(*args, **kwargs)


__all__ = [
    "_EXTRACT_SYSTEM_PROMPT",
    "_FILLER_OPENER_RE",
    "_GENERIC_PREFIXES",
    "_META_COMMENTARY_RE",
    "_READ_TOOLS",
    "_SENTENCE_SPLIT_RE",
    "_TRIGGER_PATTERNS",
    "_WRITE_TOOLS",
    "_capture_provenance",
    "_cosine",
    "_extract_and_save",
    "_extract_text",
    "_hash_assistant",
    "_jaccard",
    "_parse_exchanges",
    "_parse_transcript",
    "_passes_prefilter",
    "_passes_quality",
    "_read_last_exchange",
    "_read_recent_exchanges",
    "_strip_private",
    "_tool_activity",
    "collect_tool_files",
    "dedupe_batch",
    "extract_and_save_text",
    "extract_insights",
    "find_near_duplicate",
    "incremental_tick_due",
    "is_meta_commentary",
    "is_near_duplicate",
    "list_sessions_without_watermark",
    "reweight_ambiguous_type",
    "run_capture",
    "run_capture_incremental",
    "score_type_confidence",
    "strip_meta_commentary",
]
