"""Prompt override resolution — package default -> `state_dir/prompts/<name>.md`.

Every LLM system prompt in memo resolves through `resolve_prompt()`. Users
override any prompt by dropping a UTF-8 markdown file at
`<state_dir>/prompts/<name>.md` (whole-prompt replacement — key for PyPI
users on small MLX models, where prompt/language sensitivity is high).
`prompt_version()` hashes the RESOLVED text so caches / provenance keyed on
it invalidate the moment an override lands.

Never on the 5s recall-hook path: the recall hook makes no LLM calls, so no
call site of this module runs inside it. Resolution is one stat + optional
small file read.
"""

from __future__ import annotations

import logging
from pathlib import Path

from memo.util import sha256_short

_log = logging.getLogger(__name__)

# Canonical prompt names — one per hardcoded system prompt (see docs table in
# tests/test_prompt_overrides.py / Task G-5 for the constant <-> name map).
PROMPT_NAMES = frozenset(
    {
        "ask",  # memory/prompts.py _ASK_SYSTEM_PROMPT
        "consolidate",  # memory/prompts.py _CONSOLIDATE_SYSTEM_PROMPT
        "synthesis",  # memory/prompts.py _SYNTHESIS_SYSTEM_PROMPT
        "reflect",  # memory/prompts.py _REFLECT_SYSTEM_PROMPT
        "derive",  # memory/prompts.py _DERIVE_SYSTEM_PROMPT
        "extract_entities",  # memory/prompts.py _EXTRACT_ENTITIES_SYSTEM_PROMPT
        "contradiction",  # temporal.py _CONTRADICTION_SYSTEM_PROMPT
        "coordination",  # coordination.py _JUDGE_SYSTEM_PROMPT
        "validity_extract",  # temporal.py _VALIDITY_EXTRACT_SYSTEM_PROMPT
        "merge",  # consolidation.py _MERGE_SYSTEM_PROMPT
        "capture_extract",  # capture.py _EXTRACT_SYSTEM_PROMPT
    }
)


def resolve_prompt(name: str, default: str, state_dir: Path) -> str:
    """Return `<state_dir>/prompts/<name>.md` content if present, else `default`."""
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt name {name!r}; known: {sorted(PROMPT_NAMES)}")
    path = state_dir / "prompts" / f"{name}.md"
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except OSError as exc:
        _log.warning("prompt override %s unreadable (%s); using default", path, exc)
    return default


def prompt_version(name: str, default: str, state_dir: Path) -> str:
    """Short stable hash of the RESOLVED prompt text (cache/provenance key)."""
    return sha256_short(resolve_prompt(name, default, state_dir))
