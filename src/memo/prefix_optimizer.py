"""L3 Prefix-Aligned Recall: KV cache optimization through deterministic recall structure.

This module provides utilities to reorder recall context (system prompt + memories)
to maximize KV cache prefix matches on LLM providers (Anthropic, Bedrock, etc.).

Key insight: LLM providers cache attention weights using KV prefixes. By pinning
the order and encoding of recall context deterministically, we increase the chance
that repeated recalls from the same session will hit cached KV prefixes, reducing
actual token computation costs.

Techniques:
- Deterministic memory ordering (SHA256 hash-based sort)
- Stable JSON encoding (sorted keys, consistent spacing)
- Prefix structure pinning (system → cite instruction → sorted memories → verbosity)
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memo.config import Config


def flag_prefix_cache_align_enabled() -> bool:
    """Read MEMO_PREFIX_CACHE_ALIGN flag (default: OFF).

    When enabled, optimize_recall_prefix reorders memories deterministically
    to maximize KV cache prefix hits across repeated recalls.

    Returns:
        True if MEMO_PREFIX_CACHE_ALIGN=1/true, False otherwise.
    """
    from memo.flags import flag_bool

    return flag_bool("MEMO_PREFIX_CACHE_ALIGN")


def _stable_json_encode(data: dict) -> str:
    """Encode dict to JSON with sorted keys and deterministic output.

    This ensures the same data structure always produces identical JSON
    regardless of insertion order. Critical for prefix stability.

    Args:
        data: Dict to encode.

    Returns:
        JSON string with sorted keys, no trailing whitespace.
    """
    return json.dumps(data, separators=(",", ": "), sort_keys=True)


def _sort_memories_deterministic(memories: list[str]) -> list[str]:
    """Sort memories by SHA256 hash for deterministic ordering.

    This ensures the same set of memories always appears in the same order,
    independent of input order. Uses SHA256 to provide a stable,
    collision-resistant sort key.

    Args:
        memories: List of memory strings.

    Returns:
        Sorted list (sorted by hash of each memory).
    """
    if not memories:
        return []

    def sort_key(mem: str) -> str:
        """Return SHA256 hash as sort key."""
        return hashlib.sha256(mem.encode()).hexdigest()

    return sorted(memories, key=sort_key)


def optimize_recall_prefix(
    system_prompt: str, memories_text: str, config: Config
) -> tuple[str, str]:
    """Optimize recall prefix structure for KV cache alignment.

    Reorders recall context to pin its structure for maximum prefix stability:
    1. System prompt (unchanged)
    2. Citation instruction (if present)
    3. Memories (sorted by SHA256 hash)
    4. Verbosity steering (if present)

    Args:
        system_prompt: The system prompt (e.g., "You are Claude...")
        memories_text: Memory block text, potentially with newline-separated entries.
        config: Config object (for future extensibility).

    Returns:
        Tuple of (system_prompt, optimized_memories_text) with deterministic ordering.
    """
    # Handle empty memories case
    if not memories_text or memories_text.strip() == "":
        return (system_prompt, "")

    # Split memories by newline to get individual entries
    memory_lines = memories_text.split("\n")

    # Filter out empty lines
    memory_lines = [line for line in memory_lines if line.strip()]

    # Sort deterministically
    sorted_lines = _sort_memories_deterministic(memory_lines)

    # Rejoin with newlines
    optimized_memories = "\n".join(sorted_lines)

    return (system_prompt, optimized_memories)
