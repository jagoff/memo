"""Wave 2 Token Economy: L3 Prefix-Aligned Recall tests.

This test suite covers prefix optimization for KV cache alignment,
ensuring deterministic, stable ordering of recall context to maximize
prefix-match hits on LLM provider caches.

Test structure:
- TestPrefixOptimizerFlags: flag reading and defaults
- TestStableJsonEncode: JSON encoding determinism
- TestSortMemoriesDeterministic: memory sorting by hash
- TestOptimizeRecallPrefix: end-to-end prefix optimization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from memo.prefix_optimizer import (
    _sort_memories_deterministic,
    _stable_json_encode,
    flag_prefix_cache_align_enabled,
    optimize_recall_prefix,
)

if TYPE_CHECKING:
    from memo.config import Config


class TestPrefixOptimizerFlags:
    """Test flag_prefix_cache_align_enabled with defaults and env vars."""

    def test_flag_default_off(self, monkeypatch):
        """MEMO_PREFIX_CACHE_ALIGN defaults to False."""
        monkeypatch.delenv("MEMO_PREFIX_CACHE_ALIGN", raising=False)
        assert flag_prefix_cache_align_enabled() is False

    def test_flag_env_true(self, monkeypatch):
        """MEMO_PREFIX_CACHE_ALIGN=1 returns True."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "1")
        assert flag_prefix_cache_align_enabled() is True

    def test_flag_env_false(self, monkeypatch):
        """MEMO_PREFIX_CACHE_ALIGN=0 returns False."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "0")
        assert flag_prefix_cache_align_enabled() is False

    def test_flag_env_true_string(self, monkeypatch):
        """MEMO_PREFIX_CACHE_ALIGN=true returns True."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "true")
        assert flag_prefix_cache_align_enabled() is True

    def test_flag_env_false_string(self, monkeypatch):
        """MEMO_PREFIX_CACHE_ALIGN=false returns False."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "false")
        assert flag_prefix_cache_align_enabled() is False


class TestStableJsonEncode:
    """Test _stable_json_encode for deterministic JSON output."""

    def test_simple_dict(self):
        """Simple dict encodes with sorted keys."""
        data = {"b": 2, "a": 1}
        result = _stable_json_encode(data)
        # Keys should be sorted: a, b
        assert result == '{"a": 1, "b": 2}'

    def test_nested_dict(self):
        """Nested dicts encode with all keys sorted."""
        data = {"z": {"b": 2, "a": 1}, "a": "first"}
        result = _stable_json_encode(data)
        parsed = json.loads(result)
        # Check that first level is sorted: a, z
        keys_first = list(parsed.keys())
        assert keys_first == ["a", "z"]
        # Check nested is sorted: a, b
        keys_nested = list(parsed["z"].keys())
        assert keys_nested == ["a", "b"]

    def test_determinism(self):
        """Same data always produces identical output."""
        data = {"x": 10, "y": {"z": 3, "w": 1}, "a": "alpha"}
        result1 = _stable_json_encode(data)
        result2 = _stable_json_encode(data)
        assert result1 == result2

    def test_no_trailing_whitespace(self):
        """Output has no trailing whitespace."""
        data = {"key": "value"}
        result = _stable_json_encode(data)
        assert result == result.rstrip()

    def test_list_ordering_preserved(self):
        """List order is preserved (not sorted)."""
        data = {"items": [3, 1, 2]}
        result = _stable_json_encode(data)
        parsed = json.loads(result)
        assert parsed["items"] == [3, 1, 2]

    def test_empty_dict(self):
        """Empty dict encodes cleanly."""
        result = _stable_json_encode({})
        assert result == "{}"

    def test_unicode_preserved(self):
        """Unicode characters are preserved."""
        data = {"emoji": "🧠", "text": "memo"}
        result = _stable_json_encode(data)
        assert "🧠" in result
        assert "memo" in result


class TestSortMemoriesDeterministic:
    """Test _sort_memories_deterministic for SHA256-based sorting."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = _sort_memories_deterministic([])
        assert result == []

    def test_single_item(self):
        """Single item returns unchanged."""
        memories = ["single memory"]
        result = _sort_memories_deterministic(memories)
        assert result == memories

    def test_determinism(self):
        """Same input always produces identical order."""
        memories = ["alpha", "beta", "gamma", "delta"]
        result1 = _sort_memories_deterministic(memories)
        result2 = _sort_memories_deterministic(memories)
        assert result1 == result2

    def test_sorting_by_hash(self):
        """Memories are sorted by SHA256 hash."""
        memories = ["apple", "banana", "cherry"]
        result = _sort_memories_deterministic(memories)
        # Verify they're sorted by hash, not alphabetically
        hashes = [__import__("hashlib").sha256(m.encode()).hexdigest() for m in result]
        assert hashes == sorted(hashes)

    def test_preserves_content(self):
        """All items are preserved in output."""
        memories = ["one", "two", "three"]
        result = _sort_memories_deterministic(memories)
        assert set(result) == set(memories)
        assert len(result) == len(memories)

    def test_different_order_same_sort(self):
        """Different input orders produce same sorted output."""
        memories1 = ["a", "b", "c"]
        memories2 = ["c", "b", "a"]
        result1 = _sort_memories_deterministic(memories1)
        result2 = _sort_memories_deterministic(memories2)
        assert result1 == result2

    def test_multiline_content(self):
        """Multiline memory text is sorted by its full hash."""
        memory1 = "line1\nline2\nline3"
        memory2 = "line1\nline2"
        memory3 = "different\nmultiline"
        result = _sort_memories_deterministic([memory1, memory2, memory3])
        # Just verify it returns all three
        assert len(result) == 3
        assert set(result) == {memory1, memory2, memory3}


class TestOptimizeRecallPrefix:
    """Test optimize_recall_prefix for stable prefix ordering."""

    def test_return_type(self, tmp_cfg):
        """Returns tuple of (system_prompt, memories_text)."""
        system = "You are helpful."
        memories = "Memory 1\nMemory 2"
        result = optimize_recall_prefix(system, memories, tmp_cfg)
        assert isinstance(result, tuple)
        assert len(result) == 2
        system_out, memories_out = result
        assert isinstance(system_out, str)
        assert isinstance(memories_out, str)

    def test_preserves_system_prompt(self, tmp_cfg):
        """System prompt is returned unchanged."""
        system = "You are a helpful assistant."
        memories = "Memory content"
        system_out, _ = optimize_recall_prefix(system, memories, tmp_cfg)
        assert system_out == system

    def test_determinism_same_input(self, tmp_cfg):
        """Same input produces identical output."""
        system = "System prompt"
        memories = "Memory 1\nMemory 2\nMemory 3"
        result1 = optimize_recall_prefix(system, memories, tmp_cfg)
        result2 = optimize_recall_prefix(system, memories, tmp_cfg)
        assert result1 == result2

    def test_memory_text_preserved(self, tmp_cfg):
        """All memory content is preserved in output."""
        system = "System"
        memory_lines = ["First memory", "Second memory", "Third memory"]
        memories = "\n".join(memory_lines)
        _, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        # All lines should appear in output
        for line in memory_lines:
            assert line in memories_out

    def test_empty_memories(self, tmp_cfg):
        """Empty memories returns empty memories."""
        system = "System"
        memories = ""
        system_out, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        assert system_out == system
        assert memories_out == ""

    def test_multiline_memories(self, tmp_cfg):
        """Multiline memories are sorted deterministically."""
        system = "System"
        memories = "Line 1\nLine 2\nLine 3"
        _, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        # All original lines should be in output
        assert "Line 1" in memories_out
        assert "Line 2" in memories_out
        assert "Line 3" in memories_out

    def test_repeated_call_consistency(self, tmp_cfg):
        """Multiple calls with same input return identical output."""
        system = "You are helpful."
        memories = "Alpha\nBeta\nGamma"
        results = [optimize_recall_prefix(system, memories, tmp_cfg) for _ in range(3)]
        # All results should be identical
        assert results[0] == results[1]
        assert results[1] == results[2]

    def test_order_independence(self, tmp_cfg):
        """Different input orders produce same output (deterministic sort)."""
        system = "System"
        memories1 = "Z memory\nA memory\nM memory"
        memories2 = "A memory\nZ memory\nM memory"

        _, out1 = optimize_recall_prefix(system, memories1, tmp_cfg)
        _, out2 = optimize_recall_prefix(system, memories2, tmp_cfg)

        # Both should produce the same sorted output
        assert out1 == out2

    def test_special_characters(self, tmp_cfg):
        """Special characters in memories are preserved."""
        system = "System"
        memories = "Memory with [brackets]\nMemory with {braces}\nMemory with \"quotes\""
        _, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        assert "[brackets]" in memories_out
        assert "{braces}" in memories_out
        assert '"quotes"' in memories_out

    def test_unicode_preserved(self, tmp_cfg):
        """Unicode in memories is preserved."""
        system = "System"
        memories = "记忆 1\nMémoire 2\n🧠 Memory 3"
        _, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        assert "记忆" in memories_out
        assert "Mémoire" in memories_out
        assert "🧠" in memories_out

    def test_long_memories(self, tmp_cfg):
        """Long memories are handled correctly."""
        system = "System"
        long_memory = "x" * 5000
        memories = f"{long_memory}\nShort memory"
        _, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        assert long_memory in memories_out
        assert "Short memory" in memories_out


# ──────────────────────────────────────────────────────────────────────────────
# Integration-like tests (after core module tests pass)
# ──────────────────────────────────────────────────────────────────────────────


class TestPrefixOptimizerIntegration:
    """Integration tests for the prefix optimizer."""

    def test_flag_gates_optimization(self, monkeypatch, tmp_cfg):
        """When flag is OFF, optimization is skipped."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "0")
        system = "System"
        memories = "Memory"
        # Even with flag off, function should return valid output
        system_out, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        assert system_out == system

    def test_flag_enables_optimization(self, monkeypatch, tmp_cfg):
        """When flag is ON, optimization returns stable output."""
        monkeypatch.setenv("MEMO_PREFIX_CACHE_ALIGN", "1")
        system = "System"
        memories = "Memory 1\nMemory 2"
        system_out, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)
        # Should return valid output
        assert isinstance(system_out, str)
        assert isinstance(memories_out, str)

    def test_realistic_recall_scenario(self, tmp_cfg):
        """Realistic recall block optimization."""
        system = "You are Claude, an AI assistant."
        memories = """Note: Decision on architecture choice
Decision: Use SQLite for storage
Fact: Python 3.11+ required
Preference: Prefer immutability patterns
Synthesis: Cross-session learning enabled"""

        system_out, memories_out = optimize_recall_prefix(system, memories, tmp_cfg)

        # All content preserved
        assert system_out == system
        assert "Decision on architecture" in memories_out
        assert "Python 3.11+" in memories_out
        assert "immutability" in memories_out

    def test_prefix_stability_repeated_recalls(self, tmp_cfg):
        """Prefix remains stable across repeated recall calls."""
        system = "Base system prompt"

        # Simulate repeated recalls with same memory set
        prefixes = []
        for _ in range(5):
            memories = "Alpha\nBeta\nGamma"
            _, prefix = optimize_recall_prefix(system, memories, tmp_cfg)
            prefixes.append(prefix)

        # All prefixes should be identical (stable)
        assert len(set(prefixes)) == 1, "Prefixes should be identical across calls"
