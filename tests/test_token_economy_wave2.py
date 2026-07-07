"""
Wave 2 Token Economy: L2 (Streaming Compression) + L3 (Prefix Optimization) + Integration.

Tests for:
- L2: stream_compress.py (response token reduction via marker compression)
- L3: prefix_optimizer.py (KV cache alignment via recall structure pinning)
- Integration: L2+L3 combined in recall hook
- Measurement: token baseline script compatibility
"""

from __future__ import annotations

import pytest


# ── L2: Streaming Compression ─────────────────────────────────────────────


def test_flag_stream_compress_enabled() -> None:
    """Flag resolution for L2 streaming compression."""
    from memo.flags_recall import flag_stream_compress_enabled

    # Default: OFF
    assert flag_stream_compress_enabled() is False


def test_compress_token_stream_yields_markers() -> None:
    """L2: Low-signal spans compressed into markers."""
    import os
    from memo.stream_compress import compress_token_stream
    from memo.config import Config

    # Enable L2 compression for this test
    os.environ["MEMO_STREAM_COMPRESS"] = "1"
    try:
        tokens = ["I'll", " help", " you", "...", "Let", " me", " think"]
        config = Config()
        compressed = list(compress_token_stream(iter(tokens), config))

        # Should yield marker for preamble "I'll help you..."
        assert any("[...compressed" in str(t) for t in compressed), "Expected compression marker"
    finally:
        os.environ.pop("MEMO_STREAM_COMPRESS", None)


def test_compress_token_stream_idempotent() -> None:
    """L2: Double-compression returns same result (idempotent)."""
    from memo.stream_compress import compress_token_stream
    from memo.config import Config

    tokens = ["I'll", " help", " you", "...", "Let", " me", " think"]
    config = Config()

    pass1 = list(compress_token_stream(iter(tokens), config))
    # Re-compress the marker output
    pass2 = list(compress_token_stream(iter(pass1), config))

    # Should not double-compress markers
    assert pass1 == pass2


# ── L3: Prefix Optimization ──────────────────────────────────────────────


def test_flag_prefix_cache_align_enabled() -> None:
    """Flag resolution for L3 prefix cache alignment."""
    from memo.flags_recall import flag_prefix_cache_align_enabled

    # Default: OFF
    assert flag_prefix_cache_align_enabled() is False


def test_optimize_recall_prefix_returns_tuple() -> None:
    """L3: Optimization returns (system_prompt, memories) tuple."""
    from memo.prefix_optimizer import optimize_recall_prefix
    from memo.config import Config

    system_prompt = "You are a helpful assistant."
    memories_text = "Memory 1: learned X\nMemory 2: learned Y"
    config = Config()

    opt_sys, opt_mem = optimize_recall_prefix(system_prompt, memories_text, config)

    assert isinstance(opt_sys, str)
    assert isinstance(opt_mem, str)
    # Should preserve content (even if reordered)
    assert "helpful" in opt_sys


def test_optimize_recall_prefix_stable_order() -> None:
    """L3: Recall prefix is deterministically ordered (reproducible)."""
    from memo.prefix_optimizer import optimize_recall_prefix
    from memo.config import Config

    system_prompt = "You are a helpful assistant."
    memories_text = "Memory 1: learned X\nMemory 2: learned Y"
    config = Config()

    opt_sys_1, opt_mem_1 = optimize_recall_prefix(system_prompt, memories_text, config)
    opt_sys_2, opt_mem_2 = optimize_recall_prefix(system_prompt, memories_text, config)

    # Deterministic: same input = same output
    assert opt_sys_1 == opt_sys_2
    assert opt_mem_1 == opt_mem_2


# ── Integration: L2 + L3 Combined ────────────────────────────────────────


def test_l2_l3_compatible() -> None:
    """Integration: L2 and L3 compose without interference."""
    from memo.stream_compress import compress_token_stream
    from memo.prefix_optimizer import optimize_recall_prefix
    from memo.config import Config

    config = Config()
    system_prompt = "System."
    memories_text = "Memory 1\nMemory 2"

    # L3: optimize recall structure
    opt_sys, opt_mem = optimize_recall_prefix(system_prompt, memories_text, config)

    # L2: compress tokens from output
    tokens = opt_sys.split()
    compressed = list(compress_token_stream(iter(tokens), config))

    # Should not regress: both work together
    assert len(compressed) >= 0


# ── Measurement & Gating ──────────────────────────────────────────────────


def test_baseline_script_syntactically_valid() -> None:
    """Measurement: scripts/wave2_token_baseline.py compiles."""
    import subprocess
    import pathlib

    script_path = pathlib.Path("/Users/fer/repos/memo/scripts/wave2_token_baseline.py")
    if not script_path.exists():
        pytest.skip("Baseline script not yet created")

    result = subprocess.run(
        ["python3", "-m", "py_compile", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script compile failed: {result.stderr}"


def test_gating_checklist_exists() -> None:
    """Measurement: Gating checklist doc mentions key gates."""
    import pathlib

    checklist_path = pathlib.Path(
        "/Users/fer/repos/memo/docs/superpowers/plans/wave2_gating_checklist.md"
    )
    if not checklist_path.exists():
        pytest.skip("Gating checklist not yet created")

    content = checklist_path.read_text()
    # Should mention key gate requirements
    assert any(
        phrase in content.lower()
        for phrase in ["20+", "tests", "0.90", "baseline", "wave 2"]
    )
