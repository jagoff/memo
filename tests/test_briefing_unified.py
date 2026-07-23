"""Unified briefing tests for Memo's native operational state."""

from __future__ import annotations

import asyncio

from memo.briefing import compact_text, operational_briefing_lines
from memo.memory import Memory
from memo.server import build_server


def test_operational_briefing_is_empty_without_state(tmp_cfg) -> None:
    mem = Memory(tmp_cfg)
    try:
        assert operational_briefing_lines(mem) == []
    finally:
        mem.close()


def test_operational_briefing_renders_native_continuity(tmp_cfg) -> None:
    mem = Memory(tmp_cfg)
    try:
        mem.operational.set_focus(project="memo", summary="Finish the native memory runtime")
        mem.operational.create_handoff(
            project="memo",
            summary="Run the complete verification matrix",
            from_actor="codex",
            to_actor="next-agent",
        )
        mem.operational.add_attention(
            project="memo",
            summary="Review federation ACLs",
            severity="high",
        )
        mem.operational.open_conflict(
            topic="storage-authority",
            summary="Two candidate values need an explicit decision",
            freeze_write=True,
        )

        markdown = "\n".join(operational_briefing_lines(mem))
        assert "Operational continuity" in markdown
        assert "Finish the native memory runtime" in markdown
        assert "Run the complete verification matrix" in markdown
        assert "Review federation ACLs" in markdown
        assert "Open conflicts" in markdown
        assert "write frozen" in markdown
        assert "Memo journal: observed local head=" in markdown
    finally:
        mem.close()


def test_mcp_unified_briefing_returns_native_operational_state(tmp_cfg) -> None:
    mem = Memory(tmp_cfg)
    try:
        mem.operational.set_focus(project="memo", summary="Ship Memo 4")
        server = build_server(memory=mem)
        fn = asyncio.run(server.get_tool("memo_unified_briefing")).fn

        out = fn(cwd=str(tmp_cfg.data_dir / "memo"))

        assert out["available"] is True
        assert "Ship Memo 4" in out["markdown"]
        assert isinstance(out["lines"], list)
        assert out["notification"] == ""
    finally:
        mem.close()


def test_compact_text_preserves_limit_and_ellipsis() -> None:
    compact = compact_text("alpha\n\n" + ("beta " * 200), max_chars=80)
    assert len(compact) <= 80
    assert compact.endswith("…")
    assert "\n\n" not in compact
