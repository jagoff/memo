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
        assert "lines" not in out, "markdown must not be shipped twice"
        assert out["notification"] == ""
    finally:
        mem.close()


def test_compact_text_preserves_limit_and_ellipsis() -> None:
    compact = compact_text("alpha\n\n" + ("beta " * 200), max_chars=80)
    assert len(compact) <= 80
    assert compact.endswith("…")
    assert "\n\n" not in compact


def test_budget_sections_gives_every_section_a_share() -> None:
    """A 4,000-char first section must not consume a 900-char briefing whole.

    Regression: `compose_unified_briefing` concatenated every section and then
    hard-truncated the join, so the profile block (section 0, ~4.6k chars) ate
    100% of the budget and all 10 later sections were dropped mid-word.
    """
    from memo.briefing import budget_sections

    lines = ["# Huge"] + [f"- filler line {i} padded out to be wide" * 2 for i in range(120)]
    lines += ["### Small A", "- a1", "- a2"]
    lines += ["### Small B", "- b1"]
    lines += ["### Small C", "- c1"]

    out = budget_sections(lines, max_chars=900)

    assert len(out) <= 900
    for heading in ("# Huge", "### Small A", "### Small B", "### Small C"):
        assert heading in out, f"{heading} was starved out of the budget"


def test_budget_sections_keeps_whole_lines_not_mid_word_cuts() -> None:
    """Each section truncates at line granularity, so no bullet is cut mid-word."""
    from memo.briefing import budget_sections

    lines = ["### S", "- keep this whole", "- and this one", "- " + "x" * 400]
    out = budget_sections(lines, max_chars=60)

    assert "- keep this whole" in out
    for line in out.splitlines():
        assert line in lines, f"line {line!r} is a partial cut of a source line"


def test_budget_sections_drops_a_section_that_cannot_fit_its_heading() -> None:
    """A naked heading carries no information, so it is dropped rather than kept."""
    from memo.briefing import budget_sections

    out = budget_sections(["### Kept", "- v", "### AVeryLongHeadingThatCannotFit"], max_chars=14)

    assert "### AVeryLongHeadingThatCannotFit" not in out
    assert out.strip(), "budget large enough for the first section should emit it"


def test_unified_briefing_surfaces_more_than_one_section(tmp_cfg) -> None:
    """End-to-end: the composed briefing must carry several sections, not one."""
    from memo.briefing import budget_sections

    lines = ["# Profile — global"] + [
        f"- profile detail {i} spelled out at length" for i in range(90)
    ]
    lines += ["### Knowledge map (your hubs)", "- hub one", "- hub two"]
    lines += ["### Temporal facts", "- fact one"]
    lines += ["### Operational continuity", "- focus: ship the thing"]

    out = budget_sections(lines, max_chars=900)
    surviving = [line for line in out.splitlines() if line.startswith("#")]

    assert len(surviving) >= 4, f"only {len(surviving)} sections survived: {surviving}"
