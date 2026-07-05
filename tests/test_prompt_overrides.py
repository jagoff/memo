"""Prompt override resolution — default -> state_dir/prompts/<name>.md."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_prompt_default_when_no_override(tmp_path: Path):
    from memo.prompt_overrides import resolve_prompt

    assert resolve_prompt("ask", "DEFAULT TEXT", tmp_path) == "DEFAULT TEXT"


def test_resolve_prompt_reads_override_file(tmp_path: Path):
    from memo.prompt_overrides import resolve_prompt

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "ask.md").write_text("OVERRIDE", encoding="utf-8")
    assert resolve_prompt("ask", "DEFAULT", tmp_path) == "OVERRIDE"


def test_resolve_prompt_rejects_unknown_name(tmp_path: Path):
    from memo.prompt_overrides import resolve_prompt

    with pytest.raises(ValueError, match="unknown prompt name"):
        resolve_prompt("nope", "x", tmp_path)


def test_empty_override_falls_back_to_default(tmp_path: Path):
    from memo.prompt_overrides import resolve_prompt

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "ask.md").write_text("   \n", encoding="utf-8")
    assert resolve_prompt("ask", "DEFAULT", tmp_path) == "DEFAULT"


def test_prompt_version_changes_with_override(tmp_path: Path):
    from memo.prompt_overrides import prompt_version

    v_default = prompt_version("ask", "DEFAULT", tmp_path)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "ask.md").write_text("OVERRIDE", encoding="utf-8")
    assert prompt_version("ask", "DEFAULT", tmp_path) != v_default


def test_extract_insights_uses_override_prompt(tmp_path: Path):
    from memo.capture import extract_insights

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "capture_extract.md").write_text(
        "CUSTOM EXTRACTOR", encoding="utf-8"
    )
    seen: dict = {}

    class _Chat:
        def chat(self, model, messages, options):
            seen["system"] = messages[0]["content"]
            return {"message": {"content": "[]"}}

    extract_insights(_Chat(), "m", "user text", "assistant text", state_dir=tmp_path)
    assert seen["system"] == "CUSTOM EXTRACTOR"


_CALL_SITES = [
    ("src/memo/memory/ask_ops.py", "ask"),
    ("src/memo/memory/consolidate_ops.py", "consolidate"),
    ("src/memo/memory/consolidate_ops.py", "synthesis"),
    ("src/memo/memory/write_ops.py", "derive"),
    ("src/memo/memory/maintain_ops.py", "extract_entities"),
    ("src/memo/cli_transcripts.py", "reflect"),
    ("src/memo/temporal.py", "contradiction"),
    ("src/memo/proactive.py", "suggestion"),
    ("src/memo/consolidation.py", "merge"),
]


def test_every_llm_call_site_resolves_its_prompt():
    """Guard: no chat() call may pass a raw prompt constant — each must route
    through resolve_prompt("<name>", ...) so user overrides take effect."""
    root = Path(__file__).resolve().parents[1]
    for rel, name in _CALL_SITES:
        text = (root / rel).read_text(encoding="utf-8")
        assert f'resolve_prompt("{name}"' in text, (
            f"{rel}: chat() call does not resolve prompt {name!r} via prompt_overrides"
        )
