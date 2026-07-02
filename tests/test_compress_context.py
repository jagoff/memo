"""Tests for `memo compress-context` command (cli_compress_context.py)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from memo.cli_compress_context import compress, compress_context_cmd

# ---------------------------------------------------------------------------
# Unit tests for the compress() function
# ---------------------------------------------------------------------------


def test_horizontal_rules_removed() -> None:
    content = "# Heading\n\n---\n\nSome text.\n"
    result = compress(content)
    assert "---" not in result
    assert "# Heading" in result
    assert "Some text." in result


def test_horizontal_rules_not_greedy() -> None:
    """Lines that are not exactly '---' are kept."""
    content = "----\n---extra\nsome --- text\n"
    result = compress(content)
    # '----' is not exactly '---'
    assert "----" in result
    # '---extra' is not exactly '---'
    assert "---extra" in result


def test_long_list_item_truncated() -> None:
    long_body = "x" * 130
    content = f"- {long_body}\n"
    result = compress(content)
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].endswith("…")
    # Total line length should be <= 120 + len("- ") but in practice the prefix is part of 120
    assert len(lines[0]) <= 121  # 120 chars + "…" = up to 121


def test_long_list_item_truncated_at_word_boundary() -> None:
    content = "- word1 word2 word3 " + "a" * 100 + " lastword\n"
    result = compress(content)
    line = result.strip()
    assert line.endswith("…")
    # Should not cut in the middle of a word
    text_without_ellipsis = line[:-1]  # remove …
    # Ensure the last character before … is not in the middle of a word
    assert not text_without_ellipsis[-1].isalpha() or text_without_ellipsis.endswith(" ")


def test_short_list_item_unchanged() -> None:
    content = "- short item\n"
    assert compress(content) == "- short item\n"


def test_nested_list_item_truncated() -> None:
    long_body = "y" * 130
    content = f"  - {long_body}\n"
    result = compress(content)
    line = result.strip()
    assert line.endswith("…")


def test_numbered_list_item_truncated() -> None:
    long_body = "z" * 130
    content = f"1. {long_body}\n"
    result = compress(content)
    line = result.strip()
    assert line.endswith("…")


def test_blockquote_truncated_at_100() -> None:
    long_text = "q" * 150
    content = f"> {long_text}\n"
    result = compress(content)
    line = result.strip()
    assert line.startswith("> ")
    assert line.endswith("…")
    # text part (after "> ") should be 100 chars + "…"
    text_part = line[2:]
    assert len(text_part) == 101  # 100 chars + "…"


def test_blockquote_short_unchanged() -> None:
    content = "> short quote\n"
    assert compress(content) == "> short quote\n"


def test_html_comment_lines_removed() -> None:
    content = "# Title\n<!-- this is a comment -->\nSome text.\n"
    result = compress(content)
    assert "<!--" not in result
    assert "# Title" in result
    assert "Some text." in result


def test_html_comment_not_removed_if_not_full_line() -> None:
    """Inline comments within a line are not removed."""
    content = "text <!-- inline --> more text\n"
    result = compress(content)
    assert "<!-- inline -->" in result


def test_multiple_blank_lines_collapsed() -> None:
    content = "line1\n\n\n\nline2\n"
    result = compress(content)
    assert "\n\n\n" not in result
    assert "line1" in result
    assert "line2" in result


def test_two_blank_lines_become_one() -> None:
    content = "a\n\n\nb\n"
    result = compress(content)
    assert result == "a\n\nb\n"


def test_single_blank_line_preserved() -> None:
    content = "a\n\nb\n"
    assert compress(content) == "a\n\nb\n"


def test_trailing_whitespace_stripped() -> None:
    content = "line with spaces   \nanother line  \n"
    result = compress(content)
    for line in result.splitlines():
        assert line == line.rstrip()


def test_idempotent() -> None:
    """Compressing twice yields the same result."""
    content = (
        "# Title\n\n---\n\n- " + "word " * 30 + "\n\n\n\nSome text.  \n"
        "> " + "q" * 150 + "\n"
        "<!-- comment -->\n"
    )
    once = compress(content)
    twice = compress(once)
    assert once == twice


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    target = tmp_path / "context.md"
    original = "# Title\n\n---\n\nSome text.\n"
    target.write_text(original, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(compress_context_cmd, ["--dry-run", str(target)])

    assert result.exit_code == 0
    # File must be unchanged
    assert target.read_text(encoding="utf-8") == original
    # Combined output should contain the stats line
    assert "Compressed" in result.output


def test_backup_creates_orig(tmp_path: Path) -> None:
    target = tmp_path / "context.md"
    original = "# Title\n\n---\n\nSome text.\n"
    target.write_text(original, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(compress_context_cmd, ["--backup", str(target)])

    assert result.exit_code == 0
    backup = tmp_path / "context.md.orig"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original


def test_default_overwrites_file(tmp_path: Path) -> None:
    target = tmp_path / "context.md"
    original = "# Title\n\n---\n\nSome text.\n"
    target.write_text(original, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(compress_context_cmd, [str(target)])

    assert result.exit_code == 0
    written = target.read_text(encoding="utf-8")
    assert "---" not in written
    assert "Some text." in written


def test_stats_output_present(tmp_path: Path) -> None:
    target = tmp_path / "context.md"
    original = "# Title\n\n---\n\nSome text.\n"
    target.write_text(original, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(compress_context_cmd, ["--dry-run", str(target)])

    assert result.exit_code == 0
    assert "Compressed" in result.output
    assert "→" in result.output


def test_cli_idempotent(tmp_path: Path) -> None:
    """Running the CLI twice produces the same file."""
    content = "# Title\n\n---\n\n- " + "word " * 30 + "\n\n\n\nSome text.  \n"
    target = tmp_path / "context.md"
    target.write_text(content, encoding="utf-8")

    runner = CliRunner()
    runner.invoke(compress_context_cmd, [str(target)])
    after_first = target.read_text(encoding="utf-8")

    runner.invoke(compress_context_cmd, [str(target)])
    after_second = target.read_text(encoding="utf-8")

    assert after_first == after_second
