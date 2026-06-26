"""`memo compress-context` — rule-based compression of context/markdown files.

Reduces file size through deterministic, lossless-ish transformations:
  1. Remove horizontal rules (lines that are exactly ``---``)
  2. Truncate list items >120 chars at the last word boundary before char 120, append ``…``
  3. Compress blockquotes: keep only first 100 chars of the text part, append ``…`` if truncated
  4. Remove HTML comment-only lines (``^<!--.*-->$``)
  5. Collapse 2+ consecutive blank lines → 1 blank line
  6. Strip trailing whitespace per line

No LLM — fully deterministic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click

# --- compression rules ---

_HORIZ_RULE = re.compile(r"^---\s*$")
_HTML_COMMENT = re.compile(r"^<!--.*-->$")
# List item prefixes: ``- ``, ``* ``, ``  - `` (nested), ``  * ``, numbered ``1. `` etc.
_LIST_ITEM = re.compile(r"^(\s*(?:[-*]|\d+\.)\s)")
_BLOCKQUOTE = re.compile(r"^(>\s?)(.*)")

_LIST_TRUNCATE_AT = 120
_BLOCKQUOTE_TRUNCATE_AT = 100


def _truncate_at_word(text: str, max_len: int) -> str:
    """Truncate *text* to at most *max_len* chars at a word boundary, appending ``…``."""
    if len(text) <= max_len:
        return text
    # Find last space at or before max_len
    cut = text.rfind(" ", 0, max_len)
    if cut <= 0:
        cut = max_len
    return text[:cut] + "…"


def compress(content: str) -> str:
    """Apply all compression rules and return the compressed string."""
    lines = content.splitlines()
    out: list[str] = []
    blank_run = 0

    for line in lines:
        # 1. Strip trailing whitespace
        line = line.rstrip()

        # 2. Remove horizontal rules
        if _HORIZ_RULE.match(line):
            continue

        # 3. Remove HTML comment-only lines
        if _HTML_COMMENT.match(line):
            continue

        # 4. Truncate long list items
        m_list = _LIST_ITEM.match(line)
        if m_list and len(line) > _LIST_TRUNCATE_AT:
            prefix = m_list.group(1)
            body = line[len(prefix):]
            truncated_body = _truncate_at_word(body, _LIST_TRUNCATE_AT - len(prefix))
            line = prefix + truncated_body

        # 5. Compress blockquotes
        m_bq = _BLOCKQUOTE.match(line)
        if m_bq:
            marker = m_bq.group(1)
            text = m_bq.group(2)
            if len(text) > _BLOCKQUOTE_TRUNCATE_AT:
                text = text[:_BLOCKQUOTE_TRUNCATE_AT] + "…"
            line = marker + text

        # 6. Collapse consecutive blank lines
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0

        out.append(line)

    result = "\n".join(out)
    # Preserve a single trailing newline if original had one
    if content.endswith("\n"):
        result += "\n"
    return result


# --- CLI command ---


@click.command(name="compress-context")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Print compressed output to stdout; do not write.")
@click.option("--backup", is_flag=True, help="Save original as <path>.orig before overwriting.")
def compress_context_cmd(path: Path, dry_run: bool, backup: bool) -> None:
    """Compress a context/markdown file using rule-based transformations.

    Removes horizontal rules, truncates long list items, compresses
    blockquotes, strips HTML comments, collapses blank lines, and strips
    trailing whitespace. No LLM — deterministic.

    Examples:

      memo compress-context context.md

      memo compress-context context.md --dry-run

      memo compress-context context.md --backup
    """
    original = path.read_text(encoding="utf-8")
    compressed = compress(original)

    before = len(original)
    after = len(compressed)
    pct = round((1 - after / before) * 100) if before > 0 else 0

    if dry_run:
        click.echo(compressed, nl=False)
        click.echo(
            f"Compressed {path}: {before} → {after} chars ({pct}% reduction)",
            file=sys.stderr,
        )
        return

    if backup:
        backup_path = path.with_suffix(path.suffix + ".orig")
        backup_path.write_text(original, encoding="utf-8")

    path.write_text(compressed, encoding="utf-8")
    click.echo(
        f"Compressed {path}: {before} → {after} chars ({pct}% reduction)",
        file=sys.stderr,
    )
