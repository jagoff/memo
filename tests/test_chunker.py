"""Tests for memo.chunker — heading-aware markdown chunking."""

from __future__ import annotations

from memo.chunker import DEFAULT_TARGET_CHARS, chunk_markdown


def test_empty_body_returns_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\t ") == []


def test_short_body_is_single_chunk_unchanged() -> None:
    body = "# Title\n\nA short note."
    chunks = chunk_markdown(body)
    assert len(chunks) == 1
    assert chunks[0]["seq"] == 0
    assert chunks[0]["heading"] == ""
    assert chunks[0]["body"] == body


def test_seq_is_zero_indexed_and_contiguous() -> None:
    # Force multiple chunks with a small target.
    body = "## A\n\n" + ("alpha. " * 60) + "\n\n## B\n\n" + ("beta. " * 60)
    chunks = chunk_markdown(body, target_chars=120)
    assert len(chunks) > 1
    assert [c["seq"] for c in chunks] == list(range(len(chunks)))


def test_h2_split_carries_heading() -> None:
    body = "## First\n\n" + ("x " * 200) + "\n\n## Second\n\n" + ("y " * 200)
    chunks = chunk_markdown(body, target_chars=150)
    headings = {c["heading"] for c in chunks}
    assert "First" in headings
    assert "Second" in headings


def test_heading_line_included_in_chunk_body() -> None:
    body = "## Topic\n\n" + ("word " * 200)
    chunks = chunk_markdown(body, target_chars=150)
    # the chunk under "Topic" keeps the heading line for self-context
    assert any("## Topic" in c["body"] for c in chunks)


def test_oversized_section_subsplits() -> None:
    # Two H2 sections, the second oversized → it sub-splits, keeping its heading.
    body = "## Small\n\nshort.\n\n## Big\n\n" + ("sentence. " * 400)
    chunks = chunk_markdown(body, target_chars=200)
    assert len(chunks) > 2
    big_chunks = [c for c in chunks if c["heading"] == "Big"]
    assert len(big_chunks) > 1  # oversized section produced multiple chunks


def test_default_target_keeps_medium_doc_single() -> None:
    body = "para. " * 100  # well under 2000 chars
    assert len(body) < DEFAULT_TARGET_CHARS
    assert len(chunk_markdown(body)) == 1


def test_paragraph_fallback_when_no_headings() -> None:
    body = ("alpha. " * 80) + "\n\n" + ("beta. " * 80)
    chunks = chunk_markdown(body, target_chars=200)
    assert len(chunks) > 1
    assert all(c["heading"] == "" for c in chunks)
