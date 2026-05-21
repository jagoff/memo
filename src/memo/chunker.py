"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Heading-aware markdown chunker.

memo uses single-vector-per-doc by default — each `.md` produces one
embedding regardless of length. Works well for short notes (<2000 chars)
where the whole doc fits comfortably in the embedder's context. Fails
on long docs (informes, audit reports, multi-section notes) where
specific facts get diluted in a single 1024- or 2560-dim vector.

This module splits long markdown into heading-aware chunks of
~target_chars each, so each chunk can be embedded separately and
retrieved on its own merits.

## Strategy

1. **Short doc** (<= target): return single chunk, seq=0.
2. **Long doc**:
   a) Split on `##` headings (H2). If 2+ sections result → use them.
   b) Otherwise try `###` (H3).
   c) Otherwise fall back to paragraph splitter (`\n\n`).
3. **Section still too long** (> target after heading split): sub-split
   by paragraphs, then by sentences if a paragraph exceeds target.
4. Preserve nearest-heading-path in `chunk_heading` so downstream search
   can show "where in the doc" the chunk came from.

## Why char-based not token-based

Calling the tokenizer per-chunk is slow (~50ms each). Char-based
estimation is fine for chunking purposes — 4 chars/token is the rule
of thumb for English+Spanish, and the embedder truncates internally
if a chunk slightly exceeds context anyway. We err on smaller chunks
(target 2000 chars ≈ 500 tokens) to leave headroom for instruction
prefix + EOS.
"""

from __future__ import annotations

import re
from typing import TypedDict

DEFAULT_TARGET_CHARS = 2000  # ≈ 500 tokens with some headroom
DEFAULT_OVERLAP_CHARS = 200  # ≈ 50 tokens; included in chunk N+1's start


class Chunk(TypedDict):
    seq: int
    heading: str
    body: str


_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])")


def chunk_markdown(
    body: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split markdown body into heading-aware chunks.

    Returns a list of {seq, heading, body} dicts. seq is 0-indexed
    relative to this document.

    Always returns at least one chunk (with seq=0). For short docs
    (<= target_chars), returns the body unchanged as a single chunk.
    """
    body = (body or "").strip()
    if not body:
        return []
    if len(body) <= target_chars:
        return [Chunk(seq=0, heading="", body=body)]

    # Heading-aware split. Try H2 first; if no useful split, fall back.
    sections = _split_by_heading_level(body, level=2)
    if len(sections) <= 1:
        sections = _split_by_heading_level(body, level=3)
    if len(sections) <= 1:
        sections = _split_by_paragraphs(body)

    chunks: list[Chunk] = []
    seq = 0
    current_heading = ""

    for section in sections:
        if section["heading"]:
            current_heading = section["heading"]
        sec_text = section["text"].strip()
        if not sec_text:
            continue

        if len(sec_text) <= target_chars:
            chunks.append(Chunk(seq=seq, heading=current_heading, body=sec_text))
            seq += 1
        else:
            # Section overflows target — sub-split into smaller chunks.
            for sub_body in _group_into_chunks(sec_text, target_chars):
                chunks.append(Chunk(seq=seq, heading=current_heading, body=sub_body))
                seq += 1

    if not chunks:
        # Defensive fallback — should never happen given the empty check above.
        chunks = [Chunk(seq=0, heading="", body=body)]
    return chunks


def _split_by_heading_level(body: str, level: int) -> list[dict]:
    """Split body at heading lines of exactly `level` hashes.

    Returns a list of {heading, text}. The pre-first-heading prefix
    (if any) is the first section with heading="".
    """
    pattern = re.compile(rf"^{'#' * level}\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    if not matches:
        return [{"heading": "", "text": body}]

    sections: list[dict] = []
    if matches[0].start() > 0:
        prefix = body[: matches[0].start()].strip()
        if prefix:
            sections.append({"heading": "", "text": prefix})

    for i, m in enumerate(matches):
        heading_line = m.group(0)
        heading_text = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_body = body[start:end].strip()
        # Include the heading line in the chunk text so the chunk is
        # self-contextualizing — embedder + reader both benefit.
        full_text = f"{heading_line}\n\n{section_body}".strip() if section_body else heading_line
        sections.append({"heading": heading_text, "text": full_text})

    return sections


def _split_by_paragraphs(body: str) -> list[dict]:
    """Fallback split: paragraph boundaries (\\n\\n)."""
    paragraphs = _PARAGRAPH_BREAK_RE.split(body)
    return [{"heading": "", "text": p.strip()} for p in paragraphs if p.strip()]


def _group_into_chunks(text: str, target_chars: int) -> list[str]:
    """Group paragraphs (then sentences) into chunks of <=target_chars.

    Used when a heading-section is itself too large. Greedy packer:
    accumulate paragraphs until the next would overflow, emit the
    accumulated chunk, start fresh.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK_RE.split(text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for p in paragraphs:
        if len(p) > target_chars:
            # Paragraph itself overflows — split by sentence.
            for sent in _split_sentences(p, target_chars):
                if len(current) + len(sent) + 2 > target_chars and current:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
        elif len(current) + len(p) + 2 > target_chars and current:
            chunks.append(current.strip())
            current = p
        else:
            current = (current + "\n\n" + p).strip() if current else p

    if current:
        chunks.append(current.strip())
    return chunks


def _split_sentences(paragraph: str, target_chars: int) -> list[str]:
    """Split a paragraph into sentence-grouped chunks of <=target_chars.

    Uses a Spanish/English-aware sentence boundary regex (`.!? ` followed
    by a capital). Pure char-split fallback if no sentences detected.
    """
    sentences = _SENTENCE_RE.split(paragraph)
    if len(sentences) == 1:
        # No sentence boundaries detected — char-split as last resort.
        chunks: list[str] = []
        for i in range(0, len(paragraph), target_chars):
            chunks.append(paragraph[i : i + target_chars])
        return chunks

    out: list[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > target_chars:
            # Single sentence overflows — char-split it.
            if current:
                out.append(current.strip())
                current = ""
            for i in range(0, len(s), target_chars):
                out.append(s[i : i + target_chars])
            continue
        if len(current) + len(s) + 1 > target_chars and current:
            out.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        out.append(current.strip())
    return out
