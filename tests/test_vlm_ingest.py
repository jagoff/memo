"""VLM caption wiring in ingest — gating helper + enrich_with_ocr fallback blocks."""

from __future__ import annotations

from pathlib import Path

import memo.ingest_helpers as ih
from memo.ingest_helpers import caption_if_ocr_weak


def _vault_with_image(tmp_path: Path) -> tuple[Path, Path, Path]:
    vault = tmp_path / "vault"
    vault.mkdir()
    img = vault / "shot.png"
    img.write_bytes(b"\x89PNG fake image bytes")
    note = vault / "note.md"
    note.write_text("Diagram of the sync flow:\n\n![[shot.png]]\n", encoding="utf-8")
    return vault, note, img


def test_gate_returns_empty_when_flag_off(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMO_VLM_CAPTION_ENABLED", raising=False)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert caption_if_ocr_weak(img, "", tmp_path / "state") == ""


def test_gate_skips_caption_when_ocr_rich(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_VLM_CAPTION_ENABLED", "1")
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    # >= MEMO_VLM_CAPTION_MIN_OCR_CHARS (default 40) → no caption call
    assert caption_if_ocr_weak(img, "x" * 200, tmp_path / "state") == ""


def test_gate_captions_when_flag_on_and_ocr_weak(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_VLM_CAPTION_ENABLED", "1")
    monkeypatch.setattr(
        "memo.vlm_caption.caption_image_cached",
        lambda p, *, cache_dir: "flowchart of memo sync tiers",
    )
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert caption_if_ocr_weak(img, "", tmp_path / "state") == "flowchart of memo sync tiers"


def test_enrich_appends_vlm_block_when_ocr_empty(tmp_path: Path, monkeypatch):
    vault, note, img = _vault_with_image(tmp_path)
    monkeypatch.setenv("MEMO_VLM_CAPTION_ENABLED", "1")
    monkeypatch.setattr(ih, "extract_text_cached", lambda p, *, cache_dir: "")
    monkeypatch.setattr(
        "memo.vlm_caption.caption_image_cached",
        lambda p, *, cache_dir: "flowchart of memo sync tiers",
    )

    enriched, resolved, hashes = ih.enrich_with_ocr(
        note.read_text(), note, vault, tmp_path / "state"
    )

    assert "<!-- VLM: shot.png -->" in enriched
    assert "flowchart of memo sync tiers" in enriched
    assert resolved == [img]
    assert len(hashes) == 1


def test_enrich_unchanged_when_flag_off_and_no_ocr(tmp_path: Path, monkeypatch):
    vault, note, img = _vault_with_image(tmp_path)
    monkeypatch.delenv("MEMO_VLM_CAPTION_ENABLED", raising=False)
    monkeypatch.setattr(ih, "extract_text_cached", lambda p, *, cache_dir: "")

    enriched, resolved, _ = ih.enrich_with_ocr(note.read_text(), note, vault, tmp_path / "state")

    assert "<!-- VLM:" not in enriched
    assert enriched == note.read_text()
    assert resolved == [img]
