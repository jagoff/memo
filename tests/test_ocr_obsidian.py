"""Tests for OCR + Obsidian image embed resolution.

Vision-dependent tests are skipped automatically when PyObjC isn't
installed (Linux CI etc).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.obsidian_links import (
    find_image_embeds,
    resolve_attachment_folder,
    resolve_image_path,
)
from memo.ocr import extract_text, extract_text_cached, vision_available


# ---------- find_image_embeds ----------


def test_find_image_embeds_basic() -> None:
    body = "![[hello.png]]"
    assert find_image_embeds(body) == ["hello.png"]


def test_find_image_embeds_with_alias() -> None:
    body = "![[hello.png|alt text]]"
    assert find_image_embeds(body) == ["hello.png"]


def test_find_image_embeds_multiple_dedup() -> None:
    body = "![[a.png]]\n\n![[b.jpg]]\n\n![[a.png]]"
    assert find_image_embeds(body) == ["a.png", "b.jpg"]


def test_find_image_embeds_skips_markdown_links() -> None:
    body = "![alt](https://example.com/x.png) and [foo](bar.png)"
    assert find_image_embeds(body) == []


def test_find_image_embeds_skips_note_embeds() -> None:
    """`![[note.md]]` is an Obsidian note transclude, not an image."""
    body = "![[some-note.md]]"
    assert find_image_embeds(body) == []


def test_find_image_embeds_case_insensitive_ext() -> None:
    body = "![[Foo.PNG]] ![[Bar.JPEG]]"
    assert find_image_embeds(body) == ["Foo.PNG", "Bar.JPEG"]


def test_find_image_embeds_empty_input() -> None:
    assert find_image_embeds("") == []
    assert find_image_embeds("no images here") == []


def test_find_image_embeds_handles_spaces_in_filename() -> None:
    body = "![[Captura de pantalla 2026-03-30.png]]"
    assert find_image_embeds(body) == ["Captura de pantalla 2026-03-30.png"]


# ---------- resolve_attachment_folder ----------


def test_resolve_attachment_folder_reads_app_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "Attachments").mkdir()
    (vault / ".obsidian" / "app.json").write_text(
        json.dumps({"attachmentFolderPath": "Attachments"}),
        encoding="utf-8",
    )
    assert resolve_attachment_folder(vault) == (vault / "Attachments").resolve()


def test_resolve_attachment_folder_missing_config(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    assert resolve_attachment_folder(vault) is None


def test_resolve_attachment_folder_invalid_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".obsidian" / "app.json").write_text(
        json.dumps({"attachmentFolderPath": "Does/Not/Exist"}),
        encoding="utf-8",
    )
    assert resolve_attachment_folder(vault) is None


# ---------- resolve_image_path ----------


def test_resolve_image_path_uses_attachment_folder(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "Attachments").mkdir()
    (vault / ".obsidian" / "app.json").write_text(
        json.dumps({"attachmentFolderPath": "Attachments"}),
        encoding="utf-8",
    )
    target = vault / "Attachments" / "foo.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = resolve_image_path("foo.png", vault)
    assert out == target


def test_resolve_image_path_falls_back_to_note_dir(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note_dir = vault / "notes"
    note_dir.mkdir(parents=True)
    target = note_dir / "bar.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = resolve_image_path("bar.png", vault, note_dir=note_dir)
    assert out == target


def test_resolve_image_path_rglob_fallback(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    deep = vault / "a" / "b" / "c"
    deep.mkdir(parents=True)
    target = deep / "deep.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = resolve_image_path("deep.png", vault)
    assert out == target


def test_resolve_image_path_returns_none_when_missing(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    assert resolve_image_path("nope.png", vault) is None


# ---------- extract_text + cache ----------


_VISION_AVAILABLE = vision_available()


@pytest.mark.skipif(not _VISION_AVAILABLE, reason="Apple Vision (PyObjC) not installed")
def test_extract_text_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert extract_text(tmp_path / "missing.png") == ""


def test_extract_text_cached_caches_empty_for_unavailable(tmp_path: Path) -> None:
    """When Vision can't read the file (or isn't installed), cache
    still stores an empty result so we don't re-attempt repeatedly."""
    fake_img = tmp_path / "x.png"
    fake_img.write_bytes(b"not really a png")
    cache_dir = tmp_path / "cache"
    out = extract_text_cached(fake_img, cache_dir=cache_dir)
    assert out == ""
    cache_files = list(cache_dir.glob("*.txt"))
    assert len(cache_files) == 1
    # Second call reads from cache (deterministic, no re-OCR attempted).
    out2 = extract_text_cached(fake_img, cache_dir=cache_dir)
    assert out2 == ""


def test_extract_text_cached_missing_file_returns_empty(tmp_path: Path) -> None:
    assert extract_text_cached(tmp_path / "nope.png", cache_dir=tmp_path / "cache") == ""
