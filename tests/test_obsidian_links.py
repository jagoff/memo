"""Tests for memo.obsidian_links — Obsidian image-embed resolution."""

from __future__ import annotations

import json
from pathlib import Path

from memo.obsidian_links import (
    find_image_embeds,
    resolve_attachment_folder,
    resolve_image_path,
)


def test_find_embeds_basic() -> None:
    body = "Look ![[diagram.png]] and ![[photo.jpg]]."
    assert find_image_embeds(body) == ["diagram.png", "photo.jpg"]


def test_find_embeds_ignores_alias_and_dedups() -> None:
    body = "![[a.png|320]] then again ![[a.png]] and ![[b.webp|alt text]]"
    assert find_image_embeds(body) == ["a.png", "b.webp"]


def test_find_embeds_ignores_markdown_image_syntax() -> None:
    # ![alt](url) is NOT an Obsidian embed
    assert find_image_embeds("![alt](https://x/y.png)") == []


def test_find_embeds_empty_and_non_image() -> None:
    assert find_image_embeds("") == []
    assert find_image_embeds("![[notes.md]] ![[sheet.pdf]]") == []


def test_find_embeds_case_insensitive_ext() -> None:
    assert find_image_embeds("![[IMG.PNG]] ![[v.JPEG]]") == ["IMG.PNG", "v.JPEG"]


def test_resolve_attachment_folder_missing_config(tmp_path: Path) -> None:
    assert resolve_attachment_folder(tmp_path) is None


def test_resolve_attachment_folder_reads_app_json(tmp_path: Path) -> None:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text(
        json.dumps({"attachmentFolderPath": "assets"}), encoding="utf-8"
    )
    attach = tmp_path / "assets"
    attach.mkdir()
    assert resolve_attachment_folder(tmp_path) == attach.resolve()


def test_resolve_attachment_folder_unparseable_returns_none(tmp_path: Path) -> None:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text("{not json", encoding="utf-8")
    assert resolve_attachment_folder(tmp_path) is None


def test_resolve_image_path_prefers_attachment_folder(tmp_path: Path) -> None:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text(
        json.dumps({"attachmentFolderPath": "assets"}), encoding="utf-8"
    )
    (tmp_path / "assets").mkdir()
    img = tmp_path / "assets" / "pic.png"
    img.write_bytes(b"\x89PNG")
    assert resolve_image_path("pic.png", tmp_path) == img


def test_resolve_image_path_note_dir_fallback(tmp_path: Path) -> None:
    note_dir = tmp_path / "notes"
    note_dir.mkdir()
    img = note_dir / "local.png"
    img.write_bytes(b"x")
    assert resolve_image_path("local.png", tmp_path, note_dir=note_dir) == img


def test_resolve_image_path_rglob_fallback(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    img = deep / "buried.png"
    img.write_bytes(b"x")
    assert resolve_image_path("buried.png", tmp_path) == img


def test_resolve_image_path_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_image_path("nope.png", tmp_path) is None
    assert resolve_image_path("", tmp_path) is None
