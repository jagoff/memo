"""Tests for memo.ocr — pure helpers + cache flow (Apple Vision stubbed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from memo import ocr


def test_vision_available_returns_bool() -> None:
    assert isinstance(ocr.vision_available(), bool)


def test_hash_bytes_matches_sha256(tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(b"hello-bytes")
    assert ocr._hash_bytes(f) == hashlib.sha256(b"hello-bytes").hexdigest()


def test_ocr_enabled_via_env_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_OCR_ENABLED", raising=False)
    assert ocr.ocr_enabled_via_env() is False
    assert ocr.ocr_enabled_via_env(default=True) is True


def test_ocr_enabled_via_env_spellings(monkeypatch) -> None:
    for v in ("1", "true", "ON", "Yes"):
        monkeypatch.setenv("MEMO_OCR_ENABLED", v)
        assert ocr.ocr_enabled_via_env() is True
    for v in ("0", "false", "no", ""):
        monkeypatch.setenv("MEMO_OCR_ENABLED", v)
        assert ocr.ocr_enabled_via_env() is False


def test_extract_text_cached_missing_file_returns_empty(tmp_path: Path) -> None:
    assert ocr.extract_text_cached(tmp_path / "nope.png", cache_dir=tmp_path / "c") == ""


def test_extract_text_cached_writes_then_reads_cache(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "pic.png"
    img.write_bytes(b"PNGDATA")
    cache = tmp_path / "cache"

    calls = {"n": 0}

    def fake_extract(p, *, languages=()):
        calls["n"] += 1
        return "extracted text"

    monkeypatch.setattr(ocr, "extract_text", fake_extract)

    # miss → runs extract_text, caches
    assert ocr.extract_text_cached(img, cache_dir=cache) == "extracted text"
    assert calls["n"] == 1
    # hit → no second extract call
    assert ocr.extract_text_cached(img, cache_dir=cache) == "extracted text"
    assert calls["n"] == 1


def test_extract_text_cached_caches_empty_result(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "blank.png"
    img.write_bytes(b"x")
    cache = tmp_path / "cache"
    calls = {"n": 0}

    def fake_extract(p, *, languages=()):
        calls["n"] += 1
        return ""

    monkeypatch.setattr(ocr, "extract_text", fake_extract)
    assert ocr.extract_text_cached(img, cache_dir=cache) == ""
    assert ocr.extract_text_cached(img, cache_dir=cache) == ""
    assert calls["n"] == 1  # empty result cached, not re-attempted
