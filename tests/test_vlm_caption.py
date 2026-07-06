"""VLM caption module — availability guard, SHA256 cache, never-raise contract."""

from __future__ import annotations

from pathlib import Path

import memo.vlm_caption as vc


def test_vlm_flags_registered_default_off():
    from memo.flags import flag_bool, flag_int, flag_str

    assert flag_bool("MEMO_VLM_CAPTION_ENABLED") is False
    assert flag_int("MEMO_VLM_CAPTION_MIN_OCR_CHARS") == 40
    assert flag_str("MEMO_VLM_MODEL") == "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def test_caption_unavailable_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vc, "vlm_available", lambda: False)
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG fake")
    assert vc.caption_image(img) == ""


def test_caption_missing_file_returns_empty(tmp_path: Path):
    assert vc.caption_image_cached(tmp_path / "nope.png", cache_dir=tmp_path / "c") == ""


def test_caption_cached_hits_cache_second_time(tmp_path: Path, monkeypatch):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG fake bytes")
    calls = {"n": 0}

    def _fake_caption(path, *, max_tokens=120):
        calls["n"] += 1
        return "a whiteboard diagram of the sync flow"

    monkeypatch.setattr(vc, "caption_image", _fake_caption)
    cache = tmp_path / "vlm_cache"

    first = vc.caption_image_cached(img, cache_dir=cache)
    second = vc.caption_image_cached(img, cache_dir=cache)

    assert first == second == "a whiteboard diagram of the sync flow"
    assert calls["n"] == 1
    assert len(list(cache.glob("*.caption.json"))) == 1


def test_caption_cached_empty_result_is_cached_too(tmp_path: Path, monkeypatch):
    img = tmp_path / "blank.png"
    img.write_bytes(b"\x89PNG blank")
    calls = {"n": 0}

    def _fake_caption(path, *, max_tokens=120):
        calls["n"] += 1
        return ""

    monkeypatch.setattr(vc, "caption_image", _fake_caption)
    cache = tmp_path / "vlm_cache"

    assert vc.caption_image_cached(img, cache_dir=cache) == ""
    assert vc.caption_image_cached(img, cache_dir=cache) == ""
    assert calls["n"] == 1
