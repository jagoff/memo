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
        return "extracted text", 0.9

    monkeypatch.setattr(ocr, "extract_text_with_confidence", fake_extract)

    # miss → runs OCR, caches
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
        return "", 0.0

    monkeypatch.setattr(ocr, "extract_text_with_confidence", fake_extract)
    assert ocr.extract_text_cached(img, cache_dir=cache) == ""
    assert ocr.extract_text_cached(img, cache_dir=cache) == ""
    assert calls["n"] == 1  # empty result cached, not re-attempted


# --- OCR confidence gate --------------------------------------------------


def test_ocr_min_confidence_default(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_OCR_MIN_CONFIDENCE", raising=False)
    assert ocr.ocr_min_confidence() == ocr._DEFAULT_OCR_MIN_CONFIDENCE


def test_ocr_min_confidence_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_OCR_MIN_CONFIDENCE", "0.7")
    assert ocr.ocr_min_confidence() == 0.7


def test_ocr_min_confidence_clamps_and_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_OCR_MIN_CONFIDENCE", "5")
    assert ocr.ocr_min_confidence() == 1.0
    monkeypatch.setenv("MEMO_OCR_MIN_CONFIDENCE", "nope")
    assert ocr.ocr_min_confidence() == ocr._DEFAULT_OCR_MIN_CONFIDENCE


def test_join_confident_drops_low_confidence_lines() -> None:
    # Mirrors the real distribution: garbled lines at 0.30, legible at 0.50.
    pairs = [
        ("aws", 0.50),
        ("Lambda", 0.50),
        ("IOpth)n+51", 0.30),
        ("U�t*� St•tqSllf", 0.30),
    ]
    out = ocr._join_confident(pairs, 0.4)
    assert out == "aws\nLambda"


def test_join_confident_floor_zero_keeps_all() -> None:
    pairs = [("aws", 0.5), ("garble", 0.1)]
    assert ocr._join_confident(pairs, 0.0) == "aws\ngarble"


def test_join_confident_skips_blank_lines() -> None:
    pairs = [("  ", 0.9), ("real", 0.9)]
    assert ocr._join_confident(pairs, 0.4) == "real"


def test_confidence_floor_in_cache_key(tmp_path: Path, monkeypatch) -> None:
    # Different thresholds → different cache files → re-OCR, not stale serve.
    img = tmp_path / "img.png"
    img.write_bytes(b"bytes")
    cache = tmp_path / "cache"

    monkeypatch.setattr(
        ocr, "extract_text_with_confidence", lambda p, *, languages=(): ("text", 0.9)
    )
    monkeypatch.setenv("MEMO_OCR_MIN_CONFIDENCE", "0.4")
    ocr.extract_text_cached(img, cache_dir=cache)
    monkeypatch.setenv("MEMO_OCR_MIN_CONFIDENCE", "0.7")
    ocr.extract_text_cached(img, cache_dir=cache)
    files = sorted(p.name for p in cache.glob("*.json"))
    assert any(".c40." in f for f in files)
    assert any(".c70." in f for f in files)


# --- whole-image quality (lever 2) ----------------------------------------


def test_mean_conf() -> None:
    assert ocr._mean_conf([("a", 0.3), ("b", 0.5)]) == 0.4
    assert ocr._mean_conf([]) == 0.0


def test_ocr_low_conf_threshold_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_OCR_LOW_CONF_THRESHOLD", raising=False)
    assert ocr.ocr_low_conf_threshold() == ocr._DEFAULT_OCR_LOW_CONF_THRESHOLD
    monkeypatch.setenv("MEMO_OCR_LOW_CONF_THRESHOLD", "0.5")
    assert ocr.ocr_low_conf_threshold() == 0.5


def test_image_health_confidence_flags_low_quality(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_OCR_LOW_CONF_THRESHOLD", "0.6")
    # junk console screenshot (measured mean ~0.45) → down-weighted to its mean
    assert ocr.image_health_confidence(0.45) == 0.45
    # clean image at/above threshold → neutral (None)
    assert ocr.image_health_confidence(0.8) is None
    # floored at 0.1
    assert ocr.image_health_confidence(0.02) == 0.1


def test_image_health_confidence_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_OCR_LOW_CONF_THRESHOLD", "0")
    assert ocr.image_health_confidence(0.45) is None


def test_cached_with_confidence_roundtrip(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "img.png"
    img.write_bytes(b"bytes")
    cache = tmp_path / "cache"
    calls = {"n": 0}

    def fake(p, *, languages=()):
        calls["n"] += 1
        return "clean text", 0.42

    monkeypatch.setattr(ocr, "extract_text_with_confidence", fake)
    monkeypatch.delenv("MEMO_OCR_MIN_CONFIDENCE", raising=False)
    t, c = ocr.extract_text_cached_with_confidence(img, cache_dir=cache)
    assert (t, c) == ("clean text", 0.42)
    # second call → cache hit, no re-OCR, confidence preserved
    t2, c2 = ocr.extract_text_cached_with_confidence(img, cache_dir=cache)
    assert (t2, c2) == ("clean text", 0.42)
    assert calls["n"] == 1
