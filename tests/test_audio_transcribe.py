"""Audio transcription module — availability guard, SHA256 cache, never-raise."""

from __future__ import annotations

from pathlib import Path

import memo.audio_transcribe as at


def test_whisper_model_flag_registered_default_empty():
    from memo.flags import flag_str

    assert flag_str("MEMO_WHISPER_MODEL") == ""


def test_audio_extensions_cover_common_formats():
    assert {".m4a", ".mp3", ".wav"} <= at.AUDIO_EXTENSIONS


def test_transcribe_unavailable_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(at, "whisper_available", lambda: False)
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"fake-aac")
    assert at.transcribe_audio(audio) == ""


def test_transcribe_cached_missing_file_returns_empty(tmp_path: Path):
    assert at.transcribe_audio_cached(tmp_path / "nope.m4a", cache_dir=tmp_path / "c") == ""


def test_transcribe_cached_hits_cache_second_time(tmp_path: Path, monkeypatch):
    audio = tmp_path / "standup.m4a"
    audio.write_bytes(b"fake-aac-bytes")
    calls = {"n": 0}

    def _fake_transcribe(path):
        calls["n"] += 1
        return "decidimos migrar el deploy a uv"

    monkeypatch.setattr(at, "transcribe_audio", _fake_transcribe)
    cache = tmp_path / "audio_cache"

    first = at.transcribe_audio_cached(audio, cache_dir=cache)
    second = at.transcribe_audio_cached(audio, cache_dir=cache)

    assert first == second == "decidimos migrar el deploy a uv"
    assert calls["n"] == 1
    assert len(list(cache.glob("*.transcript.json"))) == 1
