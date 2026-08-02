"""Audio transcription via mlx-whisper (optional dependency).

Vault audio files (voice memos, meeting recordings) become searchable text
through the SAME ingest pipeline as notes/PDFs, keeping the raw file as
provenance (`abs_path`). Mirrors the `ocr.py` / `vlm_caption.py` contract:
guarded optional import, empty string on ANY failure (never raises),
SHA256 byte cache, deferred mlx imports. Ingest-time only — never the
recall-hook path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from pathlib import Path

__all__ = [
    "AUDIO_EXTENSIONS",
    "transcribe_audio",
    "transcribe_audio_cached",
    "whisper_available",
]

_log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac", ".opus"})

_WHISPER_CHECKED = False
_WHISPER_OK = False


def whisper_available() -> bool:
    """True if mlx-whisper can be imported in this interpreter."""
    global _WHISPER_CHECKED, _WHISPER_OK
    if _WHISPER_CHECKED:
        return _WHISPER_OK
    _WHISPER_CHECKED = True
    try:
        import mlx_whisper  # type: ignore[import-not-found,import-untyped]  # noqa: F401

        _WHISPER_OK = True
    except ImportError as exc:
        _log.debug("mlx-whisper not installed: %s", exc)
        _WHISPER_OK = False
    except Exception as exc:
        _log.warning("mlx-whisper import failed: %s", exc)
        _WHISPER_OK = False
    return _WHISPER_OK


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe `audio_path` with mlx-whisper. Empty string on any failure."""
    if not whisper_available():
        return ""
    p = Path(audio_path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        import mlx_whisper  # type: ignore[import-not-found,import-untyped]

        from memo.flags import flag_str

        model = flag_str("MEMO_WHISPER_MODEL")
        kwargs = {"path_or_hf_repo": model} if model else {}
        result = mlx_whisper.transcribe(str(p), **kwargs)
        return str((result or {}).get("text") or "").strip()
    except Exception as exc:
        _log.debug("whisper transcription failed for %s: %s", p, exc)
        return ""


def _hash_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def transcribe_audio_cached(audio_path: Path, *, cache_dir: Path) -> str:
    """Transcription with SHA256 caching at `<cache_dir>/<hash32>.transcript.json`."""
    p = Path(audio_path)
    if not p.exists():
        return ""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _hash_bytes(p)
    except OSError as exc:
        _log.debug("whisper: cache unavailable, bypassing: %s", exc)
        return transcribe_audio(p)
    cache_path = cache_dir / f"{digest[:32]}.transcript.json"
    if cache_path.exists():
        try:
            return str(json.loads(cache_path.read_text(encoding="utf-8")).get("t", ""))
        except (OSError, ValueError) as exc:
            _log.debug("whisper: cache read failed, re-transcribing: %s", exc)
    text = transcribe_audio(p)
    with contextlib.suppress(Exception):
        cache_path.write_text(json.dumps({"t": text}), encoding="utf-8")
    return text
