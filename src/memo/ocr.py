"""OCR de imágenes vía Apple Vision (PyObjC).

Usado por el indexer para extraer texto de screenshots/imágenes embebidas
en notas Obsidian (`![[image.png]]`). Si PyObjC `Vision` no está instalado
o el archivo no se puede leer, devuelve string vacío sin fallar — el
indexer sigue con el body original.

Cache key = SHA256 de los bytes de la imagen, persistido en
`<cache_dir>/<hash>.txt`. Idempotente entre reindex full y forzado.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path

__all__ = ["extract_text", "extract_text_cached", "vision_available"]

_log = logging.getLogger(__name__)

_VISION_CHECKED = False
_VISION_OK = False


def vision_available() -> bool:
    """True si PyObjC Vision se puede importar en este intérprete."""
    global _VISION_CHECKED, _VISION_OK
    if _VISION_CHECKED:
        return _VISION_OK
    _VISION_CHECKED = True
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
        _VISION_OK = True
    except ImportError as exc:
        _log.debug("Apple Vision not installed: %s", exc)
        _VISION_OK = False
    except Exception as exc:
        _log.warning("Apple Vision import failed: %s", exc)
        _VISION_OK = False
    return _VISION_OK


def extract_text(
    image_path: Path,
    *,
    languages: Iterable[str] = ("es-ES", "en-US"),
) -> str:
    """Run Apple Vision text recognition on ``image_path``.

    Returns concatenated recognized strings joined by newlines, or
    empty string on any failure (missing file, Vision unavailable,
    decode error). Never raises.
    """
    if not vision_available():
        return ""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        import Quartz
        import Vision
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(p))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if not src or Quartz.CGImageSourceGetCount(src) == 0:
            return ""
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg_image is None:
            return ""

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None,
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        # Methods may be unavailable on older macOS versions.
        with contextlib.suppress(AttributeError):
            request.setRecognitionLevel_(1)  # VNRequestTextRecognitionLevelAccurate
        with contextlib.suppress(AttributeError):
            request.setUsesLanguageCorrection_(True)
        with contextlib.suppress(AttributeError, TypeError):
            request.setRecognitionLanguages_(list(languages))

        ok, err = handler.performRequests_error_([request], None)
        if not ok:
            _log.debug("Vision performRequests failed for %s: %s", p, err)
            return ""
        results = request.results() or []
        chunks: list[str] = []
        for obs in results:
            try:
                top = obs.topCandidates_(1)
                if top and len(top) > 0:
                    text = str(top[0].string())
                    if text.strip():
                        chunks.append(text)
            except Exception:
                continue
        return "\n".join(chunks).strip()
    except Exception as exc:
        _log.debug("OCR error on %s: %s", p, exc)
        return ""


def _hash_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_cached(
    image_path: Path,
    *,
    cache_dir: Path,
    languages: Iterable[str] = ("es-ES", "en-US"),
) -> str:
    """OCR with SHA256-based caching at ``<cache_dir>/<hash>.txt``.

    Cache hit → read file. Miss → run :func:`extract_text`, write cache
    (empty string also cached so repeated calls don't re-attempt failing
    images). On any I/O error returns empty string without raising.
    """
    p = Path(image_path)
    if not p.exists():
        return ""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return extract_text(p, languages=languages)
    try:
        digest = _hash_bytes(p)
    except Exception:
        return extract_text(p, languages=languages)
    cache_path = cache_dir / f"{digest[:32]}.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass
    text = extract_text(p, languages=languages)
    with contextlib.suppress(Exception):
        cache_path.write_text(text, encoding="utf-8")
    return text


def ocr_enabled_via_env(default: bool = False) -> bool:
    raw = os.environ.get("MEMO_OCR_ENABLED", "1" if default else "0").strip().lower()
    return raw in {"1", "true", "on", "yes"}
