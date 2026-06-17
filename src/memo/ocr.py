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
import json
import logging
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "extract_text",
    "extract_text_cached",
    "extract_text_cached_with_confidence",
    "extract_text_with_confidence",
    "image_health_confidence",
    "ocr_low_conf_threshold",
    "ocr_min_confidence",
    "vision_available",
]

_log = logging.getLogger(__name__)

# Apple Vision (accurate mode) returns a per-line confidence in [0,1]. Empirically
# on low-res console screenshots clean lines top out at ~0.50 while mojibake
# (`IOpth)n+51`, `U�t*� St•tqSllf`) sits at ~0.30; legible text
# screenshots score 0.9-1.0. A 0.40 floor strips the garbled lines that pollute
# embeddings/BM25 without dropping the readable ones. 0 disables the filter.
_DEFAULT_OCR_MIN_CONFIDENCE = 0.4


def ocr_min_confidence(default: float = _DEFAULT_OCR_MIN_CONFIDENCE) -> float:
    """Per-line OCR confidence floor. Lines below this are dropped before the
    text is indexed, so mojibake doesn't pollute retrieval. Override with
    ``MEMO_OCR_MIN_CONFIDENCE`` (0 disables)."""
    from memo.flags import active_flags

    raw = active_flags().get("MEMO_OCR_MIN_CONFIDENCE")
    if raw is None:
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default


def _join_confident(pairs: list[tuple[str, float]], min_conf: float) -> str:
    """Join OCR line candidates, dropping any whose confidence is below
    ``min_conf``. Pure helper (no Vision) so it's unit-testable. When the floor
    is 0 every non-empty line is kept (legacy behaviour)."""
    kept = [t for t, c in pairs if t.strip() and (min_conf <= 0.0 or c >= min_conf)]
    return "\n".join(kept).strip()


# Whole-image OCR quality. Vision (accurate mode) tops out near 0.50 on legible
# low-res screenshots and sits at ~0.30 on mojibake; a junk console screenshot
# averages ~0.45 across all lines. Images whose mean line-confidence is below
# this threshold are flagged low-quality so the indexer can down-weight the
# whole record (search score x confidence), keeping clean text notes above
# garbled screenshots. 0 disables image down-weighting.
_DEFAULT_OCR_LOW_CONF_THRESHOLD = 0.6


def ocr_low_conf_threshold(default: float = _DEFAULT_OCR_LOW_CONF_THRESHOLD) -> float:
    """Mean-confidence threshold below which an OCR'd image is treated as
    low-quality. Override with ``MEMO_OCR_LOW_CONF_THRESHOLD`` (0 disables)."""
    from memo.flags import active_flags

    raw = active_flags().get("MEMO_OCR_LOW_CONF_THRESHOLD")
    if raw is None:
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default


def image_health_confidence(mean_conf: float) -> float | None:
    """Map an image's mean OCR confidence to a memory-health confidence for the
    record, or None to leave it neutral (1.0). Returns the mean (floored at 0.1)
    when it falls below :func:`ocr_low_conf_threshold` so low-quality screenshots
    rank below clean notes; None otherwise. None when the threshold is 0."""
    thr = ocr_low_conf_threshold()
    if thr <= 0.0 or mean_conf <= 0.0 or mean_conf >= thr:
        return None
    return max(0.1, mean_conf)


def _mean_conf(pairs: list[tuple[str, float]]) -> float:
    """Mean confidence over ALL recognized lines (pre per-line filter) — the
    whole-image quality signal. 0.0 when no lines."""
    if not pairs:
        return 0.0
    return sum(c for _, c in pairs) / len(pairs)

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
    """Run Apple Vision text recognition on ``image_path``; return text only.

    Thin wrapper over :func:`extract_text_with_confidence`. Returns the
    confidence-filtered text, or empty string on any failure. Never raises.
    """
    return extract_text_with_confidence(image_path, languages=languages)[0]


def extract_text_with_confidence(
    image_path: Path,
    *,
    languages: Iterable[str] = ("es-ES", "en-US"),
) -> tuple[str, float]:
    """Run Apple Vision OCR and return ``(filtered_text, mean_confidence)``.

    ``filtered_text`` drops per-line candidates below :func:`ocr_min_confidence`
    (mojibake); ``mean_confidence`` is the mean over ALL recognized lines (the
    whole-image quality signal). ``("", 0.0)`` on any failure. Never raises.
    """
    if not vision_available():
        return "", 0.0
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return "", 0.0
    try:
        import Quartz
        import Vision
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(p))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if not src or Quartz.CGImageSourceGetCount(src) == 0:
            return "", 0.0
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg_image is None:
            return "", 0.0

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image,
            None,
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
            return "", 0.0
        results = request.results() or []
        pairs: list[tuple[str, float]] = []
        for obs in results:
            try:
                top = obs.topCandidates_(1)
                if top and len(top) > 0:
                    text = str(top[0].string())
                    if text.strip():
                        try:
                            conf = float(top[0].confidence())
                        except Exception:
                            conf = 1.0
                        pairs.append((text, conf))
            except Exception:
                continue
        return _join_confident(pairs, ocr_min_confidence()), _mean_conf(pairs)
    except Exception as exc:
        _log.debug("OCR error on %s: %s", p, exc)
        return "", 0.0


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
    """OCR with SHA256-based caching; return text only.

    Thin wrapper over :func:`extract_text_cached_with_confidence` for the many
    callers that don't need the confidence. Returns empty string on any error.
    """
    return extract_text_cached_with_confidence(
        image_path, cache_dir=cache_dir, languages=languages
    )[0]


def extract_text_cached_with_confidence(
    image_path: Path,
    *,
    cache_dir: Path,
    languages: Iterable[str] = ("es-ES", "en-US"),
) -> tuple[str, float]:
    """OCR with SHA256-based JSON caching at ``<cache_dir>/<hash>.<conf>.json``.

    Returns ``(filtered_text, mean_confidence)``. Cache hit → read JSON. Miss →
    OCR + write ``{"t": text, "c": conf}`` (empty result cached too). The
    confidence floor is folded into the cache key so changing the threshold (or
    upgrading from the legacy ``.txt`` cache) re-OCRs instead of serving stale
    garbled text. On any I/O error falls back to a live OCR without raising.
    """
    p = Path(image_path)
    if not p.exists():
        return "", 0.0
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.debug("ocr: cache dir unavailable, bypassing cache: %s", exc)
        return extract_text_with_confidence(p, languages=languages)
    try:
        digest = _hash_bytes(p)
    except OSError as exc:
        _log.debug("ocr: hash failed, bypassing cache: %s", exc)
        return extract_text_with_confidence(p, languages=languages)
    conf_tag = f"c{round(ocr_min_confidence() * 100):02d}"
    cache_path = cache_dir / f"{digest[:32]}.{conf_tag}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(cached.get("t", "")), float(cached.get("c", 0.0))
        except (OSError, ValueError) as exc:
            _log.debug("ocr: cache read failed, re-extracting: %s", exc)
    text, conf = extract_text_with_confidence(p, languages=languages)
    with contextlib.suppress(Exception):
        cache_path.write_text(
            json.dumps({"t": text, "c": conf}), encoding="utf-8"
        )
    return text, conf


def ocr_enabled_via_env(default: bool = False) -> bool:
    from memo.flags import active_flags, flag_bool

    if "MEMO_OCR_ENABLED" not in active_flags():
        return default
    return flag_bool("MEMO_OCR_ENABLED")
