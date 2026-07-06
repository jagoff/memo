"""Image captioning via mlx-vlm (optional dependency).

Companion to `ocr.py`: where Apple Vision OCR yields little or no text
(text-free diagrams, photos, whiteboards), a small local VLM produces a
searchable caption at INGEST TIME only — never on the recall-hook path.

Same contract as `ocr.py`:
- optional dep guarded (`vlm_available()`); empty string on ANY failure,
  never raises;
- SHA256-of-bytes cache at `<cache_dir>/<hash32>.caption.json` so
  re-ingest is free (empty results are cached too — delete the cache dir
  to re-probe after installing mlx-vlm);
- mlx imports stay DEFERRED (inside functions) per the MLX invariants.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["caption_image", "caption_image_cached", "vlm_available"]

_log = logging.getLogger(__name__)

_VLM_CHECKED = False
_VLM_OK = False
_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}

_CAPTION_PROMPT = (
    "Describe this image in 2-3 concise sentences for a search index. "
    "Name visible apps, diagram types, UI elements, or objects. "
    "Reply with the description only."
)
_CAPTION_MAX_TOKENS = 120


def vlm_available() -> bool:
    """True if mlx-vlm can be imported in this interpreter."""
    global _VLM_CHECKED, _VLM_OK
    if _VLM_CHECKED:
        return _VLM_OK
    _VLM_CHECKED = True
    try:
        import mlx_vlm  # noqa: F401

        _VLM_OK = True
    except ImportError as exc:
        _log.debug("mlx-vlm not installed: %s", exc)
        _VLM_OK = False
    except Exception as exc:
        _log.warning("mlx-vlm import failed: %s", exc)
        _VLM_OK = False
    return _VLM_OK


def _load_model(model_name: str) -> tuple[Any, Any, Any]:
    """Load (model, processor, config) once per process. Deferred imports."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(model_name)
    config = load_config(model_name)
    _MODEL_CACHE[model_name] = (model, processor, config)
    return model, processor, config


def caption_image(image_path: Path, *, max_tokens: int = _CAPTION_MAX_TOKENS) -> str:
    """Generate a caption for `image_path`. Empty string on any failure."""
    if not vlm_available():
        return ""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        from memo.flags import flag_str

        model_name = flag_str("MEMO_VLM_MODEL")
        model, processor, config = _load_model(model_name)
        prompt = apply_chat_template(processor, config, _CAPTION_PROMPT, num_images=1)
        out = generate(
            model, processor, prompt, image=[str(p)], max_tokens=max_tokens, verbose=False
        )
        # mlx-vlm >= 0.1.x returns a GenerationResult with .text; older returns str.
        text = getattr(out, "text", out)
        return str(text or "").strip()
    except Exception as exc:
        _log.debug("vlm caption failed for %s: %s", p, exc)
        return ""


def _hash_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def caption_image_cached(image_path: Path, *, cache_dir: Path) -> str:
    """Caption with SHA256 JSON caching at `<cache_dir>/<hash32>.caption.json`.

    Cache hit → read JSON. Miss → caption + write `{"t": text}` (empty
    result cached too, so VLM-less installs don't re-probe every ingest).
    On any I/O error falls back to a live caption without raising.
    """
    p = Path(image_path)
    if not p.exists():
        return ""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _hash_bytes(p)
    except OSError as exc:
        _log.debug("vlm: cache unavailable, bypassing: %s", exc)
        return caption_image(p)
    cache_path = cache_dir / f"{digest[:32]}.caption.json"
    if cache_path.exists():
        try:
            return str(json.loads(cache_path.read_text(encoding="utf-8")).get("t", ""))
        except (OSError, ValueError) as exc:
            _log.debug("vlm: cache read failed, re-captioning: %s", exc)
    text = caption_image(p)
    with contextlib.suppress(Exception):
        cache_path.write_text(json.dumps({"t": text}), encoding="utf-8")
    return text
