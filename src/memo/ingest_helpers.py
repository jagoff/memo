"""Shared helpers used by both `memo ingest` (vault) and `memo repo index`.

Centralises three operations that previously lived inline in
`repo_index.py` so the vault ingest path can reuse them:

- `enrich_with_ocr`  — detects `![[image]]` embeds, OCRs each, appends
  `<!-- OCR: name.png -->\n<text>\n` blocks. Returns the enriched text
  plus the set of resolved image paths so the caller can track which
  attachments have been "claimed" by a note (used to compute the
  orphan-images set in vault ingest).

- `extract_pdf_text` — wraps `pdftotext -layout -enc UTF-8 -nopgbrk
  <pdf> -`. Returns empty string if the binary is unavailable or the
  PDF is unreadable; callers can decide whether to skip.

- `find_orphan_images` — walks a vault for image files NOT in the
  referenced set produced by `enrich_with_ocr`. Used to create
  standalone memories from screenshots that nobody linked.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from memo.obsidian_links import find_image_embeds, resolve_image_path
from memo.ocr import extract_text_cached

_log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".heic",
        ".bmp",
        ".tiff",
    }
)

_PDFTOTEXT = shutil.which("pdftotext")


def pdftotext_available() -> bool:
    return _PDFTOTEXT is not None


def caption_if_ocr_weak(img_path: Path, ocr_text: str, state_dir: Path) -> str:
    """mlx-vlm caption fallback for images whose OCR yielded little/no text.

    Gated by MEMO_VLM_CAPTION_ENABLED (default off). Returns "" when the
    flag is off, OCR already produced enough text
    (>= MEMO_VLM_CAPTION_MIN_OCR_CHARS), or captioning fails. Ingest-time
    only — never called from the recall-hook path.
    """
    from memo.flags import flag_bool, flag_int

    if not flag_bool("MEMO_VLM_CAPTION_ENABLED"):
        return ""
    if len((ocr_text or "").strip()) >= (flag_int("MEMO_VLM_CAPTION_MIN_OCR_CHARS") or 40):
        return ""
    from memo.vlm_caption import caption_image_cached

    return caption_image_cached(img_path, cache_dir=state_dir / "vlm_cache")


def enrich_with_ocr(
    text: str,
    note_path: Path,
    vault_root: Path,
    state_dir: Path,
) -> tuple[str, list[Path], list[bytes]]:
    """Append OCR blocks for every `![[image]]` referenced in `text`.

    Returns `(enriched_text, resolved_image_paths, image_byte_hashes)`.
    `resolved_image_paths` is what callers union into a "referenced"
    set so orphan-image detection knows which files are spoken for.
    """
    if not text or "![[" not in text:
        return text, [], []
    image_names = find_image_embeds(text)
    if not image_names:
        return text, [], []

    resolved: list[Path] = []
    ocr_blocks: list[str] = []
    img_hashes: list[bytes] = []
    cache_dir = state_dir / "ocr_cache"
    for img_name in image_names:
        img_path = resolve_image_path(img_name, vault_root, note_dir=note_path.parent)
        if img_path is None:
            continue
        resolved.append(img_path)
        ocr_text = extract_text_cached(img_path, cache_dir=cache_dir)
        caption = caption_if_ocr_weak(img_path, ocr_text, state_dir)
        if not ocr_text and not caption:
            continue
        if ocr_text:
            ocr_blocks.append(f"\n\n<!-- OCR: {img_name} -->\n{ocr_text}\n")
        if caption:
            ocr_blocks.append(f"\n\n<!-- VLM: {img_name} -->\n{caption}\n")
        try:
            img_hashes.append(hashlib.sha256(img_path.read_bytes()).digest())
        except Exception as exc:
            _log.debug("hash failed for %s: %s", img_path, exc)

    if not ocr_blocks:
        return text, resolved, []
    return text + "".join(ocr_blocks), resolved, img_hashes


def extract_pdf_text(pdf_path: Path) -> str:
    """Best-effort plain-text extraction via poppler's `pdftotext`.

    Empty string on any failure — caller decides whether to skip the
    file or fall back to OCR-as-image. `-layout` keeps columns/tables
    readable; `-nopgbrk` strips the form-feed separators that confuse
    chunk-by-paragraph splitters.
    """
    if _PDFTOTEXT is None:
        return ""
    try:
        result = subprocess.run(
            [_PDFTOTEXT, "-layout", "-enc", "UTF-8", "-nopgbrk", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.debug("pdftotext failed for %s: %s", pdf_path, exc)
        return ""
    if result.returncode != 0:
        _log.debug("pdftotext exit %d for %s: %s", result.returncode, pdf_path, result.stderr[:200])
        return ""
    return result.stdout or ""


def find_orphan_images(
    vault_root: Path,
    referenced: set[Path],
    excluded_dirs: tuple[str, ...] = (),
) -> list[Path]:
    """Walk `vault_root` for image files not present in `referenced`.

    `excluded_dirs` is the same exclusion list used by the markdown
    walker (`.obsidian`, `.git`, `Obsidian`, etc.) — we honour it
    so attachments under excluded directories are also skipped.
    """
    referenced_resolved = {p.resolve() for p in referenced}
    orphans: list[Path] = []
    for path in vault_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(vault_root)
        rel_str = str(rel)
        if any(rel_str.startswith(d) or f"/{d}/" in f"/{rel_str}/" for d in excluded_dirs):
            continue
        if path.resolve() in referenced_resolved:
            continue
        orphans.append(path)
    orphans.sort()
    return orphans
