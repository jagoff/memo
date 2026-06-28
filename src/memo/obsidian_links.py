"""Helpers for resolving images embedded in Obsidian notes.

Obsidian uses the `![[image.png]]` syntax (with an optional `|alt` alias).
The image can live in:

1. The folder configured in `.obsidian/app.json` → `attachmentFolderPath`
   (relative to the vault root). This is the most common case.
2. The same directory as the note that references it.
3. Any folder in the vault if the filename is unique (Obsidian resolves
   by basename).

This module exposes pure helpers — it doesn't touch disk except in
:func:`resolve_image_path`, which does stat/rglob.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

__all__ = [
    "find_image_embeds",
    "resolve_attachment_folder",
    "resolve_image_path",
]

_log = logging.getLogger(__name__)

_IMG_EXTS = ("png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "bmp", "tiff")
_IMG_EMBED_RE = re.compile(
    r"!\[\[(?P<name>[^\]|]+?\.(?:" + "|".join(_IMG_EXTS) + r"))(?:\|[^\]]*)?\]\]",
    re.IGNORECASE,
)


def _is_safe_embed_name(filename: str) -> bool:
    """Reject embed names that would escape the vault.

    Obsidian resolves embeds by basename, so a legitimate embed is a bare
    filename. A name that is absolute, contains a path separator, or has a
    ``..`` component (e.g. ``![[../../etc/passwd.png]]``) would let an embed
    point outside the vault and get read by OCR — reject it.
    """
    if not filename or "/" in filename or "\\" in filename:
        return False
    p = Path(filename)
    return not p.is_absolute() and ".." not in p.parts


def find_image_embeds(body: str) -> list[str]:
    """Returns the list of image filenames from `![[image.png]]` embeds.

    Order preserved. Duplicates dropped. Alias text after `|` ignored.
    Markdown image syntax `![alt](url)` is intentionally NOT matched —
    those usually point to remote URLs.
    """
    if not body:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _IMG_EMBED_RE.finditer(body):
        name = m.group("name").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def resolve_attachment_folder(vault_root: Path) -> Path | None:
    """Read `.obsidian/app.json` for `attachmentFolderPath`.

    Returns absolute path of the configured attachment folder, or None
    if config missing/unparseable. Obsidian stores the path relative to
    vault root with forward slashes.
    """
    cfg = vault_root / ".obsidian" / "app.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.debug("Could not parse %s: %s", cfg, exc)
        return None
    raw = data.get("attachmentFolderPath")
    if not isinstance(raw, str) or not raw.strip():
        return None
    folder = (vault_root / raw).resolve()
    if folder.exists() and folder.is_dir():
        return folder
    return None


def resolve_image_path(
    filename: str,
    vault_root: Path,
    *,
    note_dir: Path | None = None,
) -> Path | None:
    """Resolve `filename` to an absolute path inside `vault_root`.

    Search order:
      1. `vault_root / attachmentFolderPath / filename`
      2. `note_dir / filename` (if provided)
      3. rglob `vault_root` for any file matching `filename` (first hit)

    Returns the first existing path, or None if nothing matches. Never
    raises.
    """
    if not filename:
        return None
    if not _is_safe_embed_name(filename):
        _log.debug("Rejecting unsafe embed name (path traversal): %r", filename)
        return None
    try:
        attach = resolve_attachment_folder(vault_root)
    except Exception:
        attach = None
    if attach is not None:
        candidate = attach / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    if note_dir is not None:
        candidate = note_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    try:
        for hit in vault_root.rglob(filename):
            if hit.is_file():
                return hit
    except Exception as exc:
        _log.debug("rglob failed for %s in %s: %s", filename, vault_root, exc)
    return None
