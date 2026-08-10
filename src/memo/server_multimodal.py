"""MCP tools — OCR domain.

Registered by ``build_server()`` via ``register(server, memory)``. Hosts
``memo_ocr_image`` as an on-demand OCR fallback. The former
``memo_multimodal_*`` tools were placeholder hash-embedding stubs, removed
2026-07 — VLM captions and Whisper transcripts through the normal text index
are the cross-modal path.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_ocr_image(
        image_path: str,
    ) -> dict[str, Any]:
        """Run Apple Vision OCR on an image and return extracted text.

        Cached by SHA256 in `<state_dir>/ocr_cache`. Intended as an on-demand
        fallback for screenshots that have not been picked up by the indexer
        yet. Returns empty `text` if Vision is unavailable
        (non-macOS or missing PyObjC) or the file is missing.

        Args:
            image_path: Absolute path to the image file. The caller is
                responsible for resolving Obsidian `![[...]]` syntax
                to a real path on disk. Must live under the configured
                vault/data dir, the current working directory, Downloads,
                Desktop, or Documents.
        """
        from pathlib import Path

        from memo.ocr import extract_text_cached, vision_available
        from memo.server_import_export import _get_allowed_base_dirs, _is_subdir

        if not vision_available():
            return {"text": "", "cached": False, "error": "vision unavailable"}
        path = Path(image_path).expanduser().resolve(strict=False)
        # Allow-list: the same roots as the import/export tools plus the
        # configured vault / data dirs (where indexed screenshots live).
        # Blocks prompt-injected OCR reads of arbitrary files on disk.
        allowed = [memory.cfg.data_dir, *_get_allowed_base_dirs()]
        if memory.cfg.vault_path is not None:
            allowed.append(memory.cfg.vault_path)
        bases = [base.expanduser().resolve(strict=False) for base in allowed]
        if not any(path == base or _is_subdir(path, base) for base in bases):
            return {
                "text": "",
                "cached": False,
                "error": "image path outside allowed directories",
            }
        # `is_file`, not `exists`: a DIRECTORY passes `exists()` and then blows
        # up in `read_bytes()` below, which reached the MCP caller as a raw
        # `[Errno 21] Is a directory` instead of this tool's error envelope.
        if not path.is_file():
            return {"text": "", "cached": False, "error": "file not found"}
        cache_dir = memory.cfg.state_dir / "ocr_cache"
        sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        text = extract_text_cached(path, cache_dir=cache_dir)
        from memo.ocr import ocr_min_confidence

        conf_tag = f"c{round(ocr_min_confidence() * 100):02d}"
        cached = (cache_dir / f"{sha[:32]}.{conf_tag}.json").exists()
        return {"text": text, "cached": cached}
