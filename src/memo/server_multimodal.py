"""MCP tools — OCR domain. Registered by build_server() via register(server, memory). Hosts memo_ocr_image (used by Synapse as a chat-time OCR fallback). The former memo_multimodal_* tools were placeholder hash-embedding stubs, removed 2026-07 — VLM captions + whisper transcripts through the normal text index are the cross-modal path."""

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

        Cached by SHA256 in `<state_dir>/ocr_cache`. Used by Synapse as a
        chat-time fallback for screenshots that have not been picked up
        by the indexer yet. Returns empty `text` if Vision is unavailable
        (non-macOS or missing PyObjC) or the file is missing.

        Args:
            image_path: Absolute path to the image file. The caller is
                responsible for resolving Obsidian `![[...]]` syntax
                to a real path on disk.
        """
        from pathlib import Path

        from memo.ocr import extract_text_cached, vision_available

        if not vision_available():
            return {"text": "", "cached": False, "error": "vision unavailable"}
        path = Path(image_path)
        if not path.exists():
            return {"text": "", "cached": False, "error": "file not found"}
        cache_dir = memory.cfg.state_dir / "ocr_cache"
        sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        text = extract_text_cached(path, cache_dir=cache_dir)
        from memo.ocr import ocr_min_confidence
        conf_tag = f"c{round(ocr_min_confidence() * 100):02d}"
        cached = (cache_dir / f"{sha[:32]}.{conf_tag}.json").exists()
        return {"text": text, "cached": cached}
