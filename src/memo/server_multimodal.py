"""MCP tools — multi-modal domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
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

    @server.tool()
    def memo_multimodal_add_image(
        image_path: str,
        memory_id: str | None = None,
    ) -> dict[str, str]:
        """Add image to multi-modal corpus.

        Args:
            image_path: Path to the image file.
            memory_id: Optional associated memory ID.
        """
        from pathlib import Path

        content = memory.multimodal.add_image(Path(image_path), memory_id)
        return {"content_id": content.id, "modality": content.modality}

    @server.tool()
    def memo_multimodal_add_audio(
        audio_path: str,
        memory_id: str | None = None,
    ) -> dict[str, str]:
        """Add audio to multi-modal corpus.

        Args:
            audio_path: Path to the audio file.
            memory_id: Optional associated memory ID.
        """
        from pathlib import Path

        content = memory.multimodal.add_audio(Path(audio_path), memory_id)
        return {"content_id": content.id, "modality": content.modality}

    @server.tool()
    def memo_multimodal_search_images(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with text, find images.

        Args:
            query: Text query.
            limit: Max results.

        Returns:
            List of CrossModalResult objects.
        """
        results = memory.multimodal.search.search_text_find_images(query, limit=limit)
        return [r.__dict__ for r in results]

    @server.tool()
    def memo_multimodal_search_audio(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with text, find audio.

        Args:
            query: Text query.
            limit: Max results.

        Returns:
            List of CrossModalResult objects.
        """
        results = memory.multimodal.search.search_text_find_audio(query, limit=limit)
        return [r.__dict__ for r in results]

    @server.tool()
    def memo_multimodal_search_all(
        query: str,
        limit: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search across all modalities.

        Args:
            query: Text query.
            limit: Max results per modality.

        Returns:
            Dict with results per modality.
        """
        results = memory.multimodal.search.search_all_modalities(query, limit=limit)
        return {k: [r.__dict__ for r in v] for k, v in results.items()}
