"""`memo_ocr_image` answers with its error envelope, never a raw OSError.

`exists()` is true for a directory, so a directory path fell through the
not-found guard and died in `read_bytes()`. Measured 2026-08-09 over stdio,
`memo_ocr_image(image_path="/private/tmp")` came back as::

    Error calling tool 'memo_ocr_image': [Errno 21] Is a directory: '/private/tmp'

— an unhandled exception where every other rejected path (outside the
allow-list, missing) returns `{"text": "", "cached": False, "error": ...}`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


@pytest.fixture
def ocr_server(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    # Vision is macOS+PyObjC only; pin it available so the path checks are the
    # thing under test rather than the platform.
    from memo import ocr

    monkeypatch.setattr(ocr, "vision_available", lambda: True)
    monkeypatch.setattr(ocr, "extract_text_cached", lambda path, cache_dir: "ocr text")
    memory = Memory(tmp_cfg)
    try:
        yield build_server(memory=memory)
    finally:
        memory.close()


@pytest.mark.asyncio
async def test_a_directory_returns_the_error_envelope(ocr_server: Any, tmp_cfg: Config) -> None:
    directory = tmp_cfg.data_dir / "screenshots"
    directory.mkdir(parents=True, exist_ok=True)

    result = await ocr_server.call_tool("memo_ocr_image", {"image_path": str(directory)})

    payload = result.structured_content
    assert payload["error"] == "file not found"
    assert payload["text"] == ""


@pytest.mark.asyncio
async def test_a_missing_file_still_returns_the_error_envelope(
    ocr_server: Any, tmp_cfg: Config
) -> None:
    missing = tmp_cfg.data_dir / "absent.png"

    result = await ocr_server.call_tool("memo_ocr_image", {"image_path": str(missing)})

    assert result.structured_content["error"] == "file not found"


@pytest.mark.asyncio
async def test_a_real_file_is_read(ocr_server: Any, tmp_cfg: Config) -> None:
    """The guard must not reject the case it exists to serve."""
    image = tmp_cfg.data_dir / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await ocr_server.call_tool("memo_ocr_image", {"image_path": str(image)})

    payload = result.structured_content
    assert "error" not in payload
    assert payload["text"] == "ocr text"


def test_path_guard_uses_is_file() -> None:
    """Pin the distinction: `exists()` would let a directory through."""
    source = Path("src/memo/server_multimodal.py").read_text(encoding="utf-8")

    assert "if not path.is_file():" in source
