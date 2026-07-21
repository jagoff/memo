"""Tests for server_multimodal MCP tool registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_server_and_tools() -> tuple[MagicMock, dict]:
    """Return a (server_mock, tools_dict) pair.

    `server.tool()` is wired so each `@server.tool()` decorated function is
    captured in `tools` by its `__name__`, without going through FastMCP.
    """
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_only_ocr_tool(tmp_cfg) -> None:
    """register() exposes exactly memo_ocr_image — the CLIP-stub tools are gone
    (captions/transcripts through the text index are the cross-modal path)."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert set(tools) == {"memo_ocr_image"}, f"Tool mismatch: {set(tools)}"


def test_memo_ocr_image_vision_unavailable(tmp_cfg) -> None:
    """memo_ocr_image returns error envelope when Vision is unavailable."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.ocr.vision_available", return_value=False):
        result = tools["memo_ocr_image"](image_path="/nonexistent/image.png")

    assert result["text"] == ""
    assert result["cached"] is False
    assert "vision unavailable" in result["error"]


def test_memo_ocr_image_file_not_found(tmp_cfg) -> None:
    """memo_ocr_image returns error envelope when the file does not exist."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.ocr.vision_available", return_value=True):
        result = tools["memo_ocr_image"](image_path=str(tmp_cfg.vault_path / "does-not-exist.png"))

    assert result["text"] == ""
    assert result["cached"] is False
    assert "file not found" in result["error"]


def test_memo_ocr_image_rejects_path_outside_allowed_roots(tmp_cfg, tmp_path: Path) -> None:
    """Arbitrary absolute paths outside the allow-list are refused without a read."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    # tmp_path itself is outside every allowed root (vault/data dir, cwd,
    # Downloads/Desktop/Documents) — only its vault/data subdirs are allowed.
    outside = tmp_path / "outside" / "secret.png"
    outside.parent.mkdir()
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    with (
        patch("memo.ocr.vision_available", return_value=True),
        patch("memo.ocr.extract_text_cached") as extract,
    ):
        result = tools["memo_ocr_image"](image_path=str(outside))

    assert result["text"] == ""
    assert result["cached"] is False
    assert "outside allowed" in result["error"]
    extract.assert_not_called()


def test_memo_ocr_image_returns_extracted_text(tmp_cfg) -> None:
    """memo_ocr_image returns extracted text when Vision is available and file exists."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    # Create a real file inside an allowed root (the vault) so the allow-list
    # passes, path.exists() passes, and read_bytes() works.
    img = tmp_cfg.vault_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    with (
        patch("memo.ocr.vision_available", return_value=True),
        patch("memo.ocr.extract_text_cached", return_value="extracted text from image"),
        patch("memo.ocr.ocr_min_confidence", return_value=0.4),
    ):
        result = tools["memo_ocr_image"](image_path=str(img))

    assert result["text"] == "extracted text from image"
    assert "cached" in result
