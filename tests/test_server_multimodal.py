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


def test_register_exposes_all_six_tools(tmp_cfg) -> None:
    """register() must expose exactly the six expected MCP tools."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_ocr_image",
        "memo_multimodal_add_image",
        "memo_multimodal_add_audio",
        "memo_multimodal_search_images",
        "memo_multimodal_search_audio",
        "memo_multimodal_search_all",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


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
        result = tools["memo_ocr_image"](image_path="/nonexistent/does-not-exist.png")

    assert result["text"] == ""
    assert result["cached"] is False
    assert "file not found" in result["error"]


def test_memo_ocr_image_returns_extracted_text(tmp_cfg, tmp_path: Path) -> None:
    """memo_ocr_image returns extracted text when Vision is available and file exists."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    # Create a real file so path.exists() passes and read_bytes() works
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    with (
        patch("memo.ocr.vision_available", return_value=True),
        patch("memo.ocr.extract_text_cached", return_value="extracted text from image"),
        patch("memo.ocr.ocr_min_confidence", return_value=0.4),
    ):
        result = tools["memo_ocr_image"](image_path=str(img))

    assert result["text"] == "extracted text from image"
    assert "cached" in result


def test_memo_multimodal_add_image_calls_multimodal(tmp_cfg) -> None:
    """memo_multimodal_add_image calls memory.multimodal.add_image and returns envelope."""
    from memo.memory import Memory
    from memo.multimodal import MultiModalContent
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_content = MultiModalContent(
        id="img-001",
        memory_id="mem-xyz",
        modality="image",
        content=b"",
        embedding=[],
        metadata={},
        created_at="2024-01-01T00:00:00+00:00",
    )
    mem.multimodal.add_image.return_value = fake_content

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_add_image"](
        image_path="/some/image.png",
        memory_id="mem-xyz",
    )

    mem.multimodal.add_image.assert_called_once()
    assert result["content_id"] == "img-001"
    assert result["modality"] == "image"


def test_memo_multimodal_add_image_without_memory_id(tmp_cfg) -> None:
    """memo_multimodal_add_image passes None memory_id correctly."""
    from memo.memory import Memory
    from memo.multimodal import MultiModalContent
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_content = MultiModalContent(
        id="img-standalone",
        memory_id=None,
        modality="image",
        content=b"",
        embedding=[],
        metadata={},
        created_at="2024-01-01T00:00:00+00:00",
    )
    mem.multimodal.add_image.return_value = fake_content

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_add_image"](image_path="/standalone.png")

    _, kwargs = mem.multimodal.add_image.call_args
    # default memory_id is None
    assert kwargs.get("memory_id") is None or mem.multimodal.add_image.call_args[0][1] is None
    assert result["content_id"] == "img-standalone"


def test_memo_multimodal_add_audio_calls_multimodal(tmp_cfg) -> None:
    """memo_multimodal_add_audio calls memory.multimodal.add_audio and returns envelope."""
    from memo.memory import Memory
    from memo.multimodal import MultiModalContent
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_content = MultiModalContent(
        id="aud-002",
        memory_id=None,
        modality="audio",
        content=b"",
        embedding=[],
        metadata={},
        created_at="2024-01-01T00:00:00+00:00",
    )
    mem.multimodal.add_audio.return_value = fake_content

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_add_audio"](
        audio_path="/some/audio.mp3",
        memory_id=None,
    )

    mem.multimodal.add_audio.assert_called_once()
    assert result["content_id"] == "aud-002"
    assert result["modality"] == "audio"


def test_memo_multimodal_search_images_returns_list(tmp_cfg) -> None:
    """memo_multimodal_search_images returns a list of dicts from CrossModalResult.__dict__."""
    from memo.memory import Memory
    from memo.multimodal import CrossModalResult
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_results = [
        CrossModalResult(
            content_id="img-1",
            modality="image",
            similarity=0.95,
            metadata={"filename": "a.png"},
        ),
        CrossModalResult(
            content_id="img-2",
            modality="image",
            similarity=0.80,
            metadata={"filename": "b.png"},
        ),
    ]
    mem.multimodal.search.search_text_find_images.return_value = fake_results

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_search_images"](query="architecture diagram", limit=5)

    mem.multimodal.search.search_text_find_images.assert_called_once_with(
        "architecture diagram", limit=5
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["content_id"] == "img-1"
    assert result[0]["modality"] == "image"
    assert result[0]["similarity"] == 0.95
    assert result[1]["content_id"] == "img-2"


def test_memo_multimodal_search_images_empty_result(tmp_cfg) -> None:
    """memo_multimodal_search_images returns an empty list when no images match."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.multimodal.search.search_text_find_images.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_search_images"](query="nothing here")

    assert isinstance(result, list)
    assert result == []


def test_memo_multimodal_search_audio_returns_list(tmp_cfg) -> None:
    """memo_multimodal_search_audio returns a list of dicts from CrossModalResult.__dict__."""
    from memo.memory import Memory
    from memo.multimodal import CrossModalResult
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_results = [
        CrossModalResult(
            content_id="aud-1",
            modality="audio",
            similarity=0.72,
            metadata={"filename": "talk.mp3"},
        ),
    ]
    mem.multimodal.search.search_text_find_audio.return_value = fake_results

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_search_audio"](query="voice recording", limit=3)

    mem.multimodal.search.search_text_find_audio.assert_called_once_with("voice recording", limit=3)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["content_id"] == "aud-1"
    assert result[0]["modality"] == "audio"
    assert result[0]["similarity"] == 0.72


def test_memo_multimodal_search_all_returns_dict_of_lists(tmp_cfg) -> None:
    """memo_multimodal_search_all returns a dict mapping modality -> list of dicts."""
    from memo.memory import Memory
    from memo.multimodal import CrossModalResult
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_results: dict[str, list[CrossModalResult]] = {
        "image": [
            CrossModalResult(content_id="img-1", modality="image", similarity=0.9, metadata={}),
        ],
        "audio": [
            CrossModalResult(content_id="aud-1", modality="audio", similarity=0.75, metadata={}),
        ],
    }
    mem.multimodal.search.search_all_modalities.return_value = fake_results

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_search_all"](query="meeting notes", limit=5)

    mem.multimodal.search.search_all_modalities.assert_called_once_with("meeting notes", limit=5)
    assert isinstance(result, dict)
    assert "image" in result
    assert "audio" in result
    assert isinstance(result["image"], list)
    assert result["image"][0]["content_id"] == "img-1"
    assert result["image"][0]["similarity"] == 0.9
    assert result["audio"][0]["content_id"] == "aud-1"


def test_memo_multimodal_search_all_empty_result(tmp_cfg) -> None:
    """memo_multimodal_search_all returns an empty dict when no modalities match."""
    from memo.memory import Memory
    from memo.server_multimodal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.multimodal.search.search_all_modalities.return_value = {}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_multimodal_search_all"](query="nothing")

    assert isinstance(result, dict)
    assert result == {}
