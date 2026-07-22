"""Tests for server_links MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from memo.memory import Memory


def _make_server_and_tools():
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


def test_register_exposes_all_four_tools(tmp_cfg) -> None:
    """register() must expose exactly the four expected links MCP tools."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_links_backlinks",
        "memo_links_outlinks",
        "memo_links_suggest",
        "memo_links_format",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_links_backlinks_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_links_backlinks must convert Backlink dataclasses to plain dicts."""
    from memo.crossref import Backlink
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_backlink = Backlink(
        source_id="src-abc",
        source_title="Source Memory",
        target_id="tgt-xyz",
        link_type="wikilink",
        context="...some [[tgt-xyz]] in context...",
    )
    mem.crossref.get_backlinks.return_value = [fake_backlink]

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_links_backlinks" in tools

    result = tools["memo_links_backlinks"](memory_id="tgt-xyz")

    # get_backlinks is now called with a batched title_resolver so source_title
    # is populated from the store instead of always "".
    call = mem.crossref.get_backlinks.call_args
    assert call.args == ("tgt-xyz",)
    assert "title_resolver" in call.kwargs
    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, dict)
    assert item["source_id"] == "src-abc"
    assert item["source_title"] == "Source Memory"
    assert item["target_id"] == "tgt-xyz"
    assert item["link_type"] == "wikilink"
    assert "context" in item


def test_memo_links_backlinks_empty(tmp_cfg) -> None:
    """memo_links_backlinks must return [] when no backlinks exist."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.crossref.get_backlinks.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_backlinks"](memory_id="orphan-id")
    assert result == []
    assert mem.crossref.get_backlinks.call_args.args == ("orphan-id",)


def test_memo_links_outlinks_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_links_outlinks must convert Wikilink dataclasses to plain dicts."""
    from memo.crossref import Wikilink
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_wikilink = Wikilink(
        target="other-memory",
        alias="Other",
        position=42,
    )
    mem.crossref.get_outlinks.return_value = [fake_wikilink]

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_links_outlinks" in tools

    result = tools["memo_links_outlinks"](memory_id="src-id")

    mem.crossref.get_outlinks.assert_called_once_with("src-id")
    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, dict)
    assert item["target"] == "other-memory"
    assert item["alias"] == "Other"
    assert item["position"] == 42


def test_memo_links_outlinks_empty(tmp_cfg) -> None:
    """memo_links_outlinks must return [] when no outlinks exist."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.crossref.get_outlinks.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_outlinks"](memory_id="leaf-id")
    assert result == []
    mem.crossref.get_outlinks.assert_called_once_with("leaf-id")


def test_memo_links_suggest_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_links_suggest must convert LinkSuggestion dataclasses to plain dicts."""
    from memo.crossref import LinkSuggestion
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_suggestion = LinkSuggestion(
        memory_id="rel-001",
        title="Related Decision",
        similarity=0.87,
        reason="High semantic similarity (0.87)",
    )
    mem.link_suggester.suggest_links.return_value = [fake_suggestion]

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_links_suggest" in tools

    result = tools["memo_links_suggest"](
        content="We decided to use MLX for embeddings.",
        title="MLX Decision",
        tags=["mlx", "embeddings"],
        limit=3,
    )

    mem.link_suggester.suggest_links.assert_called_once_with(
        content="We decided to use MLX for embeddings.",
        title="MLX Decision",
        tags=["mlx", "embeddings"],
        limit=3,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, dict)
    assert item["memory_id"] == "rel-001"
    assert item["title"] == "Related Decision"
    assert item["similarity"] == 0.87
    assert "reason" in item


def test_memo_links_suggest_passes_defaults(tmp_cfg) -> None:
    """memo_links_suggest must use title='', tags=[], limit=5 when omitted."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.link_suggester.suggest_links.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_suggest"](content="some content")

    mem.link_suggester.suggest_links.assert_called_once_with(
        content="some content",
        title="",
        tags=[],
        limit=5,
    )
    assert result == []


def test_memo_links_suggest_empty(tmp_cfg) -> None:
    """memo_links_suggest must return [] when no suggestions are found."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.link_suggester.suggest_links.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_suggest"](content="unique content with no matches")
    assert result == []


def test_memo_links_format_with_title(tmp_cfg) -> None:
    """memo_links_format must return [[id|Title]] when title differs from id."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.link_suggester.format_wikilink.return_value = "[[mem-abc|My Memory]]"

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_links_format" in tools

    result = tools["memo_links_format"](memory_id="mem-abc", title="My Memory")

    mem.link_suggester.format_wikilink.assert_called_once_with("mem-abc", "My Memory")
    assert result == "[[mem-abc|My Memory]]"


def test_memo_links_format_without_title(tmp_cfg) -> None:
    """memo_links_format must return [[id]] when no title is provided."""
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.link_suggester.format_wikilink.return_value = "[[mem-xyz]]"

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_format"](memory_id="mem-xyz")

    mem.link_suggester.format_wikilink.assert_called_once_with("mem-xyz", None)
    assert result == "[[mem-xyz]]"


def test_memo_links_suggest_multiple_results(tmp_cfg) -> None:
    """memo_links_suggest must return all suggestions as a list of dicts."""
    from memo.crossref import LinkSuggestion
    from memo.server_links import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    suggestions = [
        LinkSuggestion(
            memory_id=f"rel-{i:03d}",
            title=f"Related Memory {i}",
            similarity=0.9 - i * 0.05,
            reason=f"Similarity score {0.9 - i * 0.05:.2f}",
        )
        for i in range(3)
    ]
    mem.link_suggester.suggest_links.return_value = suggestions

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_links_suggest"](content="complex query", limit=5)

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["memory_id"] == "rel-000"
    assert result[1]["memory_id"] == "rel-001"
    assert result[2]["memory_id"] == "rel-002"
    # All items must be plain dicts, not dataclass instances
    for item in result:
        assert isinstance(item, dict)
        assert "memory_id" in item
        assert "similarity" in item
