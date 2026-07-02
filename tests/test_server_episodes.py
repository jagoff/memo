"""Tests for server_episodes MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memo.resume._types import ResumeCandidate


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


def _stub_candidate(
    session_id: str = "abc123",
    agent: str = "claude",
    score: float = 0.87,
    summary: str = "worked on memo indexer",
    cwd: str = "/Users/fer/repos/memo",
    resume_command: list[str] | None = None,
) -> ResumeCandidate:
    return ResumeCandidate(
        agent=agent,
        provider="episode",
        uri=f"memo://episode/{agent}/{session_id}",
        session_id=session_id,
        title=summary,
        updated_at="2026-07-01T03:00:00Z",
        cwd=cwd,
        summary=summary,
        resume_mode="native_resume",
        resume_command=resume_command or ["claude", "--resume", session_id],
        metadata={"score": score, "episode": True},
    )


def test_register_exposes_exactly_one_tool(tmp_cfg) -> None:
    """register() must expose exactly memo_episodes_search."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert {"memo_episodes_search"} == set(tools), f"Unexpected tools registered: {set(tools)}"


def test_memo_episodes_search_is_registered(tmp_cfg) -> None:
    """memo_episodes_search must be present in tools after register()."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_episodes_search" in tools


def test_memo_episodes_search_empty_index_returns_envelope(tmp_cfg) -> None:
    """memo_episodes_search returns the correct envelope even when the index is empty."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[]) as mock_search:
        result = tools["memo_episodes_search"](query="indexer refactor")

    mock_search.assert_called_once()
    assert result["query"] == "indexer refactor"
    assert result["results"] == []
    assert isinstance(result, dict)


def test_memo_episodes_search_calls_semantic_search_with_correct_args(tmp_cfg) -> None:
    """memo_episodes_search must forward query + limit to semantic_search."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[]) as mock_search:
        tools["memo_episodes_search"](query="embed pipeline", limit=7)

    mock_search.assert_called_once_with(tmp_cfg, "embed pipeline", k=7, allow_cold=True)


def test_memo_episodes_search_default_limit_is_ten(tmp_cfg) -> None:
    """memo_episodes_search passes k=10 when limit is omitted."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[]) as mock_search:
        tools["memo_episodes_search"](query="foo")

    _, kwargs = mock_search.call_args
    assert kwargs.get("k") == 10
    assert kwargs.get("allow_cold") is True


def test_memo_episodes_search_maps_candidates_correctly(tmp_cfg) -> None:
    """Each result dict must contain exactly the fields the tool promises."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    cand = _stub_candidate(
        session_id="session-xyz",
        agent="opencode",
        score=0.91,
        summary="vector store rewrite",
        cwd="/work",
        resume_command=["opencode", "--session", "session-xyz"],
    )

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[cand]):
        result = tools["memo_episodes_search"](query="vector store")

    assert result["query"] == "vector store"
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["session_id"] == "session-xyz"
    assert hit["agent"] == "opencode"
    assert hit["score"] == 0.91
    assert hit["summary"] == "vector store rewrite"
    assert hit["cwd"] == "/work"
    assert hit["resume_command"] == ["opencode", "--session", "session-xyz"]


def test_memo_episodes_search_result_keys_are_exact(tmp_cfg) -> None:
    """Result dicts must contain exactly the documented keys, no extras."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    cand = _stub_candidate()

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[cand]):
        result = tools["memo_episodes_search"](query="anything")

    expected_keys = {"session_id", "agent", "score", "summary", "cwd", "resume_command"}
    actual_keys = set(result["results"][0].keys())
    assert actual_keys == expected_keys, f"Result key mismatch: {actual_keys}"


def test_memo_episodes_search_summary_falls_back_to_title(tmp_cfg) -> None:
    """When summary is empty the result must use title instead."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    cand = ResumeCandidate(
        agent="claude",
        provider="episode",
        uri="memo://episode/claude/no-summary",
        session_id="no-summary",
        title="title-fallback",
        updated_at="2026-07-01T03:00:00Z",
        cwd="",
        summary="",  # empty — tool must use title
        resume_mode="native_resume",
        resume_command=[],
        metadata={"score": 0.5},
    )

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=[cand]):
        result = tools["memo_episodes_search"](query="fallback test")

    assert result["results"][0]["summary"] == "title-fallback"


def test_memo_episodes_search_multiple_results_preserved(tmp_cfg) -> None:
    """All candidates returned by semantic_search appear in results, in order."""
    from memo.memory import Memory
    from memo.server_episodes import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    candidates = [_stub_candidate(session_id=f"sess-{i}", score=float(i) / 10) for i in range(5)]

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.resume._index.semantic_search", return_value=candidates):
        result = tools["memo_episodes_search"](query="multi")

    assert len(result["results"]) == 5
    for i, hit in enumerate(result["results"]):
        assert hit["session_id"] == f"sess-{i}"


def test_no_module_level_mlx_imports() -> None:
    """server_episodes must not have module-level MLX imports (deferred-import invariant)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_episodes.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations: list[str] = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"
