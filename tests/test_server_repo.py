"""Tests for server_repo MCP tool registration."""

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


def test_register_exposes_all_seven_tools(tmp_cfg) -> None:
    """register() must expose exactly the seven expected MCP tools."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_repo_index",
        "memo_repo_embed",
        "memo_repo_status",
        "memo_repo_search",
        "memo_repo_get_file",
        "memo_repo_list",
        "memo_repo_delete",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_repo_index_delegates_to_memory(tmp_cfg) -> None:
    """memo_repo_index must delegate to memory.repo_index and return its result."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_index.return_value = {"status": "ok", "chunks": 42}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_index"](
        url="https://github.com/example/repo.git",
        name="myrepo",
        ref="main",
        force=True,
        with_embeddings=False,
        include=["*.py"],
        exclude=["tests/*"],
        max_file_bytes=65536,
    )

    mem.repo_index.assert_called_once_with(
        "https://github.com/example/repo.git",
        name="myrepo",
        ref="main",
        force=True,
        with_embeddings=False,
        include=["*.py"],
        exclude=["tests/*"],
        max_file_bytes=65536,
    )
    assert result == {"status": "ok", "chunks": 42}


def test_memo_repo_index_default_args(tmp_cfg) -> None:
    """memo_repo_index must pass default arguments correctly."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_index.return_value = {"status": "ok", "chunks": 5}

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_repo_index"](url="https://github.com/example/repo.git")

    mem.repo_index.assert_called_once_with(
        "https://github.com/example/repo.git",
        name=None,
        ref=None,
        force=False,
        with_embeddings=True,
        include=None,
        exclude=None,
        max_file_bytes=None,
    )


def test_memo_repo_embed_delegates_to_memory(tmp_cfg) -> None:
    """memo_repo_embed must delegate to memory.repo_embed and return its result."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_embed.return_value = {"embedded": 10, "skipped": 2}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_embed"](repo="myrepo", force=True)

    mem.repo_embed.assert_called_once_with("myrepo", force=True)
    assert result == {"embedded": 10, "skipped": 2}


def test_memo_repo_embed_default_force(tmp_cfg) -> None:
    """memo_repo_embed must default force=False."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_embed.return_value = {"embedded": 0, "skipped": 0}

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_repo_embed"](repo="myrepo")

    mem.repo_embed.assert_called_once_with("myrepo", force=False)


def test_memo_repo_status_delegates_to_memory(tmp_cfg) -> None:
    """memo_repo_status must delegate to memory.repo_status and return its result."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_status.return_value = {"repo": "myrepo", "exact_count": 100, "vec_count": 90}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_status"](repo="myrepo")

    mem.repo_status.assert_called_once_with("myrepo")
    assert result == {"repo": "myrepo", "exact_count": 100, "vec_count": 90}


def test_memo_repo_status_returns_none_for_missing(tmp_cfg) -> None:
    """memo_repo_status must return None when memory.repo_status returns None."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_status.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_status"](repo="nonexistent")

    assert result is None


def test_memo_repo_search_calls_to_dict_on_each_hit(tmp_cfg) -> None:
    """memo_repo_search must call .to_dict() on each hit and return a list of dicts."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    hit1 = MagicMock()
    hit1.to_dict.return_value = {"path": "src/foo.py", "score": 0.9, "text": "def foo(): ..."}
    hit2 = MagicMock()
    hit2.to_dict.return_value = {"path": "src/bar.py", "score": 0.7, "text": "def bar(): ..."}
    mem.repo_search.return_value = [hit1, hit2]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_search"](
        query="find foo",
        limit=5,
        repo="myrepo",
        path="src/",
        mode="vec",
    )

    mem.repo_search.assert_called_once_with(
        "find foo",
        limit=5,
        repo="myrepo",
        path="src/",
        mode="vec",
    )
    hit1.to_dict.assert_called_once()
    hit2.to_dict.assert_called_once()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"path": "src/foo.py", "score": 0.9, "text": "def foo(): ..."}
    assert result[1] == {"path": "src/bar.py", "score": 0.7, "text": "def bar(): ..."}


def test_memo_repo_search_defaults(tmp_cfg) -> None:
    """memo_repo_search must use default limit=10, mode='hybrid', repo=None, path=None."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_search.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_search"](query="hello")

    mem.repo_search.assert_called_once_with(
        "hello",
        limit=10,
        repo=None,
        path=None,
        mode="hybrid",
    )
    assert result == []


def test_memo_repo_search_empty_result(tmp_cfg) -> None:
    """memo_repo_search must return an empty list when no hits are found."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_search.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_search"](query="zzz_no_match")

    assert isinstance(result, list)
    assert result == []


def test_memo_repo_get_file_delegates_to_memory(tmp_cfg) -> None:
    """memo_repo_get_file must delegate to memory.repo_get_file and return its result."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_get_file.return_value = {"path": "src/foo.py", "content": "def foo(): pass\n"}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_get_file"](repo="myrepo", path="src/foo.py", start=1, end=10)

    mem.repo_get_file.assert_called_once_with("myrepo", "src/foo.py", start=1, end=10)
    assert result == {"path": "src/foo.py", "content": "def foo(): pass\n"}


def test_memo_repo_get_file_default_range(tmp_cfg) -> None:
    """memo_repo_get_file must default start=None, end=None."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_get_file.return_value = {"path": "README.md", "content": "# Hello\n"}

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_repo_get_file"](repo="myrepo", path="README.md")

    mem.repo_get_file.assert_called_once_with("myrepo", "README.md", start=None, end=None)


def test_memo_repo_get_file_returns_none_for_missing(tmp_cfg) -> None:
    """memo_repo_get_file must return None when the file is not indexed."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_get_file.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_get_file"](repo="myrepo", path="does/not/exist.py")

    assert result is None


def test_memo_repo_list_delegates_to_memory(tmp_cfg) -> None:
    """memo_repo_list must delegate to memory.repo_list and return its result."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_list.return_value = [
        {"name": "repo1", "url": "https://github.com/a/b.git"},
        {"name": "repo2", "url": "https://github.com/c/d.git"},
    ]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_list"](limit=50)

    mem.repo_list.assert_called_once_with(limit=50)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "repo1"


def test_memo_repo_list_default_limit(tmp_cfg) -> None:
    """memo_repo_list must default limit=100."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_list.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_repo_list"]()

    mem.repo_list.assert_called_once_with(limit=100)


def test_memo_repo_delete_wraps_result_in_envelope(tmp_cfg) -> None:
    """memo_repo_delete must wrap memory.repo_delete's bool in a {'deleted': ...} envelope."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_delete.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_delete"](repo="myrepo", remove_clone=True)

    mem.repo_delete.assert_called_once_with("myrepo", remove_clone=True)
    assert result == {"deleted": True}


def test_memo_repo_delete_default_remove_clone(tmp_cfg) -> None:
    """memo_repo_delete must default remove_clone=True."""
    from memo.server_repo import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.repo_delete.return_value = False

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_repo_delete"](repo="myrepo")

    mem.repo_delete.assert_called_once_with("myrepo", remove_clone=True)
    assert result == {"deleted": False}


def test_no_module_level_mlx_imports() -> None:
    """server_repo must not have module-level MLX imports (deferred-import invariant)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_repo.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"
