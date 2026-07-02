"""Tests for server_backup MCP tool registration."""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

from memo.sync import BackupMetadata


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


def _fake_metadata(name: str = "backup-20260701") -> BackupMetadata:
    return BackupMetadata(
        timestamp="2026-07-01T03:00:00+00:00",
        memory_count=42,
        checksum="abc123",
        compressed_size=1024,
        original_size=4096,
        name=name,
    )


def test_register_exposes_all_three_tools(tmp_cfg) -> None:
    """register() must expose exactly the three expected MCP tools."""
    from memo.memory import Memory
    from memo.server_backup import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_backup_create", "memo_backup_list", "memo_backup_restore"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_backup_create_calls_memory_and_returns_dict(tmp_cfg) -> None:
    """memo_backup_create must delegate to memory.backup.create_backup and return asdict envelope."""
    from memo.memory import Memory
    from memo.server_backup import register

    meta = _fake_metadata("my-backup")
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.create_backup.return_value = meta

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_backup_create"](compress=True, name="my-backup")

    mem.backup.create_backup.assert_called_once_with(compress=True, name="my-backup")
    assert result == dataclasses.asdict(meta)
    assert result["name"] == "my-backup"
    assert result["memory_count"] == 42
    assert result["checksum"] == "abc123"


def test_memo_backup_create_default_args(tmp_cfg) -> None:
    """memo_backup_create with no args must pass default compress=True, name=None."""
    from memo.memory import Memory
    from memo.server_backup import register

    meta = _fake_metadata()
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.create_backup.return_value = meta

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_backup_create"]()

    mem.backup.create_backup.assert_called_once_with(compress=True, name=None)


def test_memo_backup_list_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_backup_list must call list_backups and return a list of asdict envelopes."""
    from memo.memory import Memory
    from memo.server_backup import register

    meta_a = _fake_metadata("backup-a")
    meta_b = _fake_metadata("backup-b")
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.list_backups.return_value = [meta_a, meta_b]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_backup_list"]()

    mem.backup.list_backups.assert_called_once_with()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == dataclasses.asdict(meta_a)
    assert result[1] == dataclasses.asdict(meta_b)
    assert result[0]["name"] == "backup-a"
    assert result[1]["name"] == "backup-b"


def test_memo_backup_list_empty(tmp_cfg) -> None:
    """memo_backup_list must return an empty list when no backups exist."""
    from memo.memory import Memory
    from memo.server_backup import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.list_backups.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_backup_list"]()

    assert result == []
    assert isinstance(result, list)


def test_memo_backup_restore_success(tmp_cfg) -> None:
    """memo_backup_restore must delegate to restore_backup and return a success envelope."""
    from memo.memory import Memory
    from memo.server_backup import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.restore_backup.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_backup_restore"](
        backup_name="backup-20260701",
        restore_memories=True,
        restore_dbs=True,
    )

    mem.backup.restore_backup.assert_called_once_with(
        "backup-20260701",
        restore_memories=True,
        restore_dbs=True,
    )
    assert result == {"success": True, "backup_name": "backup-20260701"}


def test_memo_backup_restore_failure(tmp_cfg) -> None:
    """memo_backup_restore must propagate a False result from restore_backup."""
    from memo.memory import Memory
    from memo.server_backup import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.restore_backup.return_value = False

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_backup_restore"](backup_name="bad-backup")

    assert result["success"] is False
    assert result["backup_name"] == "bad-backup"


def test_memo_backup_restore_partial_flags(tmp_cfg) -> None:
    """memo_backup_restore must pass through restore_memories and restore_dbs flags."""
    from memo.memory import Memory
    from memo.server_backup import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.backup.restore_backup.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_backup_restore"](
        backup_name="partial-backup",
        restore_memories=False,
        restore_dbs=True,
    )

    mem.backup.restore_backup.assert_called_once_with(
        "partial-backup",
        restore_memories=False,
        restore_dbs=True,
    )
