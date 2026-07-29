"""Elicitation gate on irreversible MCP tools (server_elicit).

Matrix per docs/SPECS/2026-07-28-mcp-elicitation-destructive-ops-design.md:
accept / decline / cancel / no-handler (fail-open) / MEMO_ELICIT_CONFIRM=0
for each gated tool, plus decline-signal on/off and ungated-tools-never-elicit.

Uses an in-process `fastmcp.Client(server, elicitation_handler=...)`: omitting
the handler makes the client NOT advertise the elicitation capability — that
IS the fail-open fixture.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from memo.memory import Memory
from memo.server import build_server
from memo.server_elicit import GATED_TOOLS


def _call(server: Any, tool: str, args: dict[str, Any], handler: Any = None) -> Any:
    from fastmcp import Client

    async def _run() -> Any:
        kwargs = {"elicitation_handler": handler} if handler is not None else {}
        async with Client(server, **kwargs) as c:
            res = await c.call_tool(tool, args)
            return res.data

    return asyncio.run(_run())


def _accept_handler(action: str, calls: list[str] | None = None) -> Any:
    async def handler(message: str, response_type: Any, params: Any, ctx: Any) -> str:
        if calls is not None:
            calls.append(message)
        return action

    return handler


def _result_handler(client_action: str, calls: list[str] | None = None) -> Any:
    """Handler replying with a raw ElicitResult action ("decline"/"cancel")."""

    async def handler(message: str, response_type: Any, params: Any, ctx: Any) -> Any:
        from fastmcp.client.elicitation import ElicitResult

        if calls is not None:
            calls.append(message)
        return ElicitResult(action=client_action)

    return handler


def _signal_count(mem: Memory) -> int:
    row = mem.store.connection.execute(
        "SELECT COUNT(*) AS c FROM meta WHERE type = 'feedback'"
    ).fetchone()
    return int(row["c"])


@dataclass
class Scenario:
    """One gated tool: how to arm it, call it, and check both outcomes."""

    tool: str
    action: str
    setup: Callable[[Memory, Path], tuple[dict[str, Any], Callable[[], bool] | None]]
    accept_ok: Callable[[dict[str, Any]], bool]
    env: dict[str, str] = field(default_factory=dict)


def _setup_delete(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], Callable[[], bool]]:
    rec = mem.save(content="doomed memory body", title="Doomed", type_="note")
    return {"id": rec.id}, lambda: mem.get(rec.id) is None


def _setup_synth(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], Callable[[], bool]]:
    rec = mem.save(content="synth insight body", title="Synth insight", type_="synthesis")
    return {"id": rec.id}, lambda: mem.get(rec.id) is None


def _setup_backup(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], None]:
    mem.save(content="pre-backup memory", title="Base", type_="note")
    # compress=False: a near-empty gz archive trips the restore-side
    # suspicious-compression-ratio (zip bomb) guard.
    meta = mem.backup.create_backup(compress=False, name="elicitbk")
    # restore_dbs=False: keep the live sqlite connection valid post-restore.
    return {"backup_name": meta.name, "restore_dbs": False}, None


def _setup_feedback(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], Callable[[], bool]]:
    rec = mem.save(content="feedback target", title="FB target", type_="note")
    mem.feedback_record(rec.id, query_text="some query", rating="up")
    return {"source_id": rec.id}, lambda: len(mem.feedback_list(source_id=rec.id)) == 0


def _setup_repo(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], Callable[[], bool]]:
    clone = tmp_path / "repo-clone"
    clone.mkdir(exist_ok=True)
    mem.store.upsert_repo_source(
        {
            "id": "repo-elicit-1",
            "name": "elicit-repo",
            "url": "https://example.invalid/elicit-repo.git",
            "ref": "main",
            "commit_sha": "deadbeef",
            "clone_path": str(clone),
            "indexed_at": "2026-07-28T00:00:00Z",
            "status": "ready",
        }
    )
    return {"repo": "elicit-repo"}, lambda: mem.store.get_repo_source("elicit-repo") is None


def _setup_cache(mem: Memory, tmp_path: Path) -> tuple[dict[str, Any], Callable[[], bool]]:
    mem.save(content="cache entry one", title="C1", type_="note")
    mem.save(content="cache entry two", title="C2", type_="note")
    before = mem.store.count()
    return {}, lambda: mem.store.count() < before


SCENARIOS: dict[str, Scenario] = {
    "memo_delete": Scenario(
        tool="memo_delete",
        action="delete",
        setup=_setup_delete,
        accept_ok=lambda d: d["deleted"] is True,
    ),
    "memo_synthesize_delete": Scenario(
        tool="memo_synthesize_delete",
        action="delete",
        setup=_setup_synth,
        accept_ok=lambda d: d["deleted"] is True,
    ),
    "memo_backup_restore": Scenario(
        tool="memo_backup_restore",
        action="restore",
        setup=_setup_backup,
        accept_ok=lambda d: d["success"] is True,
    ),
    "memo_feedback_clear": Scenario(
        tool="memo_feedback_clear",
        action="clear",
        setup=_setup_feedback,
        accept_ok=lambda d: d["deleted"] == 1,
    ),
    "memo_repo_delete": Scenario(
        tool="memo_repo_delete",
        action="delete",
        setup=_setup_repo,
        accept_ok=lambda d: d["deleted"] is True,
    ),
    "memo_cache_evict": Scenario(
        tool="memo_cache_evict",
        action="evict",
        setup=_setup_cache,
        accept_ok=lambda d: d["count"] >= 1,
        env={
            "MEMO_CACHE_MODE": "read_through",
            "MEMO_CACHE_MAX_ENTRIES": "1",
            "MEMO_CACHE_BACKEND": "none",
        },
    ),
}


def _arm(
    name: str, mem: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Scenario, dict[str, Any], Callable[[], bool] | None, Any]:
    scn = SCENARIOS[name]
    # Env AFTER setup: with the cache tier already on, the setup saves would
    # auto-evict at save time (write_ops capacity bound) and leave no
    # overflow for the tool to confirm.
    args, destroyed = scn.setup(mem, tmp_path)
    for k, v in scn.env.items():
        monkeypatch.setenv(k, v)
    if scn.env:
        mem._cache = None  # drop the memoized CacheManager so it re-reads env
    return scn, args, destroyed, build_server(mem)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gated_accept_executes(name, mock_memory, tmp_path, monkeypatch):
    scn, args, destroyed, server = _arm(name, mock_memory, tmp_path, monkeypatch)
    data = _call(server, scn.tool, args, handler=_accept_handler(scn.action))
    assert scn.accept_ok(data), data
    if destroyed is not None:
        assert destroyed()
    assert _signal_count(mock_memory) == 0  # accept writes no decline signal


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gated_decline_aborts_and_writes_signal(name, mock_memory, tmp_path, monkeypatch):
    scn, args, destroyed, server = _arm(name, mock_memory, tmp_path, monkeypatch)
    data = _call(server, scn.tool, args, handler=_result_handler("decline"))
    assert data == {"ok": False, "aborted": "declined"}
    if destroyed is not None:
        assert not destroyed()  # nothing deleted
    assert _signal_count(mock_memory) == 1  # decline-as-signal memory written


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gated_cancel_aborts_no_signal(name, mock_memory, tmp_path, monkeypatch):
    scn, args, destroyed, server = _arm(name, mock_memory, tmp_path, monkeypatch)
    data = _call(server, scn.tool, args, handler=_result_handler("cancel"))
    assert data == {"ok": False, "aborted": "cancelled"}
    if destroyed is not None:
        assert not destroyed()
    assert _signal_count(mock_memory) == 0  # cancel is a pure no-op


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gated_no_handler_fail_open(name, mock_memory, tmp_path, monkeypatch):
    """Client without the elicitation capability proceeds unconfirmed (must-not-brick)."""
    scn, args, destroyed, server = _arm(name, mock_memory, tmp_path, monkeypatch)
    data = _call(server, scn.tool, args)  # no elicitation_handler
    assert scn.accept_ok(data), data
    if destroyed is not None:
        assert destroyed()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gated_flag_off_never_elicits(name, mock_memory, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_ELICIT_CONFIRM", "0")
    scn, args, destroyed, server = _arm(name, mock_memory, tmp_path, monkeypatch)
    calls: list[str] = []
    data = _call(server, scn.tool, args, handler=_result_handler("decline", calls))
    assert calls == []  # handler never invoked
    assert scn.accept_ok(data), data
    if destroyed is not None:
        assert destroyed()


def test_decline_signal_flag_off_suppresses_write(mock_memory, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_ELICIT_DECLINE_SIGNAL", "0")
    scn, args, destroyed, server = _arm("memo_delete", mock_memory, tmp_path, monkeypatch)
    data = _call(server, scn.tool, args, handler=_result_handler("decline"))
    assert data == {"ok": False, "aborted": "declined"}
    assert destroyed is not None and not destroyed()
    assert _signal_count(mock_memory) == 0


@pytest.mark.parametrize("tool", ["memo_forget", "memo_update"])
def test_ungated_tools_never_elicit(tool, mock_memory, monkeypatch):
    rec = mock_memory.save(content="reversible target", title="Reversible", type_="note")
    server = build_server(mock_memory)
    calls: list[str] = []
    args = {"id": rec.id} if tool == "memo_forget" else {"id": rec.id, "title": "Renamed"}
    data = _call(server, tool, args, handler=_result_handler("decline", calls))
    assert calls == []  # handler present + capability advertised, but no elicit
    assert data is not None and "aborted" not in data


def test_elicit_message_states_blast_radius(mock_memory):
    rec = mock_memory.save(content="blast body", title="Blast Radius Probe", type_="note")
    server = build_server(mock_memory)
    calls: list[str] = []
    data = _call(server, "memo_delete", {"id": rec.id}, handler=_accept_handler("delete", calls))
    assert data["deleted"] is True
    assert len(calls) == 1
    assert "Blast Radius Probe" in calls[0]
    assert "No trash" in calls[0]


def test_ctx_hidden_from_client_schema(mock_memory):
    """The injected Context param must not leak into the client-facing schema."""
    server = build_server(mock_memory)

    async def _schemas() -> dict[str, dict[str, Any]]:
        from fastmcp import Client

        async with Client(server) as c:
            tools = await c.list_tools()
            return {t.name: (t.inputSchema or {}) for t in tools}

    schemas = asyncio.run(_schemas())
    for tool in sorted(GATED_TOOLS):
        assert tool in schemas
        assert "ctx" not in schemas[tool].get("properties", {}), tool
