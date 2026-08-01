from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from fastmcp import FastMCP

from memo.cli import cli
from memo.definitive_integration_runtime import _build_two_peers
from memo.errors import OperationalError
from memo.operational import OperationalStore
from memo.operational_mesh import OperationalMesh, mesh_identity
from memo.server_mesh import register


@dataclass
class _Config:
    device_id: str


class _MemoryAdapter:
    def __init__(self, peer: Any) -> None:
        self.cfg = _Config(peer.identity.device_id)
        self.operational = peer.store

    def close(self) -> None:
        return


def _invoke(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    memory: _MemoryAdapter,
    arguments: list[str],
) -> dict[str, Any]:
    monkeypatch.setattr("memo.cli_mesh._memory_from_env", lambda: memory)
    result = runner.invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert isinstance(value, dict)
    return value


def _tool(server: FastMCP[Any], name: str) -> Any:
    tool = asyncio.run(server.get_tool(name))
    assert tool is not None
    return tool.fn


def _server(memory: _MemoryAdapter) -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("memo-mesh-test")
    register(server, memory)
    return server


def test_cli_and_mcp_two_terminal_roundtrip_uses_only_memo_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    peer_a, peer_b, transport = _build_two_peers(tmp_path / "runtime", now=now)
    memory_a = _MemoryAdapter(peer_a)
    memory_b = _MemoryAdapter(peer_b)
    root_a = str(peer_a.sync.transport.root)
    root_b = str(peer_b.sync.transport.root)
    remote = str(transport.remote)
    assert Path(root_a).resolve() != Path(root_b).resolve()
    common_a = [
        "--transport",
        root_a,
        "--remote",
        remote,
        "--actor-id",
        "agent-a",
        "--session-id",
        "term-a",
    ]
    common_b = [
        "--transport",
        root_b,
        "--remote",
        remote,
        "--actor-id",
        "agent-b",
        "--session-id",
        "term-b",
    ]
    runner = CliRunner()

    sent_a = _invoke(
        runner,
        monkeypatch,
        memory_a,
        [
            "mesh",
            "message",
            "send",
            "handoff",
            "A completed the migration",
            "--to",
            "device-b:term-b",
            "--expects-ack",
            "--idempotency-key",
            "a-message-1",
            *common_a,
        ],
    )
    _invoke(runner, monkeypatch, memory_a, ["mesh", "sync", "publish", *common_a])
    before_ingest_b = _invoke(
        runner,
        monkeypatch,
        memory_b,
        ["mesh", "message", "list", "--channel", "handoff", *common_b],
    )
    assert before_ingest_b["messages"] == []
    _invoke(runner, monkeypatch, memory_b, ["mesh", "sync", "ingest", *common_b])

    listed_b = _invoke(
        runner,
        monkeypatch,
        memory_b,
        ["mesh", "message", "list", "--channel", "handoff", *common_b],
    )
    assert [row["body"] for row in listed_b["messages"]] == ["A completed the migration"]
    reserved_b = _invoke(
        runner,
        monkeypatch,
        memory_b,
        ["mesh", "delivery", "reserve", *common_b],
    )
    assert reserved_b["count"] == 1
    _invoke(
        runner,
        monkeypatch,
        memory_b,
        [
            "mesh",
            "delivery",
            "ack",
            sent_a["message_id"],
            "--idempotency-key",
            "b-ack-1",
            *common_b,
        ],
    )
    _invoke(
        runner,
        monkeypatch,
        memory_b,
        [
            "mesh",
            "presence",
            "announce",
            "memo",
            "/work/memo",
            "integration",
            "verifying A's handoff",
            "--file",
            "src/memo/operational_mesh.py",
            "--idempotency-key",
            "b-presence-1",
            *common_b,
        ],
    )
    _invoke(runner, monkeypatch, memory_b, ["mesh", "sync", "publish", *common_b])
    _invoke(runner, monkeypatch, memory_a, ["mesh", "sync", "ingest", *common_a])
    presence_at_a = _invoke(
        runner,
        monkeypatch,
        memory_a,
        ["mesh", "presence", "list", "--project", "memo", *common_a],
    )
    assert [(row["actor_id"], row["intent"]) for row in presence_at_a["presence"]] == [
        ("agent-b", "verifying A's handoff")
    ]

    server_a = _server(memory_a)
    server_b = _server(memory_b)
    send_b = _tool(server_b, "memo_mesh_message_send")
    publish_b = _tool(server_b, "memo_mesh_sync_publish")
    ingest_a = _tool(server_a, "memo_mesh_sync_ingest")
    list_a = _tool(server_a, "memo_mesh_message_list")
    reserve_a = _tool(server_a, "memo_mesh_delivery_reserve")
    ack_a = _tool(server_a, "memo_mesh_delivery_ack")
    announce_a = _tool(server_a, "memo_mesh_presence_announce")
    publish_a = _tool(server_a, "memo_mesh_sync_publish")
    ingest_b = _tool(server_b, "memo_mesh_sync_ingest")
    presence_b = _tool(server_b, "memo_mesh_presence_list")

    sent_b = send_b(
        transport_path=root_b,
        remote=remote,
        channel="handoff",
        body="B verified and acknowledged A",
        target_ids=["device-a:hermetic-session"],
        expects_ack=True,
        idempotency_key="b-message-1",
    )
    publish_b(
        transport_path=root_b,
        remote=remote,
    )
    before_ingest_a = list_a(
        transport_path=root_a,
        remote=remote,
        channel="handoff",
    )
    assert before_ingest_a["messages"] == []
    ingest_a(
        transport_path=root_a,
        remote=remote,
    )
    messages_a = list_a(
        transport_path=root_a,
        remote=remote,
        channel="handoff",
    )
    assert [row["body"] for row in messages_a["messages"]] == ["B verified and acknowledged A"]
    reserved_a = reserve_a(
        transport_path=root_a,
        remote=remote,
    )
    assert reserved_a["count"] == 1
    ack_a(
        transport_path=root_a,
        remote=remote,
        message_id=sent_b["message_id"],
        idempotency_key="a-ack-1",
    )
    announce_a(
        transport_path=root_a,
        remote=remote,
        project="memo",
        workspace="/work/memo",
        topic="integration",
        intent="received B's verification",
        idempotency_key="a-presence-1",
    )
    publish_a(
        transport_path=root_a,
        remote=remote,
    )
    ingest_b(
        transport_path=root_b,
        remote=remote,
    )
    visible_at_b = presence_b(
        transport_path=root_b,
        remote=remote,
        project="memo",
    )
    assert {row["actor_id"] for row in visible_at_b["presence"]} == {
        "agent-a",
        "agent-b",
    }


def test_mesh_fails_closed_without_v2_or_local_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OperationalStore(tmp_path / "legacy", device_id="device-a")
    identity = mesh_identity(
        device_id="device-a",
        actor_id="agent-a",
        source_client="pytest",
    )

    with pytest.raises(OperationalError, match="ledger v2"):
        OperationalMesh(
            store,
            identity=identity,
            transport_path=tmp_path / "transport",
        )

    peer_a, _peer_b, transport = _build_two_peers(
        tmp_path / "runtime",
        now=datetime.now(UTC),
    )
    foreign = mesh_identity(
        device_id="device-b",
        actor_id="agent-a",
        source_client="pytest",
    )
    with pytest.raises(OperationalError, match="local device"):
        OperationalMesh(
            peer_a.store,
            identity=foreign,
            transport_path=transport.root,
        )

    monkeypatch.setattr(peer_a.store, "_context_provider", None)
    with pytest.raises(OperationalError, match="authority is unavailable"):
        OperationalMesh(
            peer_a.store,
            identity=identity,
            transport_path=transport.root,
        )


def test_mesh_rejects_actor_impersonation_and_unqualified_recipients(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="safe terminal-principal syntax"):
        mesh_identity(
            device_id="device-a",
            actor_id="device-b:term-b",
            session_id="term-a",
            source_client="pytest",
        )

    peer_a, _peer_b, transport = _build_two_peers(
        tmp_path / "runtime",
        now=datetime.now(UTC),
    )
    mesh = OperationalMesh(
        peer_a.store,
        identity=mesh_identity(
            device_id="device-a",
            actor_id="agent-a",
            session_id="term-a",
            source_client="pytest",
        ),
        transport_path=peer_a.sync.transport.root,
        remote=str(transport.remote),
    )

    with pytest.raises(ValueError, match="device:session"):
        mesh.send_message(
            channel="handoff",
            body="attempt actor-addressed delivery",
            target_ids=("agent-b",),
            idempotency_key="unqualified-target",
        )


@pytest.mark.parametrize("command", ["events", "chat-session"])
def test_local_diagnostic_cli_help_routes_peer_coordination_to_mesh(command: str) -> None:
    result = CliRunner().invoke(cli, [command, "--help"])

    assert result.exit_code == 0
    assert "local diagnostic" in result.output.lower()
    assert "not replicated" in result.output.lower()
    assert "memo mesh" in result.output


def test_mesh_remote_argument_binds_an_isolated_local_clone(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    peer_a, _peer_b, _transport = _build_two_peers(tmp_path / "runtime", now=now)
    identity = mesh_identity(
        device_id="device-a",
        actor_id="agent-a",
        source_client="pytest",
    )

    remote = tmp_path / "remote.git"
    subprocess.run(
        ("git", "init", "--bare", "--quiet", str(remote)),
        check=True,
        capture_output=True,
        text=True,
    )

    mesh = OperationalMesh(
        peer_a.store,
        identity=identity,
        transport_path=tmp_path / "transport",
        remote=str(remote),
    )

    assert mesh.remote == str(remote)
    assert mesh.transport.remote == str(remote)
