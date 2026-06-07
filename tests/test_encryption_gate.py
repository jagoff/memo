"""`MEMO_ENCRYPTION_ENABLED` gate — CLI group + MCP tools refuse when off.

The at-rest encryption vertical (`encryption.py` + `cli_encrypt` + `server_encrypt`)
ships EXPERIMENTAL and is gated OFF by default. These tests lock the two
user-facing surfaces in both states:

- OFF (default): `memo encrypt …` exits non-zero with a disabled message and
  the `memory_encrypt_*` MCP tools return `{"ok": False, "status": "disabled"}`
  — neither touches the key manager.
- ON (`MEMO_ENCRYPTION_ENABLED=1`): both surfaces proceed to the real path.

Mirrors the `SYNAPSE_MEMFLOW_TRANSPORT` hermetic approach: the flag is set per
test via `monkeypatch.setenv`. The module-level `test_encryption.py` suite
exercises the EncryptionManager classes directly and is unaffected by the gate.
"""

from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    return Memory(cfg)


def _tool(server, name):
    tool = asyncio.run(server.get_tool(name))
    if tool is None:
        raise RuntimeError(f"tool {name!r} not registered")
    return tool.fn


# ── CLI surface ───────────────────────────────────────────────────────────────

_CLI_CMDS = (
    ["encrypt", "status"],
    ["encrypt", "lock"],
    ["encrypt", "unlock", "pw"],
)


@pytest.mark.parametrize("argv", _CLI_CMDS, ids=lambda a: a[1])
def test_cli_disabled_by_default(argv, monkeypatch, tmp_path):
    """Flag unset → non-zero exit + disabled message, key manager untouched."""
    monkeypatch.delenv("MEMO_ENCRYPTION_ENABLED", raising=False)
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    result = CliRunner().invoke(cli, argv, env=env)
    assert result.exit_code != 0, result.output
    assert "MEMO_ENCRYPTION_ENABLED" in result.output


def test_cli_enabled_proceeds(monkeypatch, tmp_path):
    """Flag on → guard passes through to the real command body."""
    monkeypatch.setenv("MEMO_ENCRYPTION_ENABLED", "1")

    class _FakeEnc:
        def is_unlocked(self) -> bool:
            return False

    class _FakeMem:
        encryption = _FakeEnc()

    monkeypatch.setattr("memo.cli_encrypt._get_memory", lambda cfg: _FakeMem())
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    result = CliRunner().invoke(cli, ["encrypt", "status"], env=env)
    assert result.exit_code == 0, result.output
    assert "locked" in result.output.lower()


# ── MCP surface ───────────────────────────────────────────────────────────────

_MCP_CALLS = (
    ("memory_encrypt_status", {}),
    ("memory_encrypt_lock", {}),
    ("memory_encrypt_unlock", {"password": "pw"}),
)


@pytest.mark.parametrize("name,kwargs", _MCP_CALLS, ids=lambda x: x if isinstance(x, str) else "")
def test_mcp_disabled_by_default(name, kwargs, mem, monkeypatch):
    """Flag unset → disabled payload, key manager untouched."""
    monkeypatch.delenv("MEMO_ENCRYPTION_ENABLED", raising=False)
    server = build_server(memory=mem)
    out = _tool(server, name)(**kwargs)
    assert out == {
        "ok": False,
        "status": "disabled",
        "error": "Encryption disabled (set MEMO_ENCRYPTION_ENABLED=1 to enable).",
    }


def test_mcp_enabled_proceeds(mem, monkeypatch):
    """Flag on → real status shape (is_unlocked key present)."""
    monkeypatch.setenv("MEMO_ENCRYPTION_ENABLED", "1")
    server = build_server(memory=mem)
    out = _tool(server, "memory_encrypt_status")()
    assert "is_unlocked" in out
    assert out["status"] in ("locked", "unlocked")
