"""End-user contract for the packaged MCP stdio process.

Most server tests use FastMCP's in-process transport.  This test crosses the
actual process/protocol boundary used by Codex, Claude Code, and other MCP
clients so entry-point, environment, serialization, and shutdown regressions
cannot hide behind the in-memory adapter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client


def _server_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_CONFIG_FILE": str(tmp_path / "missing-config.toml"),
        "MEMO_CONFIG_DIR": str(tmp_path / "missing-config-dir"),
        "MEMO_MCP_PROFILE": "full",
        "MEMO_EMBEDDER_BACKEND": "mlx",
        "MEMO_EMBEDDER_MODEL": str(tmp_path / "missing-model"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
        "MEMO_UPDATE_CHECK_ENABLED": "0",
        "MEMO_AUTO_UPDATE": "0",
        "MEMO_STATUSLINE_SELFHEAL": "0",
        "MEMO_HOOK_SELFHEAL": "0",
        "MEMO_CODEGRAPH_DISCOVERY": "0",
    }


def _stop_isolated_idle_daemon(env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "memo.cli", "idle-daemon", "stop"],
        env={**os.environ, **env},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.asyncio
async def test_stdio_process_exposes_full_surface_and_core_crud(tmp_path: Path) -> None:
    env = _server_env(tmp_path)
    config = {
        "mcpServers": {
            "memo": {
                "command": sys.executable,
                "args": ["-m", "memo.server"],
                "env": env,
            }
        }
    }

    try:
        async with Client(config, timeout=20, init_timeout=20) as client:
            tools = await client.list_tools()
            assert len(tools) == 161

            version = (await client.call_tool("memo_version", {})).data
            assert version["version"]

            saved = (
                await client.call_tool(
                    "memo_save",
                    {
                        "content": "stdio end-user protocol proof",
                        "title": "stdio journey",
                        "type": "note",
                        "tags": ["audit", "stdio"],
                        "scope": "global",
                        "defer_embed": True,
                    },
                )
            ).data
            memory_id = saved["id"]
            assert saved["action"] == "created"
            assert saved["index_pending"] is True

            fetched = (await client.call_tool("memo_get", {"id": memory_id[:12]})).data
            assert fetched["body"] == "stdio end-user protocol proof"

            searched = (
                await client.call_tool(
                    "memo_search",
                    {
                        "query": "protocol proof",
                        "mode": "bm25",
                        "source": "codex-e2e",
                    },
                )
            ).data
            assert [hit["id"] for hit in searched["hits"]] == [memory_id]

            assert (await client.call_tool("memo_rerank", {"query": "", "hits": []})).data == []
            assert (await client.call_tool("memo_consolidate_list_archived", {})).data == []
            relation_reviews = (await client.call_tool("mem_relation_reviews", {})).data
            assert relation_reviews == {"pending": [], "count": 0}

            forgotten = (
                await client.call_tool(
                    "memo_forget", {"id": memory_id, "reason": "stdio round-trip"}
                )
            ).data
            assert forgotten == {"forgotten": True, "id": memory_id}
            restored = (await client.call_tool("memo_unforget", {"id": memory_id})).data
            assert restored == {"unforgotten": True, "id": memory_id}

            updated = (
                await client.call_tool(
                    "memo_update",
                    {"id": memory_id, "title": "stdio journey updated"},
                )
            ).data
            assert updated["title"] == "stdio journey updated"

            history = (await client.call_tool("memo_history", {"id": memory_id, "limit": 10})).data
            assert {event["op"] for event in history} >= {"save", "update"}

            deleted = (await client.call_tool("memo_delete", {"id": memory_id})).data
            assert deleted == {"deleted": True}
            assert (await client.call_tool("memo_get", {"id": memory_id})).data is None
    finally:
        _stop_isolated_idle_daemon(env)
