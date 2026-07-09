from __future__ import annotations

import asyncio
import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.server import build_server


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_CONTEXT_PACK": "1",
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_RERANKER_ENABLED": "0",
    }


def test_context_pack_cli_empty_corpus_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["context-pack", "what is current?", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["question"] == "what is current?"
    assert payload["current_facts"] == []


def test_context_pack_mcp_empty_corpus(mem_with_stub) -> None:
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))
    assert tool is not None

    payload = tool.fn(question="what is current?")

    assert payload["question"] == "what is current?"
    assert payload["current_facts"] == []
    assert payload["supporting_context"] == []
    assert payload["stale_or_conflicting"] == []
