from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from memo.cli import cli
from memo.dashboard import read_recall_log
from memo.server import build_server


def _env(tmp_path: Path, *, context_pack: str = "1") -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_CONTEXT_PACK": context_pack,
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_RERANKER_ENABLED": "0",
    }


def _hit(**overrides):
    base = {
        "id": "abc12345deadbeef",
        "title": "Current status",
        "type": "note",
        "body": "The current state is documented here.",
        "score": 0.91,
        "tags": [],
        "extra": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


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


def test_context_pack_cli_disabled_by_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["context-pack", "what is current?", "--json"],
        env=_env(tmp_path, context_pack="0"),
    )
    assert result.exit_code != 0
    assert "MEMO_CONTEXT_PACK=1" in result.output


def test_context_pack_cli_logs_consult(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeMemory:
        def search(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [_hit()]

    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: _FakeMemory())

    result = CliRunner().invoke(
        cli,
        ["context-pack", "what is current?", "--json", "--source", "synapse"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    rows = read_recall_log(tmp_path / "state", limit=5)
    assert len(rows) == 1
    assert rows[0]["via"] == "cli:context_pack"
    assert rows[0]["source"] == "synapse"
    assert rows[0]["hits"] == [
        {
            "id": "abc12345",
            "score": 0.91,
            "title": "Current status",
        }
    ]
    assert captured["kwargs"]["read_through"] is False


def test_context_pack_mcp_empty_corpus(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))
    assert tool is not None

    payload = tool.fn(question="what is current?")

    assert payload["question"] == "what is current?"
    assert payload["current_facts"] == []
    assert payload["supporting_context"] == []
    assert payload["stale_or_conflicting"] == []


def test_context_pack_mcp_disabled_by_flag(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "0")
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="what is current?")

    assert payload == {
        "status": "disabled",
        "reason": "MEMO_CONTEXT_PACK=0 disables explicit context-pack tools.",
        "question": "what is current?",
    }


def test_context_pack_mcp_logs_consult(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    captured: dict[str, object] = {}

    def _search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [_hit()]

    mem_with_stub.search = _search  # type: ignore[method-assign]
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="what is current?", source="synapse")

    assert payload["question"] == "what is current?"
    rows = read_recall_log(mem_with_stub.cfg.state_dir, limit=5)
    assert len(rows) == 1
    assert rows[0]["via"] == "mcp:context_pack"
    assert rows[0]["source"] == "synapse"
    assert rows[0]["hits"] == [
        {
            "id": "abc12345",
            "score": 0.91,
            "title": "Current status",
        }
    ]
    assert captured["kwargs"]["read_through"] is False


class _CompactFakeGraph:
    def memory_entities(self, memory_id):
        if memory_id in ("m1", "m2"):
            return [{"name": "proyecto-omega", "type": "topic", "mention_count": 1}]
        return []

    def total_indexed_memories(self):
        return 10

    def entity_doc_freqs(self, names):
        return {"proyecto-omega": 1.0} if "proyecto-omega" in names else {}


def _omega_hits():
    return [
        _hit(id="m1", title="Nota uno", score=0.9, body="cuerpo uno sobre proyecto omega"),
        _hit(id="m2", title="Nota dos", score=0.85, body="cuerpo dos sobre proyecto omega"),
    ]


def test_context_pack_graph_compact_collapses_when_enabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    monkeypatch.setenv("MEMO_CONTEXT_GRAPH_COMPACT", "1")
    mem_with_stub.search = lambda *a, **kw: _omega_hits()  # type: ignore[method-assign]
    monkeypatch.setattr(mem_with_stub, "graph", _CompactFakeGraph())
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="qué sabés del proyecto omega?")

    rows = (
        payload["current_facts"] + payload["supporting_context"] + payload["stale_or_conflicting"]
    )
    ids = {row["id"] for row in rows}
    assert len(ids) == 1
    assert ids <= {"m1", "m2"}


def test_context_pack_graph_compact_noop_when_disabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    monkeypatch.setenv("MEMO_CONTEXT_GRAPH_COMPACT", "0")
    mem_with_stub.search = lambda *a, **kw: _omega_hits()  # type: ignore[method-assign]
    monkeypatch.setattr(mem_with_stub, "graph", _CompactFakeGraph())
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="qué sabés del proyecto omega?")

    rows = (
        payload["current_facts"] + payload["supporting_context"] + payload["stale_or_conflicting"]
    )
    ids = {row["id"] for row in rows}
    assert ids == {"m1", "m2"}
