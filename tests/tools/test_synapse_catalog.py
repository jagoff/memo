from __future__ import annotations

from pathlib import Path

import pytest

from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.synapse_catalog import (
    SynapseCatalogError,
    discover_synapse_operations,
)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "synapse"
    package = root / "src" / "synapse"
    (package / "cli").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "source.json").write_bytes(
        canonical_json_bytes({"source_commit": "a" * 40})
    )
    (package / "mcp_catalog.py").write_text(
        """CANONICAL_MCP_TOOLS = [
    McpToolManifest(tool_id=\"synapse.federate.query\", mcp_name=\"synapse_federate_query\"),
    McpToolManifest(tool_id=\"synapse.chat.ask\", mcp_name=\"synapse_chat_ask\"),
]
""",
        encoding="utf-8",
    )
    (package / "cli" / "parser.py").write_text(
        """def build_parser():
    sub = parser.add_subparsers()
    sub.add_parser(\"query\")
""",
        encoding="utf-8",
    )
    for filename, symbols in {
        "runtime.py": "def runtime_loop(): pass\n",
        "watcher.py": "def _emit(): pass\n",
        "morning_digest.py": "def run_morning_digest(): pass\n",
        "whatsapp_live.py": "def last_messages(): pass\ndef last_messages_multi(): pass\n",
        "vault_archive.py": "def move_to_archive(): pass\n",
    }.items():
        (package / filename).write_text(symbols, encoding="utf-8")
    (root / "tests" / "test_runtime.py").write_text("pass\n", encoding="utf-8")
    return root


def test_catalog_includes_canonical_mcp_and_live_daemon_operations(snapshot: Path) -> None:
    rows = discover_synapse_operations(snapshot)
    names = {row.source_operation for row in rows}

    assert "synapse.federate.query" in names
    assert "synapse.chat.ask" in names
    assert "synapse.runtime.loop" in names
    assert "synapse.watcher.event" in names


def test_catalog_excludes_runtime_self_audit_from_admission(snapshot: Path) -> None:
    rows = discover_synapse_operations(snapshot)

    runtime = next(row for row in rows if row.source_operation == "synapse.runtime.loop")
    assert runtime.exclusion_reason == "self_audit"


def test_catalog_rejects_noncanonical_source_record(snapshot: Path) -> None:
    (snapshot / "source.json").write_text(
        '{"source_commit": "' + "a" * 40 + '"}', encoding="utf-8"
    )

    with pytest.raises(SynapseCatalogError, match="canonical"):
        discover_synapse_operations(snapshot)
