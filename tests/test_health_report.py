"""`memo health` — corpus/index/embedder health summary.

A single read-only snapshot of operational state: corpus size, index
dims, embedder profile, health-score coverage, and derived warnings.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli
from memo.health_report import build_health_report


def test_health_report_empty_corpus(mock_memory):
    report = build_health_report(mock_memory)
    assert report["corpus"]["memorias"] == 0
    assert report["index"]["expected_dims"] == mock_memory.cfg.embedder_dims
    # An empty corpus should surface at least one warning.
    assert report["warnings"], "empty corpus should warn"


def test_health_report_counts_memorias(mock_memory):
    mock_memory.save(content="one", title="One", tags=["t"])
    mock_memory.save(content="two", title="Two", tags=["t"])
    report = build_health_report(mock_memory)
    assert report["corpus"]["memorias"] == 2


def test_health_report_warns_when_health_scores_unpopulated(mock_memory):
    mock_memory.save(content="one", title="One", tags=["t"])
    report = build_health_report(mock_memory)
    assert report["health_table"]["tracked"] == 0
    joined = " ".join(report["warnings"]).lower()
    assert "health" in joined or "dream" in joined or "contradict" in joined


def test_health_report_no_embedder_probe_by_default(mock_memory):
    report = build_health_report(mock_memory)
    assert report["embedder"]["latency_ms"] is None


def test_server_health_summary_tool(mock_memory):
    import asyncio

    from memo.server import build_server

    mock_memory.save(content="one", title="One", tags=["t"])
    server = build_server(memory=mock_memory)
    tool = asyncio.run(server.get_tool("memory_health_summary")).fn
    out = tool()
    assert out["corpus"]["memorias"] == 1


def test_cli_health_json(monkeypatch, mock_memory):
    monkeypatch.setattr("memo.cli_health._get_memory", lambda cfg: mock_memory)
    monkeypatch.setattr("memo.cli_health.Config.from_env", staticmethod(lambda: mock_memory.cfg))
    result = CliRunner().invoke(cli, ["health", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "corpus" in data
    assert data["corpus"]["memorias"] == 0
