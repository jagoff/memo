from __future__ import annotations

import json
from unittest.mock import MagicMock

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_VAULT_PATH": str(tmp_cfg.vault_path),
        "MEMO_AUTO_PROJECT_TAG": "0",
    }


def test_graph_export_json_creates_parent_directories(tmp_path, tmp_cfg) -> None:
    output = tmp_path / "nested" / "graph" / "memo-graph.json"

    result = CliRunner().invoke(
        cli,
        ["graph", "export", "--format", "json", "--output", str(output)],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data


def test_graph_rebuild_and_stats_json_report_projection(tmp_cfg) -> None:
    runner = CliRunner()
    env = _env(tmp_cfg)

    rebuild = runner.invoke(cli, ["graph", "rebuild", "--json"], env=env)
    stats = runner.invoke(cli, ["graph", "stats", "--json"], env=env)

    assert rebuild.exit_code == 0, rebuild.output
    assert stats.exit_code == 0, stats.output
    rebuilt = json.loads(rebuild.output)
    payload = json.loads(stats.output)
    assert rebuilt["projection"]["activated"] is True
    assert payload["projection"]["active_version"]
    assert "rejection_reasons" in payload["projection"]


def test_graph_trace_json_routes_to_memory_api(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_trace(self, **kwargs):
            assert kwargs == {"memory_id": "abc123", "code": None, "limit": 25}
            return {"available": True, "code_refs": [{"uri": "codegraph://repo/symbol"}]}

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())
    result = CliRunner().invoke(
        cli,
        ["graph", "trace", "--memory", "abc123", "--limit", "25", "--json"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["code_refs"][0]["uri"] == "codegraph://repo/symbol"


def test_graph_discover_json_routes_to_curated_api(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_discover(self, **kwargs):
            assert kwargs["include_code"] is True
            return {"available": True, "projection_version": "v1", "communities": [], "bridges": []}

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())
    result = CliRunner().invoke(
        cli,
        ["graph", "discover", "--include-code", "--json"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["projection_version"] == "v1"


def test_graph_stats_text_reports_active_projection(tmp_cfg, monkeypatch) -> None:
    health = {
        "raw": {"entities": 7, "links": 11},
        "edges": {
            "edges": 4,
            "edges_gt1": 2,
            "weight_min": 1.0,
            "weight_mean": 2.5,
            "weight_max": 5.0,
        },
        "projection": {
            "active_version": "abcdef123456",
            "node_count": 6,
            "edge_count": 3,
            "rejected_count": 1,
            "code_node_count": 2,
            "code_link_count": 4,
        },
    }
    memory = type("Memory", (), {"graph_health": lambda self: health})()
    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: memory)

    result = CliRunner().invoke(cli, ["graph", "stats"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "entities 7" in output
    assert "2 = 50.0% > 1" in output
    assert "projection abcdef12" in output
    assert "6 nodes / 3 edges / 1 rejected / 2 code nodes / 4 memory↔code links" in output


def test_graph_stats_text_reports_missing_projection(tmp_cfg, monkeypatch) -> None:
    health = {
        "raw": {"entities": 0, "links": 0},
        "edges": {
            "edges": 0,
            "edges_gt1": 0,
            "weight_min": 0.0,
            "weight_mean": 0.0,
            "weight_max": 0.0,
        },
        "projection": {"active_version": None},
    }
    memory = type("Memory", (), {"graph_health": lambda self: health})()
    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: memory)

    result = CliRunner().invoke(cli, ["graph", "stats"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "projection: missing" in result.output


def test_graph_trace_requires_exactly_one_direction(tmp_cfg, monkeypatch) -> None:
    get_memory = MagicMock()
    monkeypatch.setattr("memo.cli_graph._get_memory", get_memory)

    result = CliRunner().invoke(cli, ["graph", "trace"], env=_env(tmp_cfg))

    assert result.exit_code == 2
    assert "provide exactly one of --memory or --code" in result.output
    get_memory.assert_not_called()


def test_graph_trace_text_reports_unavailable_reason(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_trace(self, **kwargs):
            assert kwargs == {"memory_id": None, "code": "memo.search", "limit": 50}
            return {"available": False, "reason": "projection_stale"}

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())

    result = CliRunner().invoke(cli, ["graph", "trace", "--code", "memo.search"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "trace unavailable: projection_stale" in result.output


def test_graph_trace_text_renders_code_and_memory_evidence(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_trace(self, **kwargs):
            assert kwargs == {"memory_id": "memory-1", "code": None, "limit": 50}
            return {
                "available": True,
                "projection_version": "1234567890",
                "code_refs": [
                    {
                        "qualified_name": "memo.search",
                        "label": "search",
                        "uri": "codegraph://memo/search",
                        "relation": "mentions",
                    }
                ],
                "memories": [
                    {"id": "abcdef123", "title": "Search contract", "relation": "supports"}
                ],
            }

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())

    result = CliRunner().invoke(cli, ["graph", "trace", "--memory", "memory-1"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "projection 12345678" in result.output
    assert "memo.search codegraph://memo/search (mentions)" in result.output
    assert "abcdef12 Search contract (supports)" in result.output


def test_graph_discover_text_reports_unavailable_reason(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_discover(self, **_kwargs):
            return {"available": False, "reason": "projection_missing"}

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())

    result = CliRunner().invoke(cli, ["graph", "discover"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "discovery unavailable: projection_missing" in result.output


def test_graph_discover_text_renders_communities_and_bridges(tmp_cfg, monkeypatch) -> None:
    class _Mem:
        def graph_discover(self, **kwargs):
            assert kwargs == {
                "min_community_size": 4,
                "min_bridge_side": 2,
                "max_communities": 5,
                "max_bridges": 5,
                "include_code": False,
            }
            return {
                "available": True,
                "projection_version": "fedcba9876",
                "communities": [
                    {
                        "representative": {"label": "Retrieval", "uri": "entity://retrieval"},
                        "size": 3,
                        "memory_ids": ["m1", "m2"],
                    }
                ],
                "bridges": [
                    {
                        "left_rep": {"label": "Retrieval"},
                        "right_rep": {"label": "Storage"},
                        "bridge": {"label": "SQLite"},
                    }
                ],
            }

    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _Mem())

    result = CliRunner().invoke(cli, ["graph", "discover", "--no-code"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "1 communities / 1 bridges projection fedcba98" in result.output
    assert "Retrieval (3 nodes, 2 memories)" in result.output
    assert "Retrieval ↔ Storage via SQLite" in result.output
