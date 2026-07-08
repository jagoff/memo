from __future__ import annotations

import json

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
