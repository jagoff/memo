"""Tests for the multi-agent MCP preset registry."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from memo.runtime import agent_presets as ap


@dataclasses.dataclass(frozen=True)
class _Server:
    name: str
    command: str
    env: dict


def _server(cmd: Path) -> _Server:
    return _Server(name="memo", command=str(cmd), env={"MEMO_NONINTERACTIVE": "1"})


def test_vscode_uses_servers_key_and_type(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    res = ap.install_from_preset(ap.AGENT_PRESETS["vscode"], _server(tmp_path / "memo-mcp"), write=True)
    assert res["ok"] and res["action"] in {"created", "updated"}
    data = json.loads(Path(res["path"]).read_text())
    assert "servers" in data and "mcpServers" not in data
    entry = data["servers"]["memo"]
    assert entry["type"] == "stdio"
    assert entry["command"].endswith("memo-mcp")


def test_zed_uses_context_servers_flat_command(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "linux")
    res = ap.install_from_preset(ap.AGENT_PRESETS["zed"], _server(tmp_path / "memo-mcp"), write=True)
    data = json.loads(Path(res["path"]).read_text())
    entry = data["context_servers"]["memo"]
    assert isinstance(entry["command"], str)  # flat, not nested object
    assert "type" not in entry


def test_family_a_uses_mcpservers_no_type(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    res = ap.install_from_preset(ap.AGENT_PRESETS["windsurf"], _server(tmp_path / "memo-mcp"), write=True)
    data = json.loads(Path(res["path"]).read_text())
    assert "mcpServers" in data
    assert "type" not in data["mcpServers"]["memo"]


def test_source_injected_into_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    res = ap.install_from_preset(ap.AGENT_PRESETS["kiro"], _server(tmp_path / "memo-mcp"), write=True)
    data = json.loads(Path(res["path"]).read_text())
    assert data["mcpServers"]["memo"]["env"]["MEMO_SOURCE"] == "kiro"


def test_merge_preserves_existing_server(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    path = ap.resolve_preset_path(ap.AGENT_PRESETS["windsurf"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    ap.install_from_preset(ap.AGENT_PRESETS["windsurf"], _server(tmp_path / "memo-mcp"), write=True)
    data = json.loads(path.read_text())
    assert "other" in data["mcpServers"] and "memo" in data["mcpServers"]


def test_jetbrains_is_snippet_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    res = ap.install_from_preset(ap.AGENT_PRESETS["jetbrains"], _server(tmp_path / "memo-mcp"), write=True)
    assert res["ok"] and res["action"] == "snippet"
    assert "mcpServers" in res["snippet"]
    assert "path" not in res  # nothing written


def test_dry_run_reports_path_no_write(monkeypatch, tmp_path):
    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    res = ap.install_from_preset(ap.AGENT_PRESETS["warp"], _server(tmp_path / "memo-mcp"), write=False)
    assert res["action"] == "dry-run" and res["path"].endswith("/.warp/.mcp.json")
    assert not Path(res["path"]).exists()


def test_continue_block_file_shape(monkeypatch, tmp_path):
    import yaml

    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    res = ap.install_from_preset(ap.AGENT_PRESETS["continue"], _server(tmp_path / "memo-mcp"), write=True)
    assert res["path"].endswith("/.continue/mcpServers/memo.yaml")
    doc = yaml.safe_load(Path(res["path"]).read_text())
    assert doc["schema"] == "v1"
    assert isinstance(doc["mcpServers"], list)  # list, not object map
    assert doc["mcpServers"][0]["command"].endswith("memo-mcp")
    assert doc["mcpServers"][0]["env"]["MEMO_SOURCE"] == "continue"


def test_goose_extensions_merge_preserves_and_renames(monkeypatch, tmp_path):
    import yaml

    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "linux")
    path = ap.resolve_preset_path(ap.AGENT_PRESETS["goose"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"extensions": {"existing": {"type": "stdio"}}, "OPENAI_HOST": "x"}))
    ap.install_from_preset(ap.AGENT_PRESETS["goose"], _server(tmp_path / "memo-mcp"), write=True)
    doc = yaml.safe_load(path.read_text())
    assert "existing" in doc["extensions"]           # merge preserved
    assert doc["OPENAI_HOST"] == "x"                 # top-level preserved
    memo_ext = doc["extensions"]["memo"]
    assert memo_ext["cmd"].endswith("memo-mcp")      # `cmd`, not `command`
    assert memo_ext["envs"]["MEMO_SOURCE"] == "goose"  # `envs` value map (delivers values)


def test_goose_rejects_malformed_yaml(monkeypatch, tmp_path):
    import click

    monkeypatch.setattr(ap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ap.sys, "platform", "linux")
    path = ap.resolve_preset_path(ap.AGENT_PRESETS["goose"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("extensions: [unclosed\n")
    with pytest.raises(click.ClickException):
        ap.install_from_preset(ap.AGENT_PRESETS["goose"], _server(tmp_path / "memo-mcp"), write=True)
