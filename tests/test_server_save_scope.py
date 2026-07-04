"""MCP `memo_save` `scope` param — per-call global writes (no auto project tag)."""

from __future__ import annotations

import asyncio

import pytest

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
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    m = Memory(cfg)
    yield m
    m.close()


def _tool(server, name):
    tool = asyncio.run(server.get_tool(name))
    if tool is None:
        raise RuntimeError(f"tool {name!r} not registered")
    return tool.fn


def test_scope_global_skips_auto_project_tag(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    save = _tool(build_server(memory=mem), "memo_save")

    out = save(content="a machine-wide fact", scope="global")

    assert not any(t.startswith("project:") for t in out["tags"])


def test_scope_omitted_keeps_auto_project_tag(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    save = _tool(build_server(memory=mem), "memo_save")

    out = save(content="a project fact")

    assert "project:detected-repo" in out["tags"]


def test_scope_project_is_accepted_alias_of_default(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    save = _tool(build_server(memory=mem), "memo_save")

    out = save(content="a project fact", scope="project")

    assert "project:detected-repo" in out["tags"]


def test_scope_global_explicit_project_tag_still_wins(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    save = _tool(build_server(memory=mem), "memo_save")

    out = save(content="x", tags=["project:explicit"], scope="global")

    assert "project:explicit" in out["tags"]
    assert "project:detected-repo" not in out["tags"]


def test_invalid_scope_returns_structured_error(mem: Memory):
    save = _tool(build_server(memory=mem), "memo_save")

    out = save(content="x", scope="personal")

    assert out["error"] == "invalid_scope"
