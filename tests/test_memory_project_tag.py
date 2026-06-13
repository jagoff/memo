"""Memory.save auto-attaches `project:<repo>` tag from cwd or env."""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def mem_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir, vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir, embedder_dims=4,
    )

    def _embed(self, inputs):
        return [[0.0, 0.0, 0.0, 1.0]] * len(inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    mem = Memory(cfg)
    yield mem
    mem.close()


def test_auto_project_tag_from_env(
    mem_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "my-app")
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    rec = mem_stub.save(content="some content", title="X")
    assert "project:my-app" in rec.tags


def test_auto_project_tag_skipped_when_user_provided(
    mem_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "from-env")
    rec = mem_stub.save(content="x", title="t", tags=["project:explicit"])
    assert "project:explicit" in rec.tags
    assert "project:from-env" not in rec.tags


def test_auto_project_tag_disabled_by_env(
    mem_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "my-app")
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "0")
    rec = mem_stub.save(content="x", title="t")
    assert not any(t.startswith("project:") for t in rec.tags)


def test_auto_project_tag_opt_out_per_call(
    mem_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "my-app")
    rec = mem_stub.save(content="x", title="t", auto_project=False)
    assert not any(t.startswith("project:") for t in rec.tags)


def test_auto_project_tag_from_cwd_param(
    mem_stub: Memory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    repo = tmp_path / "repo-from-cwd"
    repo.mkdir()
    (repo / ".git").mkdir()
    rec = mem_stub.save(content="x", title="t", cwd=str(repo))
    assert "project:repo-from-cwd" in rec.tags
