"""memo mine-git — deterministic failure_pattern seeding from fix/revert commits."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@test", "-c", "user.name=T", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "feat: initial layout")
    (repo / "a.txt").write_text("two")
    _git(repo, "add", "a.txt")
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        "fix(daemon): crash-loop on missing contracts\n\n"
        "launchd plist pointed at memo's venv which lacks consciousness_contracts.",
    )
    return repo


def _pin_env(tmp_path: Path, monkeypatch) -> None:
    # conftest neutralizes the config FILE, not exported env vars — pin the
    # vault + project-tag flags too so a dev shell with MEMO_VAULT_PATH /
    # MEMO_MEMORIES_IN_VAULT exported can never write into the real vault
    # (tests/conftest.py contract; pattern: tests/test_capture_incremental.py
    # `_setup_env`). MEMO_AUTO_PROJECT_TAG=0 because these tests assert tags.
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "0")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )


def test_mine_git_saves_failure_pattern_from_fix_commit(tmp_path, monkeypatch):
    _pin_env(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    from memo.git_miner import mine_git_history

    out = mine_git_history(repo=repo)
    assert out["status"] == "ok"
    assert len(out["saved"]) == 1  # feat commit ignored, fix commit mined

    from memo.config import Config
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        rec = mem.get(out["saved"][0])
        assert rec.type == "failure_pattern"
        assert "crash-loop" in rec.title
        assert rec.extra["source"] == "mine-git"
        assert len(rec.extra["commit_sha"]) == 40
        assert "project:proj" in rec.tags
        assert "git-mined" in rec.tags
        assert "Pattern:" in (rec.body or "")
    finally:
        mem.close()


def test_mine_git_is_resumable_and_dedups_by_sha(tmp_path, monkeypatch):
    _pin_env(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    from memo.git_miner import mine_git_history

    out1 = mine_git_history(repo=repo)
    assert len(out1["saved"]) == 1
    out2 = mine_git_history(repo=repo)
    assert out2["saved"] == []
    assert out2["skipped_seen"] == 1


def test_mine_git_dry_run_saves_nothing(tmp_path, monkeypatch):
    _pin_env(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    from memo.git_miner import mine_git_history

    out = mine_git_history(repo=repo, dry_run=True)
    assert out["saved"] == ["<dry-run>"]
    # dry-run must not advance the SHA watermark:
    out2 = mine_git_history(repo=repo)
    assert len(out2["saved"]) == 1


def test_mine_git_not_a_repo(tmp_path, monkeypatch):
    _pin_env(tmp_path, monkeypatch)
    from memo.git_miner import mine_git_history

    out = mine_git_history(repo=tmp_path / "empty")
    assert out["status"] == "not_a_repo"


def test_cli_mine_git_json(tmp_path, monkeypatch):
    import json as _json

    from click.testing import CliRunner

    _pin_env(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    from memo.cli_transcripts import mine_git

    res = CliRunner().invoke(
        mine_git,
        ["--repo", str(repo), "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
        },
    )
    assert res.exit_code == 0, res.output
    data = _json.loads(res.output)
    assert data["status"] == "ok" and len(data["saved"]) == 1
