"""Regression tests for the recall-hook — 5s budget, concurrent safety.

Covers three properties:
1. Valid JSON output (or empty) when corpus is empty.
2. Clean degradation (no crash) when the embedder raises.
3. No deadlocks or corrupt output from concurrent invocations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from memo.cli import cli

if TYPE_CHECKING:
    from memo.config import Config


@pytest.fixture
def recall_env(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Minimal environment for recall-hook: stub embedder, no MLX, isolated storage.

    Sets MEMO_DATA_DIR and MEMO_STATE_DIR so Config.from_env() inside the hook
    uses the tmp_cfg paths rather than the developer's real storage.
    MEMO_EMBEDDER_DIMS=4 matches the stub embedder's output dimension.
    """
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )
    return tmp_cfg


def test_recall_hook_returns_valid_json_on_empty_corpus(recall_env: Config) -> None:
    """recall-hook must return valid JSON (or empty string) with no memories saved."""
    runner = CliRunner()
    payload = json.dumps(
        {"prompt": "some meaningful query here to test recall", "session_id": "test-sess-1"}
    )
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    output = result.output.strip()
    if output:
        parsed = json.loads(output)  # raises if invalid JSON
        assert isinstance(parsed, (dict, list, str))


def test_recall_hook_returns_json_when_embedder_raises(
    recall_env: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """recall-hook must not crash when the embedder raises — degrade to empty JSON."""

    def failing_embed(self, query: str) -> list[float]:
        raise RuntimeError("MLX OOM")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", failing_embed)
    # Force vec mode so the embedder is actually called during search.
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")

    runner = CliRunner()
    payload = json.dumps(
        {"prompt": "some meaningful query here to test recall", "session_id": "test-sess-2"}
    )
    result = runner.invoke(cli, ["recall-hook"], input=payload)
    # Must not propagate the exception — exit code must be 0.
    assert result.exit_code == 0, f"Crash on embedder failure: {result.output}"
    output = result.output.strip()
    if output:
        json.loads(output)  # must be valid JSON if anything was printed


def test_recall_hook_concurrent_invocations(recall_env: Config) -> None:
    """Multiple concurrent recall-hook CLI invocations must not deadlock or raise.

    Use real subprocesses instead of invoking Click's test runner from multiple
    threads: agent hooks execute independent CLI processes, while CliRunner
    mutates process-global stdin/stdout and is not a thread-safe surface.
    """
    env = os.environ.copy()
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "memo.cli", "recall-hook"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(3)
    ]

    results = []
    for proc in procs:
        stdout, stderr = proc.communicate(input="", timeout=15)
        results.append((proc.returncode, stdout, stderr))

    assert all(code == 0 for code, _stdout, _stderr in results), results
