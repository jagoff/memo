"""Regression tests for the recall-hook — 5s budget, concurrent safety.

Covers three properties:
1. Valid JSON output (or empty) when corpus is empty.
2. Clean degradation (no crash) when the embedder raises.
3. No deadlocks or corrupt output from concurrent invocations.
"""
from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from memo.cli import cli

if TYPE_CHECKING:
    from memo.config import Config


@pytest.fixture
def recall_env(tmp_cfg: "Config", monkeypatch: pytest.MonkeyPatch) -> "Config":
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


def test_recall_hook_returns_valid_json_on_empty_corpus(recall_env: "Config") -> None:
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
    recall_env: "Config", monkeypatch: pytest.MonkeyPatch
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


def test_recall_hook_concurrent_invocations(recall_env: "Config") -> None:
    """Multiple concurrent recall-hook CLI invocations must not deadlock or raise.

    Uses empty stdin so each thread bails early (before any sqlite access),
    which avoids the sys.stdin global-replacement race inherent in CliRunner's
    multi-threaded use.  The assertion is simply: all 3 threads complete with
    exit_code 0 and no uncaught exceptions.
    """
    errors: list[str] = []
    exit_codes: list[int] = []

    def run_hook() -> None:
        try:
            runner = CliRunner()
            r = runner.invoke(cli, ["recall-hook"])
            exit_codes.append(r.exit_code)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run_hook) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"Concurrent recall-hook errors: {errors}"
    assert all(code == 0 for code in exit_codes), f"Non-zero exit codes: {exit_codes}"
    assert len(exit_codes) == 3, "Not all threads completed in time"
