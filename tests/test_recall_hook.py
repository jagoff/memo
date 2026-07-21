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
from memo.recall_logic import RankKnobs

if TYPE_CHECKING:
    from memo.config import Config


@pytest.mark.parametrize(
    ("mode", "expected_top_k", "expected_min_sim"),
    [
        ("focus", 2, 0.65),
        ("explore", 5, 0.4),
        ("maintenance", 1, 0.7),
        ("unknown", 3, 0.5),
    ],
)
def test_apply_session_mode_is_bounded(mode: str, expected_top_k: int, expected_min_sim: float):
    from memo.cli_recall_hook import apply_session_mode

    knobs = RankKnobs(top_k=3, min_sim=0.5)

    adjusted = apply_session_mode(knobs, mode)

    assert adjusted.top_k == expected_top_k
    assert adjusted.min_sim == expected_min_sim


@pytest.mark.parametrize(
    ("budget", "prompt_length", "expected"),
    [(0, 20, 0), (500, 20, 750), (700, 20, 800), (500, 100, 500), (500, 301, 300)],
)
def test_adaptive_token_budget(budget: int, prompt_length: int, expected: int):
    from memo.cli_recall_hook import adaptive_token_budget

    assert adaptive_token_budget(budget, prompt_length) == expected


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


def test_recall_disabled_stamps_off_cohort(recall_env: Config, monkeypatch) -> None:
    """MEMO_RECALL_DISABLE must stamp the turn (via='disabled') into
    recall_hook.log so `memo roi` can compare recall-on/off cohorts."""
    from memo.dashboard import read_recall_hook_log

    monkeypatch.setenv("MEMO_RECALL_DISABLE", "1")
    runner = CliRunner()
    payload = json.dumps(
        {"prompt": "cómo configuro el sync remoto de memo?", "session_id": "abl-1"}
    )
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0
    assert result.output.strip().splitlines()[-1] == "{}"  # hook still injects nothing
    rows = [r for r in read_recall_hook_log(recall_env.state_dir) if r.get("session_id") == "abl-1"]
    assert len(rows) == 1
    assert rows[0]["via"] == "disabled"
    assert rows[0]["prompt"].startswith("cómo configuro el sync")
    assert isinstance(rows[0]["turn"], int)


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


def test_daemon_busy_marker_falls_through_to_subprocess(
    recall_env: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon {"busy": true} reply (warming / lock-bail) is NOT a recall
    result: the hook must run the subprocess fallback instead of printing it
    (a bare '{}' used to keep recall dark for the whole warmup window)."""
    from memo.memory import MemoryRecord

    hit = MemoryRecord(
        id="beefcafe11223344",
        path="notes/busy.md",
        title="Busy Fallback Memory",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="a body long enough to pass the min-body gate " * 3,
        extra={},
        score=0.9,
    )

    class _OneHitMemory:
        def __init__(self, cfg: object) -> None:
            pass

        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            return [hit]

        def close(self) -> None:
            pass

    monkeypatch.setattr("memo.recall_server.connect_and_recall", lambda *a, **k: '{"busy": true}')
    monkeypatch.setattr("memo.memory.Memory", _OneHitMemory)
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")

    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert '"busy"' not in result.output  # marker never leaks to the hook output
    parsed = json.loads(result.output.strip())
    assert "Busy Fallback Memory" in parsed["hookSpecificOutput"]["additionalContext"]


def test_daemon_legit_empty_reply_still_prints(
    recall_env: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legit daemon '{}' (empty recall) keeps the historical contract: it is
    printed verbatim and the subprocess fallback does NOT run."""

    class _NeverMemory:
        def __init__(self, cfg: object) -> None:
            raise AssertionError("subprocess fallback must not run on a legit '{}' reply")

    monkeypatch.setattr("memo.recall_server.connect_and_recall", lambda *a, **k: "{}")
    monkeypatch.setattr("memo.memory.Memory", _NeverMemory)

    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


def test_corrupt_prewarm_signal_downgrades_to_bm25(
    recall_env: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable/corrupt .prewarm_ts must count as NOT warm (downgrade to
    bm25) — failing open kept vec mode and paid a cold MLX load in the hook."""
    seen: dict[str, str] = {}

    class _ModeRecorder:
        def __init__(self, cfg: object) -> None:
            pass

        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            seen["mode"] = mode
            return []

        def close(self) -> None:
            pass

    monkeypatch.setattr("memo.memory.Memory", _ModeRecorder)
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.delenv("MEMO_RECALL_FORCE_MODE", raising=False)
    (recall_env.state_dir / ".prewarm_ts").write_text("not-a-timestamp")

    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert seen["mode"] == "bm25"
