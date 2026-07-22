"""Tests for recall-hook latency metrics (recall_metrics + memo stats section).

Covers the observability contract from the Q3 recall-observability spec:
- stamp writes one valid jsonl line ({ts, total_ms, path, hits});
- stamping failure is silent (unwritable dir) and flag-gated;
- rotation trims to the newest KEEP_LINES once past ~MAX_LINES;
- nearest-rank percentiles are correct on synthetic data;
- summarize splits daemon vs subprocess and windows to the last 7 days;
- BOTH hook execution paths (daemon socket + subprocess fallback) stamp;
- `memo stats` renders the latency section and omits it gracefully.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from memo import recall_metrics
from memo.cli import cli

if TYPE_CHECKING:
    from memo.config import Config


@pytest.fixture
def metrics_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the metrics flag to its default-ON path (developer env may export 0)."""
    monkeypatch.delenv("MEMO_RECALL_METRICS", raising=False)


@pytest.fixture
def recall_env(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Isolated env for hook/stats CLI invocations (mirrors test_recall_hook)."""
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.delenv("MEMO_RECALL_METRICS", raising=False)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )
    return tmp_cfg


# ---------------------------------------------------------------- stamp


def test_stamp_writes_valid_line(tmp_path: Path, metrics_on: None) -> None:
    recall_metrics.stamp(tmp_path, total_ms=123.45, path="daemon", hits=3)

    lines = recall_metrics.metrics_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["path"] == "daemon"
    assert entry["hits"] == 3
    assert entry["total_ms"] == pytest.approx(123.4, abs=0.11)
    parsed_ts = datetime.fromisoformat(entry["ts"])  # valid iso8601, tz-aware
    assert parsed_ts.tzinfo is not None


def test_stamp_flag_default_on() -> None:
    from memo.flags import REGISTRY

    assert REGISTRY["MEMO_RECALL_METRICS"].default is True


def test_stamp_disabled_by_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_METRICS", "0")
    recall_metrics.stamp(tmp_path, total_ms=1.0, path="daemon", hits=0)
    assert not recall_metrics.metrics_path(tmp_path).exists()


def test_stamp_silent_on_unwritable_dir(tmp_path: Path, metrics_on: None) -> None:
    """state_dir path occupied by a file → mkdir/append fails → stamp stays silent."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a dir", encoding="utf-8")
    recall_metrics.stamp(blocked, total_ms=1.0, path="subprocess", hits=1)  # must not raise
    assert blocked.read_text(encoding="utf-8") == "not a dir"


# ---------------------------------------------------------------- rotation


def _seed_metrics_file(target: Path, n: int) -> None:
    with target.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "ts": "2026-07-02T00:00:00+00:00",
                        "total_ms": 1.0,
                        "path": "daemon",
                        "hits": 0,
                        "i": i,
                        "pad": "x" * 40,
                    }
                )
                + "\n"
            )


def test_rotation_trims_via_stamp(tmp_path: Path, metrics_on: None) -> None:
    target = recall_metrics.metrics_path(tmp_path)
    _seed_metrics_file(target, recall_metrics.MAX_LINES + 100)

    recall_metrics.stamp(tmp_path, total_ms=42.0, path="subprocess", hits=2)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == recall_metrics.KEEP_LINES
    # Newest entries kept: the just-stamped line is last…
    last = json.loads(lines[-1])
    assert last["path"] == "subprocess"
    assert last["total_ms"] == 42.0
    # …and the oldest survivor is the expected seeded index (5101 total → keep 2601..5100).
    first = json.loads(lines[0])
    assert first["i"] == recall_metrics.MAX_LINES + 100 + 1 - recall_metrics.KEEP_LINES


def test_no_rotation_below_threshold(tmp_path: Path, metrics_on: None) -> None:
    target = recall_metrics.metrics_path(tmp_path)
    _seed_metrics_file(target, 10)
    recall_metrics.stamp(tmp_path, total_ms=1.0, path="daemon", hits=0)
    assert len(target.read_text(encoding="utf-8").splitlines()) == 11


def test_maybe_rotate_atomic_rewrite_keeps_newest(tmp_path: Path) -> None:
    target = tmp_path / "recall_metrics.jsonl"
    target.write_text("".join(f'{{"i": {i}}}\n' for i in range(12)), encoding="utf-8")

    recall_metrics._maybe_rotate(target, max_lines=10, keep_lines=5, size_trip_bytes=1)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["i"] for ln in lines] == [7, 8, 9, 10, 11]
    # No tmp litter left behind.
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------- percentiles


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(100, 0, -1)]  # 100..1 unsorted on purpose
    assert recall_metrics.percentile(values, 50) == 50.0
    assert recall_metrics.percentile(values, 95) == 95.0
    assert recall_metrics.percentile(values, 99) == 99.0
    assert recall_metrics.percentile([7.5], 99) == 7.5
    assert recall_metrics.percentile([], 50) == 0.0


# ---------------------------------------------------------------- summarize


def _write_entries(state_dir: Path, entries: list[dict]) -> None:
    target = recall_metrics.metrics_path(state_dir)
    with target.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_summarize_splits_by_path_and_windows(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    recent = now.isoformat(timespec="seconds")
    old = (now - timedelta(days=9)).isoformat(timespec="seconds")
    _write_entries(
        tmp_path,
        [
            {"ts": recent, "total_ms": 10.0, "path": "daemon", "hits": 1},
            {"ts": recent, "total_ms": 20.0, "path": "daemon", "hits": 2},
            {"ts": recent, "total_ms": 30.0, "path": "daemon", "hits": 0},
            {"ts": recent, "total_ms": 100.0, "path": "subprocess", "hits": 1},
            {"ts": recent, "total_ms": 200.0, "path": "subprocess", "hits": 1},
            # Outside the 7-day window — must be excluded.
            {"ts": old, "total_ms": 9999.0, "path": "daemon", "hits": 0},
        ],
    )

    summary = recall_metrics.summarize(tmp_path, days=7)

    assert summary["daemon"]["count"] == 3
    assert summary["daemon"]["p50"] == 20.0
    assert summary["daemon"]["p99"] == 30.0  # the 9999 tail was windowed out
    assert summary["subprocess"]["count"] == 2
    assert summary["subprocess"]["p50"] == 100.0
    assert summary["subprocess"]["p95"] == 200.0


def test_summarize_missing_empty_and_garbage(tmp_path: Path) -> None:
    assert recall_metrics.summarize(tmp_path) == {}  # missing file

    target = recall_metrics.metrics_path(tmp_path)
    target.write_text("", encoding="utf-8")
    assert recall_metrics.summarize(tmp_path) == {}  # empty file

    target.write_text('not json\n{"ts": "nope"}\n', encoding="utf-8")
    assert recall_metrics.summarize(tmp_path) == {}  # malformed lines skipped


# ---------------------------------------------------------------- count_hits


def test_count_hits_prefers_system_message() -> None:
    out = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "[abc12345] one",
            },
            "systemMessage": "🧠 memo · 2: one, two",
        }
    )
    assert recall_metrics.count_hits(out) == 2


def test_count_hits_falls_back_to_short_ids() -> None:
    context = (
        "<memo-recall readonly>\n[abc12345] one\n[deadbeef] two\n</memo-recall>\n"
        "_cite it inline by short id — e.g. `per your memory [a1b2c3d4]`_"
    )
    out = json.dumps(
        {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}
    )
    assert recall_metrics.count_hits(out) == 2  # cite placeholder excluded


def test_count_hits_empty_and_garbage() -> None:
    assert recall_metrics.count_hits("{}") == 0
    assert recall_metrics.count_hits("not json at all") == 0
    assert recall_metrics.count_hits(json.dumps({"hookSpecificOutput": "weird"})) == 0


# ---------------------------------------------------------------- hook paths


def test_hook_subprocess_path_stamps(recall_env: Config) -> None:
    """Empty corpus → subprocess fallback runs → one line with path=subprocess."""
    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output

    lines = recall_metrics.metrics_path(recall_env.state_dir).read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["path"] == "subprocess"
    assert entry["hits"] == 0
    assert entry["total_ms"] >= 0.0


def test_hook_subprocess_hits_are_post_session_dedup(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subprocess ``hits`` counts the POST-session-dedup injected memories —
    comparable with the daemon path — including 0 on the 'all hits already
    recalled' bail (pre-dedup counting would report 1/2/2)."""
    from memo.memory import MemoryRecord

    def _hit(hid: str, title: str) -> MemoryRecord:
        return MemoryRecord(
            id=hid,
            path=f"notes/{hid}.md",
            title=title,
            type="note",
            tags=[],
            created="2026-01-01T00:00:00+00:00",
            updated="2026-01-01T00:00:00+00:00",
            body=f"distinct body for {title}, long enough to pass the min-body gate.",
            extra={},
            score=0.8,
        )

    hit_a = _hit("aabbccdd11223344", "Memory Alpha")
    hit_b = _hit("eeff001122334455", "Memory Beta")
    # Turn 1 surfaces A; turns 2 and 3 surface A+B — session dedup must strip
    # the already-injected ids, so the stamped hits are 1, 1, 0.
    per_turn = [[hit_a], [hit_a, hit_b], [hit_a, hit_b]]
    calls = {"n": 0}

    class StubMemory:
        def __init__(self, cfg: object) -> None:
            pass

        def search(
            self,
            query: str,
            limit: int = 5,
            mode: str = "bm25",
            recency: bool = False,
            exclude_types: object = None,
            exclude_tags: object = None,
        ) -> list[MemoryRecord]:
            hits = per_turn[min(calls["n"], len(per_turn) - 1)]
            calls["n"] += 1
            return hits

        def close(self) -> None:
            pass

    monkeypatch.setattr("memo.memory.Memory", StubMemory)

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_RECALL_METRICS": "1",
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_MIN_BODY_CHARS": "0",
        "MEMO_RECALL_TOKEN_BUDGET": "0",
        "MEMO_RECALL_EXPAND_CONTEXT": "0",
        "MEMO_RECALL_ADAPTIVE_CONTEXT": "0",
        "MEMO_RECALL_CONTEXTUAL": "0",
        # Isolate SESSION dedup: the A/B fixture bodies differ by one word, so
        # the default-ON pre-top-K paraphrase collapse would strip B — off here.
        "MEMO_RECALL_DEDUP_COLLAPSE": "0",
    }
    payload = json.dumps(
        {
            "prompt": "what do you know about this recurring topic",
            "session_id": "metrics-dedup-session-001",
            "cwd": str(tmp_cfg.data_dir),
        }
    )
    runner = CliRunner()
    for _ in range(3):
        result = runner.invoke(cli, ["recall-hook"], input=payload, env=env, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    lines = recall_metrics.metrics_path(tmp_cfg.state_dir).read_text().splitlines()
    assert len(lines) == 3  # exactly one stamp per hook run
    entries = [json.loads(ln) for ln in lines]
    assert [e["path"] for e in entries] == ["subprocess"] * 3
    # Turn 1: A injected. Turn 2: A deduped, only B injected (pre-dedup = 2).
    # Turn 3: both already recalled -> dedup bail stamps 0 (pre-dedup = 2).
    assert [e["hits"] for e in entries] == [1, 1, 0]


def test_hook_daemon_path_stamps(recall_env: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Daemon socket answer → one line with path=daemon and parsed hit count."""
    daemon_result = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "[abc12345] one\n[deadbeef] two",
            },
            "systemMessage": "🧠 memo · 2: one, two",
        }
    )
    monkeypatch.setattr(
        "memo.recall_server.connect_and_recall",
        lambda *args, **kwargs: daemon_result,
    )

    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "additionalContext" in result.output

    lines = recall_metrics.metrics_path(recall_env.state_dir).read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["path"] == "daemon"
    assert entry["hits"] == 2


def test_hook_metrics_failure_is_silent(
    recall_env: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising stamp must never break the hook output contract."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("memo.recall_metrics.stamp", _boom)
    runner = CliRunner()
    payload = json.dumps({"prompt": "some meaningful query here to test recall"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    json.loads(result.output.strip())  # still valid JSON


# ---------------------------------------------------------------- memo stats


def test_stats_renders_latency_section(recall_env: Config) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    _write_entries(
        recall_env.state_dir,
        [
            {"ts": now, "total_ms": 12.0, "path": "daemon", "hits": 1},
            {"ts": now, "total_ms": 3400.0, "path": "subprocess", "hits": 3},
        ],
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["stats"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Recall Latency" in result.output
    assert "daemon" in result.output
    assert "subprocess" in result.output
    assert "p95" in result.output


def test_stats_omits_latency_section_when_no_data(recall_env: Config) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["stats"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Recall Latency" not in result.output
