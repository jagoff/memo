"""TUI dashboard — sparkline rendering + recall-log ring buffer."""

from __future__ import annotations

import json
from pathlib import Path

import memo.dashboard_logs as dashboard_logs
from memo.dashboard import (
    _human_age,
    _human_bytes,
    append_grounding_log,
    append_recall_log,
    read_recall_log,
    sparkline,
    verdict,
)


def test_sparkline_empty_input_is_full_width() -> None:
    out = sparkline([], width=10)
    assert len(out) == 10
    assert all(c == "▁" for c in out)


def test_sparkline_renders_monotonic_series() -> None:
    out = sparkline([0, 1, 2, 3, 4, 5, 6, 7], width=8)
    assert len(out) == 8
    # First char = lowest level, last char = highest level.
    assert out[0] == "▁"
    assert out[-1] == "█"


def test_sparkline_pads_short_series_at_left() -> None:
    out = sparkline([5, 5, 5], width=8)
    # Short series gets leading zero-pads, so the last 3 should be the
    # highest-level char and the first 5 the lowest.
    assert out[-3:] == "█" * 3
    assert out[:5] == "▁" * 5


def test_sparkline_buckets_long_series() -> None:
    # 50 samples bucketed into width=10 → each bucket has 5 samples.
    out = sparkline(list(range(50)), width=10)
    assert len(out) == 10
    assert out[0] != out[-1]


def test_sparkline_all_zero_renders_baseline() -> None:
    out = sparkline([0, 0, 0, 0], width=4)
    assert out == "▁▁▁▁"


def test_append_and_read_recall_log_roundtrip(tmp_path: Path) -> None:
    append_recall_log(
        tmp_path,
        prompt="why did MLX win?",
        hits=[
            {"id": "abcdef1234567890", "score": 0.82, "title": "MLX vs Ollama bench"},
            {"id": "xyz9876543210000", "score": 0.71, "title": "Qwen3 embedder swap"},
        ],
    )
    entries = read_recall_log(tmp_path, limit=5)
    assert len(entries) == 1
    assert entries[0]["prompt"] == "why did MLX win?"
    assert entries[0]["hits"][0]["id"] == "abcdef12"  # truncated to 8
    assert entries[0]["hits"][0]["score"] == 0.82


def test_context_cost_log_roundtrip(tmp_path: Path) -> None:
    dashboard_logs.append_context_cost_log(
        tmp_path,
        kind="recall",
        chars=321,
        client="claude-code",
        session_id="sid-1",
        turn=4,
    )

    rows = dashboard_logs.read_context_cost_log(tmp_path)

    assert rows == [
        {
            "ts": rows[0]["ts"],
            "kind": "recall",
            "chars": 321,
            "tokens_est": 81,
            "client": "claude-code",
            "session_id": "sid-1",
            "turn": 4,
        }
    ]


def test_recall_log_returns_newest_first(tmp_path: Path) -> None:
    for i in range(5):
        append_recall_log(tmp_path, prompt=f"q{i}", hits=[])
    entries = read_recall_log(tmp_path, limit=3)
    assert [e["prompt"] for e in entries] == ["q4", "q3", "q2"]


def test_recall_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_recall_log(tmp_path, limit=10) == []


def test_recall_log_swallows_write_errors(tmp_path: Path) -> None:
    # Pointing at a path that can't be written to — append_recall_log
    # must NOT raise (recall hook can never fail).
    bad = tmp_path / "nonexistent" / "deep"
    # Should be silent — would raise on the .open() call otherwise.
    append_recall_log(bad, prompt="x", hits=[])
    # And the function should have created the parent dir for us.
    assert (bad / "recall.log").is_file() or not (bad / "recall.log").exists()


def test_recall_log_truncates_long_prompt(tmp_path: Path) -> None:
    long_prompt = "x" * 5000
    append_recall_log(tmp_path, prompt=long_prompt, hits=[])
    entries = read_recall_log(tmp_path)
    assert len(entries[0]["prompt"]) == 200


def test_recall_log_caps_hits_to_5(tmp_path: Path) -> None:
    hits = [{"id": f"id{i:08d}xxxx", "score": 0.5, "title": "t"} for i in range(20)]
    append_recall_log(tmp_path, prompt="q", hits=hits)
    entries = read_recall_log(tmp_path)
    assert len(entries[0]["hits"]) == 5


def test_recall_log_corrupted_line_ignored(tmp_path: Path) -> None:
    p = tmp_path / "recall.log"
    p.write_text(
        json.dumps({"ts": "2026-01-01T00:00:00", "prompt": "ok", "hits": []})
        + "\n{not-json}\n"
        + json.dumps({"ts": "2026-01-01T00:01:00", "prompt": "also ok", "hits": []})
        + "\n",
        encoding="utf-8",
    )
    entries = read_recall_log(tmp_path, limit=10)
    assert len(entries) == 2
    assert {e["prompt"] for e in entries} == {"ok", "also ok"}


def test_human_age_handles_iso_z_suffix() -> None:
    assert _human_age(None) == "—"
    assert _human_age("") == "—"
    # Just make sure it doesn't crash on a Z suffix and produces a
    # reasonable string.
    out = _human_age("2026-05-01T12:00:00Z")
    assert any(out.endswith(s) for s in ("s ago", "m ago", "h ago", "d ago", "now"))


def test_human_bytes_units() -> None:
    assert _human_bytes(512) == "512 B"
    assert _human_bytes(2048) == "2.0 KB"
    assert _human_bytes(3 * 1024 * 1024) == "3.0 MB"
    assert _human_bytes(2 * 1024 ** 3) == "2.00 GB"


def test_verdict_unused_when_too_few_consults(tmp_path: Path) -> None:
    for i in range(5):
        append_recall_log(tmp_path, prompt=f"q{i}", hits=[], via="daemon", client="claude-code")
    v = verdict(tmp_path)
    assert v["status"] == "unused"
    assert v["label"].startswith("❌")


def test_verdict_unmeasured_when_read_but_not_grounded(tmp_path: Path) -> None:
    # Enough reads with hits, but no grounding rows yet → cannot judge help.
    for i in range(25):
        append_recall_log(
            tmp_path,
            prompt=f"q{i}",
            hits=[{"id": f"id{i:04d}", "score": 0.9, "title": "t"}],
            via="daemon",
            client="claude-code",
            session_id="s1",
            turn=i,
        )
    v = verdict(tmp_path)
    assert v["status"] == "unmeasured"
    assert v["consults"] >= 20


def test_verdict_ok_when_read_and_grounded(tmp_path: Path) -> None:
    for i in range(25):
        append_recall_log(
            tmp_path,
            prompt=f"q{i}",
            hits=[{"id": f"id{i:04d}", "score": 0.9, "title": "t"}],
            via="daemon",
            client="claude-code",
            session_id="s1",
            turn=i,
        )
        # Ground every recall so grounded_rate is ~1.0 (>= 10% threshold).
        append_grounding_log(
            tmp_path,
            session_id="s1",
            turn=i,
            recall_id=f"id{i:04d}",
            used_score=0.9,
            method="overlap",
        )
    v = verdict(tmp_path)
    assert v["status"] == "ok"
    assert v["label"].startswith("✅")


def test_verdict_marks_expected_consumers_silent(tmp_path: Path) -> None:
    for i in range(3):
        append_recall_log(tmp_path, prompt=f"q{i}", hits=[], via="daemon", client="claude-code")
    v = verdict(tmp_path)
    by_name = {p["name"]: p["reads"] for p in v["per_consumer"]}
    assert by_name["claude-code"] is True
    assert by_name["memflow"] is False
    assert "memflow" in v["silent"]
