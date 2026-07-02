"""Tests for per-consumer consult telemetry (`memo usefulness`).

Covers the logic that answers "who actually reads memo" without needing an
MLX forward pass: the dashboard aggregation (`consult_breakdown` /
`consumer_label`) and the server-side consult logger (`_log_consult`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import memo.dashboard_logs as dashboard_logs
from memo import dashboard
from memo.server_common import log_consult as _log_consult


def test_consumer_label_classifies_each_source() -> None:
    assert dashboard.consumer_label({"via": "daemon"}) == "claude-code"
    assert dashboard.consumer_label({"via": "subprocess"}) == "claude-code"
    assert dashboard.consumer_label({"via": "bail"}) == "claude-code"
    assert dashboard.consumer_label({"via": "mcp:ask"}) == "mcp:unknown"
    # An explicit source always wins over the via prefix.
    assert dashboard.consumer_label({"via": "mcp:ask", "source": "devin"}) == "devin"


def test_consult_breakdown_groups_and_flags_silent(tmp_path: Path) -> None:
    dashboard.append_recall_log(
        tmp_path, prompt="q1", hits=[{"id": "a" * 8, "score": 0.8, "title": "t"}], via="daemon"
    )
    dashboard.append_recall_log(tmp_path, prompt="q2", hits=[], via="bail", reason="short")
    dashboard.append_recall_log(
        tmp_path, prompt="q3", hits=[{"id": "b" * 8, "score": 0.7, "title": "t2"}],
        via="mcp:search", source="synapse",
    )

    b = dashboard.consult_breakdown(tmp_path)
    names = {c["consumer"] for c in b["consumers"]}
    assert "claude-code" in names
    assert "synapse" in names
    # memflow never consulted → surfaced as a silent gap, not hidden.
    assert "memflow" in b["silent"]
    assert "synapse" not in b["silent"]

    cc = next(c for c in b["consumers"] if c["consumer"] == "claude-code")
    assert cc["consults"] == 2 and cc["fired"] == 1 and cc["bailed"] == 1
    syn = next(c for c in b["consumers"] if c["consumer"] == "synapse")
    assert syn["consults"] == 1 and syn["hit_rate"] == 1.0


def test_log_consult_records_via_and_lowercased_source(tmp_path: Path) -> None:
    fake = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path))
    _log_consult(
        fake,  # type: ignore[arg-type]
        tool="search",
        query="hola",
        hits=[{"id": "x" * 8, "score": 0.9, "title": "t"}],
        t0_ms=0,
        source="Synapse",
    )
    rows = dashboard.read_recall_log(tmp_path, limit=5)
    assert rows, "expected a consult row to be written"
    assert rows[0]["via"] == "mcp:search"
    assert rows[0]["source"] == "synapse"
    assert dashboard.consumer_label(rows[0]) == "synapse"
    assert "synapse" in dashboard_logs.read_consumer_last_seen(tmp_path)


def test_log_consult_falls_back_to_memo_source_env(tmp_path: Path, monkeypatch) -> None:
    """A client that can't pass per-call source= (devin / opencode / devin-desktop)
    is still attributed when MEMO_SOURCE is set in the server env."""
    monkeypatch.setenv("MEMO_SOURCE", "Devin")
    fake = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path))
    _log_consult(
        fake,  # type: ignore[arg-type]
        tool="search",
        query="q",
        hits=[{"id": "y" * 8, "score": 0.9, "title": "t"}],
        t0_ms=0,
    )
    rows = dashboard.read_recall_log(tmp_path, limit=5)
    assert rows and rows[0]["source"] == "devin"
    assert dashboard.consumer_label(rows[0]) == "devin"


def test_log_consult_explicit_source_overrides_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_SOURCE", "devin")
    fake = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path))
    _log_consult(
        fake,  # type: ignore[arg-type]
        tool="ask", query="q", hits=[], t0_ms=0, source="synapse",
    )
    rows = dashboard.read_recall_log(tmp_path, limit=5)
    assert rows and rows[0]["source"] == "synapse"


def test_log_consult_never_raises_on_bad_memory(tmp_path: Path) -> None:
    # A memory object without cfg must not break the caller — telemetry is
    # best-effort and swallows its own errors.
    _log_consult(object(), tool="ask", query="q", hits=[], t0_ms=0)  # type: ignore[arg-type]
