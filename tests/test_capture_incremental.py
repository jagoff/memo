"""Incremental mid-session capture — watermark + throttle tests.

Stubs the helper LLM (`extract_insights`) and the embedder so tests run on
any platform (no MLX). Focus: the watermark advances and bounds reprocessing,
a re-run with no new turns is a no-op, and a corrupt/missing/out-of-range
watermark degrades to a safe full pass — the parts most likely to break under
refactor or to silently drop a long session's insight.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.capture import incremental_tick_due, run_capture_incremental
from memo.cli import cli


def _write_transcript(p: Path, lines: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _assistant(marker: str) -> str:
    """≥200 chars + trigger keywords so the combined turn passes the prefilter."""
    return (
        f"{marker} decidí cambiar la configuración porque encontramos un bug en el "
        "reranker que se manifestaba con bodies largos; el fix fue truncar el texto "
        "antes de rankear y la latencia bajó tres veces en el camino caliente del hook."
    )


def _exchange(user: str, marker: str) -> list[dict]:
    return [
        {"type": "user", "message": {"content": user}},
        {"type": "assistant", "message": {"content": _assistant(marker)}},
    ]


def _setup_env(tmp_path: Path, monkeypatch) -> Path:
    """Isolated data/state/vault + a deterministic 4-dim embedder stub.

    Mirrors the env-pinning conventions in tests/conftest.py: nothing reads
    the developer's real vault or the live recall daemon.
    """
    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state.mkdir()
    monkeypatch.setenv("MEMO_DATA_DIR", str(data))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "0")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )
    return state


def _watermark(state_dir: Path, session_id: str) -> dict:
    f = state_dir / ".capture_watermark" / f"{session_id}.json"
    return json.loads(f.read_text(encoding="utf-8"))


# ── watermark advances + bounds reprocessing ────────────────────────────────


def test_watermark_advances_and_bounds_reprocessing(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []

    def _record(chat, model, user_text, assistant_text):
        calls.append((user_text, assistant_text))
        return []  # no saves — this test asserts on the watermark + slicing

    monkeypatch.setattr("memo.capture.extract_insights", _record)

    transcript = tmp_path / "t.jsonl"
    sid = "sess-A"
    _write_transcript(
        transcript,
        _exchange("primera pregunta", "ALPHAMARK") + _exchange("segunda", "BETAMARK"),
    )

    out1 = run_capture_incremental(transcript, sid)
    assert out1["status"] == "ok"
    assert out1["exchange_count"] == 2
    assert _watermark(state, sid)["exchange_count"] == 2
    # First pass sees both new exchanges.
    assert len(calls) == 1
    assert "ALPHAMARK" in calls[0][1] and "BETAMARK" in calls[0][1]

    # A third exchange arrives — only IT should be reprocessed.
    _write_transcript(
        transcript,
        _exchange("primera pregunta", "ALPHAMARK")
        + _exchange("segunda", "BETAMARK")
        + _exchange("tercera", "GAMMAMARK"),
    )
    out2 = run_capture_incremental(transcript, sid)
    assert out2["status"] == "ok"
    assert out2["processed_turns"] == 1
    assert _watermark(state, sid)["exchange_count"] == 3
    assert len(calls) == 2
    # Bounded: the second pass saw GAMMA only, never the already-captured turns.
    assert "GAMMAMARK" in calls[1][1]
    assert "ALPHAMARK" not in calls[1][1]
    assert "BETAMARK" not in calls[1][1]


def test_second_immediate_pass_captures_nothing_new(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []

    def _record(chat, model, user_text, assistant_text):
        calls.append((user_text, assistant_text))
        return []

    monkeypatch.setattr("memo.capture.extract_insights", _record)

    transcript = tmp_path / "t.jsonl"
    sid = "sess-B"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    first = run_capture_incremental(transcript, sid)
    assert first["status"] == "ok"
    assert len(calls) == 1

    # No transcript change → nothing new → no extraction, watermark steady.
    second = run_capture_incremental(transcript, sid)
    assert second["status"] == "no_new"
    assert second["exchange_count"] == 1
    assert len(calls) == 1  # extractor was NOT called again
    assert _watermark(state, sid)["exchange_count"] == 1


def test_incremental_saves_new_insight(tmp_path: Path, monkeypatch):
    """End-to-end: a survived insight is saved and the watermark advances."""
    _setup_env(tmp_path, monkeypatch)

    def _one_insight(chat, model, user_text, assistant_text):
        return [
            {
                "title": "Reranker threshold 0.4 fix",
                "type": "decision",
                "body": (
                    "Cambiar el threshold del reranker a 0.4 resolvió el bug de hits "
                    "relevantes filtrados en queries difusas, porque los scores "
                    "fusionados necesitan un piso más permisivo que el coseno puro."
                ),
                "tags": ["reranker", "bug"],
            }
        ]

    monkeypatch.setattr("memo.capture.extract_insights", _one_insight)

    transcript = tmp_path / "t.jsonl"
    sid = "sess-save"
    _write_transcript(transcript, _exchange("qué arreglamos", "ALPHAMARK"))

    out = run_capture_incremental(transcript, sid)
    assert out["status"] == "ok"
    assert len(out["saved"]) == 1


# ── corrupt / missing / out-of-range watermark ──────────────────────────────


def test_missing_watermark_starts_from_zero(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "memo.capture.extract_insights",
        lambda *a: calls.append((a[2], a[3])) or [],
    )

    transcript = tmp_path / "t.jsonl"
    sid = "sess-missing"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    assert not (state / ".capture_watermark" / f"{sid}.json").exists()
    out = run_capture_incremental(transcript, sid)
    assert out["status"] == "ok"
    assert len(calls) == 1  # processed from scratch
    assert _watermark(state, sid)["exchange_count"] == 1


def test_corrupt_watermark_handled(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "memo.capture.extract_insights",
        lambda *a: calls.append((a[2], a[3])) or [],
    )

    sid = "sess-corrupt"
    wm_file = state / ".capture_watermark" / f"{sid}.json"
    wm_file.parent.mkdir(parents=True, exist_ok=True)
    wm_file.write_text("}{ not json at all", encoding="utf-8")

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    out = run_capture_incremental(transcript, sid)  # must not raise
    assert out["status"] == "ok"
    assert len(calls) == 1  # corrupt → treated as fresh, full pass
    assert _watermark(state, sid)["exchange_count"] == 1


def test_negative_watermark_resets_to_zero(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "memo.capture.extract_insights",
        lambda *a: calls.append((a[2], a[3])) or [],
    )

    sid = "sess-oob-neg"
    wm_file = state / ".capture_watermark" / f"{sid}.json"
    wm_file.parent.mkdir(parents=True, exist_ok=True)
    wm_file.write_text(json.dumps({"exchange_count": -5}), encoding="utf-8")

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    out = run_capture_incremental(transcript, sid)
    assert out["status"] == "ok"
    assert len(calls) == 1  # negative → reset to 0, reprocesses whole transcript
    assert _watermark(state, sid)["exchange_count"] == 1


def test_stale_watermark_ahead_of_transcript_returns_no_new(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "memo.capture.extract_insights",
        lambda *a: calls.append((a[2], a[3])) or [],
    )

    sid = "sess-oob-stale"
    wm_file = state / ".capture_watermark" / f"{sid}.json"
    wm_file.parent.mkdir(parents=True, exist_ok=True)
    wm_file.write_text(json.dumps({"exchange_count": 999}), encoding="utf-8")

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    out = run_capture_incremental(transcript, sid)
    assert out["status"] == "no_new"  # watermark > total → nothing new, no re-pass
    assert len(calls) == 0  # should NOT reprocess already-captured turns


def test_empty_transcript_is_no_pair(tmp_path: Path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "missing.jsonl"  # never created
    out = run_capture_incremental(transcript, "sess-empty")
    assert out["status"] == "no_pair"


# ── throttle (incremental_tick_due) ─────────────────────────────────────────


def test_tick_due_respects_interval(tmp_path: Path, monkeypatch):
    import time

    state = _setup_env(tmp_path, monkeypatch)
    sid = "sess-throttle"
    wm_file = state / ".capture_watermark" / f"{sid}.json"
    wm_file.parent.mkdir(parents=True, exist_ok=True)

    # No watermark yet → always due.
    assert incremental_tick_due(state, sid, 600) is True

    # Just stamped → not due within the interval.
    wm_file.write_text(json.dumps({"updated": time.time()}), encoding="utf-8")
    assert incremental_tick_due(state, sid, 600) is False

    # Stamped long ago → due again.
    wm_file.write_text(json.dumps({"updated": time.time() - 700}), encoding="utf-8")
    assert incremental_tick_due(state, sid, 600) is True

    # interval_s <= 0 disables the throttle (every prompt).
    wm_file.write_text(json.dumps({"updated": time.time()}), encoding="utf-8")
    assert incremental_tick_due(state, sid, 0) is True


# ── CLI: memo capture-tick ──────────────────────────────────────────────────


def _cli_env(state: Path, tmp_path: Path, interval: str) -> dict:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(state),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_AUTO_PROJECT_TAG": "0",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_CAPTURE_INTERVAL_S": interval,
    }


def test_capture_tick_cli_silent_and_advances(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr("memo.capture.extract_insights", lambda *a, **k: [])

    transcript = tmp_path / "t.jsonl"
    sid = "sess-cli"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["capture-tick"],
        input=json.dumps({"session_id": sid, "transcript_path": str(transcript)}),
        env=_cli_env(state, tmp_path, interval="0"),  # disable throttle
    )
    assert result.exit_code == 0
    assert result.output.strip() == "{}"  # always silent
    assert _watermark(state, sid)["exchange_count"] == 1


def test_capture_tick_cli_throttled_second_call(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    calls: list[int] = []
    monkeypatch.setattr(
        "memo.capture.extract_insights",
        lambda *a, **k: calls.append(1) or [],
    )

    transcript = tmp_path / "t.jsonl"
    sid = "sess-cli-throttle"
    _write_transcript(transcript, _exchange("pregunta", "ALPHAMARK"))

    runner = CliRunner()
    env = _cli_env(state, tmp_path, interval="600")  # real throttle
    payload = json.dumps({"session_id": sid, "transcript_path": str(transcript)})

    first = runner.invoke(cli, ["capture-tick"], input=payload, env=env)
    assert first.exit_code == 0
    assert len(calls) == 1

    # Second call within the interval → throttled, no extraction.
    second = runner.invoke(cli, ["capture-tick"], input=payload, env=env)
    assert second.exit_code == 0
    assert second.output.strip() == "{}"
    assert len(calls) == 1  # extractor NOT called again


def test_capture_tick_cli_no_session_is_noop(tmp_path: Path, monkeypatch):
    state = _setup_env(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["capture-tick"],
        input=json.dumps({"transcript_path": "/nope.jsonl"}),  # no session_id
        env=_cli_env(state, tmp_path, interval="0"),
    )
    assert result.exit_code == 0
    assert result.output.strip() == "{}"
