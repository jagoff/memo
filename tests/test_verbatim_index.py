"""Tests for the verbatim turn-level indexer (parser + incremental pass, Task V3)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.config import Config
from memo.store.turn_store import TurnStore
from memo.verbatim_index import parse_turns, run_verbatim_index_pass

_FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


def _line(role: str, text: str, ts: str) -> str:
    return json.dumps(
        {
            "type": role,
            "message": {"role": role, "content": text},
            "timestamp": ts,
        }
    )


def _write_transcript(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── parse_turns ──────────────────────────────────────────────────────────


def test_parse_turns_preserves_timestamp(tmp_path: Path):
    f = tmp_path / "sess.jsonl"
    _write_transcript(
        f,
        [
            _line(
                "user",
                "this is a long enough user turn to survive the min-chars filter",
                "2026-07-13T10:00:00.000Z",
            ),
            _line(
                "assistant",
                "this is a long enough assistant reply to survive the min-chars filter",
                "2026-07-13T10:00:05.000Z",
            ),
        ],
    )
    turns = parse_turns(f)
    assert len(turns) == 2
    assert turns[0]["ts"] == "2026-07-13T10:00:00+00:00"
    assert turns[0]["role"] == "user"
    assert turns[0]["idx"] == 0
    assert turns[1]["idx"] == 1


def test_parse_turns_skips_short_turns(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_VERBATIM_MIN_CHARS", "20")
    f = tmp_path / "sess.jsonl"
    _write_transcript(
        f,
        [
            _line("user", "short", "2026-07-13T10:00:00.000Z"),
            _line(
                "assistant",
                "this reply is long enough to clear the minimum characters filter",
                "2026-07-13T10:00:05.000Z",
            ),
        ],
    )
    turns = parse_turns(f)
    assert len(turns) == 1
    assert turns[0]["role"] == "assistant"


def test_parse_turns_redacts_secrets(tmp_path: Path):
    f = tmp_path / "sess.jsonl"
    _write_transcript(
        f,
        [
            _line(
                "user",
                f"here is my key {_FAKE_AWS_KEY} please do not leak it anywhere ever",
                "2026-07-13T10:00:00.000Z",
            ),
        ],
    )
    turns = parse_turns(f)
    assert len(turns) == 1
    assert _FAKE_AWS_KEY not in turns[0]["text"]
    assert "****" in turns[0]["text"]


def test_parse_turns_missing_file_returns_empty(tmp_path: Path):
    assert parse_turns(tmp_path / "nope.jsonl") == []


def test_parse_turns_skips_missing_or_invalid_timestamps(tmp_path: Path):
    f = tmp_path / "sess.jsonl"
    _write_transcript(
        f,
        [
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": _long("missing timestamp should not persist")},
                }
            ),
            _line("assistant", _long("invalid timestamp should not persist"), "not-a-date"),
        ],
    )

    assert parse_turns(f) == []


# ── run_verbatim_index_pass ──────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "none.toml"))
    return Config(data_dir=tmp_path / "data", state_dir=tmp_path / "state")


@pytest.fixture
def claude_projects_root(tmp_path: Path, monkeypatch) -> Path:
    """`run_verbatim_index_pass` walks `Path.home() / ".claude" / "projects"`."""
    root = tmp_path / ".claude" / "projects"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _long(text: str) -> str:
    """Pad text past the default MEMO_VERBATIM_MIN_CHARS (20)."""
    return text + " " * max(0, 25 - len(text))


def test_pass_ingests_turns(cfg: Config, claude_projects_root: Path):
    f = claude_projects_root / "sess-1.jsonl"
    _write_transcript(
        f,
        [
            _line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z"),
            _line("assistant", _long("hi back, also long enough"), "2026-07-13T10:00:05.000Z"),
        ],
    )
    result = run_verbatim_index_pass(cfg)
    assert result["status"] == "ok"
    assert result["sessions_indexed"] == 1
    assert result["turns_indexed"] == 2
    assert result["skipped_unchanged"] == 0

    store = TurnStore(cfg.verbatim_db)
    try:
        assert store.stats() == {"sessions": 1, "turns": 2}
    finally:
        store.close()


def test_second_run_skips_unchanged(cfg: Config, claude_projects_root: Path):
    f = claude_projects_root / "sess-1.jsonl"
    _write_transcript(
        f,
        [_line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z")],
    )
    first = run_verbatim_index_pass(cfg)
    assert first["sessions_indexed"] == 1

    second = run_verbatim_index_pass(cfg)
    assert second["sessions_indexed"] == 0
    assert second["skipped_unchanged"] == 1


def test_grown_file_reingests_whole_session(cfg: Config, claude_projects_root: Path):
    f = claude_projects_root / "sess-1.jsonl"
    _write_transcript(
        f,
        [_line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z")],
    )
    run_verbatim_index_pass(cfg)

    _write_transcript(
        f,
        [
            _line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z"),
            _line(
                "assistant", _long("a brand new second turn appended"), "2026-07-13T10:00:05.000Z"
            ),
        ],
    )
    result = run_verbatim_index_pass(cfg)
    assert result["sessions_indexed"] == 1
    assert result["turns_indexed"] == 2

    store = TurnStore(cfg.verbatim_db)
    try:
        assert store.stats() == {"sessions": 1, "turns": 2}
    finally:
        store.close()


def test_dry_run_writes_nothing(cfg: Config, tmp_path: Path):
    """Test dry run using injectable root param directly (no monkeypatch needed)."""
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    f = root / "sess-1.jsonl"
    _write_transcript(
        f,
        [_line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z")],
    )
    result = run_verbatim_index_pass(cfg, root=root, dry_run=True)
    assert result["status"] == "ok"
    assert result["sessions_indexed"] == 1
    assert result["turns_indexed"] == 1

    assert not cfg.verbatim_db.exists()
    assert not (cfg.state_dir / "verbatim-index.json").exists()


def test_pass_clamps_retention_to_at_least_one_day(cfg: Config, tmp_path: Path, monkeypatch):
    observed: dict[str, float] = {}

    def _find(root: Path, *, since_days: float):
        observed["since_days"] = since_days
        return []

    monkeypatch.setattr("memo.verbatim_index.find_transcripts", _find)
    result = run_verbatim_index_pass(cfg, root=tmp_path, max_days=-30, dry_run=True)

    assert result["status"] == "ok"
    assert observed["since_days"] == 1


def test_index_files_are_private(cfg: Config, claude_projects_root: Path):
    f = claude_projects_root / "sess-private.jsonl"
    _write_transcript(
        f,
        [_line("user", _long("private indexed transcript turn"), "2026-07-13T10:00:00Z")],
    )

    result = run_verbatim_index_pass(cfg)
    assert result["status"] == "ok"
    assert stat.S_IMODE(cfg.state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(cfg.verbatim_db.stat().st_mode) == 0o600
    watermark = cfg.state_dir / "verbatim-index.json"
    assert stat.S_IMODE(watermark.stat().st_mode) == 0o600


def test_never_raises_on_store_failure(cfg: Config, claude_projects_root: Path, monkeypatch):
    f = claude_projects_root / "sess-1.jsonl"
    _write_transcript(
        f,
        [_line("user", _long("hello there, a long enough turn"), "2026-07-13T10:00:00.000Z")],
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("store exploded")

    monkeypatch.setattr("memo.verbatim_index.TurnStore", _boom)
    result = run_verbatim_index_pass(cfg)
    assert result["status"] == "error"
    assert "error" in result
    assert "store exploded" in result["error"]


# ── CLI: memo verbatim index ─────────────────────────────────────────────


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_cli_verbatim_index_json(tmp_path: Path, monkeypatch):
    fake_result = {
        "status": "ok",
        "sessions_indexed": 2,
        "turns_indexed": 5,
        "pruned": 0,
        "skipped_unchanged": 1,
    }
    # cli_verbatim.index_cmd imports run_verbatim_index_pass lazily from
    # memo.verbatim_index — patch it at the source module.
    import memo.verbatim_index as verbatim_index_module

    monkeypatch.setattr(
        verbatim_index_module, "run_verbatim_index_pass", lambda cfg, **kw: fake_result
    )

    from memo.cli import cli

    res = CliRunner().invoke(cli, ["verbatim", "index", "--json"], env=_cli_env(tmp_path))
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data == fake_result


def test_cli_verbatim_index_help(tmp_path: Path):
    from memo.cli import cli

    res = CliRunner().invoke(cli, ["verbatim", "--help"])
    assert res.exit_code == 0
    assert "verbatim" in res.output.lower()

    res2 = CliRunner().invoke(cli, ["verbatim", "index", "--help"])
    assert res2.exit_code == 0
    assert "index" in res2.output.lower()
