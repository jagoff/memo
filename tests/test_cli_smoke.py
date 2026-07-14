"""Smoke tests for CLI commands that previously lacked coverage.

Invoking with --help is sufficient to catch Click registration errors and
ensure the command is reachable without requiring heavy backends (DB, MLX,
git). For sync clone, we monkeypatch the backend and do a real invocation
to verify the Click option→param binding.

Isolation: all tests pass MEMO_NONINTERACTIVE=1, MEMO_DATA_DIR, and
MEMO_STATE_DIR via env= (conftest already sets these as defaults, but we
pin them explicitly so tests stay hermetic even if run in isolation).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    """Minimal CLI env — no real config, no real DB."""
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


# ---------------------------------------------------------------------------
# memo contradict scan
# ---------------------------------------------------------------------------


def test_contradict_scan_help(tmp_path: Path) -> None:
    """Verify command is registered and Click options are valid."""
    res = CliRunner().invoke(cli, ["contradict", "scan", "--help"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "--top-k" in res.output


# ---------------------------------------------------------------------------
# memo sync clone
# ---------------------------------------------------------------------------


def test_sync_clone_help(tmp_path: Path) -> None:
    res = CliRunner().invoke(cli, ["sync", "clone", "--help"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "URL" in res.output


def test_sync_clone_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real invocation with mocked clone_bootstrap verifies param binding."""
    dest = tmp_path / "cloned-repo"

    def _fake_clone(url: str, dest_path: Path) -> dict:
        return {
            "cloned": str(dest_path),
            "memories": 7,
            "memories_dir": str(dest_path / "memories"),
        }

    monkeypatch.setattr("memo.sync_git.clone_bootstrap", _fake_clone)

    res = CliRunner().invoke(
        cli,
        ["sync", "clone", "https://example.com/memo-sync.git", "--dest", str(dest)],
        env=_env(tmp_path),
    )
    assert res.exit_code == 0
    assert "7" in res.output or "Cloned" in res.output


# ---------------------------------------------------------------------------
# memo map
# ---------------------------------------------------------------------------


def test_map_help(tmp_path: Path) -> None:
    res = CliRunner().invoke(cli, ["map", "--help"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "--output" in res.output


# ---------------------------------------------------------------------------
# memo record-history
# ---------------------------------------------------------------------------


def test_record_history_help(tmp_path: Path) -> None:
    res = CliRunner().invoke(cli, ["record-history", "--help"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "ID_OR_PREFIX" in res.output


def test_ask_human_output_handles_none_source_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeMemory:
        def ask(self, *_args, **_kwargs):
            return {
                "question": "q",
                "answer": "answer",
                "sources": [
                    {
                        "id_short": "abc12345",
                        "title": "Expanded source",
                        "score": None,
                    }
                ],
            }

    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: _FakeMemory())

    res = CliRunner().invoke(cli, ["ask", "q"], env=_env(tmp_path))

    assert res.exit_code == 0, res.output
    assert "Expanded source" in res.output


def test_chat_ask_human_output_handles_none_source_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMemory:
        def chat_ask(self, *_args, **_kwargs):
            return {
                "question": "q",
                "answer": "answer",
                "sources": [
                    {
                        "id_short": "abc12345",
                        "title": "Expanded source",
                        "score": None,
                    }
                ],
            }

    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: _FakeMemory())

    res = CliRunner().invoke(cli, ["chat-ask", "q"], env=_env(tmp_path))

    assert res.exit_code == 0, res.output
    assert "Expanded source" in res.output
