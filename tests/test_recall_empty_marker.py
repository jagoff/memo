"""Empty-recall epistemic marker (MEMO_RECALL_EMPTY_MARKER, default on).

When a recall search actually ran in a session and nothing qualified, the hook
must inject a one-line <memo-recall> marker so the reading agent can
distinguish "memo has no record of X" from "X is false". Bails (short/trivial
prompts, errors, session dedup) and sessionless invocations keep the historical
bare "{}" contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from memo.cli_recall_hook import recall_hook
from memo.recall_logic import (
    EMPTY_RECALL_MARKER,
    _recall_logic,
    render_empty_recall_output,
)

if TYPE_CHECKING:
    from memo.config import Config

# ---------------------------------------------------------------------------
# Flag registration + pure render helper
# ---------------------------------------------------------------------------


def test_flag_registered_default_on() -> None:
    from memo.flags import REGISTRY

    spec = REGISTRY["MEMO_RECALL_EMPTY_MARKER"]
    assert spec.kind == "bool"
    assert spec.default is True
    assert spec.group == "recall"
    assert spec.opt_out is True


def test_render_empty_recall_output_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_RECALL_EMPTY_MARKER", raising=False)
    out = render_empty_recall_output()
    assert out is not None
    parsed = json.loads(out)
    context = parsed["hookSpecificOutput"]["additionalContext"]
    assert context == EMPTY_RECALL_MARKER
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    # Marker matches the real block markup and stays one payload line.
    assert context.startswith("<memo-recall readonly>\n")
    assert context.endswith("\n</memo-recall>")
    assert "absence of record, not evidence of absence" in context


def test_render_empty_recall_output_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_EMPTY_MARKER", "0")
    assert render_empty_recall_output() is None


# ---------------------------------------------------------------------------
# Subprocess hook path (cli_recall_hook)
# ---------------------------------------------------------------------------


class _EmptySearchMemory:
    """Memory stub whose search always finds nothing."""

    def __init__(self, cfg: Any) -> None:
        pass

    def search(
        self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
    ):
        return []

    def close(self) -> None:
        pass


class _RaisingSearchMemory(_EmptySearchMemory):
    """Memory stub whose search always FAILS (e.g. locked DB)."""

    def search(
        self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
    ):
        raise RuntimeError("database is locked")


def _hook_env(tmp_cfg: Config, **extra: str) -> dict[str, str]:
    env = {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_RECALL_DISABLE": "0",
        "MEMO_RECALL_TOKEN_BUDGET": "0",
        "MEMO_RECALL_MIN_BODY_CHARS": "0",
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_SKIP_BELOW": "0.0",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_RECALL_EXPAND_CONTEXT": "0",
        "MEMO_RECALL_ADAPTIVE_CONTEXT": "0",
        "MEMO_RECALL_CONTEXTUAL": "0",
    }
    env.update(extra)
    return env


def test_subprocess_empty_search_emits_marker(tmp_cfg: Config, monkeypatch) -> None:
    """Search ran + session present + nothing qualified → one-line marker."""
    monkeypatch.setattr("memo.memory.Memory", _EmptySearchMemory)
    payload = json.dumps(
        {"prompt": "what do we know about the flux capacitor", "session_id": "sess-empty-1"}
    )
    result = CliRunner().invoke(
        recall_hook, input=payload, env=_hook_env(tmp_cfg), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["hookSpecificOutput"]["additionalContext"] == EMPTY_RECALL_MARKER


def test_subprocess_empty_search_flag_off_bails(tmp_cfg: Config, monkeypatch) -> None:
    """MEMO_RECALL_EMPTY_MARKER=0 restores the silent `{}` bail."""
    monkeypatch.setattr("memo.memory.Memory", _EmptySearchMemory)
    payload = json.dumps(
        {"prompt": "what do we know about the flux capacitor", "session_id": "sess-empty-2"}
    )
    result = CliRunner().invoke(
        recall_hook,
        input=payload,
        env=_hook_env(tmp_cfg, MEMO_RECALL_EMPTY_MARKER="0"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


def test_subprocess_sessionless_keeps_bail_contract(tmp_cfg: Config, monkeypatch) -> None:
    """No session_id in the payload → historical `{}` output, no marker."""
    monkeypatch.setattr("memo.memory.Memory", _EmptySearchMemory)
    payload = json.dumps({"prompt": "what do we know about the flux capacitor"})
    result = CliRunner().invoke(
        recall_hook, input=payload, env=_hook_env(tmp_cfg), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


def test_subprocess_short_prompt_bail_emits_no_marker(tmp_cfg: Config, monkeypatch) -> None:
    """A bail (no search ran) must stay `{}` even with a session present."""
    monkeypatch.setattr("memo.memory.Memory", _EmptySearchMemory)
    payload = json.dumps({"prompt": "hola", "session_id": "sess-empty-3"})
    result = CliRunner().invoke(
        recall_hook, input=payload, env=_hook_env(tmp_cfg), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


def test_subprocess_search_exception_bails_empty(tmp_cfg: Config, monkeypatch) -> None:
    """Search FAILED (raised) — absence unproven → `{}`, never the marker."""
    monkeypatch.setattr("memo.memory.Memory", _RaisingSearchMemory)
    payload = json.dumps(
        {"prompt": "what do we know about the flux capacitor", "session_id": "sess-err-1"}
    )
    result = CliRunner().invoke(
        recall_hook, input=payload, env=_hook_env(tmp_cfg), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


def test_subprocess_session_dedup_still_bails_empty(tmp_cfg: Config, monkeypatch) -> None:
    """All-hits-already-recalled is NOT 'no record' — no marker on turn 2."""
    from memo.memory import MemoryRecord

    hit = MemoryRecord(
        id="aabbccdd11223344",
        path="notes/test.md",
        title="Test Memory",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="This is the full body of the test memory, enough chars to matter.",
        extra={},
        score=0.80,
    )

    class _OneHitMemory(_EmptySearchMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            return [hit]

    monkeypatch.setattr("memo.memory.Memory", _OneHitMemory)
    payload = json.dumps(
        {"prompt": "what do you know about this topic", "session_id": "sess-dedup-1"}
    )
    env = _hook_env(tmp_cfg)
    runner = CliRunner()

    first = runner.invoke(recall_hook, input=payload, env=env, catch_exceptions=False)
    assert first.exit_code == 0, first.output
    assert "Test Memory" in json.loads(first.output)["hookSpecificOutput"]["additionalContext"]

    second = runner.invoke(recall_hook, input=payload, env=env, catch_exceptions=False)
    assert second.exit_code == 0, second.output
    assert second.output.strip() == "{}"


# ---------------------------------------------------------------------------
# Daemon path (_recall_logic)
# ---------------------------------------------------------------------------


def _empty_mem() -> SimpleNamespace:
    return SimpleNamespace(
        search=lambda *a, **k: [],
        embedder=SimpleNamespace(is_warm=True),
    )


def _raising_mem() -> SimpleNamespace:
    def _raise(*a: Any, **k: Any) -> list:
        raise RuntimeError("database is locked")

    return SimpleNamespace(search=_raise, embedder=SimpleNamespace(is_warm=True))


def test_recall_logic_empty_with_session_emits_marker(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    result, log_fn = _recall_logic(
        "a prompt that finds nothing at all",
        cwd=None,
        mem=_empty_mem(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        session_id="sess-daemon-empty-1",
        turn=1,
    )
    assert log_fn is None
    parsed = json.loads(result)
    assert parsed["hookSpecificOutput"]["additionalContext"] == EMPTY_RECALL_MARKER


def test_recall_logic_empty_with_session_flag_off(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    monkeypatch.setenv("MEMO_RECALL_EMPTY_MARKER", "0")
    result, log_fn = _recall_logic(
        "a prompt that finds nothing at all",
        cwd=None,
        mem=_empty_mem(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        session_id="sess-daemon-empty-2",
        turn=1,
    )
    assert result == "{}"
    assert log_fn is None


def test_recall_logic_empty_without_session_keeps_contract(monkeypatch, tmp_path) -> None:
    """Sessionless direct calls (tests, eval, debug) keep the bare `{}`."""
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    result, log_fn = _recall_logic(
        "a prompt that finds nothing at all",
        cwd=None,
        mem=_empty_mem(),
        cfg=SimpleNamespace(state_dir=tmp_path),
    )
    assert result == "{}"
    assert log_fn is None


def test_recall_logic_search_exception_bails_empty(monkeypatch, tmp_path) -> None:
    """Daemon path: search-fail returns `{}` even with a session — no marker."""
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    result, log_fn = _recall_logic(
        "a prompt whose search blows up",
        cwd=None,
        mem=_raising_mem(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        session_id="sess-daemon-err-1",
        turn=1,
    )
    assert result == "{}"
    assert log_fn is None


# ---------------------------------------------------------------------------
# Daemon/subprocess parity on search failure
# ---------------------------------------------------------------------------


def test_search_exception_parity_daemon_subprocess(tmp_cfg: Config, monkeypatch, tmp_path) -> None:
    """Both paths converge on `{}` when the search raises: a failed search is
    never 'absence of record', on either path."""
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    daemon_out, _ = _recall_logic(
        "what do we know about the flux capacitor",
        cwd=None,
        mem=_raising_mem(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        session_id="sess-parity-1",
        turn=1,
    )

    monkeypatch.setattr("memo.memory.Memory", _RaisingSearchMemory)
    payload = json.dumps(
        {"prompt": "what do we know about the flux capacitor", "session_id": "sess-parity-1"}
    )
    sub = CliRunner().invoke(
        recall_hook, input=payload, env=_hook_env(tmp_cfg), catch_exceptions=False
    )
    assert sub.exit_code == 0, sub.output
    assert daemon_out == "{}"
    assert sub.output.strip() == daemon_out
