"""`capture-tick --force` resets the emission ledger at the compaction boundary.

PreCompact already runs `memo capture-tick --force` (hooks/hooks.json,
cli_hooks.wire_precompact_hook), so `--force` doubles as the compaction-boundary
signal here — no second hook wiring, no settings.json migration.

Session id: the reset uses `identity._session_id()` (env: MEMO_SESSION_ID /
CLAUDE_SESSION_ID / CLAUDE_CODE_SESSION_ID) — the SAME resolution the
subprocess recall hook (`cli_recall_hook.py`) and the MCP tools
(`server_common.py`'s `_effective_session_id()`) use to key their own ledger
writes, per those modules' own comments. `capture-tick` is spawned the same
way (a per-invocation subprocess of the Claude Code session), so it inherits
the same env and can resolve the same id -- independent of the `session_id`
field on the hook's stdin payload, which only keys the CAPTURE watermark
(a different, pre-existing mechanism). That is also why these tests invoke
`capture_tick` with no stdin payload at all: the reset must not depend on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo import emitted_ledger as el
from memo.cli_capture import capture_tick

_SID = "sess-reset"


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated state_dir wired into MEMO_STATE_DIR so a CliRunner-invoked
    `capture_tick`'s own `Config.from_env()` resolves to this same path."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setenv("MEMO_STATE_DIR", str(d))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    return d


def _seed(state_dir: Path, sid: str = _SID) -> None:
    el.append(
        state_dir,
        sid,
        [el.Entry(id="mem_a", h=el.emitted_hash("a"), n=1, ref="memo-r/aaaaaa", t=1, src="mcp")],
    )


def test_force_clears_the_ledger(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_SESSION_ID", _SID)
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    _seed(state_dir)

    result = CliRunner().invoke(capture_tick, ["--force"])

    assert result.exit_code == 0
    assert el.read(state_dir, _SID) == {}


def test_non_force_leaves_the_ledger_alone(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMO_SESSION_ID", _SID)
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    _seed(state_dir)

    result = CliRunner().invoke(capture_tick, [])

    assert result.exit_code == 0
    assert set(el.read(state_dir, _SID)) == {"mem_a"}


def test_force_is_idempotent(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PreCompact double-fires against the plugin copy (hooks/hooks.json +
    cli_hooks.wire_precompact_hook both wire `capture-tick --force`)."""
    monkeypatch.setenv("MEMO_SESSION_ID", _SID)
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    _seed(state_dir)

    r1 = CliRunner().invoke(capture_tick, ["--force"])
    r2 = CliRunner().invoke(capture_tick, ["--force"])

    assert r1.exit_code == 0 and r2.exit_code == 0
    assert el.read(state_dir, _SID) == {}


def test_flag_unset_force_leaves_the_ledger_alone(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With MEMO_EMITTED_LEDGER unset (default off), `--force` must behave
    exactly as it did before this feature: the ledger is untouched, since the
    reset never engages the flag-gated code path at all."""
    monkeypatch.delenv("MEMO_EMITTED_LEDGER", raising=False)
    monkeypatch.setenv("MEMO_SESSION_ID", _SID)
    _seed(state_dir)

    result = CliRunner().invoke(capture_tick, ["--force"])

    assert result.exit_code == 0
    assert set(el.read(state_dir, _SID)) == {"mem_a"}


def test_no_resolvable_session_id_resets_nothing(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No MEMO_SESSION_ID/CLAUDE_SESSION_ID/CLAUDE_CODE_SESSION_ID in env ->
    identity._session_id() returns None -> reset nothing rather than guess."""
    for var in ("MEMO_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    _seed(state_dir)

    result = CliRunner().invoke(capture_tick, ["--force"])

    assert result.exit_code == 0
    assert set(el.read(state_dir, _SID)) == {"mem_a"}
