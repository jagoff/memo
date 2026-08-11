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

The reset is UNCONDITIONAL on MEMO_EMITTED_LEDGER (`--force` alone gates it,
not the flag) — see `test_force_clears_the_ledger_even_with_flag_unset` for
why: with the flag off there is no ledger file so a gate would be a pure
optimisation, and that "optimisation" is actually unsafe here because the
flag is not static across a session's lifetime in this repo (dream_flags'
flag-graduation machinery flips it through the tuned overlay between nights).
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


def test_force_clears_the_ledger_even_with_flag_unset(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reset is UNCONDITIONAL on MEMO_EMITTED_LEDGER -- do not gate it on
    the flag as an "optimisation". `MEMO_EMITTED_LEDGER` is not static across
    a session's lifetime in this repo: dream_flags' flag-graduation/
    auto-revert machinery flips default-off flags through the tuned overlay
    between nights. A flag-gated reset would make this reachable: entries
    accumulate while the flag is ON -> compaction happens while the flag is
    OFF, so a gated reset would skip and the stale file would survive -> the
    flag flips back ON. Every ledger reader (apply_ledger, recall_logic._log,
    the subprocess hook) gates on the flag's value at READ time, not on
    whether an entry predates the last compaction, so those stale entries
    would resurface and get digested as if the model could still see them --
    the exact bug this task exists to prevent. With the feature off there is
    no ledger file to begin with, so the unconditional reset costs a single
    free `stat()` on this --force-only path either way (this test still
    seeds one, to prove the reset doesn't skip merely because the flag looks
    off at reset time)."""
    monkeypatch.delenv("MEMO_EMITTED_LEDGER", raising=False)
    monkeypatch.setenv("MEMO_SESSION_ID", _SID)
    _seed(state_dir)

    result = CliRunner().invoke(capture_tick, ["--force"])

    assert result.exit_code == 0
    assert el.read(state_dir, _SID) == {}


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
