"""`memo terminal receiver` CLI: the flag gate, the capability file, and the
error translation.

The transport is opt-in and writes into a live PTY, so the CLI's job is to
refuse unless explicitly enabled, to persist the capability privately, and to
turn transport failures into clean ClickExceptions instead of tracebacks.
"""

from __future__ import annotations

import json
import os
import pty
import stat

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.terminal_receiver import ReceiverSession, ReceiverSupervisor

pytestmark = pytest.mark.skipif(os.name != "posix", reason="receiver transport is Unix-only")


def _env(tmp_cfg, *, enabled: bool) -> dict[str, str]:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }
    if enabled:
        env["MEMO_TERMINAL_RECEIVER_ENABLED"] = "1"
    return env


def _session() -> ReceiverSession:
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("/bin/cat", ["cat"])  # noqa: S606 - test child, no shell
    return ReceiverSession(fd, pid)


@pytest.mark.parametrize(
    "argv",
    [
        ["terminal", "receiver", "attach", "/bin/cat"],
        [
            "terminal",
            "receiver",
            "send",
            "--socket",
            "/tmp/x",
            "--capability-file",
            "/tmp/c",
            "--message-id",
            "m",
            "--message",
            "hi",
        ],
        [
            "terminal",
            "receiver",
            "enter",
            "--socket",
            "/tmp/x",
            "--capability-file",
            "/tmp/c",
            "--message-id",
            "m",
        ],
    ],
)
def test_every_receiver_verb_is_gated_by_the_flag(tmp_cfg, argv) -> None:
    result = CliRunner().invoke(cli, argv, env=_env(tmp_cfg, enabled=False))

    assert result.exit_code != 0
    assert "MEMO_TERMINAL_RECEIVER_ENABLED" in result.output


def test_send_and_enter_round_trip_through_a_capability_file(tmp_cfg, tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    socket_path = supervisor.start()
    capability_file = tmp_path / "cap"
    capability_file.write_text(supervisor.capability, encoding="ascii")
    capability_file.chmod(0o600)
    runner = CliRunner()
    env = _env(tmp_cfg, enabled=True)

    try:
        sent = runner.invoke(
            cli,
            [
                "terminal",
                "receiver",
                "send",
                "--socket",
                str(socket_path),
                "--capability-file",
                str(capability_file),
                "--message-id",
                "cli-1",
                "--message",
                "hello",
            ],
            env=env,
        )
        assert sent.exit_code == 0, sent.output
        assert json.loads(sent.output)["ok"] is True

        entered = runner.invoke(
            cli,
            [
                "terminal",
                "receiver",
                "enter",
                "--socket",
                str(socket_path),
                "--capability-file",
                str(capability_file),
                "--message-id",
                "cli-2",
            ],
            env=env,
        )
        assert entered.exit_code == 0, entered.output
        assert json.loads(entered.output)["ok"] is True
    finally:
        supervisor.close()


def test_a_world_readable_capability_file_is_refused(tmp_cfg, tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    socket_path = supervisor.start()
    capability_file = tmp_path / "cap"
    capability_file.write_text(supervisor.capability, encoding="ascii")
    capability_file.chmod(0o644)

    try:
        result = CliRunner().invoke(
            cli,
            [
                "terminal",
                "receiver",
                "send",
                "--socket",
                str(socket_path),
                "--capability-file",
                str(capability_file),
                "--message-id",
                "cli-3",
                "--message",
                "hello",
            ],
            env=_env(tmp_cfg, enabled=True),
        )
        assert result.exit_code != 0
        assert "0600" in result.output
        assert "Traceback" not in result.output
    finally:
        supervisor.close()


def test_an_unreachable_socket_is_a_clean_error(tmp_cfg, tmp_path) -> None:
    capability_file = tmp_path / "cap"
    capability_file.write_text("whatever", encoding="ascii")
    capability_file.chmod(0o600)

    result = CliRunner().invoke(
        cli,
        [
            "terminal",
            "receiver",
            "enter",
            "--socket",
            str(tmp_path / "missing.sock"),
            "--capability-file",
            str(capability_file),
            "--message-id",
            "cli-4",
        ],
        env=_env(tmp_cfg, enabled=True),
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_attach_refuses_a_command_that_cannot_start(tmp_cfg) -> None:
    result = CliRunner().invoke(
        cli,
        ["terminal", "receiver", "attach", "   "],
        env=_env(tmp_cfg, enabled=True),
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_capability_file_is_written_private_and_never_overwritten(tmp_path) -> None:
    from memo.cli_terminal import _write_capability

    target = tmp_path / "cap"
    _write_capability(str(target), "s3cret")

    assert target.read_text(encoding="ascii") == "s3cret"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    # O_EXCL: an existing capability is never silently replaced.
    with pytest.raises(FileExistsError):
        _write_capability(str(target), "other")
