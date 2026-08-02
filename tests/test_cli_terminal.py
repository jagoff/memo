"""End-user CLI surface for live terminal coordination."""

from __future__ import annotations

import json
import os
import pty
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.terminal_live import ProcessSnapshot, TerminalBridge


def _env(tmp_cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }


def test_terminal_command_exposes_live_user_workflow(tmp_cfg) -> None:
    result = CliRunner().invoke(
        cli,
        ["terminal", "--help"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert all(command in result.output for command in ("register", "list", "send", "enter"))


def test_terminal_send_cli_returns_json_receipt(tmp_cfg, monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    tty_path = Path(os.ttyname(slave_fd))
    payloads: list[bytes] = []

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty_path,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def present(_path: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append(payload)
        return "test"

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe, presenter=present)
        target = bridge.register(agent="codex", tty=tty_path, pid=4242)
        monkeypatch.setattr("memo.cli_terminal._bridge", lambda: bridge)

        result = CliRunner().invoke(
            cli,
            [
                "terminal",
                "send",
                "--to",
                target.id,
                "--message",
                "ping",
                "--message-id",
                "cli-msg-1",
                "--json",
            ],
            env=_env(tmp_cfg),
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["status"] == "delivered"
        assert payloads == [b"ping\r"]
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_terminal_register_list_enter_and_history_cli(tmp_cfg, monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    tty_path = Path(os.ttyname(slave_fd))
    payloads: list[bytes] = []

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty_path,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def present(_path: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append(payload)
        return "test"

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe, presenter=present)
        monkeypatch.setattr("memo.cli_terminal._bridge", lambda: bridge)
        runner = CliRunner()

        registered = runner.invoke(
            cli,
            [
                "terminal",
                "register",
                "--agent",
                "codex",
                "--tty",
                str(tty_path),
                "--pid",
                "4242",
                "--terminal-app",
                "Ghostty",
                "--project",
                "/tmp/memo",
                "--json",
            ],
            env=_env(tmp_cfg),
        )
        assert registered.exit_code == 0, registered.output
        registration_id = json.loads(registered.output)["id"]

        listed = runner.invoke(cli, ["terminal", "list"], env=_env(tmp_cfg))
        assert listed.exit_code == 0, listed.output
        assert f"{registration_id}\tcodex\t{tty_path}\t/tmp/memo" in listed.output
        listed_json = runner.invoke(cli, ["terminal", "list", "--json"], env=_env(tmp_cfg))
        assert listed_json.exit_code == 0, listed_json.output
        assert json.loads(listed_json.output)[0]["id"] == registration_id

        entered = runner.invoke(
            cli,
            [
                "terminal",
                "enter",
                "--to",
                registration_id,
                "--message-id",
                "cli-enter-1",
                "--json",
            ],
            env=_env(tmp_cfg),
        )
        assert entered.exit_code == 0, entered.output
        assert json.loads(entered.output)["kind"] == "enter"
        assert payloads == [b"\r"]

        entered_text = runner.invoke(
            cli,
            [
                "terminal",
                "enter",
                "--to",
                registration_id,
                "--message-id",
                "cli-enter-2",
            ],
            env=_env(tmp_cfg),
        )
        assert entered_text.exit_code == 0, entered_text.output
        assert "delivered" in entered_text.output
        assert "via test" in entered_text.output
        assert payloads == [b"\r", b"\r"]

        history = runner.invoke(cli, ["terminal", "history"], env=_env(tmp_cfg))
        assert history.exit_code == 0, history.output
        assert registration_id in history.output
        assert "delivered" in history.output
        history_json = runner.invoke(
            cli,
            ["terminal", "history", "--limit", "1", "--json"],
            env=_env(tmp_cfg),
        )
        assert history_json.exit_code == 0, history_json.output
        rows = json.loads(history_json.output)
        assert len(rows) == 1
        assert rows[0]["target_id"] == registration_id
        assert rows[0]["status"] == "delivered"
        assert rows[0]["message_id"] == "cli-enter-2"
    finally:
        os.close(slave_fd)
        os.close(master_fd)
