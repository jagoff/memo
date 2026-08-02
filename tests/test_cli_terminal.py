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
