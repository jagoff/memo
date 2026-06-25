from __future__ import annotations

import json
import selectors
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console


def _codex_home() -> Path:
    import os

    raw = os.environ.get("CODEX_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def _copy_slash_skill(root: Path, dst: Path, *, dry_run: bool) -> None:
    src = root / "skills" / "memo" / "SKILL.md"
    if dry_run:
        click.echo(f"copy {src} -> {dst}  # /memo")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    console.print(f"[green]✓[/green] copied {src} -> {dst}  [dim](/memo)[/dim]")


def _codex_send_app_server_request(
    proc: subprocess.Popen[str],
    *,
    request_id: int,
    method: str,
    params: dict[str, Any] | None,
) -> None:
    if proc.stdin is None:
        raise click.ClickException("Codex app-server stdin is unavailable.")
    payload: dict[str, Any] = {"id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def _codex_read_app_server_response(
    proc: subprocess.Popen[str],
    request_id: int,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    if proc.stdout is None:
        raise click.ClickException("Codex app-server stdout is unavailable.")

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_s
    seen: list[str] = []

    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not selector.select(remaining):
                break
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            seen.append(line)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg["error"]
                message = err.get("message", err) if isinstance(err, dict) else err
                raise click.ClickException(f"Codex app-server {request_id} failed: {message}")
            result = msg.get("result")
            return result if isinstance(result, dict) else {}
    finally:
        selector.close()

    preview = "\n".join(seen[-5:])
    raise click.ClickException(
        f"Timed out waiting for Codex app-server response id={request_id}."
        + (f"\nLast output:\n{preview}" if preview else "")
    )


def _install_codex_plugin(root: Path, *, dry_run: bool) -> None:
    from memo.runtime.mcp import _format_command

    marketplace = root / ".agents" / "plugins" / "marketplace.json"
    if not marketplace.is_file():
        raise click.ClickException(f"Codex marketplace manifest not found: {marketplace}")
    args = ["codex", "app-server", "--listen", "stdio://", "--enable", "plugins"]
    if dry_run:
        click.echo(f"$ {_format_command(args)}  # plugin/install memo from {marketplace}")
        return

    try:
        proc = subprocess.Popen(
            args,
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise click.ClickException("`codex` not found on PATH; install Codex first.") from exc

    try:
        _codex_send_app_server_request(
            proc,
            request_id=0,
            method="initialize",
            params={
                "clientInfo": {
                    "name": "memo-installer",
                    "title": "memo installer",
                    "version": "0.1",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        _codex_read_app_server_response(proc, 0)
        _codex_send_app_server_request(
            proc,
            request_id=1,
            method="plugin/install",
            params={
                "marketplacePath": str(marketplace),
                "pluginName": "memo",
            },
        )
        _codex_read_app_server_response(proc, 1)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    console.print("[green]✓[/green] codex plugin/install memo@memo")
