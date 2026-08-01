"""Launchd install/uninstall/status for memo-owned agents (chat service)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

CHAT_LABEL = "com.memo.chat"


def render_chat_plist(
    memo_bin: str, home: str, *, port: int = 8765, dist: str | None = None
) -> str:
    args = [memo_bin, "chat", "serve", "--host", "127.0.0.1", "--port", str(port)]
    if dist:
        args += ["--dist", dist]
    args_xml = "\n".join(f"      <string>{escape(a)}</string>" for a in args)
    log = escape(f"{home}/Library/Logs/memo/chat.log")
    path_env = escape(f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    # launchd agents don't inherit the shell env — forward MEMO_* vars from the
    # installing shell so the daemon uses the same embedder/vault/state config
    # as `memo` in a terminal (mirrors watcher.py's render_plist).
    memo_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MEMO_")}
    memo_env_xml = "".join(
        f"      <key>{escape(k)}</key>\n      <string>{escape(v)}</string>\n"
        for k, v in memo_env.items()
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>{CHAT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>{path_env}</string>
{memo_env_xml}    </dict>
  </dict>
</plist>
"""


def parse_launchctl_list(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2].startswith("com.memo."):
            continue
        pid_raw, exit_raw, label = parts
        rows.append(
            {
                "label": label.strip(),
                "pid": int(pid_raw) if pid_raw.strip().isdigit() else None,
                "last_exit": int(exit_raw) if exit_raw.strip().lstrip("-").isdigit() else 0,
            }
        )
    return rows


def _plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{CHAT_LABEL}.plist"


def install_chat(memo_bin: str, home: Path, *, port: int = 8765, dist: str | None = None) -> Path:
    (home / "Library" / "Logs" / "memo").mkdir(parents=True, exist_ok=True)
    path = _plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_chat_plist(memo_bin, str(home), port=port, dist=dist), encoding="utf-8")
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False
    )
    try:
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"launchctl bootstrap failed: {stderr or exc}") from exc
    return path


def uninstall_chat(home: Path) -> bool:
    path = _plist_path(home)
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False
    )
    if path.exists():
        path.unlink()
        return True
    return False
