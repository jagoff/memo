"""Launchd install/uninstall/status for memo-owned agents (chat service)."""

from __future__ import annotations

import subprocess
from pathlib import Path

CHAT_LABEL = "com.memo.chat"


def render_chat_plist(
    memo_bin: str, home: str, *, port: int = 8765, dist: str | None = None
) -> str:
    args = [memo_bin, "chat", "serve", "--host", "127.0.0.1", "--port", str(port)]
    if dist:
        args += ["--dist", dist]
    args_xml = "\n".join(f"      <string>{a}</string>" for a in args)
    log = f"{home}/Library/Logs/memo/chat.log"
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
      <string>{home}/.local/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
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
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(path)], capture_output=True, check=True
    )
    return path


def uninstall_chat(home: Path) -> bool:
    path = _plist_path(home)
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False
    )
    if path.exists():
        path.unlink()
        return True
    return False
