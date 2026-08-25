"""Launchd install/uninstall/status for memo-owned agents (chat service)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from xml.sax.saxutils import escape

from memo import proxy_wiring

PROXY_LABEL = "com.memo.proxy"

CHAT_LABEL = "com.memo.chat"

# MEMO_* vars that belong to the terminal running the install, never to a
# long-lived daemon (superset of `tui.config.catalog.RUNTIME_ONLY_ENV_NAMES`,
# duplicated here so plist rendering doesn't import the TUI). MEMO_AGENT_TTY is
# the sharp one: runtime/codex_notify.py writes escape sequences to that path,
# and by the next boot /dev/ttysNNN belongs to an unrelated session.
_SESSION_SCOPED_ENV = frozenset(
    {
        # A bearer token belongs in the 0600 token file, not in a plist that
        # lands 0644 under ~/Library/LaunchAgents. `http_auth` reads the file
        # when the env var is absent, so dropping it here loses nothing.
        "MEMO_HTTP_API_TOKEN",
        "MEMO_AGENT_TTY",
        "MEMO_CODEX_BADGE_SHOWN",
        "MEMO_NONINTERACTIVE",
        "MEMO_SESSION_ID",
        "MEMO_STARTUP_BANNER_SHOWN",
        "MEMO_TERMINAL_ID",
        "MEMO_TERMINAL_REGISTRATION_ATTEMPTED",
    }
)


def daemon_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The MEMO_* vars it is safe to freeze into a LaunchAgent plist."""
    src = os.environ if environ is None else environ
    return {
        k: v
        for k, v in sorted(src.items())
        if k.startswith("MEMO_") and k not in _SESSION_SCOPED_ENV
    }


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
    memo_env = daemon_env()
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


def render_proxy_plist(memo_bin: str, home: str, *, port: int = 8768) -> str:
    args = [memo_bin, "proxy", "serve", "--host", "127.0.0.1", "--port", str(port)]
    args_xml = "\n".join(f"      <string>{escape(a)}</string>" for a in args)
    log = escape(f"{home}/Library/Logs/memo/proxy.log")
    path_env = escape(f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    # MEMO_* only — never ANTHROPIC_API_KEY or any other credential.
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
    <string>{PROXY_LABEL}</string>
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
    # The plist carries the daemon's whole MEMO_* environment — not world-readable.
    path.chmod(0o600)
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


def _proxy_plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{PROXY_LABEL}.plist"


def _port_owner(port: int) -> str | None:
    """Describe the process listening on 127.0.0.1:port, or None if free.

    `lsof -F pc` prints one line per field (`p<pid>`, `c<comm>`), which keeps
    the parse independent of lsof's display-width column layout.
    """
    proc = subprocess.run(
        ["lsof", "-nP", "-iTCP", f"127.0.0.1:{port}", "-sTCP:LISTEN", "-F", "pc"],
        capture_output=True,
        text=True,
        check=False,
    )
    pid: str | None = None
    comm: str | None = None
    for field in proc.stdout.splitlines():
        if field.startswith("p") and field[1:].isdigit():
            pid = field[1:]
        elif field.startswith("c"):
            comm = field[1:]
    if pid is None:
        return None
    return f"{comm or 'unknown'} (pid {pid})"


def _label_loaded(label: str) -> bool:
    """True when `launchctl list` shows `label` (used to allow proxy reinstall)."""
    proc = subprocess.run(["launchctl", "list"], capture_output=True, text=True, check=False)
    return any(row["label"] == label for row in parse_launchctl_list(proc.stdout))


def install_proxy(memo_bin: str, home: Path, *, port: int = 8768) -> Path:
    """Install the proxy LaunchAgent. Refuses a port another process owns."""
    import click

    owner = _port_owner(port)
    if owner and not _label_loaded(PROXY_LABEL):
        raise click.ClickException(
            f"port {port} is already in use by {owner} — free the port or pick another with --port"
        )
    (home / "Library" / "Logs" / "memo").mkdir(parents=True, exist_ok=True)
    path = _proxy_plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_proxy_plist(memo_bin, str(home), port=port), encoding="utf-8")
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
    if not wait_until_listening(port):
        # The gate below already protects the CLIENT -- ANTHROPIC_BASE_URL is
        # never written at a dead port -- but silence protected nobody else:
        # `memo ops install proxy` exited 0 and install.sh announced "Claude
        # Code now routes through it" over a crashlooping agent. Say it out
        # loud; the caller decides what to do about it.
        raise RuntimeError(
            f"proxy agent bootstrapped but never answered on 127.0.0.1:{port} — "
            f"check ~/Library/Logs/memo/proxy.log (a missing [http] extra is the "
            f"usual cause: pip install 'mlx-memo[http]')"
        )
    proxy_wiring.wire(home / ".claude", port)
    return path


def wait_until_listening(port: int, timeout_s: float = 10.0, interval_s: float = 0.25) -> bool:
    """Block until something answers on 127.0.0.1:port, or give up.

    The gate in front of `proxy_wiring.wire`. ANTHROPIC_BASE_URL is a hard
    dependency -- pointed at a dead port, Claude Code fails like a dead
    network -- so the variable is only ever written once the listener is real.
    A bootstrap that succeeded but whose process then exited (a port stolen
    between the check and the bind, a broken runtime) therefore leaves the
    user's settings.json untouched instead of silently breaking their client.
    """
    import socket
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(interval_s)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
        except OSError:
            pass
        time.sleep(interval_s)
    return False


def uninstall_proxy(home: Path) -> bool:
    path = _proxy_plist_path(home)
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False
    )
    # Un-point the client BEFORE the agent is gone for good: a settings.json
    # still naming a port nothing listens on is the outage this whole module
    # is careful about.
    proxy_wiring.unwire(home / ".claude")
    if path.exists():
        path.unlink()
        return True
    return False
