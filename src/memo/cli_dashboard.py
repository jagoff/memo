"""`memo dashboard` — serve the health dashboard on localhost with live refresh.

The static `web/health.html` is a build-time snapshot opened over ``file://``.
This command serves the same dashboard over ``http://127.0.0.1:<port>`` and adds
a tiny ``/api/data.json`` endpoint the page polls, so the metrics refresh in
place without re-running the build. Bind is **localhost-only** — never exposed
to the network.

The expensive 3-D projection (read every vector + PCA) is computed once for the
initial page; each poll recomputes only the cheap operational metrics.
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import click

from memo.config import Config
from memo.errors import MemoError
from memo.flags import flag_int

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_DIR = _REPO_ROOT / "web"


def _load_builder() -> Any:
    """Import the dashboard builder from ``web/build.py`` (kept out of the
    package so the static build stays a standalone script)."""
    if str(_WEB_DIR) not in sys.path:
        sys.path.insert(0, str(_WEB_DIR))
    try:
        import build  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only on broken installs
        raise MemoError(
            f"dashboard builder not found at {_WEB_DIR / 'build.py'} "
            "(expected in a source/editable install)"
        ) from exc
    return build


def _make_handler(builder: Any, cfg: Config, shell_html: str, interval: int) -> type:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # silence per-request logging
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, shell_html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/data.json":
                try:
                    data = builder.collect_data(cfg, include_projection=False)
                    data["refresh_interval_s"] = interval
                    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                except Exception as exc:  # keep the daemon alive on a bad poll
                    body = json.dumps({"error": str(exc)}).encode("utf-8")
                    self._send(500, body, "application/json; charset=utf-8")
                    return
                self._send(200, body, "application/json; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return _Handler


def _spawn_background(port: int, interval: int, log_path: Path) -> int:
    """Re-exec this command detached so the server outlives the terminal.

    Returns the child PID. Keeps the dashboard up across shells (the recurring
    "web server must start manually" pain), without a launchd plist."""
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child, closed on exit
    try:
        args = [
            sys.argv[0],
            "dashboard",
            "--no-open",
            "--foreground-only",
            "--port",
            str(port),
            "--interval",
            str(interval),
        ]
        proc = subprocess.Popen(
            args,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from the controlling terminal
        )
    except Exception:
        log.close()
        raise
    return proc.pid


@click.command("dashboard")
@click.option(
    "--port", type=int, default=None, help="Bind port (default MEMO_DASHBOARD_PORT or 8787)."
)
@click.option(
    "--interval", type=int, default=5, show_default=True, help="Live-refresh interval (seconds)."
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the dashboard in a browser.",
)
@click.option(
    "--background",
    "-b",
    is_flag=True,
    help="Run detached so it stays up after you close the terminal.",
)
@click.option(
    "--foreground-only",
    is_flag=True,
    hidden=True,
    help="Internal: serve without re-spawning (used by --background).",
)
def dashboard_cmd(
    port: int | None, interval: int, open_browser: bool, background: bool, foreground_only: bool
) -> None:
    """Serve the health dashboard on localhost with live auto-refresh."""
    builder = _load_builder()
    cfg = Config.from_env()
    _flag_port = flag_int("MEMO_DASHBOARD_PORT")
    resolved_port = port if port is not None else (8787 if _flag_port is None else _flag_port)
    interval = max(1, interval)
    url = f"http://127.0.0.1:{resolved_port}"

    if background and not foreground_only:
        log_path = cfg.state_dir / "dashboard.log"
        pid = _spawn_background(resolved_port, interval, log_path)
        click.echo(f"memo dashboard (background, pid {pid}) → {url}")
        click.echo(f"  log:  {log_path}")
        click.echo(f"  stop: kill {pid}")
        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        return

    click.echo("Building dashboard (initial snapshot + 3-D projection)…")
    data = builder.collect_data(cfg, include_projection=True)
    data["refresh_interval_s"] = interval
    shell_html = builder._render_html(data)

    handler = _make_handler(builder, cfg, shell_html, interval)
    server = ThreadingHTTPServer(("127.0.0.1", resolved_port), handler)
    click.echo(f"memo dashboard → {url}  (refresh {interval}s · Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nstopping dashboard…")
    finally:
        server.shutdown()
        server.server_close()
