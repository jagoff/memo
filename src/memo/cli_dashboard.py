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
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import click

from memo.config import Config
from memo.flags import flag_int
from memo.html_security import content_security_policy, new_csp_nonce


def _load_builder() -> Any:
    """Import the builder from the installed ``memo`` wheel."""
    from memo import web_build

    return web_build


def _make_handler(
    builder: Any,
    cfg: Config,
    shell_html: str,
    interval: int,
    *,
    csp: str | None = None,
    capability_token: str,
) -> type:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # silence per-request logging
            pass

        def _request_is_local(self) -> bool:
            """Reject DNS rebinding and cross-origin reads of the local dashboard."""
            host_header = self.headers.get("Host", "")
            try:
                host = urlsplit(f"//{host_header}")
                hostname = (host.hostname or "").lower().rstrip(".")
                port = host.port
            except ValueError:
                return False
            if host.username is not None or host.password is not None:
                return False
            if hostname not in {"127.0.0.1", "localhost", "::1"}:
                return False

            server_port = int(cast(ThreadingHTTPServer, self.server).server_port)
            if (port if port is not None else 80) != server_port:
                return False
            if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                return False

            origin_header = self.headers.get("Origin")
            if origin_header:
                try:
                    origin = urlsplit(origin_header)
                    origin_hostname = (origin.hostname or "").lower().rstrip(".")
                    origin_port = origin.port
                except ValueError:
                    return False
                if (
                    origin.scheme != "http"
                    or origin.username is not None
                    or origin.password is not None
                    or origin_hostname != hostname
                    or (origin_port if origin_port is not None else 80) != server_port
                ):
                    return False
            return True

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("x-frame-options", "DENY")
            self.send_header("cross-origin-resource-policy", "same-origin")
            self.send_header("cross-origin-opener-policy", "same-origin")
            if csp:
                self.send_header("content-security-policy", csp)
            self.end_headers()
            self.wfile.write(body)

        def _has_capability(self) -> bool:
            supplied = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if supplied and secrets.compare_digest(supplied, capability_token):
                return True
            cookie = self.headers.get("Cookie", "")
            for item in cookie.split(";"):
                name, separator, value = item.strip().partition("=")
                if (
                    separator
                    and name == "memo_dashboard"
                    and secrets.compare_digest(value, capability_token)
                ):
                    return True
            return False

        def do_GET(self) -> None:
            if not self._request_is_local():
                self._send(403, b"forbidden", "text/plain; charset=utf-8")
                return
            if not self._has_capability():
                self._send(403, b"forbidden", "text/plain; charset=utf-8")
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = shell_html.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.send_header("cache-control", "no-store")
                self.send_header("x-content-type-options", "nosniff")
                self.send_header("referrer-policy", "no-referrer")
                self.send_header("x-frame-options", "DENY")
                self.send_header("cross-origin-resource-policy", "same-origin")
                self.send_header("cross-origin-opener-policy", "same-origin")
                self.send_header(
                    "set-cookie",
                    f"memo_dashboard={capability_token}; HttpOnly; SameSite=Strict; Path=/",
                )
                if csp:
                    self.send_header("content-security-policy", csp)
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/data.json":
                try:
                    data = builder.collect_data(cfg, include_projection=False)
                    data["refresh_interval_s"] = interval
                    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                except Exception:  # keep the daemon alive on a bad poll
                    body = json.dumps({"error": "dashboard data unavailable"}).encode("utf-8")
                    self._send(500, body, "application/json; charset=utf-8")
                    return
                self._send(200, body, "application/json; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return _Handler


def _spawn_background(port: int, interval: int, log_path: Path, token: str) -> int:
    """Re-exec this command detached so the server outlives the terminal.

    Returns the child PID. Keeps the dashboard up across shells (the recurring
    "web server must start manually" pain), without a launchd plist."""
    import subprocess
    import tempfile

    log_path.parent.mkdir(parents=True, exist_ok=True)
    token_descriptor, token_name = tempfile.mkstemp(
        dir=log_path.parent,
        prefix=".dashboard-capability-",
    )
    with open(token_descriptor, "w", encoding="utf-8", closefd=True) as token_file:
        token_file.write(token)
        token_file.flush()
    Path(token_name).chmod(0o600)
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
            "--token-file",
            token_name,
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
        Path(token_name).unlink(missing_ok=True)
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
@click.option("--token", hidden=True, default=None)
@click.option("--token-file", hidden=True, type=click.Path(path_type=Path), default=None)
def dashboard_cmd(
    port: int | None,
    interval: int,
    open_browser: bool,
    background: bool,
    foreground_only: bool,
    token: str | None,
    token_file: Path | None,
) -> None:
    """Serve the health dashboard on localhost with live auto-refresh."""
    builder = _load_builder()
    cfg = Config.from_env()
    _flag_port = flag_int("MEMO_DASHBOARD_PORT")
    resolved_port = port if port is not None else (8787 if _flag_port is None else _flag_port)
    interval = max(1, interval)
    if token_file is not None:
        try:
            if token_file.is_symlink():
                raise OSError("symlink")
            token = token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise click.ClickException("could not read dashboard capability") from exc
        finally:
            token_file.unlink(missing_ok=True)
    capability_token = token or secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{resolved_port}/?token={capability_token}"

    if background and not foreground_only:
        log_path = cfg.state_dir / "dashboard.log"
        pid = _spawn_background(resolved_port, interval, log_path, capability_token)
        click.echo(f"memo dashboard (background, pid {pid}) → {url}")
        click.echo(f"  log:  {log_path}")
        click.echo(f"  stop: kill {pid}")
        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        return

    click.echo("Building dashboard (initial snapshot + 3-D projection)…")
    data = builder.collect_data(cfg, include_projection=True)
    data["refresh_interval_s"] = interval
    nonce = new_csp_nonce()
    shell_html = builder._render_html(data, nonce=nonce)

    handler = _make_handler(
        builder,
        cfg,
        shell_html,
        interval,
        csp=content_security_policy(nonce, allow_local_fetch=True),
        capability_token=capability_token,
    )
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
