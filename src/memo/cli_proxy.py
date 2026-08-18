"""`memo proxy` — the context-compression proxy.

Point Claude Code at it with ANTHROPIC_BASE_URL. Put that variable in the `env`
block of ~/.claude/settings.json, not a shell export: the background-agent
supervisor inherits only the environment of whichever shell cold-started it, so
an export reaches background sessions unpredictably.
"""

from __future__ import annotations

import socket

import click

from memo.cli_common import console


@click.group(name="proxy")
def proxy_group() -> None:
    """Context-compression proxy commands."""


@proxy_group.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8768, show_default=True, type=int)
@click.option("--upstream", default="https://api.anthropic.com", show_default=True)
def proxy_serve(host: str, port: int, upstream: str) -> None:
    """Run the proxy in the foreground."""
    try:
        import uvicorn

        from memo.proxy.server import build_app
    except ImportError as exc:  # missing [http] extra — a clean CLI error
        raise click.ClickException(
            "memo proxy needs the [http] extra: pip install 'mlx-memo[http]'"
        ) from exc
    uvicorn.run(build_app(upstream), host=host, port=port, log_level="warning")


@proxy_group.command("off")
def proxy_off() -> None:
    """Turn payload rewriting off everywhere (daemon included)."""
    from memo.config_md import set_value

    set_value("proxy.enabled", "false")
    console.print("proxy.enabled = false  (rewriting off; still forwards and measures)")


@proxy_group.command("on")
def proxy_on() -> None:
    """Turn payload rewriting back on."""
    from memo.config_md import set_value

    set_value("proxy.enabled", "true")
    console.print("proxy.enabled = true")


@proxy_group.command("status")
@click.option("--port", default=8768, show_default=True, type=int)
def proxy_status(port: int) -> None:
    """Report whether the proxy is listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        listening = sock.connect_ex(("127.0.0.1", port)) == 0
    console.print(
        f"proxy on 127.0.0.1:{port}: {'listening' if listening else 'not running'}"
    )