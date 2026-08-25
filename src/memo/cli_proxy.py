"""`memo proxy` — the context-compression proxy.

Point Claude Code at it with ANTHROPIC_BASE_URL. Put that variable in the `env`
block of ~/.claude/settings.json, not a shell export: the background-agent
supervisor inherits only the environment of whichever shell cold-started it, so
an export reaches background sessions unpredictably.
"""

from __future__ import annotations

import socket
from pathlib import Path

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

        # build_app() must be INSIDE the guard: it imports fastapi lazily, and
        # fastapi is the one [http] member a default `pip install mlx-memo`
        # actually lacks -- uvicorn and httpx arrive transitively via fastmcp.
        # Called outside, its ImportError escaped as a raw ModuleNotFoundError,
        # so the launchd agent crashlooped instead of printing this message.
        app = build_app(upstream)
    except ImportError as exc:  # missing [http] extra — a clean CLI error
        raise click.ClickException(
            "memo proxy needs the [http] extra: pip install 'mlx-memo[http]'"
        ) from exc
    uvicorn.run(app, host=host, port=port, log_level="warning")


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


@proxy_group.command("tool-savings")
@click.argument("payload_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--session",
    default="tool-savings-report",
    show_default=True,
    help="Session key to freeze the keep-set under (report-only; no real traffic).",
)
def proxy_tool_savings(payload_path: str, session: str) -> None:
    """Report what ToolSchemas would prune from a captured request body.

    PAYLOAD_PATH is a JSON file shaped like an Anthropic Messages API
    request (a `tools` array, optionally `system`/`messages`) — e.g. a
    request captured from real traffic. Runs the real pruning transform
    against it and prints tools kept/pruned and the schema token cost
    before/after. Makes no network calls.
    """
    import json

    from memo.config import Config
    from memo.mcp_budget import est_tokens
    from memo.proxy.plan import Context
    from memo.proxy.transforms.toolschemas import ToolSchemas
    from memo.proxy.zones import split

    try:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"could not read {payload_path} as JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{payload_path} is not a JSON object")

    zones = split(payload)
    total = len(zones.tools)
    before = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))

    cfg = Config.from_env()
    ctx = Context(state_dir=cfg.state_dir, session_key=session, project=None)
    ToolSchemas().apply(zones, ctx)

    kept = len(zones.tools)
    after = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))
    console.print(f"tools: {total} total, {kept} kept, {total - kept} pruned")
    console.print(f"schema tokens: {before} before -> {after} after (saved {before - after})")


@proxy_group.command("status")
@click.option("--port", default=None, type=int, help="Loopback port (default: MEMO_PROXY_PORT).")
def proxy_status(port: int | None) -> None:
    """Report whether the proxy is listening."""
    from memo.flags import flag_int

    port = port or int(flag_int("MEMO_PROXY_PORT") or 8768)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        listening = sock.connect_ex(("127.0.0.1", port)) == 0
    console.print(f"proxy on 127.0.0.1:{port}: {'listening' if listening else 'not running'}")
