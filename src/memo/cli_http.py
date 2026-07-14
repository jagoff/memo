"""`memo http-api` command — start the HTTP REST API server."""

from __future__ import annotations

import logging
import os

import click


@click.command(name="http-api")
@click.option(
    "--port",
    default=8080,
    type=int,
    help="Port to listen on (default: 8080)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--allow-no-auth",
    is_flag=True,
    help="Disable bearer auth for loopback-only development.",
)
@click.option(
    "--allow-non-loopback",
    is_flag=True,
    help="Acknowledge authenticated exposure beyond this machine.",
)
def http_api(
    port: int,
    host: str,
    reload: bool,
    allow_no_auth: bool,
    allow_non_loopback: bool,
) -> None:
    """Start the HTTP REST API server for external clients.

    Run this in a terminal to expose memo operations as HTTP endpoints.
    The API mirrors the MCP tools but returns plain JSON.

    Example:
        memo http-api --port 8080
    """
    from memo.http_auth import HttpApiAuthError, load_http_auth_config, validate_http_bind

    try:
        auth_cfg = load_http_auth_config(host=host, allow_no_auth=allow_no_auth)
        validate_http_bind(
            host,
            auth_cfg,
            allow_non_loopback=allow_non_loopback,
        )
    except HttpApiAuthError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        from memo.server_http import configure_auth, run_server
    except ImportError as exc:
        raise click.ClickException(
            "The HTTP API requires fastapi and uvicorn.\n"
            "Install with: pip install 'mlx-memo[http]'\n"
            "  or: uv tool install 'mlx-memo[http]'"
        ) from exc

    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(__name__)
    _log.info("Starting memo HTTP API on %s:%d", host, port)

    if reload:
        import uvicorn

        # The reload worker imports memo.server_http in a child process, so
        # carry the already-validated startup policy through its environment.
        os.environ["MEMO_HTTP_HOST"] = host
        os.environ["MEMO_HTTP_ALLOW_NO_AUTH"] = "1" if allow_no_auth else "0"
        configure_auth(host=host, allow_no_auth=allow_no_auth)
        uvicorn.run(
            "memo.server_http:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    else:
        run_server(
            port=port,
            host=host,
            allow_no_auth=allow_no_auth,
            allow_non_loopback=allow_non_loopback,
        )
