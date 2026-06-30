"""`memo http-api` command — start the HTTP REST API server."""

from __future__ import annotations

import logging

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
def http_api(port: int, host: str, reload: bool) -> None:
    """Start the HTTP REST API server for external clients.

    Run this in a terminal to expose memo operations as HTTP endpoints.
    The API mirrors the MCP tools but returns plain JSON.

    Example:
        memo http-api --port 8080
    """
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(__name__)
    _log.info("Starting memo HTTP API on %s:%d", host, port)

    from memo.server_http import run_server

    if reload:
        import uvicorn

        uvicorn.run(
            "memo.server_http:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    else:
        run_server(port=port, host=host)