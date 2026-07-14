from __future__ import annotations

# These tests intentionally exercise rejection/acknowledgement of wildcard binds.
# ruff: noqa: S104
import stat

import pytest
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

_TOKEN = "test-token-" + ("x" * 32)


def test_http_auth_creates_private_token_and_rejects_missing_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MEMO_HTTP_API_TOKEN", raising=False)

    from memo.http_auth import HttpAuthRejected, load_http_auth_config, verify_http_auth

    cfg = load_http_auth_config(host="127.0.0.1")

    assert cfg.token is not None
    token_path = tmp_path / "state" / "http-api-token"
    assert token_path.read_text(encoding="utf-8").strip() == cfg.token
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    with pytest.raises(HttpAuthRejected) as exc_info:
        verify_http_auth(None, cfg)
    assert exc_info.value.status_code == 401


def test_http_auth_accepts_only_matching_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)

    from memo.http_auth import HttpAuthRejected, load_http_auth_config, verify_http_auth

    cfg = load_http_auth_config(host="localhost")
    verify_http_auth(f"Bearer {_TOKEN}", cfg)
    verify_http_auth(f"bearer {_TOKEN}", cfg)

    for authorization in ("Basic Zm9vOmJhcg==", "Bearer wrong", "Bearer"):
        with pytest.raises(HttpAuthRejected):
            verify_http_auth(authorization, cfg)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "memo.local"])
def test_http_bind_rejects_non_loopback_without_explicit_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    host: str,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)

    from memo.http_auth import HttpApiAuthError, load_http_auth_config, validate_http_bind

    cfg = load_http_auth_config(host=host)
    with pytest.raises(HttpApiAuthError, match="non-loopback"):
        validate_http_bind(host, cfg, allow_non_loopback=False)


def test_http_bind_never_allows_unauthenticated_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))

    from memo.http_auth import HttpApiAuthError, load_http_auth_config, validate_http_bind

    cfg = load_http_auth_config(host="0.0.0.0", allow_no_auth=True)
    with pytest.raises(HttpApiAuthError, match="cannot run without authentication"):
        validate_http_bind("0.0.0.0", cfg, allow_non_loopback=True)


def test_mcp_http_auth_protects_protocol_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)

    from memo.http_auth import build_mcp_auth, load_http_auth_config

    cfg = load_http_auth_config(host="127.0.0.1")
    server = FastMCP("auth-test", auth=build_mcp_auth(cfg))

    @server.tool()
    def ping() -> str:
        return "pong"

    app = server.http_app(stateless_http=True)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    accept = {"Accept": "application/json, text/event-stream"}

    with TestClient(app) as client:
        assert client.post("/mcp", json=payload, headers=accept).status_code == 401
        assert (
            client.post(
                "/mcp",
                json=payload,
                headers={**accept, "Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/mcp",
                json=payload,
                headers={**accept, "Authorization": f"Bearer {_TOKEN}"},
            ).status_code
            == 200
        )


def test_shared_http_middleware_rate_limits_and_hardens_responses() -> None:
    from memo.http_auth import RateLimitMiddleware, SecurityHeadersMiddleware

    async def ok(_request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", ok)])
    app.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=60)
    app.add_middleware(SecurityHeadersMiddleware)

    with TestClient(app) as client:
        first = client.get("/")
        assert client.get("/").status_code == 200
        limited = client.get("/")

    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert first.headers["x-frame-options"] == "DENY"


def test_memo_mcp_server_attaches_shared_auth_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mock_memory,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)

    from memo.http_auth import build_mcp_auth, load_http_auth_config
    from memo.server import build_server

    auth = build_mcp_auth(load_http_auth_config(host="127.0.0.1"))
    server = build_server(memory=mock_memory, auth=auth)

    assert server.auth is auth


def test_mcp_main_rejects_unacknowledged_network_bind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import memo.flags
    import memo.server as server_module

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    monkeypatch.setattr(
        memo.flags,
        "flag_str",
        lambda name: {"MEMO_MCP_TRANSPORT": "http", "MEMO_MCP_HOST": "0.0.0.0"}.get(name),
    )
    monkeypatch.setattr(memo.flags, "flag_bool", lambda _name: False)
    monkeypatch.setattr(memo.flags, "flag_int", lambda _name: 18768)
    monkeypatch.setattr(server_module, "_start_background_tasks", lambda: ())
    monkeypatch.setattr(
        server_module,
        "build_server",
        lambda **_kwargs: pytest.fail("server built before bind validation"),
    )

    from memo.http_auth import HttpApiAuthError

    with pytest.raises(HttpApiAuthError, match="non-loopback"):
        server_module.main()


def test_mcp_main_runs_http_with_shared_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import memo.flags
    import memo.server as server_module

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    monkeypatch.setattr(
        memo.flags,
        "flag_str",
        lambda name: {
            "MEMO_MCP_TRANSPORT": "http",
            "MEMO_MCP_HOST": "127.0.0.1",
        }.get(name),
    )
    monkeypatch.setattr(memo.flags, "flag_bool", lambda _name: False)
    monkeypatch.setattr(memo.flags, "flag_int", lambda _name: 18768)
    monkeypatch.setattr(server_module, "_start_background_tasks", lambda: ())
    monkeypatch.setattr(server_module, "_ensure_idle_daemon", lambda: None)
    captured = {}

    class _Server:
        def run(self, **kwargs) -> None:
            captured["run"] = kwargs

    def _build_server(**kwargs):
        captured["auth"] = kwargs["auth"]
        return _Server()

    monkeypatch.setattr(server_module, "build_server", _build_server)

    server_module.main()

    assert captured["auth"] is not None
    run_kwargs = dict(captured["run"])
    middleware = run_kwargs.pop("middleware")
    assert run_kwargs == {
        "transport": "http",
        "host": "127.0.0.1",
        "port": 18768,
    }
    assert len(middleware) == 2
