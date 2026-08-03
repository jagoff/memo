from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

from memo.cli_dashboard import _make_handler


class _Builder:
    calls = 0

    @classmethod
    def collect_data(cls, _cfg, *, include_projection: bool):
        cls.calls += 1
        return {"projection": include_projection}


def test_dashboard_rejects_dns_rebinding_host_before_reading_data() -> None:
    _Builder.calls = 0
    handler = _make_handler(
        _Builder,
        object(),
        "<html></html>",
        5,
        capability_token="test-token",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request(
            "GET",
            "/api/data.json",
            headers={"Host": "attacker.example", "Origin": "https://attacker.example"},
        )
        response = connection.getresponse()
        response.read()

        assert response.status == 403
        assert _Builder.calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_allows_same_origin_loopback_request() -> None:
    _Builder.calls = 0
    handler = _make_handler(
        _Builder,
        object(),
        "<html></html>",
        5,
        capability_token="test-token",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = f"127.0.0.1:{server.server_port}"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request(
            "GET",
            "/api/data.json?token=test-token",
            headers={"Host": host, "Origin": f"http://{host}"},
        )
        response = connection.getresponse()
        response.read()

        assert response.status == 200
        assert _Builder.calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_rejects_local_client_without_capability() -> None:
    _Builder.calls = 0
    handler = _make_handler(
        _Builder,
        object(),
        "<html></html>",
        5,
        capability_token="test-token",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = f"127.0.0.1:{server.server_port}"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/data.json", headers={"Host": host})
        response = connection.getresponse()
        response.read()
        assert response.status == 403
        assert _Builder.calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
