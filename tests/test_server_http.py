from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from memo import __version__


class _FakeFastAPI:
    def __init__(self, **kwargs):
        self.version = kwargs["version"]

    def get(self, _path):
        return lambda fn: fn

    def post(self, _path):
        return lambda fn: fn

    def delete(self, _path):
        return lambda fn: fn


class _FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


def test_http_api_uses_runtime_package_version(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "fastapi",
        SimpleNamespace(FastAPI=_FakeFastAPI, HTTPException=_FakeHTTPException),
    )
    sys.modules.pop("memo.server_http", None)
    try:
        server_http = importlib.import_module("memo.server_http")

        assert server_http.app.version == __version__
        assert server_http.health()["version"] == __version__
    finally:
        # Evict the module built against the fake fastapi: leaving it cached
        # would hand any later importer a _FakeFastAPI-backed app.
        sys.modules.pop("memo.server_http", None)
