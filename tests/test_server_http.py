from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memo import __version__

_TOKEN = "test-token-" + ("x" * 32)


class _FakeFastAPI:
    def __init__(self, **kwargs):
        self.version = kwargs["version"]

    def get(self, _path, **_kwargs):
        return lambda fn: fn

    def post(self, _path, **_kwargs):
        return lambda fn: fn

    def delete(self, _path, **_kwargs):
        return lambda fn: fn

    def add_middleware(self, *_args, **_kwargs):
        return None


class _FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


def test_http_api_uses_runtime_package_version(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "fastapi",
        SimpleNamespace(
            Depends=lambda dependency: dependency,
            FastAPI=_FakeFastAPI,
            Header=lambda **_kwargs: None,
            HTTPException=_FakeHTTPException,
            Query=lambda **_kwargs: None,
        ),
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


def _load_server(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    sys.modules.pop("memo.server_http", None)
    module = importlib.import_module("memo.server_http")
    module.configure_auth(host="127.0.0.1")
    return module


class _Store:
    def count(self) -> int:
        return 3

    def count_by_type(self) -> dict[str, int]:
        return {"decision": 1, "note": 2}


@dataclass
class _BackupMetadata:
    name: str = "safe-backup"
    checksum: str = "abc123"
    compressed_size: int = 123
    original_size: int = 456
    memory_count: int = 7


class _Backup:
    def create_backup(self, **_kwargs) -> _BackupMetadata:
        return _BackupMetadata()

    def list_backups(self) -> list[_BackupMetadata]:
        return [_BackupMetadata()]


class _Memory:
    store = _Store()
    backup = _Backup()


def test_rest_health_is_public_and_every_api_route_requires_auth(monkeypatch, tmp_path) -> None:
    server_http = _load_server(monkeypatch, tmp_path)
    monkeypatch.setattr(server_http, "_memory", _Memory())

    with TestClient(server_http.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/stats").status_code == 401
        assert (
            client.get("/api/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        response = client.get(
            "/api/stats",
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json() == {"total": 3, "by_type": {"decision": 1, "note": 2}}
    for route in server_http.app.routes:
        if getattr(route, "path", "").startswith("/api/"):
            assert route.dependencies, f"{route.path} is missing auth"


def test_rest_request_and_parameter_limits_are_enforced(monkeypatch, tmp_path) -> None:
    server_http = _load_server(monkeypatch, tmp_path)
    auth = {"Authorization": f"Bearer {_TOKEN}"}

    with TestClient(server_http.app) as client:
        assert client.get("/api/memory?limit=101", headers=auth).status_code == 422
        assert (
            client.post(
                "/api/search",
                headers=auth,
                json={"query": "x" * 4097, "limit": 5, "mode": "hybrid"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/memory",
                headers=auth,
                content=b"x" * (server_http.MAX_REQUEST_BYTES + 1),
            ).status_code
            == 413
        )


def test_rest_backup_uses_real_metadata_fields(monkeypatch, tmp_path) -> None:
    server_http = _load_server(monkeypatch, tmp_path)
    monkeypatch.setattr(server_http, "_memory", _Memory())
    auth = {"Authorization": f"Bearer {_TOKEN}"}

    with TestClient(server_http.app) as client:
        created = client.post("/api/backup", headers=auth).json()
        listed = client.get("/api/backup", headers=auth).json()

    assert created == {
        "name": "safe-backup",
        "size": 123,
        "original_size": 456,
        "memory_count": 7,
        "checksum": "abc123",
    }
    assert listed == {
        "backups": [
            {
                "name": "safe-backup",
                "size": 123,
                "original_size": 456,
                "memory_count": 7,
            }
        ]
    }
