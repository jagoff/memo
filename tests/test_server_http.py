from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _evict_server_http() -> None:
    """Drop `memo.server_http` from BOTH places Python caches it.

    `sys.modules.pop` alone is not an eviction: the parent package keeps its
    own attribute bound to the module object, so `monkeypatch.setattr` on a
    dotted string still resolves to the stale module while the code under test
    re-imports a fresh one. The two then disagree, the patch lands nowhere,
    and a test that meant to stub `run_server` starts a real server instead.
    """
    import memo

    sys.modules.pop("memo.server_http", None)
    if hasattr(memo, "server_http"):
        delattr(memo, "server_http")


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
    _evict_server_http()
    try:
        server_http = importlib.import_module("memo.server_http")

        assert server_http.app.version == __version__
        assert server_http.health()["version"] == __version__
    finally:
        # Evict the module built against the fake fastapi: leaving it cached
        # would hand any later importer a _FakeFastAPI-backed app.
        _evict_server_http()


def _load_server(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    _evict_server_http()
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


def test_rest_rejects_oversized_chunked_body(monkeypatch, tmp_path) -> None:
    server_http = _load_server(monkeypatch, tmp_path)
    auth = {"Authorization": f"Bearer {_TOKEN}"}

    def body_chunks():
        yield b"{"
        yield b"x" * server_http.MAX_REQUEST_BYTES

    with TestClient(server_http.app) as client:
        response = client.post(
            "/api/memory",
            headers={**auth, "Content-Type": "application/json"},
            content=body_chunks(),
        )

    assert response.status_code == 413


def test_rest_rejects_oversized_chunked_body_even_when_route_ignores_body(
    monkeypatch, tmp_path
) -> None:
    server_http = _load_server(monkeypatch, tmp_path)
    memory = MagicMock()
    monkeypatch.setattr(server_http, "_memory", memory)
    auth = {"Authorization": f"Bearer {_TOKEN}"}

    def body_chunks():
        yield b"x" * (server_http.MAX_REQUEST_BYTES // 2)
        yield b"x" * (server_http.MAX_REQUEST_BYTES // 2 + 1)

    with TestClient(server_http.app) as client:
        response = client.post("/api/backup", headers=auth, content=body_chunks())

    assert response.status_code == 413
    memory.backup.create_backup.assert_not_called()


def test_rest_no_auth_mode_installs_local_request_guard(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MEMO_HTTP_API_TOKEN", raising=False)
    _evict_server_http()
    server_http = importlib.import_module("memo.server_http")
    server_http.configure_auth(host="127.0.0.1", allow_no_auth=True)
    memory = MagicMock()
    monkeypatch.setattr(server_http, "_memory", memory)

    with TestClient(server_http.app, base_url="http://127.0.0.1:18769") as client:
        response = client.post(
            "/api/backup",
            headers={
                "Host": "attacker.example",
                "Origin": "https://attacker.example",
                "Content-Type": "text/plain",
            },
            content=b"{}",
        )

    assert response.status_code == 403
    memory.backup.create_backup.assert_not_called()


def test_rest_direct_import_no_auth_mode_installs_local_request_guard(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_ALLOW_NO_AUTH", "1")
    monkeypatch.setenv("MEMO_HTTP_HOST", "127.0.0.1")
    monkeypatch.delenv("MEMO_HTTP_API_TOKEN", raising=False)
    _evict_server_http()
    server_http = importlib.import_module("memo.server_http")
    memory = MagicMock()
    monkeypatch.setattr(server_http, "_memory", memory)

    with TestClient(server_http.app, base_url="http://127.0.0.1:18769") as client:
        response = client.post(
            "/api/backup",
            headers={
                "Host": "attacker.example",
                "Origin": "https://attacker.example",
                "Content-Type": "text/plain",
            },
            content=b"{}",
        )

    assert response.status_code == 403
    memory.backup.create_backup.assert_not_called()


def test_rest_delete_reports_404_missing_and_409_ambiguous(monkeypatch, tmp_path) -> None:
    from memo.errors import AmbiguousIdError

    server_http = _load_server(monkeypatch, tmp_path)
    memory = MagicMock()
    monkeypatch.setattr(server_http, "_memory", memory)
    auth = {"Authorization": f"Bearer {_TOKEN}"}

    with TestClient(server_http.app) as client:
        memory.delete.return_value = True
        deleted = client.delete("/api/memory/feedbeef", headers=auth)
        memory.delete.return_value = False
        missing = client.delete("/api/memory/deadbeef", headers=auth)
        memory.delete.side_effect = AmbiguousIdError("a1", ["a1b2c3d4", "a1ffeedd"])
        ambiguous = client.delete("/api/memory/a1", headers=auth)

    assert deleted.status_code == 200
    assert deleted.json() == {"id": "feedbeef", "status": "deleted"}
    assert missing.status_code == 404
    assert ambiguous.status_code == 409
    assert ambiguous.json()["detail"] == {
        "error": "ambiguous",
        "prefix": "a1",
        "matches": ["a1b2c3d4", "a1ffeedd"],
    }


def test_rest_get_reports_409_for_ambiguous_prefix(monkeypatch, tmp_path) -> None:
    from memo.errors import AmbiguousIdError

    server_http = _load_server(monkeypatch, tmp_path)
    memory = MagicMock()
    memory.get.side_effect = AmbiguousIdError("a1", ["a1b2c3d4", "a1ffeedd"])
    monkeypatch.setattr(server_http, "_memory", memory)

    with TestClient(server_http.app) as client:
        response = client.get("/api/memory/a1", headers={"Authorization": f"Bearer {_TOKEN}"})

    assert response.status_code == 409
    assert response.json()["detail"]["matches"] == ["a1b2c3d4", "a1ffeedd"]


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


def test_evicting_the_module_does_not_strand_the_parent_attribute():
    """`sys.modules` and `memo.server_http` must never disagree.

    Popping only `sys.modules["memo.server_http"]` leaves the attribute on the
    parent `memo` package bound to the evicted module object. Anything that
    then does `monkeypatch.setattr("memo.server_http.run_server", ...)`
    patches that STALE object, while the code under test re-imports a fresh
    module and gets the REAL `run_server` — which starts a live uvicorn server
    and hangs until pytest-timeout kills it 120s later.

    That is the contamination behind `test_cli_http`'s intermittent hang
    (3 failures / 23 passes since 2026-08-02), and it only reproduces when the
    two files run in the same session in the wrong order — which is why it
    looked like flakiness rather than a bug.
    """
    import memo

    before = importlib.import_module("memo.server_http")
    assert getattr(memo, "server_http", None) is before

    _evict_server_http()

    assert "memo.server_http" not in sys.modules
    assert getattr(memo, "server_http", None) is None, (
        "parent attribute still points at the evicted module"
    )

    after = importlib.import_module("memo.server_http")
    assert getattr(memo, "server_http", None) is after
