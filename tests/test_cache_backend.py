"""Tests for Memo's dependency-free cache archive."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memo import cache_backend as cb


def _record(**overrides):
    values = {
        "id": "abc-123",
        "title": "Native archive",
        "type": "decision",
        "body": "Memo owns this durable cache record",
        "tags": ("memo", "native"),
        "created": "2026-07-23T00:00:00Z",
        "updated": "2026-07-23T01:00:00Z",
        "extra": {"trust_tier": "agent_verified"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_null_backend_is_inert() -> None:
    backend = cb.NullBackend()
    assert backend.push(object()) is False
    assert backend.fetch("anything") == []


@pytest.mark.parametrize("name", ["none", "bogus", "", "NONE"])
def test_make_backend_falls_back_to_null(name: str, tmp_path) -> None:
    assert isinstance(cb.make_backend(name, root=tmp_path), cb.NullBackend)


@pytest.mark.parametrize("name", ["vault", "native"])
def test_make_backend_builds_native_archive(name: str, tmp_path) -> None:
    assert isinstance(cb.make_backend(name, root=tmp_path), cb.NativeVaultBackend)


def test_native_archive_round_trip(tmp_path) -> None:
    backend = cb.NativeVaultBackend(tmp_path)
    assert backend.push(_record()) is True

    hits = backend.fetch("durable memo", limit=5)

    assert len(hits) == 1
    assert hits[0]["id"] == "abc-123"
    assert hits[0]["schema"] == "memo.cache_archive.v1"
    assert hits[0]["from_backend"] is True


def test_native_archive_rejects_empty_and_path_like_ids(tmp_path) -> None:
    backend = cb.NativeVaultBackend(tmp_path)
    assert backend.push(_record(id="", title="", body="")) is False
    assert backend.push(_record(id="../../outside")) is True
    assert not (tmp_path.parent / "outside.json").exists()
