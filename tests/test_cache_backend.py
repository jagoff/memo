"""Tests for cache_backend — NullBackend, factory routing, meta coercion.

MemflowBackend's subprocess path is not exercised (external CLI); the factory
falls back to NullBackend when the binary/project-root is absent, which is what
matters for correctness here.
"""

from __future__ import annotations

import pytest

from memo import cache_backend as cb


def test_null_backend_is_inert() -> None:
    nb = cb.NullBackend()
    assert nb.push(object()) is False  # never claims a write succeeded
    assert nb.fetch("anything") == []
    assert nb.has_current("id", "hash") is False


def test_coerce_meta() -> None:
    assert cb._coerce_meta(None) == ""
    assert cb._coerce_meta(True) == "true"
    assert cb._coerce_meta(False) == "false"
    assert cb._coerce_meta("a\nb") == "a b"
    assert cb._coerce_meta(42) == "42"
    assert len(cb._coerce_meta("x" * 1000)) == 500


@pytest.mark.parametrize("name", ["none", "vault", "bogus", "", "NONE"])
def test_make_backend_falls_back_to_null(name: str) -> None:
    assert isinstance(cb.make_backend(name), cb.NullBackend)


def test_make_backend_memflow_unavailable_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    # No memflow binary → MemflowBackend.available is False → NullBackend.
    monkeypatch.setattr(cb, "_binary", lambda: None)
    assert isinstance(cb.make_backend("memflow"), cb.NullBackend)


def test_make_backend_memflow_available_is_memflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cb, "_binary", lambda: "/usr/bin/true")
    monkeypatch.setattr(cb, "_project_root", lambda: tmp_path)
    assert isinstance(cb.make_backend("memflow"), cb.MemflowBackend)
