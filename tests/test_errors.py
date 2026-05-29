"""Tests for the memo domain error hierarchy (memo.errors)."""

from __future__ import annotations

import pytest

from memo import errors


@pytest.mark.parametrize("cls", [
    errors.NotFoundError,
    errors.ValidationError,
    errors.StorageError,
    errors.FederationError,
    errors.AmbiguousIdError,
    errors.WriteRefused,
])
def test_all_errors_subclass_memoerror(cls: type) -> None:
    assert issubclass(cls, errors.MemoError)


def test_builtin_bases_preserved() -> None:
    # legacy `except ValueError` / `except KeyError` handlers keep catching
    assert issubclass(errors.ValidationError, ValueError)
    assert issubclass(errors.AmbiguousIdError, ValueError)
    assert issubclass(errors.NotFoundError, KeyError)
    assert issubclass(errors.StorageError, RuntimeError)
    assert issubclass(errors.FederationError, RuntimeError)
    assert issubclass(errors.WriteRefused, RuntimeError)


def test_memoerror_does_not_shadow_builtin_memoryerror() -> None:
    assert errors.MemoError is not MemoryError
    assert not issubclass(errors.MemoError, MemoryError)


def test_memory_reexports_match() -> None:
    # back-compat: the names importable from memo.memory are the same objects
    from memo import memory
    assert memory.MemoError is errors.MemoError
    assert memory.AmbiguousIdError is errors.AmbiguousIdError
    assert memory.WriteRefused is errors.WriteRefused


def test_ambiguous_id_error_carries_matches() -> None:
    exc = errors.AmbiguousIdError("ab", ["abc123", "abd456"])
    assert exc.prefix == "ab"
    assert exc.matches == ["abc123", "abd456"]


def test_write_refused_carries_conflict() -> None:
    exc = errors.WriteRefused({"conflict_id": "c1", "summary": "overlap"})
    assert exc.conflict["conflict_id"] == "c1"
    assert "c1" in str(exc)
