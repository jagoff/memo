"""Tests for secret storage Memory integration."""

import pytest

# Memory mixin tests (placeholder)
# Full tests require working Memory + conftest fixtures
# Smoke test: verify Memory has secret methods
def test_memory_has_secret_methods():
    """Memory should have secret operation methods."""
    from memo.memory import Memory

    assert hasattr(Memory, "save_secret")
    assert hasattr(Memory, "get_secret")
    assert hasattr(Memory, "list_secrets")
    assert hasattr(Memory, "forget_secret")
