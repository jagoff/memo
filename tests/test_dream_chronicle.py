"""Tests for the nightly chronicle dream pass."""
from __future__ import annotations


def test_chronicle_flags_registered_default_off():
    from memo.flags import REGISTRY

    for name in ("MEMO_DREAM_CHRONICLE_ENABLED", "MEMO_CHRONICLE_WEEKLY"):
        assert name in REGISTRY
        assert REGISTRY[name].default is False
