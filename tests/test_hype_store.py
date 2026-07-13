"""Tests for the HyPE question-space index (flags + HypeStore)."""
from __future__ import annotations


def test_hype_flags_registered_defaults():
    from memo.flags import REGISTRY

    assert REGISTRY["MEMO_HYPE_ENABLED"].default is False
    assert REGISTRY["MEMO_DREAM_HYPE_ENABLED"].default is False
    assert REGISTRY["MEMO_HYPE_QUESTIONS_PER_MEMORY"].default == 3
    assert REGISTRY["MEMO_HYPE_NIGHT_CAP"].default == 400
    assert REGISTRY["MEMO_HYPE_FOLD_POOL"].default == 30
