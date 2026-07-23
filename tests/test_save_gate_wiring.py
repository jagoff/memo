"""gate-presets wired into save()'s near-duplicate gate (default off)."""

from __future__ import annotations

import pytest


def _seed_and_dup(mock_memory, type_, *, exact: bool = True):
    """Save an original, then attempt a near-identical second save of the same type.

    `mock_memory`'s stub embedder hashes the exact composed text, and the
    near-dup gate's query text (title + truncated body) differs slightly from
    the stored text (title + full body), so two hash-derived vectors land at
    an effectively random cosine rather than a smooth near-1.0 similarity.
    Pin a constant vector here (same trick as test_dedup_scope.py's
    `mem_const_embed`) so the second save is a guaranteed near-duplicate.
    """
    const = [1.0] + [0.0] * (mock_memory.cfg.embedder_dims - 1)
    mock_memory.embedder.embed = lambda inputs: [const for _ in inputs]
    mock_memory.embedder.embed_query = lambda query: const
    body = "El dashboard de synapse corre en el puerto 8765 y sirve el chat federado."
    mock_memory.save(content=body, title="dashboard port", type_=type_)
    duplicate_body = body if exact else f"{body} Confirmado por una segunda fuente."
    return lambda: mock_memory.save(
        content=duplicate_body,
        title="dashboard port",
        type_=type_,
    )


def test_strict_decision_near_dup_refused(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "strict"}')
    monkeypatch.setenv("MEMO_SAVE_DEDUP_CHECK", "1")
    dup = _seed_and_dup(mock_memory, "decision", exact=False)
    with pytest.raises(ValueError, match="near-duplicate"):
        dup()


def test_strict_decision_exact_repeat_corroborates(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "strict"}')
    monkeypatch.setenv("MEMO_SAVE_DEDUP_CHECK", "1")
    dup = _seed_and_dup(mock_memory, "decision")

    rec = dup()

    assert rec.action == "corroborated"


def test_note_near_dup_admitted_under_same_preset(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "strict"}')
    monkeypatch.setenv("MEMO_SAVE_DEDUP_CHECK", "1")
    dup = _seed_and_dup(mock_memory, "note")
    rec = dup()  # note is 'balanced' -> warn+admit
    assert rec is not None


def test_flag_off_admits_near_dup(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_SAVE_GATE_PRESETS", raising=False)
    monkeypatch.setenv("MEMO_SAVE_DEDUP_CHECK", "1")
    dup = _seed_and_dup(mock_memory, "decision")
    rec = dup()
    assert rec is not None
