"""Unmatched-term gate: weak + zero lexical overlap -> honest empty."""
from __future__ import annotations

from types import SimpleNamespace


def _hit(score: float, title: str = "Nota del reranker", body: str = "detalles del reranker de producción"):
    return SimpleNamespace(id="a" * 32, title=title, tags=["memo"], body=body, score=score)


def test_gate_fires_on_weak_no_overlap():
    from memo.recall_logic import unmatched_term_gate

    assert unmatched_term_gate("kubernetes ingress timeout", [_hit(0.42)]) is True


def test_gate_never_fires_on_strong_semantic_match():
    from memo.recall_logic import unmatched_term_gate

    # paraphrase recall: no lexical overlap but high cosine — must survive
    assert unmatched_term_gate("kubernetes ingress timeout", [_hit(0.80)]) is False


def test_gate_never_fires_when_any_term_matches():
    from memo.recall_logic import unmatched_term_gate

    assert unmatched_term_gate("bug del reranker", [_hit(0.42)]) is False


def test_gate_ignores_stopword_only_prompts_and_empty_hits():
    from memo.recall_logic import unmatched_term_gate

    assert unmatched_term_gate("what about this", [_hit(0.42)]) is False
    assert unmatched_term_gate("kubernetes ingress", []) is False
