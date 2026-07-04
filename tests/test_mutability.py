"""C4 mutability classes: regex classifier + volatile-vs-volatile downgrade
in the contradiction scanner."""

from __future__ import annotations

from memo.contradict import classify_mutability, downgrade_volatile_contradiction
from memo.temporal import Contradiction


def _contr(relationship="contradiction", confidence=0.95):
    return Contradiction(
        memory_id_a="a" * 32,
        memory_id_b="b" * 32,
        title_a="Puerto viejo",
        title_b="Puerto nuevo",
        date_a="2026-01-01",
        date_b="2026-06-01",
        relationship=relationship,
        rationale="ports differ",
        confidence=confidence,
    )


def test_classify_volatile_versions_ports_status():
    assert classify_mutability("El dashboard corre en el puerto 8765") == "volatile"
    assert classify_mutability("estamos en la versión 2.9.5 de memo") == "volatile"
    assert classify_mutability("el deploy está pending") == "volatile"
    assert classify_mutability("dashboard on :8765") == "volatile"


def test_classify_ephemeral_now_references():
    assert classify_mutability("hoy el build está roto") == "ephemeral"
    assert classify_mutability("this week we focus on the launch") == "ephemeral"


def test_classify_stable_default():
    assert classify_mutability("Fernando prefiere respuestas en español") == "stable"
    assert classify_mutability("") == "stable"


def test_classify_stable_spanish_common_words():
    # Review fix (C4): 'todo' (everything/all) and bare 'estado' (state, an
    # ordinary Spanish noun) are extremely common on this corpus — they must
    # NOT flip genuinely STABLE memories to volatile, or real contradictions
    # between them get silently downgraded to 'evolution'.
    assert classify_mutability("todo el equipo prefiere python") == "stable"
    assert classify_mutability("documentamos el estado de las decisiones en el vault") == "stable"
    # ...while the tightened status idiom still classifies real status prose:
    assert classify_mutability("el estado actual del deploy es pending") == "volatile"


def test_downgrade_volatile_pair_to_evolution():
    out = downgrade_volatile_contradiction(
        _contr(), "el dashboard usa el puerto 8080", "el dashboard usa el puerto 8765"
    )
    assert out.relationship == "evolution"
    assert "volatile-vs-volatile" in out.rationale


def test_no_downgrade_when_one_side_stable():
    out = downgrade_volatile_contradiction(
        _contr(), "el dashboard usa el puerto 8080", "preferimos python para scripts"
    )
    assert out.relationship == "contradiction"


def test_no_downgrade_for_non_contradiction():
    out = downgrade_volatile_contradiction(
        _contr(relationship="evolution"), "puerto 8080", "puerto 8765"
    )
    assert out.relationship == "evolution"
    assert "volatile-vs-volatile" not in out.rationale


def test_scan_corpus_downgrades_volatile_pair(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CONTRADICT_MUTABILITY", "1")
    mock_memory.save(content="El dashboard corre en el puerto 8080", title="Puerto A")
    mock_memory.save(content="El dashboard corre en el puerto 8765", title="Puerto B")

    def _fake_classify(self, r1, r2):
        return Contradiction(
            memory_id_a=r1.id,
            memory_id_b=r2.id,
            title_a=r1.title,
            title_b=r2.title,
            date_a=r1.updated,
            date_b=r2.updated,
            relationship="contradiction",
            rationale="ports differ",
            confidence=0.95,
        )

    monkeypatch.setattr("memo.temporal.TemporalAnalyzer._classify_pair", _fake_classify)
    res = mock_memory.contradict_scanner.scan_corpus(
        sim_floor=-1.0, confidence_threshold=0.9, max_pairs=10, min_days_apart=0
    )
    assert res.evolutions_found >= 1
    assert res.contradictions_found == 0
