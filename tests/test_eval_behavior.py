"""Behavior-eval harness: schema validation, gate scoring, error surfacing.

The pure layers are tested without MLX. The end-to-end path (real store, real
`memo recall-hook` subprocess, real model) is one `requires_mlx`/`slow` test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo import eval_behavior as eb
from memo.eval_behavior import (
    Gate,
    GateResult,
    Scenario,
    SeedMemory,
    load_scenarios,
    run_scenario,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _scenario_doc(**overrides) -> dict:
    scenario = {
        "scenario_id": "example",
        "title": "Example",
        "why": "incident abc1234",
        "prompt": "¿qué embedder usamos?",
        "seed_memories": [
            {"title": "Embedder", "content": "El embedder es 4B, 2560 dims.", "type": "decision"}
        ],
        "gates": [{"kind": "answer_must_contain_any", "patterns": ["2560"]}],
    }
    scenario.update(overrides)
    return {"schema_version": eb.SCHEMA_VERSION, "scenarios": [scenario]}


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --- loading / validation -----------------------------------------------------


def test_loads_a_wellformed_scenario(tmp_path: Path) -> None:
    scenarios = load_scenarios(_write(tmp_path, _scenario_doc()))

    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "example"
    assert scenarios[0].seed_memories[0].type == "decision"


def test_scenario_with_only_recall_gates_is_rejected(tmp_path: Path) -> None:
    """A scenario whose every gate is must_recall/must_not_recall measures
    retrieval, which `memo eval recall` already covers against a much larger
    corpus. Letting it in would quietly turn this harness into a worse copy of
    that one."""
    doc = _scenario_doc(gates=[{"kind": "must_recall", "seed_index": 0}])

    with pytest.raises(ValueError, match="no answer-layer gate"):
        load_scenarios(_write(tmp_path, doc))


def test_unknown_gate_kind_is_rejected(tmp_path: Path) -> None:
    doc = _scenario_doc(gates=[{"kind": "vibes"}])

    with pytest.raises(ValueError, match="unknown gate kind"):
        load_scenarios(_write(tmp_path, doc))


def test_recall_gate_pointing_past_the_seeded_memories_is_rejected(tmp_path: Path) -> None:
    doc = _scenario_doc(
        gates=[
            {"kind": "must_recall", "seed_index": 7},
            {"kind": "answer_must_contain_any", "patterns": ["x"]},
        ]
    )

    with pytest.raises(ValueError, match="seed_index"):
        load_scenarios(_write(tmp_path, doc))


def test_pattern_gate_without_patterns_is_rejected(tmp_path: Path) -> None:
    doc = _scenario_doc(gates=[{"kind": "answer_must_contain_any", "patterns": []}])

    with pytest.raises(ValueError, match="no patterns"):
        load_scenarios(_write(tmp_path, doc))


def test_semantic_gate_without_a_statement_is_rejected(tmp_path: Path) -> None:
    doc = _scenario_doc(gates=[{"kind": "semantic", "statement": ""}])

    with pytest.raises(ValueError, match="empty semantic gate"):
        load_scenarios(_write(tmp_path, doc))


def test_scenario_seeding_no_memories_is_rejected(tmp_path: Path) -> None:
    doc = _scenario_doc(seed_memories=[])

    with pytest.raises(ValueError, match="seeds no memories"):
        load_scenarios(_write(tmp_path, doc))


def test_unreadable_corpus_raises_valueerror(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not read"):
        load_scenarios(tmp_path / "missing.json")


# --- gate scoring -------------------------------------------------------------


def test_must_recall_matches_on_the_short_id_prefix() -> None:
    gate = Gate(kind="must_recall", seed_index=0)
    ids = ["abcdef1234567890"]

    assert eb._eval_recall_gate(gate, "…[abcdef12] Embedder…", ids).passed
    assert not eb._eval_recall_gate(gate, "…nothing here…", ids).passed


def test_must_not_recall_inverts_the_verdict() -> None:
    gate = Gate(kind="must_not_recall", seed_index=0)
    ids = ["abcdef1234567890"]

    assert eb._eval_recall_gate(gate, "…nothing here…", ids).passed
    result = eb._eval_recall_gate(gate, "…[abcdef12]…", ids)
    assert not result.passed
    assert "must not" in result.detail


def test_contain_any_passes_on_any_pattern_and_ignores_case() -> None:
    gate = Gate(kind="answer_must_contain_any", patterns=("2560", "cuatro-be"))

    assert eb._eval_answer_gate(gate, "son 2560 dims", _judge_never).passed
    assert eb._eval_answer_gate(gate, "el modelo CUATRO-BE", _judge_never).passed
    assert not eb._eval_answer_gate(gate, "no lo sé", _judge_never).passed


def test_not_contain_any_fails_on_a_forbidden_pattern() -> None:
    gate = Gate(kind="answer_must_not_contain_any", patterns=("batchear",))

    assert eb._eval_answer_gate(gate, "usá head-slice", _judge_never).passed
    result = eb._eval_answer_gate(gate, "podés batchear los pares", _judge_never)
    assert not result.passed
    assert "batchear" in result.detail


def _judge_never(answer: str, statement: str) -> bool:
    return False


def _judge_always(answer: str, statement: str) -> bool:
    return True


def test_semantic_gate_delegates_to_the_judge() -> None:
    gate = Gate(kind="semantic", statement="defers to the stored decision")

    assert eb._eval_answer_gate(gate, "cualquier cosa", _judge_always).passed
    assert not eb._eval_answer_gate(gate, "cualquier cosa", _judge_never).passed


# --- run_scenario -------------------------------------------------------------


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="s1",
        title="t",
        why="w",
        prompt="p",
        seed_memories=(SeedMemory(title="a", content="b"),),
        gates=(
            Gate(kind="must_recall", seed_index=0),
            Gate(kind="answer_must_contain_any", patterns=("2560",)),
        ),
    )


def test_run_scenario_scores_both_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eb, "seed_store", lambda sc, wd: ["abcdef1234567890"])
    monkeypatch.setattr(eb, "run_recall_hook", lambda prompt, wd: "## Memory\n- [abcdef12] a")

    result = run_scenario(
        _scenario(), answerer=lambda ctx, prompt: "son 2560 dims", judge=_judge_never
    )

    assert result.passed
    assert [g.gate.kind for g in result.gates] == ["must_recall", "answer_must_contain_any"]


def test_recall_only_skips_the_answer_layer_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eb, "seed_store", lambda sc, wd: ["abcdef1234567890"])
    monkeypatch.setattr(eb, "run_recall_hook", lambda prompt, wd: "- [abcdef12] a")

    def _explode(context: str, prompt: str) -> str:
        raise AssertionError("recall_only must not invoke the answerer")

    result = run_scenario(_scenario(), answerer=_explode, judge=_judge_never, recall_only=True)

    assert result.passed
    assert [g.gate.kind for g in result.gates] == ["must_recall"]
    assert result.answer == ""


def test_an_empty_recall_block_fails_the_must_recall_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eb, "seed_store", lambda sc, wd: ["abcdef1234567890"])
    monkeypatch.setattr(eb, "run_recall_hook", lambda prompt, wd: "")

    result = run_scenario(_scenario(), answerer=lambda c, p: "2560", judge=_judge_never)

    assert not result.passed


def test_a_failing_hook_is_reported_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An errored scenario must never read as a pass — that is the failure mode
    this whole harness exists to catch."""

    def _boom(prompt: str, workdir: Path) -> str:
        raise RuntimeError("recall-hook exited 1")

    monkeypatch.setattr(eb, "seed_store", lambda sc, wd: ["abcdef1234567890"])
    monkeypatch.setattr(eb, "run_recall_hook", _boom)

    result = run_scenario(_scenario(), answerer=lambda c, p: "2560", judge=_judge_never)

    assert not result.passed
    assert "recall-hook exited 1" in result.error
    assert result.as_dict()["passed"] is False


# --- the committed corpus -----------------------------------------------------


def test_committed_corpus_is_valid() -> None:
    """The shipped corpus must parse and satisfy every schema rule — including
    the answer-layer requirement, so a retrieval-only scenario cannot be
    committed by hand."""
    corpus = REPO_ROOT / "eval" / "behavior_scenarios.json"
    if not corpus.is_file():
        pytest.skip("no committed behavior corpus yet")

    scenarios = load_scenarios(corpus)

    assert scenarios
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)), f"duplicate scenario ids: {ids}"
    for scenario in scenarios:
        assert scenario.answer_gates, scenario.scenario_id
        assert scenario.why, f"{scenario.scenario_id} must cite the incident it generalizes"


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_end_to_end_against_a_real_seeded_store(tmp_path: Path) -> None:
    """The real path: seed a real store, run the real `memo recall-hook`
    subprocess against it, and confirm the seeded memory reaches the block."""
    scenario = Scenario(
        scenario_id="e2e",
        title="e2e",
        why="harness self-check",
        prompt="cuantas dimensiones tiene el embedder que usamos",
        seed_memories=(
            SeedMemory(
                title="Embedder dims",
                content="El embedder de esta maquina es el Qwen3-Embedding-4B y usa 2560 dimensiones.",
                type="decision",
            ),
        ),
        gates=(
            Gate(kind="must_recall", seed_index=0),
            Gate(kind="answer_must_contain_any", patterns=("2560",)),
        ),
    )

    result = run_scenario(scenario, recall_only=True, workdir=tmp_path)

    assert not result.error, result.error
    assert result.recall_block, "the real hook returned an empty block for a seeded memory"
    assert all(g.passed for g in result.gates), [
        (g.gate.kind, g.detail) for g in result.gates if not g.passed
    ]


def test_gate_result_detail_is_empty_on_pass() -> None:
    assert GateResult(Gate(kind="semantic", statement="s"), True).detail == ""


def test_a_missing_fact_gate_says_whether_the_payload_carried_it() -> None:
    """The two verdicts look identical without this and mean opposite things:
    a fact absent from the block is memo-side and unambiguous; a fact present
    in the block that the answer ignored may just be a weak answerer."""
    gate = Gate(kind="answer_must_contain_any", patterns=("2560",))

    absent = eb._eval_answer_gate(gate, "no sé", _judge_never, block="## Memory\n- nada")
    present = eb._eval_answer_gate(gate, "no sé", _judge_never, block="## Memory\n- son 2560 dims")

    assert not absent.passed and "never carried the fact" in absent.detail
    assert not present.passed and "IS in the recall block" in present.detail
