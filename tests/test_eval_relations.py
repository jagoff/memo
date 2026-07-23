from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from click import unstyle
from click.testing import CliRunner

from memo.cli_eval import eval_group
from memo.eval_relations import evaluate

LABELS = Path(__file__).parents[1] / "eval" / "relation_candidate_labels.json"


def test_fixed_relation_corpus_passes_without_noise() -> None:
    result = evaluate(LABELS)

    assert result.passed is True
    assert result.recall == 1.0
    assert result.noise == 0.0
    assert result.true_positive == 5


def test_relation_eval_cli_gate_json() -> None:
    result = CliRunner().invoke(
        eval_group, ["relations", "--labels", str(LABELS), "--gate", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert '"passed": true' in result.output


def test_relation_eval_cli_human_gate_failure(monkeypatch) -> None:
    failed = replace(evaluate(LABELS), passed=False)
    monkeypatch.setattr("memo.eval_relations.evaluate", lambda _path: failed)

    result = CliRunner().invoke(eval_group, ["relations", "--labels", str(LABELS), "--gate"])

    assert result.exit_code == 1
    assert "relation gate: recall=1.000 precision=1.000 noise=0.000" in unstyle(result.output)


def test_relation_eval_cli_rejects_invalid_label_schema(tmp_path: Path) -> None:
    labels = tmp_path / "invalid-relations.json"
    labels.write_text(json.dumps({"schema": "invalid", "cases": []}), encoding="utf-8")

    result = CliRunner().invoke(eval_group, ["relations", "--labels", str(labels)])

    assert result.exit_code == 1
    assert "invalid relation label set" in result.output
