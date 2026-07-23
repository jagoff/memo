from __future__ import annotations

from pathlib import Path

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
