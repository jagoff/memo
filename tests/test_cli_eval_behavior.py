"""`memo eval behavior` CLI surface: reporting, gating, and error handling.

The harness itself is covered in tests/test_eval_behavior.py. These exercise
the Click layer, including the paths a green run never touches — a malformed
corpus, an unknown scenario id, and a scenario that errored rather than failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo import eval_behavior as eb
from memo.cli import cli

SCENARIO = {
    "schema_version": eb.SCHEMA_VERSION,
    "scenarios": [
        {
            "scenario_id": "s1",
            "title": "t",
            "why": "incident abc1234",
            "prompt": "p",
            "seed_memories": [{"title": "a", "content": "b", "type": "note"}],
            "gates": [
                {"kind": "must_recall", "seed_index": 0},
                {"kind": "answer_must_contain_any", "patterns": ["2560"]},
            ],
        }
    ],
}


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(SCENARIO), encoding="utf-8")
    return path


@pytest.fixture
def stub_run(monkeypatch: pytest.MonkeyPatch):
    """Stub the store+hook so the CLI layer is exercised without MLX."""

    def _install(*, block: str, answer: str = "", error: Exception | None = None) -> None:
        monkeypatch.setattr(eb, "seed_store", lambda sc, wd: ["abcdef1234567890"])
        if error is not None:

            def _boom(prompt, workdir, **kw):
                raise error

            monkeypatch.setattr(eb, "run_recall_hook", _boom)
        else:
            monkeypatch.setattr(eb, "run_recall_hook", lambda prompt, wd, **kw: block)
        monkeypatch.setattr(eb, "default_answerer", lambda ctx, prompt: answer)
        monkeypatch.setattr(eb, "default_judge", lambda answer, statement: True)

    return _install


def test_passing_scenario_reports_and_exits_zero(corpus: Path, tmp_path: Path, stub_run) -> None:
    stub_run(block="## Memory\n- [abcdef12] a", answer="son 2560 dims")

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus), "--gate"], env=_env(tmp_path)
    )

    assert result.exit_code == 0, result.output
    assert "1/1 scenarios passed" in result.output


def test_gate_exits_non_zero_when_a_scenario_fails(corpus: Path, tmp_path: Path, stub_run) -> None:
    stub_run(block="## Memory\n- [abcdef12] a", answer="no idea")

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus), "--gate"], env=_env(tmp_path)
    )

    assert result.exit_code == 1
    assert "0/1 scenarios passed" in result.output


def test_without_gate_a_failure_still_exits_zero(corpus: Path, tmp_path: Path, stub_run) -> None:
    stub_run(block="## Memory\n- [abcdef12] a", answer="no idea")

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus)], env=_env(tmp_path)
    )

    assert result.exit_code == 0
    assert "0/1 scenarios passed" in result.output


def test_an_errored_scenario_prints_the_error(corpus: Path, tmp_path: Path, stub_run) -> None:
    """An error must be visible in the report, not collapse into a plain red."""
    stub_run(block="", error=RuntimeError("recall-hook exited 1"))

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus)], env=_env(tmp_path)
    )

    assert result.exit_code == 0
    assert "error:" in result.output
    assert "recall-hook exited 1" in result.output


def test_recall_only_reports_the_mode(corpus: Path, tmp_path: Path, stub_run) -> None:
    stub_run(block="## Memory\n- [abcdef12] a")

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus), "--recall-only"], env=_env(tmp_path)
    )

    assert result.exit_code == 0
    assert "recall-layer only" in result.output


def test_json_output_carries_the_schema_and_per_gate_detail(
    corpus: Path, tmp_path: Path, stub_run
) -> None:
    stub_run(block="## Memory\n- [abcdef12] a", answer="son 2560 dims")

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(corpus), "--json"], env=_env(tmp_path)
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "memo.eval_behavior.report.v1"
    assert payload["scenarios"] == 1 and payload["passed"] == 1
    assert [g["kind"] for g in payload["results"][0]["gates"]] == [
        "must_recall",
        "answer_must_contain_any",
    ]


def test_only_selects_a_single_scenario(corpus: Path, tmp_path: Path, stub_run) -> None:
    stub_run(block="## Memory\n- [abcdef12] a", answer="son 2560 dims")

    result = CliRunner().invoke(
        cli,
        ["eval", "behavior", "--scenarios", str(corpus), "--only", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "1/1 scenarios passed" in result.output


def test_only_with_an_unknown_id_is_a_clean_cli_error(corpus: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["eval", "behavior", "--scenarios", str(corpus), "--only", "nope"],
        env=_env(tmp_path),
    )

    assert result.exit_code != 0
    assert "no scenario with id 'nope'" in result.output


def test_a_malformed_corpus_is_a_clean_cli_error(tmp_path: Path) -> None:
    """A schema violation must surface as a ClickException, not a traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "only-retrieval",
                        "prompt": "p",
                        "seed_memories": [{"title": "a", "content": "b"}],
                        "gates": [{"kind": "must_recall", "seed_index": 0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli, ["eval", "behavior", "--scenarios", str(bad)], env=_env(tmp_path)
    )

    assert result.exit_code != 0
    assert "no answer-layer gate" in result.output
    assert "Traceback" not in result.output
