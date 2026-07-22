import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CHECKOUT_ACTION = "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0"
SETUP_PYTHON_ACTION = "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
UPLOAD_ARTIFACT_ACTION = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML 1.1 treats the unquoted GitHub key `on` as boolean true.
    return workflow.get("on", workflow[True])


def _named_steps(job: dict) -> dict[str, dict]:
    return {step["name"]: step for step in job["steps"] if "name" in step}


def _mutation_run_script() -> str:
    config = _workflow("mutation-tests.yml")
    steps = config["jobs"]["mutate-core"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == "Run scoped mutations")


def test_pr_workflow_enforces_resource_and_diff_coverage_gates() -> None:
    workflow = _workflow("test.yml")
    assert _triggers(workflow) == {
        "push": {"branches": ["master"]},
        "pull_request": {"branches": ["master"]},
    }
    job = workflow["jobs"]["test"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"] == {"python-version": ["3.13", "3.14"]}
    assert job["steps"][0] == {
        "uses": CHECKOUT_ACTION,
        "with": {"persist-credentials": False, "fetch-depth": 0},
    }
    steps = _named_steps(job)
    expected_order = [
        "Install package + dev deps",
        "Format check (ruff)",
        "Lint (ruff)",
        "Type check (mypy)",
        "Progressive quality budget",
        "Resource hygiene (serial ownership gate)",
        "Tests (excluding slow real-MLX smoke)",
        "Changed-lines coverage",
    ]
    positions = [
        next(i for i, step in enumerate(job["steps"]) if step.get("name") == name)
        for name in expected_order
    ]
    assert positions == sorted(positions)
    assert steps["Resource hygiene (serial ownership gate)"]["run"] == (
        '.venv/bin/python -m pytest -m "resource_hygiene" -n 0 --timeout=120 --resource-hygiene'
    )
    assert steps["Tests (excluding slow real-MLX smoke)"]["run"] == (
        '.venv/bin/python -m pytest -m "not slow" -n auto --timeout=120 '
        "--cov=memo --cov-report=term-missing --cov-report=xml"
    )
    changed = steps["Changed-lines coverage"]
    assert changed["if"] == "github.event_name == 'pull_request'"
    assert changed["run"] == (
        ".venv/bin/diff-cover coverage.xml --compare-branch=origin/master "
        "--fail-under=90 --show-uncovered"
    )


def test_stability_workflow_is_replayable_and_never_masks_flakes() -> None:
    workflow = _workflow("test-stability.yml")
    assert _triggers(workflow) == {
        "schedule": [{"cron": "30 4 * * *"}],
        "workflow_dispatch": None,
    }
    job = workflow["jobs"]["stability"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 30
    assert job["steps"][:2] == [
        {"uses": CHECKOUT_ACTION, "with": {"persist-credentials": False}},
        {"uses": SETUP_PYTHON_ACTION, "with": {"python-version": "3.13"}},
    ]
    steps = _named_steps(job)
    expected_order = [
        "Install stability environment",
        "Full non-slow suite in replayable random order",
        "Repeat concurrency and resource ownership sessions",
        "Upload stability reports",
    ]
    positions = [
        next(i for i, step in enumerate(job["steps"]) if step.get("name") == name)
        for name in expected_order
    ]
    assert positions == sorted(positions)
    assert steps["Install stability environment"]["run"] == (
        "uv sync --frozen --extra dev --extra cpu --extra http --extra test-stability --python 3.13"
    )
    seed_env = {"RANDOM_SEED": "${{ github.run_id }}"}
    full = steps["Full non-slow suite in replayable random order"]
    assert full["env"] == seed_env
    assert full["run"] == (
        "echo \"Replay: pytest -m 'not slow' -n 0 --randomly-seed=$RANDOM_SEED\"\n"
        '.venv/bin/python -m pytest -m "not slow" -n 0 --timeout=120 \\\n'
        '  --randomly-seed="$RANDOM_SEED" --junitxml=stability-full.xml\n'
    )
    repeat = steps["Repeat concurrency and resource ownership sessions"]
    assert repeat["env"] == seed_env
    assert repeat["run"] == (
        '.venv/bin/python -m pytest -m "concurrency or resource_hygiene" '
        '-n 0 -x --timeout=120 --randomly-seed="$RANDOM_SEED" --count=10 '
        "--repeat-scope=session --resource-hygiene --junitxml=stability-repeat.xml"
    )
    upload = steps["Upload stability reports"]
    assert upload == {
        "name": "Upload stability reports",
        "if": "always()",
        "uses": UPLOAD_ARTIFACT_ACTION,
        "with": {
            "name": "stability-results-${{ github.run_id }}",
            "path": "stability-*.xml",
            "if-no-files-found": "warn",
        },
    }


def test_mutation_workflow_is_scoped_scheduled_and_retains_results() -> None:
    workflow = (WORKFLOWS / "mutation-tests.yml").read_text(encoding="utf-8")
    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "timeout-minutes: 30" in workflow
    assert "--extra test-mutation" in workflow
    assert "mutmut run" in workflow
    assert "mutmut results" in workflow
    assert "scripts/check_mutation_results.py mutants" in workflow
    assert workflow.count("${PIPESTATUS[0]}") == 3
    assert (
        "          .venv/bin/mutmut run 2>&1 | tee mutation-results.txt\n"
        "          run_status=${PIPESTATUS[0]}\n"
        "          .venv/bin/mutmut results 2>&1 | tee -a mutation-results.txt\n"
        "          results_status=${PIPESTATUS[0]}\n"
        '          if [ "$run_status" -ne 0 ]; then\n'
        '            exit "$run_status"\n'
        "          fi\n"
        '          if [ "$results_status" -ne 0 ]; then\n'
        '            exit "$results_status"\n'
        "          fi\n"
        "          .venv/bin/python scripts/check_mutation_results.py mutants \\\n"
        "            2>&1 | tee -a mutation-results.txt\n"
        "          gate_status=${PIPESTATUS[0]}\n"
        '          exit "$gate_status"\n'
    ) in workflow
    assert "mutation-results.txt" in workflow
    assert "if: always()" in workflow
    assert "upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow


@pytest.mark.parametrize(
    ("run_status", "results_status", "gate_status", "expected_status"),
    [(7, 0, 0, 7), (0, 8, 0, 8), (0, 0, 9, 9)],
)
def test_mutation_workflow_propagates_each_pipeline_failure(
    tmp_path: Path,
    run_status: int,
    results_status: int,
    gate_status: int,
    expected_status: int,
) -> None:
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    mutmut = bin_dir / "mutmut"
    mutmut.write_text(
        "#!/usr/bin/env bash\n"
        'printf "mutmut %s\\n" "$1"\n'
        'if [ "$1" = "run" ]; then exit "$FAKE_RUN_STATUS"; fi\n'
        'exit "$FAKE_RESULTS_STATUS"\n',
        encoding="utf-8",
    )
    mutmut.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(
        '#!/usr/bin/env bash\nprintf "gate\\n"\nexit "$FAKE_GATE_STATUS"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)

    completed = subprocess.run(
        ["bash", "-c", _mutation_run_script()],
        cwd=tmp_path,
        env={
            **os.environ,
            "FAKE_RUN_STATUS": str(run_status),
            "FAKE_RESULTS_STATUS": str(results_status),
            "FAKE_GATE_STATUS": str(gate_status),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_status
    retained = (tmp_path / "mutation-results.txt").read_text(encoding="utf-8")
    assert "mutmut run" in retained
    assert "mutmut results" in retained
    assert ("gate" in retained) is (run_status == results_status == 0)
