from __future__ import annotations

from pathlib import Path

from memo import eval_against


def _row(config: str, precision: float, noise: float) -> dict[str, float | str]:
    return {
        "config": config,
        "precision_at_k": precision,
        "noise_at_k": noise,
        "avoid_at_k": 1.0,
        "avoid_leak_at_k": 0.0,
    }


def test_build_eval_argv_always_disables_the_shared_cache() -> None:
    argv = eval_against.build_eval_argv(
        labels_path="eval/regression_labels.json", k=5, profile="pre-push", configs=()
    )

    # The eval cache is keyed on corpus+labels+configs+k and lives in state_dir,
    # shared across worktrees — without --no-cache the ref run reads the
    # current run's numbers and every comparison is a false tie.
    assert "--no-cache" in argv
    assert "--json" in argv
    assert argv[:3] == ["eval", "recall", "--k"]
    assert "--profile" in argv and "pre-push" in argv


def test_build_eval_argv_passes_explicit_configs_through() -> None:
    argv = eval_against.build_eval_argv(labels_path=None, k=3, profile=None, configs=("A", "B"))

    assert argv.count("--config") == 2
    assert "A" in argv and "B" in argv
    assert "--profile" not in argv


def test_compare_rows_passes_when_the_diff_holds_precision() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [_row("A", 0.70, 0.10)])

    assert result.passed
    assert result.config == "A"
    assert "PASS" in result.message


def test_compare_rows_fails_when_the_diff_drops_precision() -> None:
    result = eval_against.compare_rows([_row("A", 0.55, 0.10)], [_row("A", 0.70, 0.10)])

    assert not result.passed
    assert "precision@k" in result.message
    assert result.current_precision == 0.55
    assert result.ref_precision == 0.70


def test_compare_rows_fails_when_the_diff_raises_noise() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.25)], [_row("A", 0.70, 0.10)])

    assert not result.passed
    assert "noise@k" in result.message


def test_compare_rows_pins_the_comparison_to_the_same_config() -> None:
    current = [_row("A", 0.55, 0.10), _row("B", 0.90, 0.05)]
    ref = [_row("A", 0.70, 0.10)]

    result = eval_against.compare_rows(current, ref)

    # B winning the current run must not mask A's regression.
    assert not result.passed
    assert result.config == "A"


def test_compare_rows_fails_loudly_when_the_ref_run_produced_nothing() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [])

    assert not result.passed
    assert "no rows" in result.message


def test_run_against_puts_the_worktree_src_first_on_pythonpath(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
        seen["argv"] = argv
        seen["env"] = env
        seen["cwd"] = cwd
        return '{"rows": [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}]}'

    monkeypatch.setattr(eval_against, "_add_worktree", lambda ref, root, dest: dest)
    monkeypatch.setattr(eval_against, "_remove_worktree", lambda root, dest: None)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda root: tmp_path / "wt")

    rows = eval_against.run_against(
        "origin/master",
        repo_root=tmp_path,
        argv=["eval", "recall", "--json", "--no-cache"],
        runner=_fake_runner,
    )

    assert rows == [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}]
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"].startswith(str(tmp_path / "wt" / "src"))
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[1:3] == ["-m", "memo"]
