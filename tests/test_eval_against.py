from __future__ import annotations

import os
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


# --- m1: negative-recall ⛔ floors -------------------------------------------
# `--update-baseline` once shipped bare `gate_metrics`, dropping avoid_at_k /
# avoid_leak_at_k to "vacuously true forever". `compare_rows` must not repeat
# that mistake now that it carries the same two fields.


def test_compare_rows_fails_when_the_diff_drops_avoid_coverage() -> None:
    current = [{**_row("A", 0.70, 0.10), "avoid_at_k": 0.50}]
    ref = [{**_row("A", 0.70, 0.10), "avoid_at_k": 0.90}]

    result = eval_against.compare_rows(current, ref)

    assert not result.passed
    assert "avoid@k" in result.message


def test_compare_rows_fails_when_the_diff_raises_avoid_leak() -> None:
    current = [{**_row("A", 0.70, 0.10), "avoid_leak_at_k": 0.30}]
    ref = [{**_row("A", 0.70, 0.10), "avoid_leak_at_k": 0.0}]

    result = eval_against.compare_rows(current, ref)

    assert not result.passed
    assert "avoid_leak@k" in result.message


# --- C2: label-set identity guard --------------------------------------------
# A relative --labels path resolves inside the ref worktree's cwd, not the
# caller's — silently scoring the two sides against different label sets. The
# fingerprint check catches that (and any other cause of a label-set mismatch)
# regardless of how it happened.


def test_compare_rows_refuses_when_label_sets_differ() -> None:
    result = eval_against.compare_rows(
        [_row("A", 0.70, 0.10)],
        [_row("A", 0.70, 0.10)],
        current_labels_fingerprint="fp-current",
        ref_labels_fingerprint="fp-ref",
    )

    assert not result.passed
    assert "label sets differ" in result.message


def test_compare_rows_allows_matching_label_sets() -> None:
    result = eval_against.compare_rows(
        [_row("A", 0.70, 0.10)],
        [_row("A", 0.70, 0.10)],
        current_labels_fingerprint="fp-same",
        ref_labels_fingerprint="fp-same",
    )

    assert result.passed


def test_compare_rows_skips_the_fingerprint_check_when_unknown() -> None:
    # Existing callers (and the tests above) that don't have a fingerprint to
    # give must be unaffected — this is what keeps compare_rows callable with
    # only `current`/`ref`.
    result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [_row("A", 0.70, 0.10)])

    assert result.passed


def test_run_against_prepends_the_worktree_src_onto_pythonpath(tmp_path, monkeypatch) -> None:
    # I3: PYTHONPATH must be non-empty in the environment BEFORE run_against
    # is called, and the assertion must pin the FULL resulting string — under
    # `uv run pytest` the ambient PYTHONPATH is empty, so `.startswith(wt_src)`
    # can't distinguish prepend from append; an append-bugged implementation
    # would pass it verbatim.
    monkeypatch.setenv("PYTHONPATH", "/opt/installed")
    seen: dict[str, object] = {}

    def _fake_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
        seen["argv"] = argv
        seen["env"] = env
        seen["cwd"] = cwd
        return (
            '{"rows": [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}], '
            '"labels_fingerprint": "fp-abc"}'
        )

    monkeypatch.setattr(eval_against, "_add_worktree", lambda ref, root, dest: dest)
    monkeypatch.setattr(eval_against, "_remove_worktree", lambda root, dest: None)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda root: tmp_path / "wt")

    result = eval_against.run_against(
        "origin/master",
        repo_root=tmp_path,
        argv=["eval", "recall", "--json", "--no-cache"],
        runner=_fake_runner,
    )

    assert result.rows == [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}]
    assert result.labels_fingerprint == "fp-abc"
    env = seen["env"]
    assert isinstance(env, dict)
    wt_src = str(tmp_path / "wt" / "src")
    assert env["PYTHONPATH"] == f"{wt_src}{os.pathsep}/opt/installed"
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[1:3] == ["-m", "memo"]
