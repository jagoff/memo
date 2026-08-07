from __future__ import annotations

import os
from pathlib import Path

import pytest

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


def test_compare_rows_refuses_when_the_ref_fingerprint_is_missing() -> None:
    # LOW-4: the CLI always supplies a current-side fingerprint. Once it has,
    # a MISSING ref fingerprint must fail closed (not silently skip the
    # guard) — an absent value means "we don't know if it's the same label
    # set", which is exactly what this guard exists to catch.
    result = eval_against.compare_rows(
        [_row("A", 0.70, 0.10)],
        [_row("A", 0.70, 0.10)],
        current_labels_fingerprint="fp-current",
        ref_labels_fingerprint="",
    )

    assert not result.passed
    assert "did not report" in result.message
    assert "differ" not in result.message  # distinct wording from a mismatch


def test_compare_rows_populates_avoid_and_leak_fields_in_the_result() -> None:
    # Deferred-2: m1's floors can fail a comparison on avoid@k/avoid_leak@k,
    # but the machine-readable AgainstResult carried no fields to explain
    # that failure with — only precision/noise.
    current = [{**_row("A", 0.70, 0.10), "avoid_at_k": 0.50, "avoid_leak_at_k": 0.20}]
    ref = [{**_row("A", 0.70, 0.10), "avoid_at_k": 0.90, "avoid_leak_at_k": 0.05}]

    result = eval_against.compare_rows(current, ref)

    assert not result.passed
    assert result.current_avoid == 0.50
    assert result.ref_avoid == 0.90
    assert result.current_leak == 0.20
    assert result.ref_leak == 0.05


def test_compare_rows_uses_check_gates_permissive_default_for_a_missing_ref_leak() -> None:
    # Deferred-3: check_gate defaults a MISSING baseline avoid_leak_at_k to
    # 1.0 (permissive — any leak rate clears it), not 0.0. The old
    # `.get(key) or 0.0` would make a missing ref value maximally STRICT
    # instead — pin the aligned convention so this stays a decision, not an
    # accident.
    current = [{**_row("A", 0.70, 0.10), "avoid_leak_at_k": 0.40}]
    ref = [
        {
            "config": "A",
            "precision_at_k": 0.70,
            "noise_at_k": 0.10,
            "avoid_at_k": 1.0,
            # avoid_leak_at_k deliberately absent.
        }
    ]

    result = eval_against.compare_rows(current, ref)

    assert result.passed
    assert result.ref_leak == 1.0


def test_compare_rows_raises_a_clear_error_on_a_malformed_ref_value() -> None:
    # Round-3 item 2: a differently-shaped ref payload (an older `memo eval
    # recall`, a schema change) must not surface as a raw ValueError deep
    # inside a comparison — the same class of bug MEDIUM-2 fixed one level up
    # (git/subprocess/JSON), applied here to the payload's own values.
    current = [_row("A", 0.70, 0.10)]
    ref = [{**_row("A", 0.70, 0.10), "precision_at_k": "not-a-number"}]

    with pytest.raises(eval_against.AgainstError, match="precision_at_k"):
        eval_against.compare_rows(current, ref)


def test_compare_rows_ref_leak_placeholder_is_consistent_across_early_returns() -> None:
    # Round-3 item 3: the guard/no-rows early returns used to leave
    # ref_leak=0.0 while the "config not evaluated" branch used 1.0 for the
    # same "we don't have a real ref value" situation. Both now use 1.0 (the
    # permissive default for a leak CEILING — see AgainstResult's docstring).
    guard_result = eval_against.compare_rows(
        [_row("A", 0.70, 0.10)],
        [_row("A", 0.70, 0.10)],
        current_labels_fingerprint="fp-current",
        ref_labels_fingerprint="",
    )
    no_ref_result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [])
    no_current_result = eval_against.compare_rows([], [_row("A", 0.70, 0.10)])
    # avoid_leak_at_k deliberately absent, so the branch's own `.get(...,
    # 1.0)` default is what's under test here — not a real measured value.
    ref_missing_leak = [
        {"config": "A", "precision_at_k": 0.70, "noise_at_k": 0.10, "avoid_at_k": 1.0}
    ]
    config_not_evaluated_result = eval_against.compare_rows(
        [_row("B", 0.90, 0.05)], ref_missing_leak
    )

    assert guard_result.ref_leak == 1.0
    assert no_ref_result.ref_leak == 1.0
    assert no_current_result.ref_leak == 1.0
    assert config_not_evaluated_result.ref_leak == 1.0


def test_remove_worktree_falls_back_to_rmtree_when_git_fails_despite_being_added(
    tmp_path, monkeypatch
) -> None:
    # LOW-1: if bookkeeping says the worktree WAS added but `git worktree
    # remove` still fails for some other reason, still fall back to rmtree —
    # and it's fine (expected, even) to warn in that case, since it's
    # genuinely unexpected.
    leaked = tmp_path / "leaked-wt"
    leaked.mkdir()

    def _fail(args, *, cwd):
        raise eval_against.AgainstError("git worktree remove ... failed: fatal: not a working tree")

    monkeypatch.setattr(eval_against, "_run_git", _fail)

    eval_against._remove_worktree(tmp_path, leaked, added=True)

    assert not leaked.exists()


def test_remove_worktree_skips_git_and_the_warning_when_never_added(
    tmp_path, monkeypatch, capsys
) -> None:
    # Round-3 item 1: when `_add_worktree` itself failed, `dest` was never
    # registered as a worktree — calling `git worktree remove` on it (and
    # printing its failure as a warning) is pure noise; the caller already
    # knows exactly why. Skip straight to removing the directory, and don't
    # shell out to git at all.
    leaked = tmp_path / "never-added"
    leaked.mkdir()

    def _fail(args, *, cwd):
        raise AssertionError("git must not be invoked when the worktree was never added")

    monkeypatch.setattr(eval_against, "_run_git", _fail)

    eval_against._remove_worktree(tmp_path, leaked, added=False)

    assert not leaked.exists()
    captured = capsys.readouterr()
    assert captured.err == ""


def test_run_against_cleans_up_even_when_add_worktree_fails(tmp_path, monkeypatch) -> None:
    # LOW-1: _add_worktree used to run OUTSIDE the try/finally, so a failure
    # there (e.g. a typo'd ref) skipped _remove_worktree entirely.
    calls: list[str] = []

    monkeypatch.setattr(eval_against, "_ref_exists", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_ref_has_main_entrypoint", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda: tmp_path / "wt")

    def _fail_add(ref, root, dest):
        calls.append("add")
        raise eval_against.AgainstError("git worktree add ... failed: fatal: invalid reference")

    def _record_remove(root, dest, *, added):
        calls.append("remove")
        assert added is False, "add failed, so added must be False by the time remove runs"

    monkeypatch.setattr(eval_against, "_add_worktree", _fail_add)
    monkeypatch.setattr(eval_against, "_remove_worktree", _record_remove)

    with pytest.raises(eval_against.AgainstError):
        eval_against.run_against(
            "typo-ref",
            repo_root=tmp_path,
            argv=["eval", "recall"],
            runner=lambda argv, env, cwd: "{}",
        )

    assert calls == ["add", "remove"]


def test_run_against_raises_a_distinct_error_for_an_unknown_ref(tmp_path, monkeypatch) -> None:
    # MEDIUM (round 3): `git cat-file -e <ref>:path` fails identically for
    # "ref exists but lacks the path" and "ref does not exist" — without
    # checking existence FIRST, a typo'd ref (the common case for a
    # hand-typed --against value) was misdiagnosed as "predates the
    # entrypoint" and told to check out an unrelated commit.
    monkeypatch.setattr(eval_against, "_ref_exists", lambda ref, root: False)

    def _entrypoint_check_must_not_run(ref, root):
        raise AssertionError("the entrypoint check must not run for an unknown ref")

    monkeypatch.setattr(eval_against, "_ref_has_main_entrypoint", _entrypoint_check_must_not_run)

    with pytest.raises(eval_against.AgainstError, match="unknown ref") as excinfo:
        eval_against.run_against(
            "orgin/mstr",
            repo_root=tmp_path,
            argv=["eval", "recall", "--json", "--no-cache"],
            runner=lambda argv, env, cwd: "{}",
        )

    # Distinct from the "predates the entrypoint" wording.
    assert "predates" not in str(excinfo.value)


def test_run_against_refuses_a_ref_that_predates_the_entrypoint(tmp_path, monkeypatch) -> None:
    # Deferred-1: src/memo/__main__.py doesn't exist at origin/master until
    # this branch merges — PYTHONPATH=<wt>/src python -m memo has no fallback,
    # so any older ref fails deep inside the subprocess with an opaque error.
    # A pre-flight `git cat-file -e` check should catch it before that.
    monkeypatch.setattr(eval_against, "_ref_exists", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_ref_has_main_entrypoint", lambda ref, root: False)
    monkeypatch.setattr(eval_against, "_first_commit_with_main_entrypoint", lambda root: "abc1234")

    with pytest.raises(eval_against.AgainstError, match="abc1234"):
        eval_against.run_against(
            "origin/master",
            repo_root=tmp_path,
            argv=["eval", "recall", "--json", "--no-cache"],
            runner=lambda argv, env, cwd: "{}",
        )


def test_run_against_raises_a_clear_error_on_invalid_json_output(tmp_path, monkeypatch) -> None:
    # MEDIUM-2: a polluted-stdout parse failure must not surface as a raw
    # traceback indistinguishable from a genuine ranking regression.
    monkeypatch.setattr(eval_against, "_ref_exists", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_ref_has_main_entrypoint", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_add_worktree", lambda ref, root, dest: dest)
    monkeypatch.setattr(eval_against, "_remove_worktree", lambda root, dest, *, added: None)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda: tmp_path / "wt")

    garbage = "warning: some deprecation notice\n" + "not json at all"

    with pytest.raises(eval_against.AgainstError) as excinfo:
        eval_against.run_against(
            "origin/master",
            repo_root=tmp_path,
            argv=["eval", "recall", "--json", "--no-cache"],
            runner=lambda argv, env, cwd: garbage,
        )

    # The message quotes the excerpt via repr() (so control characters like
    # the embedded newline stay legible on one line) — assert against the
    # same repr, not the raw string.
    assert repr(garbage[:200]) in str(excinfo.value)


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

    monkeypatch.setattr(eval_against, "_ref_exists", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_ref_has_main_entrypoint", lambda ref, root: True)
    monkeypatch.setattr(eval_against, "_add_worktree", lambda ref, root, dest: dest)
    monkeypatch.setattr(eval_against, "_remove_worktree", lambda root, dest, *, added: None)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda: tmp_path / "wt")

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
