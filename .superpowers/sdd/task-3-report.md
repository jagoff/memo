# Task 3 Report: Eval Recall Profile Source Of Truth

Status: done

Commit: this report is included in the `feat: name recall eval profiles` commit; exact SHA is listed in the final response.

Files changed:
- `src/memo/eval_recall.py`
- `src/memo/cli_eval.py`
- `tests/test_eval_recall.py`
- `.superpowers/sdd/task-3-report.md`

Local operational update:
- Updated `.git/hooks/pre-push` locally to run `memo eval recall --labels eval/regression_labels.json --k 5 --force --gate --profile pre-push`.
- The hook remains machine-local and was not staged for commit.

Implementation:
- Added `EvalProfile = Literal["quick", "default", "pre-push", "matrix", "expensive"]`.
- Added `profile_configs()` as the source of truth for named recall eval profiles.
- Added `--profile` to `memo eval recall`.
- Preserved config selection order: explicit `--config`, then `--profile`, then existing `select_configs(None, quick=quick)` behavior.

Tests:
- Failing-first check:
  `uv run --no-sync pytest tests/test_eval_recall.py::test_profile_configs_name_eval_roles tests/test_eval_recall.py::test_cli_eval_recall_profile_pre_push_selects_named_subset -q`
  failed as expected before implementation because `profile_configs` and `--profile` did not exist.
- Focused tests:
  `uv run --no-sync pytest tests/test_eval_recall.py::test_cli_eval_recall_help_lists_options tests/test_eval_recall.py::test_profile_configs_name_eval_roles tests/test_eval_recall.py::test_cli_eval_recall_profile_pre_push_selects_named_subset -q`
  passed: 3 passed.
- Quick eval smoke:
  `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force --quick --max-prompts 1`
  passed; ran 1 config x 1 prompt.
- Ruff:
  `uv run --no-sync ruff check src/memo/eval_recall.py src/memo/cli_eval.py tests/test_eval_recall.py`
  passed.

Concerns:
- Pre-existing unrelated dirty files remain in the worktree: `.superpowers/sdd/progress.md`, `.superpowers/sdd/task-1-brief.md`, `.superpowers/sdd/task-2-brief.md`.

---

# Task 3 Report: Markdown Config Flag Integration

Status: done

Commits:
- `feat(config): read markdown config in flags`

Tests:
- `uv run --no-sync pytest tests/test_flags.py tests/test_config_md.py -v` (28 passed)
- `uv run --no-sync ruff check src/memo/flags.py tests/test_flags.py` (passed)

Self-review:
- `flag()` now resolves values in order: environment, Markdown config, tuned overlay, default.
- `active_flags()` remains environment-only; `active_config_values()` delegates to the Markdown-backed config API.
- `validate()` includes Markdown configuration diagnostics before unknown environment-variable checks.
- Boolean Markdown values are asserted in canonical `on`/`off` form and are accepted through the existing flag coercion path.

Concerns:
- The worktree contains unrelated pre-existing `.superpowers/sdd` changes, including a whitespace warning in `task-2-brief.md`; they were not modified or staged.
