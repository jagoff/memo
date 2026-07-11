## Task 5: Markdown config CLI

Status: complete

Implemented `memo config init`, `path`, `set`, `unset`, `show`, and `migrate` in `src/memo/cli_config.py`. `show --effective` reports environment flag overrides over Markdown values; existing `flags` and Markdown-aware `validate` behavior remain intact.

Tests: `uv run --no-sync pytest tests/test_cli_config.py -v` (5 passed); `uv run --no-sync ruff check src/memo/cli_config.py tests/test_cli_config.py` (passed).

Concern: repository-wide `git diff --check` reports a pre-existing missing final newline in `.superpowers/sdd/task-2-brief.md`, which is outside this task's ownership.
