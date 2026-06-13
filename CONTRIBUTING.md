# Contributing to memo

Thanks for your interest. memo is a small, focused project — a clear bug
report or a tight PR is the best way to help.

## Quick start

```bash
git clone https://github.com/jagoff/memo
cd memo
uv pip install -e '.[dev]'        # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# CI-parity checks
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/memo

# Fast test suite (skips MLX smoke tests)
.venv/bin/python -m pytest -q -m "not slow"

# Real-MLX smoke (Apple Silicon only — auto-skipped elsewhere)
.venv/bin/python -m pytest -q -m requires_mlx
```

CI runs Linux parity checks in `.github/workflows/test.yml` and a real macOS
MLX/runtime smoke in `.github/workflows/macos-smoke.yml`. Keep both paths green
before you push if your change touches runtime, hooks, retrieval, or MLX code.

## Reporting bugs

Open an issue at <https://github.com/jagoff/memo/issues> with:

- `memo --version` output
- macOS version + chip (`uname -mrs`)
- A minimal reproducer (CLI command or MCP tool call)
- Expected vs actual behaviour
- Relevant log lines (set `MEMO_DEBUG=1` for verbose output)

## Pull requests

- Keep PRs scoped to one change. A bug fix doesn't need surrounding
  cleanup; one new feature per PR.
- Add a test that fails before your change and passes after.
- Update `CHANGELOG.md` under `## [Unreleased]` with a one-line bullet.
- Match the existing code style (`ruff check` enforces it).
- Don't introduce new dependencies without discussing first — every new
  package is a long-tail maintenance burden.

If you change retrieval, ranking, or ingest behavior, also run:

```bash
.venv/bin/memo eval recall --labels eval/regression_labels.json --k 5 --force
```

If you change hooks, daemon lifecycle, install/runtime plumbing, or migration
paths, prefer focused verification over only the broad suite. The most useful
targets are in:

- `tests/test_recall_hooks.py`
- `tests/test_recall_server.py`
- `tests/test_runtime_isolation.py`
- `tests/test_cli_migrate_vault.py`
- `tests/test_ingest_daemon.py`
- `tests/test_maint_daemon.py`

## Scope

memo is intentionally narrow: persistent semantic memory for AI agents,
running 100 % locally on Apple Silicon via MLX. Features that pull it
toward cloud sync, multi-tenant SaaS, or non-Mac platforms are out of
scope. The stable core is capture, retrieval, ambient recall/briefing,
history/time-machine, and runtime health. Experimental modules and
advanced surfaces (see `src/memo/experimental_index.md`) may change
without notice.

## License

By contributing you agree that your contributions are released under the
MIT License (see `LICENSE`).
