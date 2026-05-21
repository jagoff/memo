# Contributing to memo

Thanks for your interest. memo is a small, focused project — a clear bug
report or a tight PR is the best way to help.

## Quick start

```bash
git clone https://github.com/jagoff/memo
cd memo
uv pip install -e '.[dev]'        # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# Fast test suite (skips MLX smoke tests)
.venv/bin/python -m pytest -q -m "not slow"

# Real-MLX smoke (Apple Silicon only — auto-skipped elsewhere)
.venv/bin/python -m pytest -q -m requires_mlx

# Lint
.venv/bin/ruff check src/ tests/
```

CI runs `ruff check src/ tests/` and `pytest -q -m "not slow"` on Ubuntu
(see `.github/workflows/test.yml`). Keep both green before you push.

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

## Scope

memo is intentionally narrow: persistent semantic memory for AI agents,
running 100 % locally on Apple Silicon via MLX. Features that pull it
toward cloud sync, multi-tenant SaaS, or non-Mac platforms are out of
scope. Experimental modules (graph queries, federation, multimodal,
collaborative — see `src/memo/experimental_index.md`) are explicitly
unsupported and may change without notice.

## License

By contributing you agree that your contributions are released under the
MIT License (see `LICENSE`).
