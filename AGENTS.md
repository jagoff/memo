# AGENTS.md

## Fast Start
- Dev install: `uv pip install -e '.[dev]'`
- Run CLI from the repo: `uv run --no-sync memo <cmd>`
- Run MCP server from the repo: `uv run --no-sync memo-mcp`
- Validate `MEMO_*` flags with `uv run --no-sync memo config validate`

## Verification
- CI order is `ruff -> mypy -> pytest`.
- Linux CI-parity checks:
  - `uv run --no-sync ruff check src/ tests/`
  - `uv run --no-sync mypy src/memo`
  - `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`
- macOS/MLX smoke also exists in `.github/workflows/macos-smoke.yml`; changes to runtime, hooks, retrieval, installer, or MLX paths should keep that workflow green too.
- Slow suite is separate and serial on purpose: `uv run --no-sync pytest -m "slow" --timeout=300 -v`
- Focused test: `uv run --no-sync pytest tests/test_foo.py::test_bar -v`
- If you change retrieval/ranking/ingest behavior, also run `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`.
- If you change hooks, daemon lifecycle, install/runtime plumbing, or migrations, prefer focused checks in `tests/test_hook_contract.py`, `tests/test_recall_hooks.py`, `tests/test_recall_server.py`, `tests/test_runtime_isolation.py`, and `tests/test_cli_migrate_vault.py`.

## Architecture
- `src/memo/cli.py` is wiring only. New CLI surface belongs in `src/memo/cli_<domain>.py` and must be registered in `cli.py`.
- `src/memo/server.py` is MCP wiring only. New MCP tools usually live in `src/memo/server_<domain>.py` with `register(server, memory)`, then get registered from `build_server()`.
- Core app API is `memo.memory.Memory` in `src/memo/memory/facade.py`. It composes operation mixins; do not import mixins directly.
- Storage entrypoint is `VecStore` in `src/memo/store/queries.py`. It uses thread-local sqlite connections; write paths go through `_tx()`.
- The stable core is intentionally narrow: CRUD/retrieval, ambient recall/briefing, reindex/doctor/runtime health, and history/`as-of`/`diff`. Treat other CLI/MCP surfaces as advanced or experimental unless their docs say otherwise; `src/memo/experimental_index.md` is the boundary marker.

## Hard Constraints
- Do not read `MEMO_*` flags with raw `os.environ.get(...)` in app code. Register/access behavioral flags via `src/memo/flags.py`; storage/model config belongs in `src/memo/config.py`.
- Keep `mlx` / `mlx_lm` imports deferred inside functions. Module-level imports hurt every CLI invocation and the recall hook budget.
- `MLXEmbedder.embed()` takes `Sequence[str]`, never a bare string.
- `MEMO_EMBEDDER_DIMS` must match the selected Qwen embedder model dims.
- Query prefixing is asymmetric: the retrieval instruction prefix is for queries only, not stored documents.

## Storage Truths
- Markdown files are the source of truth; sqlite is rebuildable.
- Rebuild with `memo reindex --rebuild`, not by deleting `memvec.db`; rebuild preserves user-signal tables that are not derivable from markdown.
- Hand-edited markdown wins on reindex.

## Testing Quirks
- Tests must never touch the real vault or default state dir. Use `tmp_cfg` or an explicitly isolated `Config`.
- `tests/conftest.py` hard-disables daemon embedding by default (`MEMO_EMBEDDER_VIA_DAEMON=0`) to keep tests hermetic; opt back in only in tests that need it.
- `CliRunner` tests should pass isolated `env=` values for `MEMO_NONINTERACTIVE`, `MEMO_DATA_DIR`, and `MEMO_STATE_DIR`.
- Real MLX tests are guarded with `@pytest.mark.requires_mlx` and auto-skip when `mlx_lm` is unavailable.

## Hook And Runtime Gotchas
- `hooks/hooks.json` prefixes hook commands with `MEMO_NONINTERACTIVE=1` so first-run setup never blocks hooks; preserve that for new hook commands.
- Recall-hook changes are latency-sensitive. The hot path is expected to fit within the hook timeout budget and prefers the recall daemon socket when warm.
- `memo` and `memo-mcp` are expected to come from the same isolated runtime. Use `memo doctor --strict-runtime` when touching install/runtime behavior.

## Repo Conventions
- Use `memo.errors.MemoError` subclasses for domain errors instead of raising bare `Exception` in normal code paths.
- If a search result is wrong, do not patch one query. Make a systemic retrieval change and prove it on `eval/regression_labels.json`.
- Release version bumps must stay in sync across `pyproject.toml`, `.claude-plugin/plugin.json`, `server.json`, and `CHANGELOG.md`.
