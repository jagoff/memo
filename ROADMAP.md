# ROADMAP

Scope: execute the highest-leverage improvements for `memo` without expanding platform support. This plan assumes the current product focus stays Apple Silicon + MLX.

## 0-30 Days

### 1. Add real macOS/Apple Silicon CI
Goal: validate the runtime users actually run.

Deliverables:
- Add a GitHub Actions workflow on `macos-latest` with Python `3.13`.
- Run:
  - `uv pip install -e '.[dev]'`
  - `ruff check src/ tests/`
  - `mypy src/memo`
  - targeted real-MLX smoke tests
  - `memo doctor --strict-runtime`
- Add a lightweight installer smoke that exercises `install.sh` or an equivalent install path without widening the workflow beyond a practical runtime budget.

Likely files:
- `.github/workflows/test.yml`
- new `.github/workflows/macos-smoke.yml`
- `tests/test_smoke_mlx.py`
- `install.sh`

Success criteria:
- A PR can fail on macOS-specific regressions before release.
- At least one workflow exercises the real MLX path instead of relying on `requires_mlx` auto-skip.

### 2. Define the stable core product surface
Goal: make the product easier to understand, test, and release confidently.

Stable core:
- CRUD + retrieval: `save`, `search`, `ask`, `list`, `get`, `update`, `delete`
- maintenance: `reindex`, `briefing`, `recall-hook`
- history/time: `history`, `as-of`, `diff`
- runtime sanity: `doctor`, `prewarm`, `recall-daemon`

Deliverables:
- Mark the stable core explicitly in docs.
- Mark experimental or lower-confidence surfaces explicitly in docs and help text.
- Ensure release notes and README lead with the stable core instead of the full feature inventory.

Likely files:
- `README.md`
- `CONTRIBUTING.md`
- `src/memo/experimental_index.md`
- `src/memo/cli.py`
- `src/memo/server.py`

Success criteria:
- A new contributor can tell which commands and MCP tools are core without reading the whole repo.
- Experimental features stop reading as equally supported to the core path.

## 31-60 Days

### 3. Separate core and experimental surfaces more aggressively
Goal: reduce public complexity without deleting useful experiments.

Deliverables:
- Group experimental CLI commands into clearer namespaces or help sections.
- Move new non-core MCP tools toward explicit registration patterns that make stability boundaries visible.
- Reduce the amount of top-level surface that appears “first-class” by default.

Likely files:
- `src/memo/cli.py`
- `src/memo/server.py`
- `src/memo/mcp_tools.py`
- `src/memo/experimental_index.md`

Success criteria:
- `memo --help` and MCP tool listings communicate a clearer distinction between core and experimental capabilities.
- Adding experimental features no longer expands the apparent stable surface by default.

### 4. Strengthen retrieval regression gates
Goal: make retrieval quality changes measurable and harder to regress.

Deliverables:
- Standardize `memo eval recall --labels eval/regression_labels.json --k 5 --force` as the required check for retrieval/ranking/ingest changes.
- Expand `eval/regression_labels.json` with real failure cases.
- Track warm vs cold recall latency as part of retrieval validation.

Likely files:
- `eval/regression_labels.json`
- `src/memo/eval_recall.py`
- retrieval-related docs under `README.md` / `CLAUDE.md`
- any retrieval/ranking/ingest modules touched by future fixes

Success criteria:
- Retrieval fixes are justified with corpus-level evidence, not one-off query tuning.
- Latency-sensitive recall changes are validated against both quality and budget.

Verification commands:
- `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`
- `uv run --no-sync pytest tests/test_eval_recall.py -v`

## 61-90 Days

### 5. Release polish around the stable core
Goal: align product story, packaging, and release mechanics with the actual supported path.

Deliverables:
- Rework README ordering so the stable core comes first.
- Ensure installer, runtime doctor, and plugin metadata all reinforce the same product shape.
- Keep version bumps and release flow disciplined across all package metadata files.

Likely files:
- `README.md`
- `install.sh`
- `pyproject.toml`
- `.claude-plugin/plugin.json`
- `server.json`
- `CHANGELOG.md`

Success criteria:
- The release artifact, docs, and install path all describe the same product.
- Fewer “works in CLI, broken in MCP” or “feature looked supported but isn’t hardened” surprises.

### 6. Harden reliability around operational paths
Goal: catch failures that hurt real usage even when Linux CI stays green.

Focus areas:
- hooks lifecycle
- recall daemon lifecycle
- `reindex --rebuild`
- upgrade/migration paths
- runtime isolation checks

Likely files:
- `hooks/hooks.json`
- `tests/test_recall_server.py`
- `tests/test_recall_hooks.py`
- `tests/test_ingest_daemon.py`
- `tests/test_maint_daemon.py`
- `tests/test_runtime_isolation.py`
- `tests/test_cli_migrate_vault.py`

Success criteria:
- Operational regressions are covered by targeted tests instead of being found only on a live machine.
- Runtime/install/hook paths get the same attention as pure library behavior.

## Priority Order
1. macOS/Apple Silicon CI
2. stable core definition
3. core vs experimental separation
4. retrieval regression gates
5. release polish
6. operational reliability hardening

## Explicit Non-Goals
- No cross-platform fallback mode in this roadmap.
- No expansion of experimental surface before the stable core is clearer and better tested.
