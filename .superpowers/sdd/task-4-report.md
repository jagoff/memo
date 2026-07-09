# Task 4 Report: Explicit CLI And MCP Context-Pack Surface

## Scope

Implemented the explicit, read-only context-pack surfaces requested in Task 4:

- Added CLI command: `memo context-pack QUERY --k N --json`
- Added MCP tool: `memo_context_pack(question, k=5, type_=None, snippet_chars=800)`

The implementation stays explicit and read-only. It does not alter ambient recall, chat paths, hybrid candidate generation, or default ranking behavior outside the opt-in command/tool invocation.

## What changed

### CLI

- Added `context_pack_cmd` to [`src/memo/cli_search.py`](/Users/fer/repos/memo/src/memo/cli_search.py)
  - Uses `Config.from_env()` and the existing `get_memory()` helper.
  - Retrieves candidates through `mem.search(...)` with:
    - `mode="hybrid"`
    - `disable_reranker=True`
    - `read_through=True`
    - `quality_rerank=True`
  - Builds the pack through the existing public API in [`src/memo/context_pack.py`](/Users/fer/repos/memo/src/memo/context_pack.py).
  - Emits JSON via `dataclasses.asdict(pack)` or renders `pack.to_prompt()` in a panel for human use.

- Registered the command in [`src/memo/cli.py`](/Users/fer/repos/memo/src/memo/cli.py)
  - Imported `context_pack_cmd`
  - Added it to the root CLI
  - Placed it in the `Core` help section

- Added `context-pack` to `CORE_CLI_COMMANDS` in [`src/memo/surface.py`](/Users/fer/repos/memo/src/memo/surface.py) so it remains visible on the reduced core CLI surface.

### MCP

- Added new module [`src/memo/server_context_pack.py`](/Users/fer/repos/memo/src/memo/server_context_pack.py)
  - Follows the existing `register(server, memory)` pattern
  - Uses `@annotated_tool(server, **READ_ONLY)`
  - Uses the same retrieval settings as the CLI surface
  - Returns a plain JSON-serialisable dict via `asdict(pack)`

- Registered the tool from [`src/memo/server.py`](/Users/fer/repos/memo/src/memo/server.py)
  - Imported `server_context_pack`
  - Registered it inside the advanced-tools gate, matching the task brief and existing MCP surface structure

## Tests added

Added [`tests/test_context_pack_surface.py`](/Users/fer/repos/memo/tests/test_context_pack_surface.py) with:

- `test_context_pack_cli_empty_corpus_json`
  - Initially failed with `No such command 'context-pack'`
  - Now verifies the command returns a JSON payload with the expected question and empty `current_facts`

- `test_context_pack_mcp_empty_corpus`
  - Initially failed because `memo_context_pack` was not registered
  - Now verifies MCP registration and empty-corpus behavior

## Surface follow-through

Because the new MCP tool increases the full MCP surface by one tool, I also updated:

- full-surface test expectations in:
  - [`tests/test_cli_mcp_surface_smoke.py`](/Users/fer/repos/memo/tests/test_cli_mcp_surface_smoke.py)
  - [`tests/test_surface_profiles.py`](/Users/fer/repos/memo/tests/test_surface_profiles.py)
- user-facing full-profile count text in:
  - [`src/memo/__init__.py`](/Users/fer/repos/memo/src/memo/__init__.py)
  - [`src/memo/cli_install_mcp.py`](/Users/fer/repos/memo/src/memo/cli_install_mcp.py)
  - [`src/memo/surface.py`](/Users/fer/repos/memo/src/memo/surface.py)

The full MCP surface count is now `125`.

## Verification

Ran:

```bash
uv run --no-sync pytest tests/test_context_pack.py tests/test_context_pack_surface.py -v
uv run --no-sync pytest tests/test_server_annotations.py tests/test_cli_mcp_surface_smoke.py tests/test_surface_profiles.py -v
uv run --no-sync ruff check src/memo/__init__.py src/memo/cli.py src/memo/cli_install_mcp.py src/memo/cli_search.py src/memo/server.py src/memo/server_context_pack.py src/memo/surface.py tests/test_context_pack_surface.py tests/test_cli_mcp_surface_smoke.py tests/test_surface_profiles.py
```

Final verification pass:

```bash
uv run --no-sync pytest tests/test_context_pack.py tests/test_context_pack_surface.py tests/test_server_annotations.py tests/test_cli_mcp_surface_smoke.py tests/test_surface_profiles.py -q
```

Results:

- `302 passed`
- Ruff clean on all touched files

## Self-review

Checked for:

- alignment with existing CLI registration patterns
- alignment with existing MCP registration and annotation patterns
- use of the existing public `build_context_pack()` API
- explicit, read-only behavior only
- no changes to chat path opt-out behavior
- no changes to ambient recall hot path
- no changes to hybrid/vector/BM25 candidate generation logic
- no inclusion of unrelated working tree changes in the commit

No functional concerns found in the implemented scope.

## Review Fixes

Addressed the Task 4 review findings with a narrow follow-up patch:

- CLI `memo context-pack` now respects `MEMO_CONTEXT_PACK`.
  - When disabled, it raises a clear `ClickException` telling the caller to set `MEMO_CONTEXT_PACK=1`.
  - It now accepts `--source` and logs consults through `log_cli_consult(...)` as `via="cli:context_pack"`.
  - Consult hits are derived from the built context-pack rows, so the recall log captures the ids/scores/titles actually surfaced by the pack.

- MCP `memo_context_pack` now respects `MEMO_CONTEXT_PACK`.
  - When disabled, it returns a clear disabled payload:
    - `{"status": "disabled", "reason": "MEMO_CONTEXT_PACK=0 disables explicit context-pack tools.", "question": ...}`
  - It now accepts `source` and logs consults through `server_common.log_consult(...)` as `via="mcp:context_pack"`.
  - MCP consult hits are likewise derived from the built pack rows.

- Added `consult_hits_from_pack(...)` in [`src/memo/context_pack.py`](/Users/fer/repos/memo/src/memo/context_pack.py) so CLI and MCP use one consistent consult-hit projection from explicit context-pack results.

### Focused tests added

Extended [`tests/test_context_pack_surface.py`](/Users/fer/repos/memo/tests/test_context_pack_surface.py) to cover:

- CLI disabled-by-flag behavior
- CLI consult logging and source attribution
- MCP disabled-by-flag behavior
- MCP consult logging and source attribution

### Verification for review fixes

Ran:

```bash
uv run --no-sync pytest tests/test_context_pack.py tests/test_context_pack_surface.py tests/test_cli_consult_attribution.py tests/test_usefulness.py -q
uv run --no-sync ruff check src/memo/context_pack.py src/memo/cli_search.py src/memo/server_context_pack.py tests/test_context_pack_surface.py
```

Results:

- `35 passed`
- Ruff clean on touched files
