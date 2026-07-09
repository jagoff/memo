# Task 3 Report: Context Pack Builder

## Status

Completed.

## Commit

- `1a0eda7` `feat: build context packs for ask`

## What changed

### 1. Added `src/memo/context_pack.py`

Implemented:

- `ContextPack`
- `build_context_pack(question, hits, *, snippet_chars, budget_chars=4000)`

Behavior:

- Classifies memory hits with `memo.quality.classify_quality()`
- Separates results into:
  - `current_facts`
  - `supporting_context`
  - `stale_or_conflicting`
- Includes quality bucket and reasons per packed memory row
- Trims lower-priority sections to a character budget
- Omits sensitive/secret memories from compacted prompt output and records that omission

### 2. Integrated context packs into ask formatting

Updated [src/memo/memory/ask_ops.py](/Users/fer/repos/memo/src/memo/memory/ask_ops.py) so:

- `Memory.ask()` reads `MEMO_CONTEXT_PACK` through `memo.flags.flag_bool`
- When the flag is off, the existing ask prompt format stays unchanged
- When the flag is on, ask builds a context pack and uses it for the memory portion of the user prompt
- Existing repo snippets and lazy-expanded synthesis source memories are still appended to the prompt in the flag-on branch
- Memory sources returned from `ask()` get `quality_bucket` and `quality_reasons` metadata in the flag-on path
- Session RAG cache keys now include whether the context-pack branch was used

### 3. Kept chat ask on the old prompt path

Updated [src/memo/memory/chat_ask_ops.py](/Users/fer/repos/memo/src/memo/memory/chat_ask_ops.py) to call `ask(..., use_context_pack=False)`.

Reason:

The live codebase routes `chat_ask()` through `ask()`. Without this adaptation, enabling `MEMO_CONTEXT_PACK` would have changed chat ask behavior too, which is broader than the Task 3 requirement to integrate it into ask formatting only.

### 4. Added focused tests

Created [tests/test_context_pack.py](/Users/fer/repos/memo/tests/test_context_pack.py) covering:

- current vs stale separation
- budget trimming order
- direct ask flag-off vs flag-on formatting
- repo snippet preservation in the context-pack branch
- `chat_ask()` staying on the standard prompt path even when the flag is enabled

## Required adaptation from the brief

I did not apply the brief's `_build_ask_context()` replacement literally.

Why:

- In the current code, `_build_ask_context()` also carries:
  - repo snippets
  - lazy-expanded synthesis provenance snippets
- A literal swap to `pack.to_prompt()` would have dropped those sections from the ask prompt when `MEMO_CONTEXT_PACK=1`
- `chat_ask()` delegates to `ask()`, so a direct flag check inside `ask()` without an override would have unintentionally changed chat behavior

Conservative adaptation:

- preserve repo snippets and expanded source memories in the flag-on branch
- add an internal `use_context_pack` control so direct `ask()` can opt in while `chat_ask()` stays unchanged

## Verification

Ran:

```bash
uv run --no-sync pytest tests/test_context_pack.py -v
uv run --no-sync pytest tests/test_context_pack.py tests/test_quality.py -v
uv run --no-sync pytest tests/test_rag_cache.py -v
uv run --no-sync pytest tests/test_context_pack.py tests/test_quality.py tests/test_rag_cache.py -v
uv run --no-sync ruff check src/memo/context_pack.py src/memo/memory/ask_ops.py src/memo/memory/chat_ask_ops.py tests/test_context_pack.py
```

Result:

- all focused tests passed
- cache smoke passed
- targeted Ruff check passed

## Self-review

Checked for the main regression risks called out by the task:

- `MEMO_CONTEXT_PACK=0` keeps the legacy ask prompt path
- ambient recall was not touched
- retrieval/ranking path was not changed
- repo citations remain available to ask in the flag-on branch
- chat ask was explicitly kept off the new formatting path
- tests are isolated and use existing temporary fixtures

## Concerns

None blocking.

## Review Fix Addendum

Addressed the Task 3 review findings in the `MEMO_CONTEXT_PACK` ask path.

### Fixes

- Expanded synthesis source memories now go through the same prompt-row builder used by primary memory hits, so sensitive expanded memories are filtered before prompt assembly and surviving expanded memories carry `quality_bucket` / `quality_reasons`.
- The flag-on ask branch now applies one deterministic final budget to the full context section, not just `pack.to_prompt()`.
- Final trim order is explicit:
  1. supporting context
  2. stale/conflicting context
  3. expanded source memories
  4. repo snippets
- Repo snippets are still included in the flag-on branch when budget allows.
- The returned `sources` list in the flag-on branch is rebuilt from the kept prompt rows, so memory sources that survive the prompt budget/filter path keep quality metadata and sensitive expanded memories are not reintroduced outside the pack filter.

### Added regression coverage

- final context budget enforcement with expanded memory and repo sections
- trim-order coverage for supporting/stale before expanded/repo
- sensitive expanded memory omission
- non-sensitive expanded memory quality metadata

### Verification

```bash
uv run --no-sync pytest tests/test_context_pack.py tests/test_quality.py tests/test_rag_cache.py -v
uv run --no-sync ruff check src/memo/context_pack.py src/memo/memory/ask_ops.py tests/test_context_pack.py
```

## Re-review Fix Addendum

Addressed the follow-up review findings for Task 3.

### Fixes

- Verbatim short-circuiting now uses only memory hits that actually survived the ask context path when `MEMO_CONTEXT_PACK=1`, so omitted sensitive memories cannot be returned verbatim after prompt filtering.
- `ask_stream()` now resolves and passes the same `MEMO_CONTEXT_PACK` behavior as `ask()`, including the explicit per-call override.
- `chat_ask_stream()` now passes `use_context_pack=False` so the streaming chat surface stays on the legacy prompt path, matching `chat_ask()`.

### Added regression coverage

- sensitive top-hit verbatim bypass is blocked when context-pack filtering is active
- flag-off verbatim short-circuit behavior is preserved
- direct streaming ask adopts context-pack prompt construction and source metadata only when the flag is enabled
- streaming chat ask remains opted out of context-pack formatting

### Verification

```bash
uv run --no-sync pytest tests/test_context_pack.py tests/test_quality.py tests/test_rag_cache.py -v
uv run --no-sync ruff check src/memo/memory/ask_ops.py src/memo/memory/chat_ask_ops.py tests/test_context_pack.py
```
