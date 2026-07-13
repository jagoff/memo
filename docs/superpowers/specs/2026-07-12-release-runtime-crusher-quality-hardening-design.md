# Release, Runtime, Crusher, and Quality Hardening Design

**Date:** 2026-07-12
**Status:** Approved
**Scope:** Audit follow-ups 3–6: reproducible release artifacts, offline-by-default
runtime behavior, production SmartCrusher wiring, and progressive quality gates.

## Objective

Make memo's distributed artifacts match the source checkout, keep the default
runtime offline and free of configuration mutations, connect SmartCrusher to the
shared capture path without ever increasing or irreversibly losing content, and
prevent the measured quality debt from growing.

The work is split into four independently testable and revertible phases. Existing
user changes in `src/memo/capture_core.py` and
`tests/test_token_economy_wave1.py` must be preserved and extended rather than
discarded.

## Global Constraints

- Python remains `>=3.13`; Linux and Apple Silicon paths must remain supported.
- Markdown remains the source of truth; sqlite and caches remain derived state.
- `mlx` and `mlx_lm` imports stay deferred inside functions.
- Behavioral flags are registered and read through `memo.flags`.
- The default MCP profile remains the 14-tool `agent` profile.
- New behavior follows test-driven development: each production change starts
  with a focused failing test.
- CI continues in the order `ruff -> mypy -> pytest`, with the quality budget
  check added before the full pytest run.

## Phase 1: Reproducible Release Artifacts

### MCPB build and validation

Introduce one deterministic MCPB builder owned by the release domain. It reads
`packaging/mcpb/manifest.json`, `packaging/mcpb/icon.png`, and
`packaging/mcpb/server/main.py`, writes them in stable lexical order, uses a fixed
ZIP timestamp, and produces `packaging/memo.mcpb`.

`memo release check` must inspect both the source manifest and the binary MCPB.
It fails when:

- the archive is missing a required member;
- the archived manifest is not valid JSON;
- its release version differs from `pyproject.toml`;
- its `uvx --from` package requirement differs from the source manifest; or
- any archived member differs byte-for-byte from its source file.

The builder is idempotent: two builds from the same checkout produce identical
SHA-256 hashes. The binary artifact remains tracked because the repository
already distributes it directly, but it can no longer drift silently.

### Docker image

Replace the PyPI-latest install with a multi-stage build:

1. The builder stage copies the checkout and builds a wheel.
2. The runtime stage installs that wheel plus the CPU extra dependencies.
3. The build verifies the installed `memo` version against the version supplied
   by the release workflow.

The Python base image is pinned by digest. The Qwen embedding model is downloaded
at an explicit Hugging Face revision. Release workflows pass the expected version
from the tag and fail on mismatch. Docker no longer depends on PyPI propagation,
so the tag workflow cannot accidentally publish an older package as a newer
image.

### Publishing supply chain

GitHub Actions are pinned to full commit SHAs. `mcp-publisher` is downloaded at
an explicit version and verified against a committed SHA-256 before execution.
The PyPI propagation loop exits non-zero after its final unsuccessful attempt.
Development/release resolution is captured in `uv.lock`, and CI installs with a
frozen lock rather than resolving arbitrary new transitive versions.

## Phase 2: Offline-By-Default Runtime

The normal `memo-mcp` startup performs no network request and does not modify
Claude/Codex configuration.

### Flags

- `MEMO_UPDATE_CHECK_ENABLED`: boolean, default `false`. Enables the throttled
  remote version notification check.
- `MEMO_AUTO_UPDATE`: boolean, default `false`. Enables background installation
  of a newer tagged version and implicitly permits the required remote check.
- `MEMO_STATUSLINE_SELFHEAL`: boolean, default `false`.
- `MEMO_HOOK_SELFHEAL`: boolean, default `false`.

Plugin manifests stop forcing `MEMO_AUTO_UPDATE=1`. Explicit `memo update`
commands remain available and retain their current user-visible error behavior.

### Startup flow

`server.main()` resolves the four flags before starting background threads:

- no update threads are created when both update flags are false;
- only the notification thread is created when update checking is enabled;
- auto-update may create both the notification and installer work, while sharing
  the existing throttle/stamp state;
- each self-heal thread is created only when its corresponding flag is true.

Opted-in background failures remain non-fatal to MCP startup and are recorded in
the existing logs. Default startup tests replace the network and self-heal entry
points with sentinels and assert that none are called.

Documentation and privacy claims are updated to distinguish the offline default
from explicit update, sync, benchmark-download, and model-download operations.

## Phase 3: Production SmartCrusher

### Integration point

`_extract_and_save()` is the single live integration point because it is shared
by Stop capture, incremental capture, and `extract_and_save_text()`. When the
flag is enabled, it passes `assistant_text` through
`maybe_crush_json_capture()` before calling `extract_insights()`, using
`user_text` as relevance context. The unchanged original remains the grounding
source so later grounding and claim-support checks retain complete evidence.

The feature remains opt-in (`MEMO_CRUSHER_ENABLED=false`) until its committed
P2 token-quality gate is intentionally promoted.

### Safe transformation contract

SmartCrusher returns the original content and no hash unless every condition is
true:

- the input is a top-level JSON array with at least ten rows;
- `MEMO_CRUSHER_ROWS_KEEP_RATIO < 1.0`;
- at least one row is removed;
- the final JSON, including its recovery marker, is at least 5% smaller than the
  original; and
- the original is written successfully to `CrushCache`.

Cache writes happen only after the candidate output passes the size test. If the
cache write raises or cannot be verified, the original is returned. This makes
the operation fail-open and prevents irreversible compression.

### Scoring

Rows are tokenized deterministically from sorted JSON. The distinctiveness term
uses mean IDF rather than total IDF, preventing long rows from winning solely by
length. Context overlap contributes a bounded, IDF-weighted bonus so context can
break close scores without overwhelming row distinctiveness. Ties preserve
original row order.

Tests cover ratio `1.0`, exactly ten rows, marker-induced expansion, cache
failure, must-keep relevance, long-row bias, deterministic ties, and the shared
production call-site.

## Phase 4: Progressive Quality Gates

### Baseline

Track `eval/quality_baseline.json` with machine-readable budgets generated from
the current checkout. `scripts/quality_gate.py` supports read-only checking by
default and an explicit `--update` mode. Update mode prints every debt increase
and requires the caller to commit the changed baseline for review.

### Complexity

Ruff C901 diagnostics are keyed by relative path and function name. Existing
violations may remain, but their individual complexity cannot increase. A new
function may not introduce a C901 violation. Deleting or reducing a violation is
always accepted without requiring an immediate baseline rewrite.

### Broad exceptions

AST-derived `except Exception` counts are budgeted per file. An existing file
cannot exceed its baseline count; a new file starts with a zero allowance. The
gate complements the more detailed intent classifications in `memo.dev_audit`
without depending on fragile line numbers.

### Strict typing

Add strict mypy coverage for the modules changed by these phases:

- `memo.cli_release`
- `memo.runtime.autoupdate`
- `memo.server`
- `memo.surface`
- `memo.capture_core`

The strict set is expanded only after all current errors in those modules are
fixed. Existing broader mypy coverage remains unchanged.

### Coverage

Raise the global coverage floor from 68% to 72%, leaving headroom below the
measured 74.22%. Add focused tests for release artifacts, offline startup, and
SmartCrusher instead of relying only on the aggregate number.

### CI order

The Linux CI job runs:

1. `ruff check src/ tests/ scripts/`
2. `mypy src/memo`
3. `python scripts/quality_gate.py`
4. the existing non-slow pytest command with coverage

macOS smoke retains its real MLX checks. Release/runtime/capture changes also run
the existing focused runtime, hook, and release tests.

## Error Handling

- Release artifact drift is a hard release failure with the exact mismatched
  member or version in the message.
- Docker version mismatch is a hard build failure.
- Explicitly opted-in update checks remain best-effort during MCP startup.
- SmartCrusher failures always return the original content; cache failure never
  permits lossy output.
- Quality gates report actionable current and baseline values and never mutate
  the baseline unless `--update` is passed.

## Acceptance Criteria

- Two consecutive MCPB builds have identical SHA-256 values and match all source
  members.
- The tracked MCPB and all release manifests report one version.
- Docker installs the wheel built from the checkout and proves its version.
- Default `memo-mcp` startup invokes no network updater and no self-heal entry
  point.
- Each opt-in flag activates only its intended behavior.
- SmartCrusher is reached through `_extract_and_save()`, preserves the labeled
  relevant row, and never increases the payload.
- The original JSON is retrievable whenever compressed output is returned.
- Complexity and broad-exception debt cannot grow beyond the committed baseline.
- The selected modules pass strict mypy.
- Coverage remains at or above 72%.
- `ruff`, mypy, the quality gate, non-slow tests, slow/MLX tests, wheel/sdist
  build, release check, and MCPB reproducibility checks all pass.

## Out of Scope

- Refactoring all existing complex functions or broad exception handlers.
- Enabling SmartCrusher by default.
- Changing MCP surface profiles or HTTP authentication.
- Redesigning secret storage.
- Publishing artifacts or pushing commits to a remote.
