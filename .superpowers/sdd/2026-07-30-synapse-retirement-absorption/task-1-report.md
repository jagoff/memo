# Task 1 report — Synapse operation manifest

## Delivered

- Added static, fail-closed Synapse catalog discovery from the pinned MCP
  catalog, CLI parser, and five daemon modules.  Snapshot code is parsed, not
  imported; symlinks, non-canonical `source.json`, invalid OIDs, missing
  canonical sources, malformed AST, and duplicate operation names are rejected.
- Added the `SynapseOperation` evidence row and evolved
  `SynapseRetirementManifest` to `memo.synapse_retirement.v2`, binding the
  complete discovered operation list into its canonical signed bytes.
- Added `build_synapse_capability_manifest`, which binds the discovered rows to
  existing capability authority, exact two-Mac usage-proof bytes, signed
  exclusions, and in-tree fixture SHA-256 values.
- Strengthened route validation to reject overlapping closed predicates and
  added optional route fixture paths for source-byte binding.
- Added dry-run-only `synapse-catalog` and `synapse-manifest` CLI inspection.
  They emit the catalog/manifest digest, operation/disposition counts, source
  commit, blocker list, and two proof identifiers; neither writes production
  state.

## Verification

```text
uv run --no-sync pytest tests/tools/test_synapse_catalog.py tests/tools/test_absorption_inventory.py tests/tools/test_absorption_manifest.py -q
21 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 9 source files

git diff --check
passed
```

## Scope / concerns

- The approved Ed25519 signature domain remains
  `memo.cutover.synapse_retirement.v1`; only `memo.synapse_retirement.v2` is
  pre-authorized as the manifest schema.  The signing implementation rejects
  unregistered domains, and Task 1 ownership does not include that authority
  allow-list.
- Existing unrelated changes in secure-enclave files, prior SDD ledgers,
  `pyproject.toml`, and `hatch_build.py` were preserved and excluded from this
  task's commit.
