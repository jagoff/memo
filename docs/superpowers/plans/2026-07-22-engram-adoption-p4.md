# Engram Learnings — P4 Adoption and Setup Plan

**Goal:** Provide one explicit, idempotent `memo setup` entry point backed by a
declarative agent registry while preserving existing installer aliases.

## Contract

- `memo setup codex|claude-code|all`, `--detect`, and `--dry-run`.
- Every adapter declares detection, MCP strategy/profile, instructions,
  protocol mode, verification, restart guidance, and rollback limitations.
- Setup builds a plan before mutation. File writes are atomic, backed up,
  marker-delimited, and preserve unknown content/keys.
- External CLI mutations report partial success plus exact remediation when
  they cannot be rolled back.
- Normal startup and upgrade never rewrite agent configuration.

## Implementation

1. Add `runtime/agent_registry.py` using shared presets plus memo-local adapter
   metadata for Codex and Claude Code.
2. Add plan/apply/verify receipts and disposable-home-safe file primitives.
3. Add `cli_setup.py`; register it from wiring-only `cli.py`.
4. Delegate `install-mcp`, `install-slash`, and `mandate` compatibility paths
   to the registry where their selected agents overlap.
5. Add `memo doctor --agent <slug>` registry verification without touching the
   real vault; smoke operations use isolated temporary directories.

## Gates

- Dry-run purity, repeat idempotency, unknown-content preservation, backup and
  rollback, valid config formats, profile/marker/runtime verification, partial
  external failure receipts, and alias parity.
