# Public Release Audit — memo (mlx-memo)

**Date:** 2026-06-25  
**Scope:** Correctness + Installability audit before making repo public on GitHub and releasing to PyPI  
**Approach:** Combined outside-in simulation + targeted correctness audit + static analysis  

---

## Context

- `mlx-memo` is a local-first semantic memory system: MLX embeddings + sqlite-vec + BM25 hybrid search, MCP server, CLI.
- The GitHub repo (`jagoff/memo`) is **already public**. Prior git history contains personal paths (`/Users/fer/`), IPs, and `com.fer.*` labels — this must be addressed.
- The PyPI package `mlx-memo` is also public but intended for wider distribution post-audit.
- README exists but is incomplete/outdated. No separate quickstart for new users.
- Test suite: 1535 passing, 25 skipped, 0 failing (as of start of audit).

---

## Goals

1. A stranger with an M-series Mac can follow the README, install `mlx-memo`, and have a working `memo recall-hook` in Claude Code within 10 minutes.
2. All 5 critical user flows work correctly including error paths.
3. No personal data, credentials, or internal paths in source code or `pyproject.toml`.
4. `pyproject.toml` is complete and correct for PyPI public listing.
5. `ruff` and `mypy` clean.

---

## Phases

### Phase 0 — Pre-flight (before any audit work)

- [ ] Grep entire source tree for `/Users/fer`, `fernandoferrari`, personal IPs (`192.168.`, `fferrari`), `com.fer.*` labels
- [ ] Grep `pyproject.toml` for internal/personal metadata
- [ ] Check git history for personal paths (surface only — git history rewrite is a separate operation but must be flagged)
- [ ] Verify `ruff check src/` baseline
- [ ] Verify `mypy src/memo/` baseline
- [ ] Confirm test suite 100% green

### Phase 1 — Fresh install simulation

Simulate a brand-new user on a clean M-series Mac:

**1a. Installation**
- Install via `uv tool install mlx-memo` (primary) and `pip install mlx-memo` (secondary)
- Verify `memo --help` works, version is correct, no import errors on startup
- Verify `memo-mcp --help` works from same isolated runtime
- Check `memo doctor` output on a clean system — all checks should pass or give actionable error messages

**1b. First use flow**
- `memo save "test memory"` — does it work without prior config?
- `memo search "test"` — returns results?
- `memo recall-hook` — output in 5s budget? Correct JSON?
- `memo install-mcp` (if it exists) — wires correctly into Claude Code?

**1c. MCP configuration**
- Manual config of `memo-mcp` in Claude Code settings
- Verify MCP tools are callable: `memo_save`, `memo_search`, `memo_unified_briefing`
- Verify `memo_start_session` doesn't spawn LLM-heavy background workers without explicit opt-in

**1d. README gaps**
- Document every step where the README was wrong, missing, or unclear
- Update README inline as each gap is found

### Phase 2 — Correctness audit of 5 critical flows

For each flow: read the code, trace all branches, check error handling, verify tests cover the unhappy paths.

**Flow 1: save → search round-trip**
- `memo save` writes `.md` file first, then indexes (source of truth = markdown)
- Embedding dimension mismatch detection
- Duplicate detection behavior
- What happens on disk full / permission error?
- BM25 tokenization for non-ASCII (Spanish diacritics)

**Flow 2: recall-hook (5s budget)**
- Cold start vs warm daemon path
- Token budget enforcement (`MEMO_RECALL_TOKEN_BUDGET`)
- What if sqlite is locked? What if embedder fails?
- Output format: must be valid for Claude Code UserPromptSubmit hook
- Concurrent recall-hook calls (multiple sessions firing simultaneously)

**Flow 3: memo-mcp server**
- Startup: does it load MLX at import time (violates deferred-import invariant)?
- `memo_start_session`: what background workers does it spawn? Are they bounded?
- Tool schemas: any tool that exposes internal paths or personal data in its description?
- `MEMO_MCP_PROFILE=core` vs `agent` — does `core` work for a new user?
- HTTP transport health endpoint (`/health`)
- Concurrent tool calls (FastMCP threadpool + thread-local sqlite connections)

**Flow 4: memo sync (git cross-machine)**
- `memo sync init` (new command added today) — happy path + `gh` not installed
- `memo sync clone <url>` — fresh clone on new machine
- `memo sync pull` — rebase conflict handling
- `memo sync once` — flock behavior under concurrent sessions
- What if no git remote? Should degrade gracefully with clear message.

**Flow 5: session lifecycle**
- `memo_start_session` → idle capture → `memo capture-stop`
- `memo session idle-maintenance --mode reflect` — flock now prevents concurrent runs (fix applied today)
- `memo dream if-due` — respects `MEMO_MAINTAIN_DISABLE`?
- Session files not leaking personal paths in their content

### Phase 3 — Static analysis

**3a. ruff**
```bash
uv run --no-sync ruff check src/ --select ALL --ignore ANN,D,ERA,FIX,TD
```
Fix or explicitly ignore with comment any finding in: security rules (S), bugbear (B), simplify (SIM), correctness (E/W/F).

**3b. mypy**
```bash
uv run --no-sync mypy src/memo/ --ignore-missing-imports
```
Fix type errors in public-facing functions (CLI entrypoints, MCP tools, Memory facade).

**3c. Secret / personal data scan**
```bash
grep -rn "/Users/fer\|fernandoferrari\|192\.168\.\|com\.fer\.\|fferrari" src/ pyproject.toml README* hooks/
```
Any hit = blocker for public release.

**3d. pyproject.toml audit**
- `description` — clear, public-friendly?
- `license` — specified?
- `authors` — uses public email only?
- `python_requires` — correct minimum (3.11+)?
- `classifiers` — `Operating System :: MacOS`, `Environment :: GPU`, etc.
- `urls` — `Homepage`, `Repository`, `Changelog` present?
- No internal paths in `[tool.*]` sections

**3e. Git history surface scan**
```bash
git log --all --oneline | head -50   # check commit messages for personal info
git grep -i "192\.168\." $(git rev-list --all) -- '*.py' '*.toml' '*.json' 2>/dev/null | head -20
```
Flag any findings. A full `git filter-repo` rewrite is out of scope for this audit but must be tracked as a follow-up if findings exist.

---

## Deliverables

| # | Deliverable | Done when |
|---|-------------|-----------|
| 1 | All bugs found in Phases 1-2 fixed with regression tests | `pytest` still 100% green |
| 2 | README reflects actual golden path | A stranger can follow it without Claude's help |
| 3 | `pyproject.toml` complete for PyPI | All required fields present, no internal data |
| 4 | ruff + mypy clean | CI-equivalent passes |
| 5 | No personal data in source | grep scan returns 0 hits |
| 6 | Audit report | List of all findings + disposition (fixed/deferred/won't fix) |

---

## Out of Scope

- Performance benchmarks
- `pip-audit` of transitive dependencies
- Git history rewrite (`git filter-repo`) — flagged if needed, done separately
- Internationalization / non-macOS support
- Visual/TUI testing
- Windows or Linux support

---

## Known Findings (pre-audit, found during design)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | `bootstrap_clone` called itself recursively in `sync_git.py:138` | High | Fixed (commit c590fbc) |
| 2 | `test_capture.py` assertions used outdated `"※ auto save:"` prefix | Low | Fixed (commit c590fbc) |
| 3 | Git history likely contains `/Users/fer/` paths | High | Flagged — needs `git filter-repo` |
| 4 | `memo-mcp` not in PATH for Claude Code plugin on remote Mac | Medium | Fixed in plugin .mcp.json |
