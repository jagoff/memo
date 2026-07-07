# Task 1: L1 Crusher Infrastructure (Scoring + Cache Module)

**Context:** Wave 1 of memo token economy overhaul. Task 1 builds the foundation for JSON ingest compression. L1 (SmartCrusher) drops low-relevance JSON rows before indexing, saving 60–92% on structured data without losing retrieval quality. This task creates the cache module and configuration flags. Tasks 2–4 will wire the crusher into the capture pipeline, add retrieval commands, and add output verbosity steering.

**What to build:** Cache module + config flags for reversible JSON compression.

**Deliverables:**
1. `src/memo/store/crush_cache.py` — Cache class with store/retrieve/evict methods
2. `src/memo/flags_capture.py` — Three crusher flags (enabled, keep_ratio, ttl_days)
3. `tests/test_token_economy_wave1.py` — Test suite (11+ tests passing)

**Interfaces:**

**Produces (Task 1 output):**
- `CrushCache(state_dir: Path)` class:
  - `cache(hash: str, content: str) -> None` — store original JSON
  - `retrieve(hash: str, ttl_days: int = 30) -> str | None` — get cached JSON (TTL check, None if expired/missing)
  - `evict_expired(ttl_days: int = 30) -> int` — GC old entries, return count evicted
- `crush_marker(dropped_count: int, hash_val: str) -> dict` — sentinel object: `{"_compressed": "N rows offloaded — ask `memo retrieve <<memo-crush:HASH>>` for full"}`
- Flags in `flags_capture.py`:
  - `flag_crusher_enabled() -> bool` (default: True)
  - `flag_crusher_keep_ratio() -> float` (default: 0.2, clamped [0.05, 1.0])
  - `flag_crusher_cache_ttl_days() -> int` (default: 30)

**Global Constraints:**
- Token gate requirement: Wave 1 blocks on real token measurement (baseline vs enabled, ≥5% improvement) before `git tag v2.13.0`
- Backward-compat: All new flags default to current behavior or OFF
- File collision: Sequential execution only (no parallel worktrees)
- Measurement base: Use `token_meter.py` (actual LLM API tokens), not estimates
- Rollback: Flags disable independently; caches are append-only

**Key Details:**
- CrushCache stores metadata + content as JSON: `{"ts": "2026-07-07T...", "content": "..."}`
- TTL enforcement happens at retrieval time (lazy eviction)
- `evict_expired()` is for `memo maintain` to call during GC
- Sentinel marker format is exact (copy verbatim)
- Use `datetime.fromisoformat()` + UTC for timestamp handling
- All code Python 3.10+, use `from __future__ import annotations`

**Steps:** Follow the plan exactly (Section 1.1–1.3). Write test first (RED), then code (GREEN). Run tests after each step. Commit after each step's tests pass.

**Test locations:**
- Unit tests for CrushCache: `test_crush_cache_stores_and_retrieves`, `test_crush_cache_ttl_expiration`, `test_crush_marker_format`, etc.
- Flag tests: `test_flag_crusher_enabled`, etc.

**Report:** Write to `.superpowers/sdd/task-1-report.md` with sections:
- Completed (list of files created/modified)
- Tests (command run + output)
- Commits (git log output)
- Self-review (did you miss anything?)
- Concerns (if any)

---

**Read the plan's Task 1 section for exact code, step-by-step requirements, and test commands.**
