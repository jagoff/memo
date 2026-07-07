# Task 2: L1 JSON Crushing on Ingest + Retrieval Command

**Context:** Wave 1, Task 2 of 4. Task 1 (crusher infrastructure) is complete. This task wires the crusher into the capture pipeline, adds the retrieve command, and registers an MCP tool for LLM access.

**What to build:** JSON compression on ingest + retrieval command.

**Deliverables:**
1. `maybe_crush_json_capture()` function in `src/memo/capture_core.py`
2. `src/memo/cli_retrieve.py` — new retrieve command module
3. Wire retrieve into `src/memo/cli.py` and `src/memo/server.py`
4. Tests: 5+ covering crusher logic + retrieval

**Interfaces:**

**Consumes from Task 1:**
- `CrushCache(state_dir)` class (cache/retrieve/evict_expired methods)
- `crush_marker(dropped_count, hash_val)` function
- Flags: `flag_crusher_enabled()`, `flag_crusher_keep_ratio()`, `flag_crusher_cache_ttl_days()`

**Produces (for Task 3+):**
- `maybe_crush_json_capture(content: str, context: str, config: Config) -> tuple[str, str | None]`
  - Returns: (crushed_content, crush_hash if crushed else None)
- CLI command: `memo retrieve <<memo-crush:HASH>>`
- MCP tool: `memo_crush_retrieve(hash_marker: str) -> dict`

**Global Constraints:**
- Token gate requirement: Wave 1 must show ≥5% actual token savings before v2.13.0 ship
- Flags default to ON (crusher enabled by default)
- Crusher only activates on JSON arrays ≥10 rows
- Scoring uses 0.5 placeholder (TBD: integrate real BM25+vec scorer)
- Reversible: original JSON stored in cache, marker + hash in crushed output
- No regressions in existing tests

**Key Details:**
- JSON detection: `[...]` structure check, no schema validation
- Keep ratio: default 0.2 (top 20% of rows by score)
- Marker format: `{"_compressed": "N rows offloaded — ask `memo retrieve <<memo-crush:HASH>>` for full"}`
- Retrieve output: JSON `{"original": json_string, "hash": hash_val}`
- Scorer placeholder in code (line 389–399 of plan): call real hybrid_score TBD

**Steps:** Follow the plan exactly (Section Task 2, Step 2.1–2.3). TDD workflow: write tests RED, implement GREEN, commit after each step passes.

**Test locations:**
- `test_maybe_crush_json_capture_detects_json` — JSON detection
- `test_maybe_crush_json_respects_disable_flag` — flag gating
- `test_crush_preserves_structure` — top-K + marker format
- `test_retrieve_command_returns_original` — retrieve via CLI
- `test_mcp_tool_retrieves_cached_json` — MCP tool integration

**Report:** Write to `.superpowers/sdd/task-2-report.md` with sections:
- Completed (files modified/created)
- Tests (command run + output summary)
- Commits (git log output)
- Self-review (spec compliance vs plan, scorer integration status, any concerns)
- Concerns (if any)

---

**Read the plan's Task 2 section (lines 276–559) for exact code, step-by-step requirements, and test commands.**
