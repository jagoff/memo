# Wave 1 Token Economy Gating Checklist

Before shipping memo v2.13.0, complete these gates to verify:
1. Functional correctness (all tests pass, no regressions)
2. Token savings verified (≥5% actual reduction vs baseline)
3. Rollback readiness (all flags independently disableable)
4. MCP + CLI integration (crushing, retrieval, verbosity steering work end-to-end)

---

## 1. Functional Tests (LOCAL)

- [ ] **Wave 1 integration tests pass:**
  ```bash
  cd ~/repos/memo
  pytest tests/test_token_economy_wave1.py -v
  # Expected: 15+ tests PASS
  # - test_crush_cache_* (8 tests)
  # - test_crush_marker_* (2 tests)
  # - test_maybe_inject_verbosity_* (2 tests)
  # - test_flag_recall_verbosity_level (1 test)
  # - test_maybe_crush_json_capture_* (4+ tests)
  # - test_retrieve_mcp_tool_* (3 tests)
  # - test_wave1_end_to_end_crusher_and_verbosity (1 test) [NEW]
  # - test_wave1_flags_integration (1 test) [NEW]
  ```

- [ ] **No regressions in existing suite:**
  ```bash
  pytest tests/ -k "not requires_mlx" --ignore=tests/test_token_economy_wave1.py -v
  # Expected: All existing tests still PASS
  ```

- [ ] **Config validation passes:**
  ```bash
  memo config validate
  # Expected: No errors on MEMO_CRUSHER_* and MEMO_RECALL_VERBOSITY_LEVEL flags
  ```

---

## 2. Token Measurement (LOCAL, REQUIRES TEST CORPUS)

### Phase 1: Baseline (Week -1) — WITHOUT Wave 1 features

Collect token usage on a representative set of 50+ recall-hook prompts.

- [ ] **Create test prompt corpus** (if not already available):
  ```bash
  # Use existing eval/regression_labels.json prompts
  # Or create a new corpus: eval/test_prompts_wave1.json
  # Format: [{"prompt": "...", "context": "..."}, ...]
  ```

- [ ] **Disable all Wave 1 features:**
  ```bash
  export MEMO_CRUSHER_ENABLED=0
  export MEMO_RECALL_VERBOSITY_LEVEL=0
  ```

- [ ] **Run baseline measurement:**
  ```bash
  python scripts/wave1_token_baseline.py \
    --prompts eval/test_prompts_wave1.json \
    --output baseline_tokens.json
  # Output: baseline_tokens.json with "total" field
  # Record this value: BASELINE_TOTAL
  ```

### Phase 2: Wave 1 Enabled (Week 0) — WITH features ON

- [ ] **Enable Wave 1 features:**
  ```bash
  export MEMO_CRUSHER_ENABLED=1
  export MEMO_RECALL_VERBOSITY_LEVEL=2
  export MEMO_CRUSHER_KEEP_RATIO=0.2  # Optional: tune if needed
  ```

- [ ] **Warm up the embedder (if using recall or hybrid):**
  ```bash
  memo recall "dummy warmup query" > /dev/null
  # Ensures MLX cold-start doesn't affect timing
  ```

- [ ] **Run Wave 1 measurement (same prompts):**
  ```bash
  python scripts/wave1_token_baseline.py \
    --prompts eval/test_prompts_wave1.json \
    --output wave1_tokens.json
  # Output: wave1_tokens.json with "total" field
  # Record this value: WAVE1_TOTAL
  ```

### Phase 3: Gate Decision

- [ ] **Verify token savings gate:**
  ```bash
  # Requirement: WAVE1_TOTAL < 0.95 * BASELINE_TOTAL
  # Example:
  #   BASELINE_TOTAL = 100,000 tokens
  #   WAVE1_TOTAL = 94,000 tokens (94% of baseline)
  #   Gate: 94,000 < 95,000 ✓ PASS
  
  python -c "
  import json
  baseline = json.load(open('baseline_tokens.json'))['total']
  wave1 = json.load(open('wave1_tokens.json'))['total']
  ratio = wave1 / baseline
  gate_pass = ratio < 0.95
  print(f'Baseline: {baseline} tokens')
  print(f'Wave 1: {wave1} tokens ({ratio:.1%})')
  print(f'Gate: {'✓ PASS' if gate_pass else '✗ FAIL'} (need < 95%)')
  "
  ```

- [ ] **Estimate token savings:**
  ```bash
  python -c "
  import json
  baseline = json.load(open('baseline_tokens.json'))['total']
  wave1 = json.load(open('wave1_tokens.json'))['total']
  savings = baseline - wave1
  print(f'Tokens saved: {savings:,} ({savings/baseline:.1%})')
  "
  ```

---

## 3. Feature Verification (LOCAL)

Test each Wave 1 feature independently to confirm behavior:

### L1: Crusher (JSON array compression on ingest)

- [ ] **Crusher enabled → compresses JSON arrays:**
  ```bash
  export MEMO_CRUSHER_ENABLED=1
  
  # Test data: large JSON array
  cat > /tmp/test_data.json << 'EOF'
  [{"id": 1, "text": "row one"}, {"id": 2, "text": "row two"}, ...]  # 100+ rows
  EOF
  
  # Run a capture with crusher enabled
  # Verify: last element has "_compressed" marker
  ```

- [ ] **Crusher disabled → no compression:**
  ```bash
  export MEMO_CRUSHER_ENABLED=0
  
  # Same test data
  # Verify: data is returned as-is, no marker
  ```

- [ ] **Retrieve works:**
  ```bash
  export MEMO_CRUSHER_ENABLED=1
  
  # After capturing crushed JSON, retrieve it
  memo retrieve <<memo-crush:HASH_VALUE>>
  # Expected: original full JSON returned
  ```

### L4: Verbosity Steering (recall output reduction)

- [ ] **Verbosity level 0 (disabled) → no steering:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=0
  
  prompt="Describe the concept."
  # Internally: steering NOT injected
  # Output should be normal length
  ```

- [ ] **Verbosity level 1 → basic steering:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=1
  
  # Steering injected: "Skip preamble", concise language hint
  # Should reduce output by ~5–10%
  ```

- [ ] **Verbosity level 2 → moderate steering:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=2
  
  # Steering injected: more aggressive hints
  # Should reduce output by ~7–15%
  ```

- [ ] **Verbosity level 3 → maximum steering:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=3
  
  # Steering injected: "Minimum tokens", fragments OK
  # Should reduce output by ~15–25%
  ```

---

## 4. Rollback Verification (LOCAL)

Confirm that all flags can be disabled to roll back Wave 1:

- [ ] **Disable crusher:**
  ```bash
  export MEMO_CRUSHER_ENABLED=0
  
  # Capture JSON → verify no compression
  # Recall hook → verify no markers
  ```

- [ ] **Disable verbosity steering:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=0
  
  # Recall hook output → verify no steering injected
  ```

- [ ] **Mixed states work:**
  ```bash
  # Crusher ON, verbosity OFF
  export MEMO_CRUSHER_ENABLED=1
  export MEMO_RECALL_VERBOSITY_LEVEL=0
  # Expected: JSON compressed, recall output normal
  
  # Crusher OFF, verbosity ON
  export MEMO_CRUSHER_ENABLED=0
  export MEMO_RECALL_VERBOSITY_LEVEL=2
  # Expected: JSON not compressed, recall output steered
  ```

---

## 5. Regression Checks (LOCAL)

- [ ] **Crush cache eviction works:**
  ```bash
  memo maintain --dry-run
  # Expected: no errors, cache TTL respected
  ```

- [ ] **Retrieve command handles edge cases:**
  ```bash
  # Invalid marker format
  memo retrieve "invalid-marker"
  # Expected: error message (not crash)
  
  # Missing cache entry
  memo retrieve <<memo-crush:nonexistent>>
  # Expected: error message (cache miss)
  ```

- [ ] **Verbosity levels are byte-stable:**
  ```bash
  # Same prompt + same level should always produce identical output
  
  prompt="Test"
  out1=$(memo recall --prompt="$prompt" 2>/dev/null | head -c 1000)
  out2=$(memo recall --prompt="$prompt" 2>/dev/null | head -c 1000)
  # out1 == out2 (idempotent)
  ```

- [ ] **Flag parsing validates bounds:**
  ```bash
  export MEMO_RECALL_VERBOSITY_LEVEL=99  # Out of bounds
  memo config validate
  # Expected: warning or error about invalid level (clamped to [0, 3])
  ```

---

## 6. Integration Tests (LOCAL)

- [ ] **End-to-end pipeline works:**
  ```bash
  # In conftest/test environment:
  pytest tests/test_token_economy_wave1.py::test_wave1_end_to_end_crusher_and_verbosity -v
  # Expected: PASS
  ```

- [ ] **Flag interaction tests pass:**
  ```bash
  pytest tests/test_token_economy_wave1.py::test_wave1_flags_integration -v
  # Expected: PASS (all 4 flag combinations tested)
  ```

---

## 7. Documentation + Release Notes

- [ ] **Update CHANGELOG.md:**
  ```markdown
  ## [2.13.0] - 2026-07-07

  ### Added (Wave 1 Token Economy)
  - **L1: SmartCrusher** — JSON array compression on ingest (60–92% reduction possible)
    - New `MEMO_CRUSHER_ENABLED` flag (default: ON)
    - Cache-based retrieval via `memo retrieve <<memo-crush:HASH>>`
    - MCP tool: `memo_crush_retrieve`
  - **L4: Verbosity Steering** — Recall output reduction (5–25% possible)
    - New `MEMO_RECALL_VERBOSITY_LEVEL` flag (0–3, default: 0)
    - Byte-stable levels for idempotent output
  
  ### Measured
  - Token baseline: X,XXX tokens (50-prompt corpus)
  - Wave 1 enabled: X,XXX tokens (X% reduction)
  - Gate requirement: ≥5% savings — **✓ PASS**
  
  ### Known Limitations
  - Crusher best on structured data (JSON arrays); minimal impact on prose
  - Verbosity steering reduces token usage at potential cost of detail loss
  - Cache TTL: 30 days (configurable via `MEMO_CRUSHER_CACHE_TTL_DAYS`)
  ```

- [ ] **Update docs/superpowers/specs/** (if relevant):
  - Link to Token Economy spec (links to this checklist)

---

## 8. Pre-Release Commit

- [ ] **All changes committed:**
  ```bash
  git status
  # Expected: clean working tree or only expected files
  
  git log --oneline -10
  # Expected: commit messages follow "feat(wave1): ..." convention
  ```

- [ ] **Commit message captures Wave 1 summary:**
  ```
  feat(wave1): JSON crushing + verbosity steering — token economy enabled
  
  L1 (Crusher): JSON arrays compressed 60–92% on ingest
  L4 (Verbosity): Output reduced 5–25% via steering hints
  
  Token baseline: X,XXX → X,XXX (X% savings) ✓ gate pass
  
  All 15+ tests pass. See: docs/superpowers/plans/wave1_gating_checklist.md
  ```

---

## 9. Release (Tag + PyPI)

- [ ] **Bump version:**
  ```bash
  # Edit pyproject.toml, .claude-plugin/plugin.json, server.json, CHANGELOG.md
  memo release bump 2.13.0
  ```

- [ ] **Create git tag:**
  ```bash
  git tag -a v2.13.0 -m "Wave 1: JSON crushing + verbosity steering"
  git push origin v2.13.0
  ```

- [ ] **Publish to PyPI:**
  ```bash
  uv publish
  # Or: python -m build && twine upload dist/*
  ```

- [ ] **Verify installation:**
  ```bash
  pip install --upgrade mlx-memo
  memo --version
  # Expected: v2.13.0
  ```

---

## 10. Post-Release Monitoring

- [ ] **Monitor for issues** (first 48 hours):
  - Crush cache eviction problems
  - Retrieve command failures
  - Verbosity steering side effects (e.g., lost information)
  - Token savings lower than expected

- [ ] **Customer feedback collected:**
  - Does crushing + retrieval feel natural?
  - Is verbosity steering too aggressive?
  - Any performance regressions?

- [ ] **If regressions found, rollback plan:**
  ```bash
  # Disable Wave 1 features via env variables
  export MEMO_CRUSHER_ENABLED=0
  export MEMO_RECALL_VERBOSITY_LEVEL=0
  
  # Release patch v2.13.1 with this as default until fix
  ```

---

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Feature Owner | — | — | Pending |
| Test Lead | — | — | Pending |
| Release Lead | — | — | Pending |

---

## References

- **Wave 1 Brief:** `.superpowers/sdd/task-4-brief.md`
- **Wave 1 Plan:** `docs/superpowers/plans/2026-07-07-memo-token-economy-wave1.md` (Section "Task 4")
- **Token Baseline Script:** `scripts/wave1_token_baseline.py`
- **Integration Tests:** `tests/test_token_economy_wave1.py` (lines 486+)
- **Token Measurement:** `scripts/wave1_token_baseline.py` (schema: `memo.token_baseline.v1`)
