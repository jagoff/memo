# SmartCrusher Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect reversible JSON compression to live capture while guaranteeing that output is smaller and the original is recoverable.

**Architecture:** `maybe_crush_json_capture` remains a pure-looking fail-open boundary around deterministic scoring and `CrushCache`. `_extract_and_save` transforms only the assistant text sent to extraction while retaining original transcript text for grounding.

**Tech Stack:** Python 3.13, JSON, mean IDF scoring, filesystem cache, pytest.

## Global Constraints

- Preserve the existing uncommitted scorer work in `src/memo/capture_core.py` and `tests/test_token_economy_wave1.py`.
- `MEMO_CRUSHER_ENABLED` remains false by default.
- Compression never increases content and never emits lossy output without verified cache recovery.
- Every production change starts with a focused failing test.

---

### Task 1: Safe compression contract and normalized scoring

**Files:**
- Modify: `src/memo/capture_core.py`
- Test: `tests/test_capture_crusher.py`

**Interfaces:**
- Produces: `_score_rows_by_relevance(rows: list[Any], context: str) -> list[float]`.
- Produces: `maybe_crush_json_capture(content: str, context: str, config: Config) -> tuple[str, str | None]`.

- [ ] **Step 1: Add failing tests for ratio 1.0, ten rows, marker expansion, cache failure, context relevance, length bias, and stable ties**

```python
assert maybe_crush_json_capture(ten_rows, "", cfg) == (ten_rows, None)
assert maybe_crush_json_capture(rows, "", cfg_with_ratio_one) == (rows, None)
assert len(crushed.encode()) <= int(len(original.encode()) * 0.95)
assert maybe_crush_json_capture(original, "needle", cfg)[0] == original_on_cache_failure
assert scores[relevant_index] > scores[noise_index]
assert scores[short_distinctive] > scores[long_boilerplate]
```

- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_capture_crusher.py -v` and observe failures in the old edge cases**

- [ ] **Step 3: Implement mean-IDF base score, bounded context bonus, size-before-cache checks, cache verification, and fail-open exceptions**

```python
base = sum(_idf(token) for token in tokens) / max(len(tokens), 1)
overlap = sum(_idf(token) for token in tokens & ctx_tokens) / max(len(tokens), 1)
bonus = min(overlap, base * 0.25)
candidate = json.dumps(crushed_array + [crush_marker(dropped_count, hash_val)], ensure_ascii=False)
if len(candidate.encode("utf-8")) > len(content.encode("utf-8")) * 0.95:
    return content, None
```

- [ ] **Step 4: Run new crusher tests plus the user's existing Wave 1 tests**

Run: `uv run --no-sync pytest tests/test_capture_crusher.py tests/test_token_economy_wave1.py -v`

- [ ] **Step 5: Commit only production changes and the new test file; leave the user's pre-existing test diff unstaged**

```bash
git add src/memo/capture_core.py tests/test_capture_crusher.py
git commit -m "fix: make SmartCrusher compression fail open"
```

### Task 2: Shared live-capture integration

**Files:**
- Modify: `src/memo/capture_core.py`
- Test: `tests/test_capture_crusher.py`
- Test: `tests/test_capture_core.py`

**Interfaces:**
- Consumes: `maybe_crush_json_capture(content, context, config)`.
- Produces: extraction sees transformed assistant text; grounding sees original assistant text.

- [ ] **Step 1: Add a failing `_extract_and_save` test that records extractor and grounding arguments**

```python
assert extractor_call.assistant_text == crushed
assert grounding_call.source_text == original
assert crusher_call.context == user_text
```

- [ ] **Step 2: Run the single test and observe that the crusher has no live call-site**

- [ ] **Step 3: Invoke `maybe_crush_json_capture` immediately before `extract_insights` and keep original variables for all grounding checks**

```python
assistant_for_extraction, _crush_hash = maybe_crush_json_capture(
    assistant_text, user_text, memory.cfg
)
extracted = extract_insights(helper, model, user_text, assistant_for_extraction)
```

- [ ] **Step 4: Run capture-core, incremental-capture, grounding, and crusher tests**

Run: `uv run --no-sync pytest tests/test_capture_crusher.py tests/test_capture_core.py tests/test_capture_incremental.py tests/test_capture_grounding.py -v`

- [ ] **Step 5: Commit live integration**

```bash
git add src/memo/capture_core.py tests/test_capture_crusher.py tests/test_capture_core.py
git commit -m "feat: use SmartCrusher in shared capture"
```
