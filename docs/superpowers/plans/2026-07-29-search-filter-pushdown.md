# Search Filter Pushdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent BM25, exact, and fuzzy search from spending their candidate limit on rows excluded by date or tag filters.

**Architecture:** Extend existing store search signatures with the common filters and perform bounded pre-limit filtering at the closest capable layer. FTS5 pushes exact tag exclusions into SQL; offset-safe date filtering and Tantivy filtering use bounded over-fetch plus one shared row predicate. `Memory.search` remains the only orchestration surface; no planner abstraction is introduced.

**Tech Stack:** Python, SQLite FTS5, optional Tantivy, pytest, recall eval.

## Global Constraints

- Do not add `SearchPlan`, `SearchFilter`, a flag, a database, or a ranking change.
- Unfiltered search must preserve current ranking and candidate limits.
- Date comparisons normalize instants to UTC; do not compare mixed-offset ISO text.
- Filtered backend candidate fetch is capped at 1,000 rows.
- Existing final filters remain as defense in depth.
- CI order is Ruff, mypy, then pytest; retrieval changes also run `memo eval recall`.

---

### Task 1: Extend the backend contract and common predicate

**Files:**
- Modify: `src/memo/store/bm25_queries.py`
- Modify: `src/memo/store/queries.py`
- Modify: `src/memo/store/vec_base.py`
- Test: `tests/test_search_date_filters.py`

**Interfaces:**
- Produces: `search_bm25` and `search_fuzzy` optional keyword parameters
  `date_from: str | None`, `date_to: str | None`, and
  `exclude_tags: set[str] | None`.
- Produces: `_parse_filter_ts(value: str | None) -> datetime | None` from
  `bm25_queries.py`, imported by `queries.py`.

- [ ] **Step 1: Remove both Phase 0 `xfail` decorators**

The two crowd-out tests become ordinary failing regression tests.

- [ ] **Step 2: Run them and verify RED**

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_search_date_filters.py::test_bm25_date_filter_does_not_spend_limit_on_ineligible_hits \
  tests/test_search_date_filters.py::test_bm25_tag_filter_does_not_spend_limit_on_ineligible_hits -q
```

Expected: two FAIL results.

- [ ] **Step 3: Add bounded common filtering**

Add:

```python
_FILTER_CANDIDATE_MULT = 64
_FILTER_CANDIDATE_CAP = 1_000


def _parse_filter_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _matches_common_filters(
    row: dict[str, Any],
    *,
    date_from: datetime | None,
    date_to: datetime | None,
    exclude_tags: set[str],
) -> bool:
    if date_from is not None or date_to is not None:
        updated = _parse_filter_ts(str(row.get("updated") or ""))
        if updated is None:
            return False
        if date_from is not None and updated < date_from:
            return False
        if date_to is not None and updated > date_to:
            return False
    return not (exclude_tags & {str(tag) for tag in row.get("tags") or ()})
```

Use `min(max(limit, limit * 64), 1_000)` only when a date/tag filter is active.
Apply `_matches_common_filters` before slicing to `limit`. Add exact-tag
`meta.tags NOT LIKE ?` clauses to FTS5 SQL.

- [ ] **Step 4: Pass common filters through `Memory.search`**

Pass the three keyword arguments in BM25, exact, fuzzy, and both lexical hybrid
legs.

- [ ] **Step 5: Run the adversarial matrix**

Add `exact` and `fuzzy` parameterization to the two tests, add the `mode`
argument to each function, and replace the literal `mode="bm25"` call with
`mode=mode`:

```python
@pytest.mark.parametrize("mode", ["bm25", "exact", "fuzzy"])
```

Run the whole file; expect PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/store/bm25_queries.py src/memo/store/queries.py \
  src/memo/store/vec_base.py src/memo/memory/search_ops.py \
  tests/test_search_date_filters.py
git commit -m "fix: push common filters into lexical search"
```

### Task 2: Prove no unfiltered ranking or resource regression

**Files:**
- Modify: `tests/test_search_date_filters.py`

**Interfaces:**
- Consumes: filtered backend signatures.
- Produces: bounded-fetch and unfiltered-equivalence regression coverage.

- [ ] **Step 1: Add a spy test for the 1,000-row cap**

Use a fake Tantivy index returning requested `k` and assert a filtered
`limit=100` request asks for exactly 1,000, while an unfiltered request asks for
exactly 100.

- [ ] **Step 2: Add unfiltered equivalence**

Seed deterministic rows and assert:

```python
plain = store.search_bm25("stabletoken", limit=10)
explicit_empty = store.search_bm25(
    "stabletoken",
    limit=10,
    date_from=None,
    date_to=None,
    exclude_tags=set(),
)
assert [row["id"] for row in plain] == [row["id"] for row in explicit_empty]
```

- [ ] **Step 3: Run focused search tests**

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_search_date_filters.py tests/test_memory_search.py \
  tests/test_validity_filter.py tests/test_store.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_search_date_filters.py
git commit -m "test: bound lexical filter candidate work"
```

### Task 3: Run retrieval and quality gates

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

- [ ] **Step 1: Run static and unit gates**

```bash
uv run --no-sync ruff check src/memo tests
uv run --no-sync mypy src/memo
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_search_date_filters.py tests/test_memory_search.py \
  tests/test_validity_filter.py tests/test_store.py tests/test_perf.py -q
```

- [ ] **Step 2: Run the committed recall evaluation**

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json
```

Expected: no expected-ID regression. If the local corpus cannot satisfy the
committed label set, record that environment limitation and run the repository's
deterministic eval tests instead; do not claim a live-corpus result.

- [ ] **Step 3: Record the result and commit**

Append:

```markdown
Implementation result: **Admitted and shipped on the feature branch.** Lexical backends
apply common date/tag eligibility before their final result limit with a 1,000-candidate
cap; unfiltered ranking is unchanged and no planner abstraction was added.
```

Then:

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: record lexical filter improvement"
```
