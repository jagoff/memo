### Task 2: Wire Quality Rerank Into Explicit Search

**Files:**
- Modify: `src/memo/memory/search_scoring_ops.py`
- Modify: `src/memo/memory/search_ops.py`
- Test: `tests/test_quality_search.py`

**Interfaces:**
- Consumes: `memo.quality.apply_quality_rerank(hits, explain=None)`
- Produces: `_SearchScoringMixin._apply_quality_rerank(results: list[MemoryRecord]) -> list[MemoryRecord]`

- [ ] **Step 1: Write failing search integration tests**

Add `tests/test_quality_search.py`:

```python
from __future__ import annotations

from dataclasses import replace

from memo.memory.record import MemoryRecord
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.verification import VerificationState


def _rec(id_: str, score: float, **extra):
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title=id_,
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="body",
        extra=dict(extra),
        score=score,
    )


class _Harness(_SearchScoringMixin):
    pass


def test_apply_quality_rerank_is_flag_gated(monkeypatch) -> None:
    hits = [_rec("old", 0.9, superseded_by="new"), _rec("new", 0.7)]
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "0")
    assert [h.id for h in _Harness()._apply_quality_rerank(hits)] == ["old", "new"]

    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    out = _Harness()._apply_quality_rerank(hits)
    assert [h.id for h in out] == ["new", "old"]


def test_apply_quality_rerank_boosts_verified(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    verified = replace(_rec("verified", 0.7), verification_state=VerificationState.VERIFIED)
    out = _Harness()._apply_quality_rerank([_rec("plain", 0.72), verified])
    assert out[0].id == "verified"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --no-sync pytest tests/test_quality_search.py -v`

Expected: FAIL with `AttributeError: '_Harness' object has no attribute '_apply_quality_rerank'`.

- [ ] **Step 3: Add search-scoring helper**

In `src/memo/memory/search_scoring_ops.py`, add this method to `_SearchScoringMixin` after `_apply_contradict_penalty`:

```python
    def _apply_quality_rerank(self, results: list[MemoryRecord]) -> list[MemoryRecord]:
        """Quality-aware reranking for explicit search/ask paths.

        Default-off via MEMO_QUALITY_RERANK. Best-effort: malformed optional
        quality metadata never breaks retrieval.
        """
        if not flag_bool("MEMO_QUALITY_RERANK"):
            return results
        try:
            from memo.quality import apply_quality_rerank

            return apply_quality_rerank(results)
        except Exception as exc:
            _log.debug("quality_rerank failed: %s", exc)
            return results
```

- [ ] **Step 4: Wire helper into search pipeline**

In `src/memo/memory/search_ops.py`, after the existing health-score block and before co-recall/reference-floor logic, add:

```python
        if out and flag_bool("MEMO_QUALITY_RERANK"):
            before = len(out)
            out = self._apply_quality_rerank(out)
            _add_trace("quality_rerank", input_count=before, output_count=len(out))
```

- [ ] **Step 5: Run focused tests**

Run: `uv run --no-sync pytest tests/test_quality.py tests/test_quality_search.py -v`

Expected: PASS.

- [ ] **Step 6: Run search trace smoke**

Run: `uv run --no-sync pytest tests/test_cli_debug_recall.py::test_rank_hits_explain_none_path_is_identical -v`

Expected: PASS; this confirms the recall ranking pure path is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/memo/memory/search_scoring_ops.py src/memo/memory/search_ops.py tests/test_quality_search.py
git commit -m "feat: wire quality rerank into search"
```

---

