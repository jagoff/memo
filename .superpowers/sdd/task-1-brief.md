### Task 1: Quality Signals And Pure Reranker

**Files:**
- Create: `src/memo/quality.py`
- Modify: `src/memo/flags_search.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Produces: `QualityDecision`, `classify_quality(hit: Any) -> QualityDecision`, `apply_quality_rerank(hits: list[Any], *, explain: dict[str, dict[str, Any]] | None = None) -> list[Any]`
- Consumes: hit objects with `id`, `score`, `type`, `tags`, `extra`, `verification_state`, and `verified_at` attributes.

- [ ] **Step 1: Write failing pure-quality tests**

Add `tests/test_quality.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo.quality import apply_quality_rerank, classify_quality
from memo.verification import VerificationState


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = "durable body"
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    verified_at: int | None = None


def test_classify_quality_marks_superseded_as_stale() -> None:
    decision = classify_quality(_Hit("old", 0.9, extra={"superseded_by": "new"}))
    assert decision.bucket == "stale_or_conflicting"
    assert "superseded_by" in decision.reasons
    assert decision.multiplier < 1.0


def test_classify_quality_boosts_verified_and_supported_hit() -> None:
    hit = _Hit(
        "good",
        0.5,
        extra={"support_count": 3, "roi_score": 1.3},
        verification_state=VerificationState.VERIFIED,
    )
    decision = classify_quality(hit)
    assert decision.bucket == "current"
    assert "verified" in decision.reasons
    assert "support_count" in decision.reasons
    assert decision.multiplier > 1.0


def test_apply_quality_rerank_demotes_stale_but_keeps_it_visible() -> None:
    old = _Hit("old", 0.9, extra={"superseded_by": "new"})
    current = _Hit("new", 0.7, verification_state=VerificationState.VERIFIED)
    out = apply_quality_rerank([old, current])
    assert [h.id for h in out] == ["new", "old"]


def test_apply_quality_rerank_populates_explain() -> None:
    explain: dict[str, dict[str, Any]] = {}
    out = apply_quality_rerank(
        [_Hit("old", 0.9, extra={"invalidated": True}), _Hit("new", 0.7)],
        explain=explain,
    )
    assert [h.id for h in out] == ["new", "old"]
    assert explain["old"]["quality_bucket"] == "stale_or_conflicting"
    assert explain["old"]["quality_multiplier"] < 1.0
    assert "invalidated" in explain["old"]["quality_reasons"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --no-sync pytest tests/test_quality.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.quality'`.

- [ ] **Step 3: Add quality module**

Create `src/memo/quality.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any

from memo.verification import VerificationState


@dataclass(frozen=True)
class QualityDecision:
    bucket: str
    multiplier: float
    reasons: tuple[str, ...]


def _extra(hit: Any) -> dict[str, Any]:
    raw = getattr(hit, "extra", None)
    return raw if isinstance(raw, dict) else {}


def _verification_value(hit: Any) -> str:
    value = getattr(hit, "verification_state", VerificationState.UNVERIFIED)
    return getattr(value, "value", str(value))


def classify_quality(hit: Any) -> QualityDecision:
    extra = _extra(hit)
    reasons: list[str] = []
    multiplier = 1.0
    bucket = "current"

    if extra.get("superseded_by"):
        reasons.append("superseded_by")
        multiplier *= 0.35
        bucket = "stale_or_conflicting"
    if extra.get("invalidated") or extra.get("invalidated_at"):
        reasons.append("invalidated")
        multiplier *= 0.25
        bucket = "stale_or_conflicting"
    if extra.get("contradiction_status") in {"lost", "resolved_loser", "kept_other"}:
        reasons.append("contradiction_loser")
        multiplier *= 0.35
        bucket = "stale_or_conflicting"
    if bool(extra.get("secret")) or "secret" in set(getattr(hit, "tags", []) or []):
        reasons.append("sensitive")

    verification = _verification_value(hit)
    if verification == "verified":
        reasons.append("verified")
        multiplier *= 1.10
    elif verification in {"rejected", "invalid"}:
        reasons.append("verification_rejected")
        multiplier *= 0.30
        bucket = "stale_or_conflicting"

    support_count = int(extra.get("support_count") or 0)
    if support_count > 0:
        reasons.append("support_count")
        multiplier *= min(1.20, 1.0 + support_count * 0.03)

    roi_score = extra.get("roi_score")
    if isinstance(roi_score, (int, float)):
        if roi_score > 1.0:
            reasons.append("positive_roi")
            multiplier *= min(1.15, float(roi_score))
        elif roi_score < 0.5:
            reasons.append("low_roi")
            multiplier *= max(0.50, float(roi_score))

    if extra.get("canonical_id") or extra.get("synthesis_source_memories"):
        reasons.append("canonical_or_synthesis")
        multiplier *= 1.05

    return QualityDecision(bucket=bucket, multiplier=round(multiplier, 6), reasons=tuple(reasons))


def apply_quality_rerank(
    hits: list[Any],
    *,
    explain: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    scored: list[tuple[float, int, Any, QualityDecision]] = []
    for index, hit in enumerate(hits):
        decision = classify_quality(hit)
        base = float(getattr(hit, "score", None) or 0.0)
        score = base * decision.multiplier
        if explain is not None:
            hid = str(getattr(hit, "id", ""))
            entry = explain.setdefault(hid, {})
            entry["quality_bucket"] = decision.bucket
            entry["quality_multiplier"] = decision.multiplier
            entry["quality_reasons"] = list(decision.reasons)
            entry["quality_score"] = score
        try:
            hit = dc_replace(hit, score=score)
        except TypeError:
            setattr(hit, "score", score)
        scored.append((score, -index, hit, decision))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [hit for _score, _index, hit, _decision in scored]
```

- [ ] **Step 4: Register search flags**

In `src/memo/flags_search.py`, add specs near the other search ranking flags:

```python
    _spec(
        "MEMO_QUALITY_RERANK",
        "bool",
        False,
        "search",
        "Enable quality-aware post-retrieval reranking for explicit search/ask paths. "
        "Demotes invalidated/superseded/contradicted hits and boosts verified/supported hits. "
        "Default off to preserve ranking baselines.",
    ),
    _spec(
        "MEMO_CONTEXT_PACK",
        "bool",
        False,
        "search",
        "Enable context-pack construction for memo ask and explicit context-pack tools. "
        "Default off; ambient recall does not use context packs.",
    ),
```

- [ ] **Step 5: Run focused tests**

Run: `uv run --no-sync pytest tests/test_quality.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/quality.py src/memo/flags_search.py tests/test_quality.py
git commit -m "feat: add quality signal reranker"
```

---

