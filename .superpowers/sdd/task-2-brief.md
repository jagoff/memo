### Task 2: `Nudge` model

**Files:**
- Create: `src/memo/proactive/__init__.py` (empty)
- Create: `src/memo/proactive/nudge.py`
- Test: `tests/test_proactive_nudge.py`

**Interfaces:**
- Produces: `KIND_CONTINUITY`, `KIND_RELIABILITY`, `KIND_DEJAVU`, `KIND_HEALTH`, `KIND_ROI` (str constants); `Nudge` frozen dataclass with fields `id: str`, `kind: str`, `urgency: float`, `value: float`, `title: str`, `detail: str`, `evidence: tuple[str, ...]`, `action: str | None`, `created_at: str`, `ttl_days: int`; classmethod `Nudge.make(kind, *, subject_id, urgency, value, title, evidence, detail="", action=None, created_at, ttl_days=14) -> Nudge` that content-addresses `id = sha256(kind + ":" + subject_id)[:16]` and rejects empty `evidence`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_proactive_nudge.py
import pytest
from memo.proactive.nudge import Nudge, KIND_RELIABILITY


def test_make_hashes_id_and_requires_evidence():
    n = Nudge.make(
        KIND_RELIABILITY, subject_id="abc123", urgency=0.9, value=0.8,
        title="fact superseded", evidence=("abc123",),
        created_at="2026-07-21T00:00:00Z",
    )
    assert n.kind == KIND_RELIABILITY
    assert len(n.id) == 16
    # deterministic content address
    assert n.id == Nudge.make(
        KIND_RELIABILITY, subject_id="abc123", urgency=0.1, value=0.1,
        title="other", evidence=("abc123",), created_at="x",
    ).id


def test_make_rejects_empty_evidence():
    with pytest.raises(ValueError, match="evidence"):
        Nudge.make(KIND_RELIABILITY, subject_id="x", urgency=0.5, value=0.5,
                   title="t", evidence=(), created_at="x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proactive_nudge.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**
```python
# src/memo/proactive/nudge.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass

KIND_CONTINUITY = "continuity"
KIND_RELIABILITY = "reliability"
KIND_DEJAVU = "dejavu"
KIND_HEALTH = "health"
KIND_ROI = "roi"


@dataclass(frozen=True)
class Nudge:
    id: str
    kind: str
    urgency: float
    value: float
    title: str
    evidence: tuple[str, ...]
    created_at: str
    detail: str = ""
    action: str | None = None
    ttl_days: int = 14

    @classmethod
    def make(
        cls,
        kind: str,
        *,
        subject_id: str,
        urgency: float,
        value: float,
        title: str,
        evidence: tuple[str, ...],
        created_at: str,
        detail: str = "",
        action: str | None = None,
        ttl_days: int = 14,
    ) -> Nudge:
        if not evidence:
            raise ValueError("Nudge.evidence must be non-empty (never fabricate)")
        nid = hashlib.sha256(f"{kind}:{subject_id}".encode()).hexdigest()[:16]
        return cls(
            id=nid, kind=kind, urgency=urgency, value=value, title=title,
            evidence=tuple(evidence), created_at=created_at, detail=detail,
            action=action, ttl_days=ttl_days,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_proactive_nudge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/memo/proactive/__init__.py src/memo/proactive/nudge.py tests/test_proactive_nudge.py
git commit -m "feat(proactive): Nudge model (content-addressed, evidence-required)"
```

---

