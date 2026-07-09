# Memory Quality Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build default-off memory quality ranking, context packs, and reversible quality compaction so memo demotes stale/noisy memories and gives agents composed context instead of loose hits.

**Architecture:** Add pure quality modules first, then wire them into existing retrieval and ask surfaces behind flags. Keep compaction as a separate maintenance pass with preview/apply/undo receipts, preserving Markdown as the source of truth.

**Tech Stack:** Python 3.13, Click, FastMCP, pytest, existing `MemoryRecord`, `Memory.search`, `memo.flags`, `memo maintain`, and MCP registration modules.

## Global Constraints

- Do not replace the existing hybrid/vector/BM25 candidate retrieval path.
- Do not enable new ranking or compaction behavior by default.
- Do not delete Markdown memories.
- Do not compact memories across project or scope boundaries.
- Do not compact encrypted secrets or sensitive memories.
- Do not put context-pack construction on the ambient recall hot path until it has passed latency and quality gates.
- Flags must be registered and accessed through `src/memo/flags.py`; app code must not read `MEMO_*` values with raw environment access.
- Domain failures should use `memo.errors.MemoError` subclasses rather than bare exceptions.
- Tests must use isolated config/state and must not touch the real vault.

---

## File Structure

- Create `src/memo/quality.py`: pure quality signal extraction and reranking over `MemoryRecord`-shaped objects.
- Create `src/memo/context_pack.py`: pure context-pack construction from loaded memory hits and optional repo snippets.
- Create `src/memo/quality_compact.py`: compaction candidate planning, preview payloads, apply helpers, and receipt/undo target extraction.
- Create `src/memo/server_context_pack.py`: MCP read-only tool for explicit context packs.
- Modify `src/memo/flags_search.py`: add `MEMO_QUALITY_RERANK` and `MEMO_CONTEXT_PACK`.
- Modify `src/memo/flags_behavior.py`: add `MEMO_QUALITY_COMPACT`.
- Modify `src/memo/memory/search_scoring_ops.py` and `src/memo/memory/search_ops.py`: wire default-off quality rerank and trace stage.
- Modify `src/memo/memory/ask_ops.py`: use context-pack formatting for `memo ask` when enabled.
- Modify `src/memo/cli_search.py` and `src/memo/cli.py`: add explicit `memo context-pack`.
- Modify `src/memo/server.py`: register the context-pack MCP module under advanced tools.
- Modify `src/memo/cli_maintain.py`: add `memo maintain quality-compact --preview/--apply`.
- Modify tests under `tests/`: add focused unit and integration coverage for every flag-gated behavior.

---

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

### Task 3: Context Pack Builder

**Files:**
- Create: `src/memo/context_pack.py`
- Modify: `src/memo/memory/ask_ops.py`
- Test: `tests/test_context_pack.py`

**Interfaces:**
- Produces: `ContextPack`, `build_context_pack(question: str, hits: list[Any], *, snippet_chars: int, budget_chars: int = 4000) -> ContextPack`
- Consumes: loaded memory hits with `body`, `id`, `title`, `type`, `tags`, `score`, and `extra`

- [ ] **Step 1: Write failing context-pack tests**

Add `tests/test_context_pack.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo.context_pack import build_context_pack


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float
    title: str
    body: str
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def test_build_context_pack_separates_current_and_stale() -> None:
    pack = build_context_pack(
        "what is current?",
        [
            _Hit("new-current-id", 0.8, "Current", "Use the new plan."),
            _Hit("old-stale-id", 0.9, "Old", "Use the old plan.", extra={"superseded_by": "new"}),
        ],
        snippet_chars=80,
    )
    assert [s["id"] for s in pack.current_facts] == ["new-current-id"]
    assert [s["id"] for s in pack.stale_or_conflicting] == ["old-stale-id"]
    assert "current" in pack.summary.lower()
    assert "stale/conflicting" in pack.to_prompt()


def test_build_context_pack_budget_trims_supporting_before_current() -> None:
    hits = [
        _Hit("current", 0.8, "Current", "A" * 200),
        _Hit("support", 0.7, "Support", "B" * 200),
        _Hit("stale", 0.9, "Stale", "C" * 200, extra={"invalidated": True}),
    ]
    pack = build_context_pack("q", hits, snippet_chars=200, budget_chars=500)
    prompt = pack.to_prompt()
    assert "[current]" in prompt
    assert len(prompt) <= 500
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --no-sync pytest tests/test_context_pack.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.context_pack'`.

- [ ] **Step 3: Add context-pack module**

Create `src/memo/context_pack.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memo.quality import classify_quality


@dataclass(frozen=True)
class ContextPack:
    question: str
    summary: str
    current_facts: list[dict[str, Any]]
    supporting_context: list[dict[str, Any]]
    stale_or_conflicting: list[dict[str, Any]]
    omissions: str

    def to_prompt(self) -> str:
        sections = [
            f"Context summary:\n{self.summary}",
            _format_section("Current facts", self.current_facts),
            _format_section("Supporting context", self.supporting_context),
            _format_section("Stale/conflicting context", self.stale_or_conflicting),
        ]
        if self.omissions:
            sections.append(f"Omissions:\n{self.omissions}")
        return "\n\n".join(s for s in sections if s.strip())


def _snippet(hit: Any, snippet_chars: int) -> dict[str, Any]:
    body = str(getattr(hit, "body", "") or "")
    snippet = body[:snippet_chars]
    if len(body) > snippet_chars:
        snippet = snippet.rstrip() + "…"
    return {
        "source": "memory",
        "id": str(getattr(hit, "id", "")),
        "id_short": str(getattr(hit, "id", ""))[:8],
        "title": str(getattr(hit, "title", "")),
        "type": str(getattr(hit, "type", "")),
        "score": getattr(hit, "score", None),
        "snippet": snippet,
        "quality_bucket": classify_quality(hit).bucket,
        "quality_reasons": list(classify_quality(hit).reasons),
    }


def _format_section(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [f"{title}:"]
    for row in rows:
        lines.append(
            f"[{row['id_short']}] title: {row['title']} | type: {row['type']} | "
            f"quality: {row['quality_bucket']}\n{row['snippet']}"
        )
    return "\n\n".join(lines)


def _trim_to_budget(pack: ContextPack, budget_chars: int) -> ContextPack:
    if budget_chars <= 0 or len(pack.to_prompt()) <= budget_chars:
        return pack
    supporting = list(pack.supporting_context)
    stale = list(pack.stale_or_conflicting)
    current = list(pack.current_facts)
    omitted = 0
    while len(ContextPack(pack.question, pack.summary, current, supporting, stale, pack.omissions).to_prompt()) > budget_chars:
        if supporting:
            supporting.pop()
            omitted += 1
        elif stale:
            stale.pop()
            omitted += 1
        elif len(current) > 1:
            current.pop()
            omitted += 1
        else:
            break
    omissions = pack.omissions
    if omitted:
        omissions = f"{omissions}; +{omitted} trimmed by budget" if omissions else f"+{omitted} trimmed by budget"
    return ContextPack(pack.question, pack.summary, current, supporting, stale, omissions)


def build_context_pack(
    question: str,
    hits: list[Any],
    *,
    snippet_chars: int,
    budget_chars: int = 4000,
) -> ContextPack:
    current: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        decision = classify_quality(hit)
        row = _snippet(hit, snippet_chars)
        if decision.bucket == "stale_or_conflicting":
            stale.append(row)
        elif index == 0 or not current:
            current.append(row)
        else:
            supporting.append(row)
    if current and stale:
        summary = "Current context is available; stale/conflicting memories are included only as history."
    elif current:
        summary = "Current context is available from the retrieved memories."
    elif stale:
        summary = "Only stale/conflicting context was retrieved; answer cautiously."
    else:
        summary = "No memory context was retrieved."
    omitted = ""
    pack = ContextPack(question, summary, current, supporting, stale, omitted)
    return _trim_to_budget(pack, budget_chars)
```

- [ ] **Step 4: Wire context pack into ask**

In `src/memo/memory/ask_ops.py`, replace the `user_msg = (...)` construction with:

```python
        if flag_bool("MEMO_CONTEXT_PACK"):
            from memo.context_pack import build_context_pack

            pack = build_context_pack(
                question,
                hits,
                snippet_chars=snippet_chars,
                budget_chars=max(snippet_chars * max(k, 1) + 1200, 2000),
            )
            user_msg = (
                f"User question:\n{question}\n\n"
                f"Relevant context pack ({len(hits)} memories, {appended_repo} repo snippets):\n\n"
                f"{pack.to_prompt()}"
            )
            for source in sources:
                if source.get("source") == "memory":
                    matching = next(
                        (row for row in pack.current_facts + pack.supporting_context + pack.stale_or_conflicting if row["id"] == source["id"]),
                        None,
                    )
                    if matching:
                        source["quality_bucket"] = matching["quality_bucket"]
                        source["quality_reasons"] = matching["quality_reasons"]
        else:
            user_msg = (
                f"User question:\n{question}\n\n"
                f"Relevant context ({len(hits)} memories, {appended_repo} repo snippets):\n\n"
                + "\n---\n".join(snippet_lines)
            )
```

- [ ] **Step 5: Run focused tests**

Run: `uv run --no-sync pytest tests/test_context_pack.py tests/test_quality.py -v`

Expected: PASS.

- [ ] **Step 6: Run ask cache smoke**

Run: `uv run --no-sync pytest tests/test_rag_cache.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/context_pack.py src/memo/memory/ask_ops.py tests/test_context_pack.py
git commit -m "feat: build context packs for ask"
```

---

### Task 4: Explicit CLI And MCP Context-Pack Surface

**Files:**
- Create: `src/memo/server_context_pack.py`
- Modify: `src/memo/server.py`
- Modify: `src/memo/cli_search.py`
- Modify: `src/memo/cli.py`
- Test: `tests/test_context_pack_surface.py`

**Interfaces:**
- Produces: CLI command `memo context-pack QUERY --k N --json`
- Produces: MCP tool `memo_context_pack(question: str, k: int = 5, type_: str | None = None, snippet_chars: int = 800) -> dict[str, Any]`

- [ ] **Step 1: Write failing surface tests**

Add `tests/test_context_pack_surface.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_CONTEXT_PACK": "1",
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_RERANKER_ENABLED": "0",
    }


def test_context_pack_cli_empty_corpus_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["context-pack", "what is current?", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["question"] == "what is current?"
    assert payload["current_facts"] == []
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --no-sync pytest tests/test_context_pack_surface.py -v`

Expected: FAIL with `No such command 'context-pack'`.

- [ ] **Step 3: Add CLI command**

In `src/memo/cli_search.py`, add:

```python
@click.command(name="context-pack")
@click.argument("question")
@click.option("--k", default=5, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Restrict retrieval to one record type.")
@click.option("--snippet-chars", default=800, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def context_pack_cmd(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int,
    as_json: bool,
) -> None:
    """Build an explicit composed context pack without running the LLM."""
    from memo.context_pack import build_context_pack

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    hits = mem.search(
        question,
        limit=k,
        type_=type_,
        mode="hybrid",
        disable_reranker=True,
        read_through=True,
    )
    pack = build_context_pack(question, hits, snippet_chars=snippet_chars)
    payload = {
        "question": pack.question,
        "summary": pack.summary,
        "current_facts": pack.current_facts,
        "supporting_context": pack.supporting_context,
        "stale_or_conflicting": pack.stale_or_conflicting,
        "omissions": pack.omissions,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(pack.to_prompt(), title=f"context-pack: {question[:60]}", border_style="cyan"))
```

In `src/memo/cli.py`, update the import and command registration:

```python
from memo.cli_search import ask, chat_ask, context_pack_cmd, embed_cmd, recall, rerank_cmd, search
```

Add `context-pack` to `_COMMAND_SECTIONS` under Core or Recall, and add the command where other search commands are registered:

```python
cli.add_command(context_pack_cmd)
```

- [ ] **Step 4: Add MCP module**

Create `src/memo/server_context_pack.py`:

```python
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.context_pack import build_context_pack
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_context_pack(
        question: str,
        k: int = 5,
        type_: str | None = None,
        snippet_chars: int = 800,
    ) -> dict[str, Any]:
        """Build a composed context pack for a question without calling the LLM.

        Returns current facts, supporting context, stale/conflicting context, and
        a compact summary. Use this when search hits need interpretation before
        answering.
        """
        hits = memory.search(
            question,
            limit=k,
            type_=type_,
            mode="hybrid",
            disable_reranker=True,
            read_through=True,
        )
        pack = build_context_pack(question, hits, snippet_chars=snippet_chars)
        return {
            "question": pack.question,
            "summary": pack.summary,
            "current_facts": pack.current_facts,
            "supporting_context": pack.supporting_context,
            "stale_or_conflicting": pack.stale_or_conflicting,
            "omissions": pack.omissions,
        }
```

In `src/memo/server.py`, import and register it with advanced tools:

```python
from memo import server_context_pack as _srv_context_pack
```

Inside `if mcp_include_advanced_tools():`, add:

```python
        _srv_context_pack.register(server, memory)
```

- [ ] **Step 5: Run focused tests**

Run: `uv run --no-sync pytest tests/test_context_pack.py tests/test_context_pack_surface.py -v`

Expected: PASS.

- [ ] **Step 6: Run MCP annotation smoke**

Run: `uv run --no-sync pytest tests/test_server_annotations.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/server_context_pack.py src/memo/server.py src/memo/cli_search.py src/memo/cli.py tests/test_context_pack_surface.py
git commit -m "feat: expose context pack surfaces"
```

---

### Task 5: Quality Compaction Preview

**Files:**
- Create: `src/memo/quality_compact.py`
- Modify: `src/memo/flags_behavior.py`
- Modify: `src/memo/cli_maintain.py`
- Test: `tests/test_quality_compact.py`

**Interfaces:**
- Produces: `QualityCompactProposal`
- Produces: `preview_quality_compaction(memory: Any, *, limit: int = 20) -> dict[str, Any]`
- Produces: CLI `memo maintain quality-compact --preview --json`

- [ ] **Step 1: Write failing preview tests**

Add `tests/test_quality_compact.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_QUALITY_COMPACT": "1",
        "MEMO_RERANKER_ENABLED": "0",
    }


def test_quality_compact_preview_empty_corpus_is_read_only(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "preview"
    assert payload["proposals"] == []
    assert payload["applied"] == []
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --no-sync pytest tests/test_quality_compact.py -v`

Expected: FAIL with `No such command 'quality-compact'`.

- [ ] **Step 3: Register compaction flag**

In `src/memo/flags_behavior.py`, add:

```python
    _spec(
        "MEMO_QUALITY_COMPACT",
        "bool",
        False,
        "maintenance",
        "Enable the quality-compaction maintenance command. Default off; preview is read-only and apply is explicit.",
    ),
```

- [ ] **Step 4: Add preview module**

Create `src/memo/quality_compact.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityCompactProposal:
    proposal_id: str
    source_ids: list[str]
    canonical_title: str
    reasons: list[str]
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_ids": list(self.source_ids),
            "canonical_title": self.canonical_title,
            "reasons": list(self.reasons),
            "scope": self.scope,
        }


def _scope_key(record: Any) -> str:
    tags = [str(t) for t in getattr(record, "tags", []) or []]
    project = next((t for t in tags if t.startswith("project:")), "global")
    return project


def preview_quality_compaction(memory: Any, *, limit: int = 20) -> dict[str, Any]:
    """Return read-only quality compaction proposals.

    Initial implementation is conservative: it groups exact canonical ids and
    explicit superseded_by chains only. Near-duplicate clustering can extend
    this later without changing the CLI contract.
    """
    proposals: list[QualityCompactProposal] = []
    rows = memory.list(limit=limit) if hasattr(memory, "list") else []
    by_canonical: dict[str, list[Any]] = {}
    for rec in rows:
        extra = getattr(rec, "extra", {}) or {}
        canonical = extra.get("canonical_id") or extra.get("superseded_by")
        if canonical:
            by_canonical.setdefault(str(canonical), []).append(rec)
    for canonical, sources in by_canonical.items():
        if len(sources) < 1:
            continue
        scopes = {_scope_key(s) for s in sources}
        if len(scopes) != 1:
            continue
        source_ids = [s.id for s in sources]
        proposals.append(
            QualityCompactProposal(
                proposal_id=f"quality-compact-{canonical[:8]}",
                source_ids=source_ids,
                canonical_title=f"Canonical memory {canonical[:8]}",
                reasons=["explicit_canonical_or_superseded_by"],
                scope=next(iter(scopes)),
            )
        )
    return {
        "mode": "preview",
        "proposals": [p.to_dict() for p in proposals],
        "applied": [],
        "errors": [],
    }
```

- [ ] **Step 5: Add maintain subcommand**

In `src/memo/cli_maintain.py`, after `maintain_undo_cmd`, add:

```python
@maintain_cmd.command(name="quality-compact")
@click.option("--preview", is_flag=True, help="Preview proposals without changing memories.")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply proposals and write a receipt.")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def quality_compact_cmd(preview: bool, apply_changes: bool, limit: int, as_json: bool) -> None:
    """Preview or apply quality compaction proposals."""
    from memo.flags import flag_bool as _flag_bool
    from memo.quality_compact import preview_quality_compaction

    if not _flag_bool("MEMO_QUALITY_COMPACT"):
        raise click.ClickException("MEMO_QUALITY_COMPACT=1 is required")
    if apply_changes and preview:
        raise click.ClickException("choose either --preview or --apply")
    if not apply_changes:
        preview = True
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    receipt = preview_quality_compaction(mem, limit=limit)
    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return
    tag = "[dim](preview)[/dim] " if preview else ""
    console.print(f"{tag}[bold]memo maintain quality-compact[/bold] — {len(receipt['proposals'])} proposals")
```

- [ ] **Step 6: Run focused tests**

Run: `uv run --no-sync pytest tests/test_quality_compact.py tests/test_maintain.py::test_dry_run_on_empty_corpus_is_safe_noop -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/quality_compact.py src/memo/flags_behavior.py src/memo/cli_maintain.py tests/test_quality_compact.py
git commit -m "feat: preview quality compaction"
```

---

### Task 6: Quality Compaction Apply And Undo Receipt

**Files:**
- Modify: `src/memo/quality_compact.py`
- Modify: `src/memo/cli_maintain.py`
- Test: `tests/test_quality_compact.py`

**Interfaces:**
- Produces: `apply_quality_compaction(memory: Any, proposals: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]`
- Extends maintain receipts with `quality_compacted`.
- Extends `_undo_targets(receipt)` so `memo maintain undo` restores archived compaction sources.

- [ ] **Step 1: Add failing apply/undo tests**

Append to `tests/test_quality_compact.py`:

```python
def test_quality_compact_apply_writes_receipt_shape(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert "quality_compacted" in payload
    assert payload["errors"] == []
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --no-sync pytest tests/test_quality_compact.py -v`

Expected: FAIL because apply still returns preview mode or lacks `quality_compacted`.

- [ ] **Step 3: Implement apply helper**

In `src/memo/quality_compact.py`, add:

```python
def apply_quality_compaction(
    memory: Any,
    proposals: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    applied: list[dict[str, Any]] = []
    errors: list[str] = []
    for proposal in proposals:
        source_ids = [str(x) for x in proposal.get("source_ids") or []]
        archived: list[str] = []
        for source_id in source_ids:
            if dry_run:
                archived.append(source_id)
                continue
            try:
                ok = memory.lifecycle.archive_memory(
                    source_id,
                    superseded_by=str(proposal.get("proposal_id") or ""),
                )
                if ok:
                    archived.append(source_id)
            except Exception as exc:
                errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
        applied.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "archived_ids": archived,
                "source_ids": source_ids,
            }
        )
    return {"quality_compacted": applied, "errors": errors}
```

- [ ] **Step 4: Wire apply mode and receipt persistence**

In `quality_compact_cmd`, replace the receipt body with:

```python
    preview_receipt = preview_quality_compaction(mem, limit=limit)
    if apply_changes:
        from memo.quality_compact import apply_quality_compaction

        applied = apply_quality_compaction(mem, preview_receipt["proposals"], dry_run=False)
        receipt = {
            "mode": "apply",
            "proposals": preview_receipt["proposals"],
            "applied": applied["quality_compacted"],
            "quality_compacted": applied["quality_compacted"],
            "errors": [*preview_receipt.get("errors", []), *applied.get("errors", [])],
        }
        if not receipt["errors"]:
            d = _state_path(cfg)
            runs_dir = d / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_stamp = str(int(time.time()))
            payload = json.dumps({"ts": time.time(), "run": run_stamp, **receipt}, ensure_ascii=False, indent=2)
            (d / "last.json").write_text(payload, encoding="utf-8")
            (runs_dir / f"{run_stamp}.json").write_text(payload, encoding="utf-8")
    else:
        receipt = preview_receipt
```

In `_undo_targets`, add:

```python
    for q in receipt.get("quality_compacted", []):
        if isinstance(q, dict):
            archived.extend(q.get("archived_ids") or [])
```

- [ ] **Step 5: Run focused tests**

Run: `uv run --no-sync pytest tests/test_quality_compact.py tests/test_maintain.py::test_maintain_undo_cli_dry_run_reads_receipt -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/quality_compact.py src/memo/cli_maintain.py tests/test_quality_compact.py
git commit -m "feat: apply quality compaction with undo"
```

---

### Task 7: Evaluation Metrics And Final Verification

**Files:**
- Modify: `src/memo/eval_recall.py` or the existing eval module that owns recall metrics in this checkout.
- Modify: `tests/test_eval_recall.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces metrics keys `stale_at_k`, `canonical_hit_at_k`, `pack_answerability`, and `compaction_safety` when the relevant labels/metadata are present.

- [ ] **Step 1: Locate eval metric owner**

Run: `rg -n "precision|recall|ndcg|mrr|Recall@|precision@|metrics" src/memo tests/test_eval_recall.py`

Expected: identify the function that computes recall metric dictionaries.

- [ ] **Step 2: Write failing metric tests**

Add focused tests in `tests/test_eval_recall.py` near existing metric tests. Use existing helper style from that file. The core assertions should be:

```python
assert metrics["stale_at_k"] == 0.5
assert metrics["canonical_hit_at_k"] == 1.0
```

Use two fake hits: one with `extra={"superseded_by": "canonical"}` and one with `extra={"canonical_id": "canonical"}`.

- [ ] **Step 3: Implement metric calculation**

In the eval metric owner, compute:

```python
stale_hits = [
    h for h in hits[:k]
    if (getattr(h, "extra", {}) or {}).get("superseded_by")
    or (getattr(h, "extra", {}) or {}).get("invalidated")
]
metrics["stale_at_k"] = len(stale_hits) / max(len(hits[:k]), 1)
canonical_hits = [
    h for h in hits[:k]
    if (getattr(h, "extra", {}) or {}).get("canonical_id")
    or getattr(h, "type", "") in {"synthesis", "profile"}
]
metrics["canonical_hit_at_k"] = 1.0 if canonical_hits else 0.0
```

Add `pack_answerability` and `compaction_safety` only where enough fixture data exists; otherwise set them to `None` and document that they require context-pack/compaction eval labels.

- [ ] **Step 4: Add changelog entry**

In `CHANGELOG.md` under `[Unreleased]`, add:

```markdown
### Added

- Design and gated implementation path for Memory Quality Loop: quality-aware reranking, context packs, and reversible quality compaction.
```

- [ ] **Step 5: Run focused test suite**

Run:

```bash
uv run --no-sync pytest tests/test_quality.py tests/test_quality_search.py tests/test_context_pack.py tests/test_context_pack_surface.py tests/test_quality_compact.py tests/test_eval_recall.py -v
```

Expected: PASS.

- [ ] **Step 6: Run CI-parity checks**

Run:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: all PASS.

- [ ] **Step 7: Run retrieval eval gate**

Run:

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```

Expected: completes successfully. Compare stale/canonical metrics against baseline if a prior result file exists; otherwise record the new baseline in the PR/commit notes.

- [ ] **Step 8: Commit**

```bash
git add src/memo/eval_recall.py tests/test_eval_recall.py CHANGELOG.md
git commit -m "test: add memory quality eval metrics"
```

---

## Self-Review

- Spec coverage: quality signals, quality reranking, context packs, explicit MCP/CLI surface, compact preview/apply/undo, flags, error handling, and evaluation gates are each mapped to a task.
- Red-flag scan: no task uses deferred-work placeholders; every new interface has names, paths, commands, and expected outcomes.
- Type consistency: `QualityDecision`, `ContextPack`, `build_context_pack`, `preview_quality_compaction`, and `apply_quality_compaction` are defined before later tasks consume them.
- Scope check: this remains one implementation plan because each phase is sequential and the later compaction tasks depend on the same quality metadata contract.
