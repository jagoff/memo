# Memo Chat (rescate synapse) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** memo gana un chat web sobre su propia memoria en :8765 (pipeline de calidad + síntesis MLX warm + feedback 👍👎 + gate de regresión), rescatando lean las capacidades del chat de synapse (archivado en `~/repos/_archived/synapse`).

**Architecture:** paquete nuevo `src/memo/chat/` con etapas puras (fusión RRF, normalización por grupo, dedup de chunks, rewrite de follow-ups, multi-query gateado, feedback, floor de relevancia, fulldoc) orquestadas por un generador `chat_stream(memory, ...)` que emite eventos SSE con el contrato que ya consume la UI React de synapse (`web-chat/src/api.ts`). Superficie HTTP = FastAPI (`memo chat serve --port 8765`, extra `[http]`), sirviendo la SPA copiada a `web-chat/`. Síntesis vía el `MLXChat` que `Memory` ya posee (`_ensure_chat()`), un solo modelo warm.

**Tech Stack:** Python ≥3.13, click, FastAPI+uvicorn (extra `[http]`), mlx-lm ≥0.18, pytest; React 18 + Vite 5 (UI copiada, sin cambios de lógica).

## Global Constraints

- Spec fuente: `docs/SPECS/2026-07-30-memo-chat-design.md`. Código fuente de referencia (solo lectura): `~/repos/_archived/synapse`.
- Repo: `~/repos/memo`. Master protegido → trabajar en branch `feat/memo-chat`, PR al final. Commits `<type>: <descripción>`, SIN líneas de atribución.
- Tests: `uv run --no-sync pytest tests/ -x -q` (o el test puntual con `-v`). Lint: `uv run --no-sync ruff check src tests && uv run --no-sync ruff format --check src tests`. Types: `uv run --no-sync mypy src/memo` (si el repo lo corre así; ver `Makefile`/CI si falla).
- TDD estricto por task: test RED → implementación → GREEN → commit.
- Sin deps core nuevas. `fastapi`/`uvicorn` SOLO en el extra `[http]` (ya existen ahí, pyproject `[project.optional-dependencies].http`). En tests, todo lo que importe fastapi usa `pytest.importorskip("fastapi")`.
- Knobs `MEMO_CHAT_*` con defaults de producción (valores del plist synapse): `BASE_K=20`, `RELEVANCE_FLOOR=0.25`, `VOTE_BOOST=1.5`, `SEMANTIC_THRESHOLD=0.75`, `MULTI_QUERY=1`, `MULTI_QUERY_N=2`, `FULLDOC=1`, `ANSWER_MAX_TOKENS=1200`, `SYNTH_HEAD=8`, puerto `8765`. Rewrite de queries: SOLO reglas (prod usaba `rules`; el path LLM no se porta).
- Modelo de síntesis: `memory.cfg.llm_model` (env `MEMO_LLM_MODEL`) — no se agrega knob de modelo propio del chat.
- Estado del chat: `cfg.state_dir / "chat"` (feedback en `chat/feedback/`, sesiones en `chat/sessions/`). `state_dir` default `~/.local/share/memo` (`src/memo/config.py:65`).
- Cero referencias a memflow ni imports de `synapse` (bloqueado por `_RETIRED_IMPORTS`, `src/memo/definitive.py:21`). El código de synapse se LEE y se reescribe lean; nunca se importa ni se copia archivo a archivo (excepción: `web-chat/` UI y el corpus JSON, que se copian como datos).
- Diferido a plan 2 (NO implementar): insight/crystallize, briefing/digest, whatsapp (contacts_alias/whatsapp_live), SSM, rerank cross-encoder, HyDE, query_decompose, multi_hop expansion, OCR. Los endpoints de UI que tocan eso devuelven `501` (ver Task 10).

---

### Task 0: Branch

- [ ] **Step 1:** `cd ~/repos/memo && git fetch origin && git checkout -b feat/memo-chat origin/master`
- [ ] **Step 2:** Verificar limpio: `git status --short` → vacío.

---

### Task 1: Config del chat (`memo/chat/config.py`)

**Files:**
- Create: `src/memo/chat/__init__.py` (vacío, solo docstring)
- Create: `src/memo/chat/config.py`
- Test: `tests/test_chat_config.py`

**Interfaces:**
- Produces: `ChatConfig` dataclass con `load(state_dir: Path) -> ChatConfig`; campos `base_k:int, relevance_floor:float, vote_boost:float, semantic_threshold:float, multi_query:bool, multi_query_n:int, fulldoc:bool, answer_max_tokens:int, synth_head:int, feedback_dir:Path, sessions_dir:Path`. Todas las tasks posteriores consumen esto.

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_config.py
from pathlib import Path

from memo.chat.config import ChatConfig


def test_defaults_match_production(tmp_path: Path) -> None:
    cfg = ChatConfig.load(tmp_path)
    assert cfg.base_k == 20
    assert cfg.relevance_floor == 0.25
    assert cfg.vote_boost == 1.5
    assert cfg.semantic_threshold == 0.75
    assert cfg.multi_query is True
    assert cfg.multi_query_n == 2
    assert cfg.fulldoc is True
    assert cfg.answer_max_tokens == 1200
    assert cfg.synth_head == 8
    assert cfg.feedback_dir == tmp_path / "chat" / "feedback"
    assert cfg.sessions_dir == tmp_path / "chat" / "sessions"


def test_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_BASE_K", "5")
    monkeypatch.setenv("MEMO_CHAT_MULTI_QUERY", "0")
    monkeypatch.setenv("MEMO_CHAT_RELEVANCE_FLOOR", "0.4")
    cfg = ChatConfig.load(tmp_path)
    assert cfg.base_k == 5
    assert cfg.multi_query is False
    assert cfg.relevance_floor == 0.4
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_chat_config.py -v` → FAIL (`ModuleNotFoundError: memo.chat`).
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/config.py
"""Chat knobs — production synapse plist values baked in as code defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ChatConfig:
    base_k: int
    relevance_floor: float
    vote_boost: float
    semantic_threshold: float
    multi_query: bool
    multi_query_n: int
    fulldoc: bool
    answer_max_tokens: int
    synth_head: int
    feedback_dir: Path
    sessions_dir: Path

    @classmethod
    def load(cls, state_dir: Path) -> "ChatConfig":
        chat_root = state_dir / "chat"
        return cls(
            base_k=_env_int("MEMO_CHAT_BASE_K", 20),
            relevance_floor=_env_float("MEMO_CHAT_RELEVANCE_FLOOR", 0.25),
            vote_boost=_env_float("MEMO_CHAT_VOTE_BOOST", 1.5),
            semantic_threshold=_env_float("MEMO_CHAT_SEMANTIC_THRESHOLD", 0.75),
            multi_query=_env_bool("MEMO_CHAT_MULTI_QUERY", True),
            multi_query_n=_env_int("MEMO_CHAT_MULTI_QUERY_N", 2),
            fulldoc=_env_bool("MEMO_CHAT_FULLDOC", True),
            answer_max_tokens=_env_int("MEMO_CHAT_ANSWER_MAX_TOKENS", 1200),
            synth_head=_env_int("MEMO_CHAT_SYNTH_HEAD", 8),
            feedback_dir=chat_root / "feedback",
            sessions_dir=chat_root / "sessions",
        )
```

`src/memo/chat/__init__.py`:

```python
"""memo.chat — chat pipeline over memory + vault (lean rescue of synapse chat)."""
```

- [ ] **Step 4:** `uv run --no-sync pytest tests/test_chat_config.py -v` → PASS.
- [ ] **Step 5:** `git add src/memo/chat tests/test_chat_config.py && git commit -m "feat(chat): chat config with production defaults"`

---

### Task 2: Fusión RRF + normalización de scores (`memo/chat/fusion.py`)

**Files:**
- Create: `src/memo/chat/fusion.py`
- Test: `tests/test_chat_fusion.py`

**Interfaces:**
- Produces: `rrf_fuse(rankings: list[list[dict]], *, k: int = 60, limit: int | None = None) -> list[dict]`; `normalize_scores(sources: list[dict]) -> list[dict]` (agrega `normalized_score` y `score_group`, no muta la entrada); `source_dedup_key(item: dict) -> str`.
- Formato "source dict" (lingua franca de todo el pipeline, lo producen los adaptadores de Task 9): claves `source` ("memory"|"vault"), `id`, `title`, `type`, `score: float`, `snippet`, y opcionales `path`, `repo_name`, `locator`.

Algoritmos fieles a synapse: RRF con k=60, contribución `1/(k+rank)` (`fusion.py:85` del archivado); min-max por grupo con neutral 0.5 para singleton/cluster apretado, ratio de compresión 0.15 (`score_norm.py:47,116,120`).

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_fusion.py
from memo.chat.fusion import normalize_scores, rrf_fuse, source_dedup_key


def _src(id_: str, score: float, source: str = "memory", **kw) -> dict:
    return {"source": source, "id": id_, "title": id_, "type": "note", "score": score, "snippet": "x", **kw}


def test_rrf_prefers_items_in_both_rankings() -> None:
    a = [_src("shared", 0.9), _src("only-a", 0.8)]
    b = [_src("shared", 0.7, source="vault"), _src("only-b", 0.6, source="vault")]
    fused = rrf_fuse([a, b])
    assert fused[0]["id"] == "shared"
    assert fused[0]["rrf_origins"] == [0, 1]
    assert {s["id"] for s in fused} == {"shared", "only-a", "only-b"}


def test_rrf_limit_and_empty() -> None:
    assert rrf_fuse([]) == []
    fused = rrf_fuse([[_src("a", 1.0), _src("b", 0.5)]], limit=1)
    assert len(fused) == 1


def test_dedup_key_precedence() -> None:
    assert source_dedup_key({"locator": "repo:x:a.md:1-2@abc"}).startswith("loc::")
    assert source_dedup_key({"id": "m1"}) == "id::m1"
    assert source_dedup_key({"title": " Foo "}) == "title::foo"


def test_normalize_minmax_per_group() -> None:
    out = normalize_scores([_src("a", 10.0), _src("b", 5.0), _src("c", 0.0)])
    by_id = {s["id"]: s for s in out}
    assert by_id["a"]["normalized_score"] == 1.0
    assert by_id["c"]["normalized_score"] == 0.0
    assert by_id["a"]["score_group"] == "memory"


def test_normalize_singleton_and_tight_cluster_are_neutral() -> None:
    single = normalize_scores([_src("a", 3.0)])
    assert single[0]["normalized_score"] == 0.5
    tight = normalize_scores([_src("a", 100.0), _src("b", 99.0)])  # span 1 < 100*0.15
    assert all(s["normalized_score"] == 0.5 for s in tight)
```

- [ ] **Step 2:** Correr → FAIL (módulo no existe).
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/fusion.py
"""RRF fusion and per-group score normalization (ported lean from synapse)."""

from __future__ import annotations

from typing import Any

DEFAULT_RRF_K = 60
_COMPRESSION_RATIO = 0.15


def source_dedup_key(item: dict[str, Any]) -> str:
    if item.get("locator"):
        return f"loc::{item['locator']}"
    if item.get("id"):
        return f"id::{item['id']}"
    if item.get("path"):
        return f"path::{item['path']}"
    return f"title::{str(item.get('title', '')).strip().lower()}"


def rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rankings = [r for r in (rankings or []) if r]
    if not rankings:
        return []
    score_by_key: dict[str, float] = {}
    origins_by_key: dict[str, list[int]] = {}
    canonical_by_key: dict[str, dict[str, Any]] = {}
    best_rank_by_key: dict[str, int] = {}
    for list_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict):
                continue
            key = source_dedup_key(item)
            score_by_key[key] = score_by_key.get(key, 0.0) + 1.0 / (k + rank)
            origins_by_key.setdefault(key, []).append(list_idx)
            if key not in canonical_by_key:
                canonical_by_key[key] = item
                best_rank_by_key[key] = rank
            elif rank < best_rank_by_key[key]:
                best_rank_by_key[key] = rank
    fused = []
    for key, score in score_by_key.items():
        canonical = dict(canonical_by_key[key])
        canonical["rrf_score"] = round(score, 6)
        canonical["rrf_origins"] = list(origins_by_key[key])
        fused.append((score, best_rank_by_key[key], canonical))
    fused.sort(key=lambda t: (-t[0], t[1]))
    out = [item for _, _, item in fused]
    return out[: max(0, limit)] if limit is not None else out


def _group_of(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "")
    if source == "memory":
        return "memory"
    if source in {"vault", "repo"} or item.get("type") == "repo":
        return "vault"
    return "other"


def normalize_scores(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(s) for s in sources]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(out):
        groups.setdefault(_group_of(s), []).append(i)
    for label, indices in groups.items():
        scores = [float(out[i].get("score") or 0.0) for i in indices]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if len(indices) == 1 or hi <= lo or span < hi * _COMPRESSION_RATIO:
            for i in indices:
                out[i]["normalized_score"] = 0.5
                out[i]["score_group"] = label
            continue
        for i, s in zip(indices, scores, strict=False):
            out[i]["normalized_score"] = round((s - lo) / span, 6)
            out[i]["score_group"] = label
    return out
```

- [ ] **Step 4:** PASS.
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): RRF fusion and per-group score normalization"`

---

### Task 3: Dedup de fuentes y merge de chunks (`memo/chat/dedup.py`)

**Files:**
- Create: `src/memo/chat/dedup.py`
- Test: `tests/test_chat_dedup.py`

**Interfaces:**
- Produces: `collapse_near_duplicates(sources: list[dict]) -> list[dict]`; `dedup_key(s: dict) -> tuple[str, str, str]`; `normalize_title(title: str) -> str`; `CHUNK_NUM` (regex compilada, la consume Task 8); `score_of(s: dict) -> float` (precedencia `rerank_score > normalized_score > score`; la consumen Tasks 6, 7, 9).

Fiel a synapse: título normalizado quita el marcador `(§N/M)` (`sources_dedup.py:41`); key = `(source, título_normalizado, path_root)`; survivor por mejor score; snippets de chunks se concatenan ordenados por N, cap 6000 (`sources_dedup.py:521,609`); filas sin título no colapsan.

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_dedup.py
from memo.chat.dedup import collapse_near_duplicates, dedup_key, normalize_title, score_of


def _chunk(n: int, total: int, score: float, snippet: str) -> dict:
    return {
        "source": "memory", "id": f"m{n}", "type": "note", "score": score,
        "title": f"Proyecto X (§{n}/{total})", "snippet": snippet, "path": "notes/proyecto-x.md",
    }


def test_normalize_title_strips_chunk_marker() -> None:
    assert normalize_title("Proyecto X (§2/3)") == "proyecto x"
    assert normalize_title("  Plain  ") == "plain"


def test_chunks_collapse_and_merge_ordered() -> None:
    out = collapse_near_duplicates([_chunk(2, 3, 0.9, "parte dos"), _chunk(1, 3, 0.5, "parte uno")])
    assert len(out) == 1
    assert out[0]["snippet"] == "parte uno\n\nparte dos"
    assert out[0]["collapsed_variants"] == 1
    assert out[0]["id"] == "m2"  # survivor = mejor score


def test_untitled_rows_never_collapse() -> None:
    rows = [
        {"source": "memory", "id": "a", "title": "", "score": 1.0, "snippet": "x"},
        {"source": "memory", "id": "b", "title": "", "score": 0.5, "snippet": "y"},
    ]
    assert len(collapse_near_duplicates(rows)) == 2


def test_score_of_precedence() -> None:
    assert score_of({"score": 1.0, "normalized_score": 0.3}) == 0.3
    assert score_of({"score": 1.0, "normalized_score": 0.3, "rerank_score": 0.9}) == 0.9
    assert score_of({}) == 0.0


def test_distinct_docs_stay_separate() -> None:
    a = _chunk(1, 2, 0.9, "a")
    b = dict(_chunk(1, 2, 0.8, "b"), title="Otro doc (§1/2)", path="notes/otro.md")
    assert len(collapse_near_duplicates([a, b])) == 2
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/dedup.py
"""Source dedup: collapse (§N/M) chunk siblings of the same doc into one row."""

from __future__ import annotations

import re
from typing import Any

CHUNK_MARKER = re.compile(r"\s*\(§\d+\s*/\s*\d+\)\s*$")
CHUNK_NUM = re.compile(r"\(§(\d+)\s*/\s*(\d+)\)\s*$")
_MERGED_SNIPPET_MAX = 6000
_SCORE_FIELDS = ("rerank_score", "normalized_score", "score")


def normalize_title(title: str | None) -> str:
    return CHUNK_MARKER.sub("", (title or "").strip()).lower()


def _path_root(s: dict[str, Any]) -> str:
    raw = str(s.get("path") or s.get("locator") or "")
    return CHUNK_MARKER.sub("", raw.split("@", 1)[0]).strip().lower()


def dedup_key(s: dict[str, Any]) -> tuple[str, str, str]:
    return (str(s.get("source") or ""), normalize_title(s.get("title")), _path_root(s))


def score_of(s: dict[str, Any]) -> float:
    for field in _SCORE_FIELDS:
        value = s.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _merge_chunk_snippets(members: list[dict[str, Any]]) -> str | None:
    chunks: list[tuple[int, str]] = []
    for m in members:
        match = CHUNK_NUM.search(str(m.get("title") or ""))
        snippet = str(m.get("snippet") or "").strip()
        if match and snippet:
            chunks.append((int(match.group(1)), snippet))
    if len(chunks) < 2:
        return None
    chunks.sort(key=lambda t: t[0])
    seen: set[str] = set()
    parts: list[str] = []
    for _, text in chunks:
        if text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)[:_MERGED_SNIPPET_MAX]


def collapse_near_duplicates(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[object, list[dict[str, Any]]] = {}
    order: list[object] = []
    for idx, s in enumerate(sources):
        key: object = dedup_key(s) if str(s.get("title") or "").strip() else ("__untitled__", idx)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(s)
    out: list[dict[str, Any]] = []
    for key in order:
        members = grouped[key]
        if len(members) == 1:
            out.append(members[0])
            continue
        survivor = dict(max(members, key=lambda m: (score_of(m), len(str(m.get("snippet") or "")))))
        merged = _merge_chunk_snippets(members)
        if merged:
            survivor["snippet"] = merged
        survivor["collapsed_variants"] = len(members) - 1
        out.append(survivor)
    out.sort(key=score_of, reverse=True)
    return out
```

- [ ] **Step 4:** PASS.
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): source dedup with chunk snippet merge"`

---

### Task 4: Rewrite de follow-ups por reglas (`memo/chat/rewrite.py`)

**Files:**
- Create: `src/memo/chat/rewrite.py`
- Test: `tests/test_chat_rewrite.py`

**Interfaces:**
- Consumes: nada.
- Produces: `rewrite_query(question: str, history: list[dict[str, str]] | None) -> str`. `history` = lista de turnos `{"role": "user"|"assistant", "content": str}` (mismo shape que valida `server_chat.validate_chat_payload`).

Solo el path de reglas de synapse (`query_rewrite.py:584-625`, prod `SYNAPSE_QUERY_REWRITE_MODE=rules`): follow-up de resumen ("resumime/ampliá/más detalle") → tópico del historial; pregunta-info ("qué sabés de X") → X sin fillers; prefijo pronominal ("y él/ella/eso") → inyectar tópico. El path LLM NO se porta.

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_rewrite.py
from memo.chat.rewrite import rewrite_query

_HISTORY = [
    {"role": "user", "content": "qué sabés del proyecto memo daemon"},
    {"role": "assistant", "content": "Memo daemon es ..."},
]


def test_info_question_extracts_topic() -> None:
    assert rewrite_query("qué sabés de avature?", None) == "avature"
    assert rewrite_query("tell me about the recall daemon", None) == "the recall daemon"


def test_summary_followup_uses_history_topic() -> None:
    out = rewrite_query("resumime eso", _HISTORY)
    assert "memo" in out and "daemon" in out


def test_pronoun_prefix_injects_topic() -> None:
    out = rewrite_query("y eso cuándo fue?", _HISTORY)
    assert "daemon" in out and "cuándo" in out


def test_plain_question_passthrough() -> None:
    q = "cómo configuro el embedder"
    assert rewrite_query(q, _HISTORY) == q
    assert rewrite_query("resumime eso", None) == "resumime eso"  # sin historial no hay tópico
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/rewrite.py
"""Rules-only follow-up rewrite (the LLM paraphrase path was not rescued)."""

from __future__ import annotations

import re

_SUMMARY_FOLLOWUP_RE = re.compile(
    r"^\s*(resum[ií](me)?(lo)?|ampli[aá]|expand[ií]|m[aá]s detalles?|contame m[aá]s"
    r"|y de eso|tell me more|summar(y|ize))\b",
    re.IGNORECASE,
)
_INFO_QUESTION_RE = re.compile(
    r"^\s*(qu[eé]\s+(sab[eé]s|sabes|conoc[eé]s)\s+(de|sobre|del?)"
    r"|tell me about|what do you know about)\s+(?P<topic>.+?)[?\s]*$",
    re.IGNORECASE,
)
_PRONOUN_PREFIX_RE = re.compile(
    r"^\s*(y\s+(él|ella|eso|esa|ese|esto)|and\s+(he|she|it|that))\b", re.IGNORECASE
)
_FILLERS = {
    "que", "qué", "del", "de", "la", "el", "los", "las", "un", "una", "sobre",
    "sabes", "sabés", "conocés", "proyecto", "the", "about", "tell", "what",
}
_WORD_RE = re.compile(r"[\wáéíóúñü\-]{3,}", re.IGNORECASE)


def _history_topic(history: list[dict[str, str]] | None) -> str | None:
    if not history:
        return None
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        words = [w for w in _WORD_RE.findall(str(turn.get("content", ""))) if w.lower() not in _FILLERS]
        if words:
            return " ".join(words[:8])
    return None


def rewrite_query(question: str, history: list[dict[str, str]] | None) -> str:
    q = (question or "").strip()
    info = _INFO_QUESTION_RE.match(q)
    if info:
        return info.group("topic").strip()
    if _SUMMARY_FOLLOWUP_RE.match(q):
        topic = _history_topic(history)
        if topic:
            return topic
    if _PRONOUN_PREFIX_RE.match(q):
        topic = _history_topic(history)
        if topic:
            return f"{topic} {q}"
    return q
```

- [ ] **Step 4:** PASS. Si algún regex no matchea el caso del test, ajustar el regex (no el test).
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): rules-based follow-up query rewrite"`

---

### Task 5: Categoría de query + multi-query (`memo/chat/expand.py`)

**Files:**
- Create: `src/memo/chat/expand.py`
- Test: `tests/test_chat_expand.py`

**Interfaces:**
- Consumes: un chat backend con `chat(model, messages, options) -> dict` (protocolo `ChatBackend`, `src/memo/llm.py:43-58`).
- Produces: `classify_query(q: str) -> str` (`"lexical_exact" | "multi_hop" | "semantic_fuzzy"`); `allows_multi_query(category: str) -> bool`; `expand_query(chat, model: str, question: str, *, n: int = 2) -> list[str]` (nunca lanza; `[]` en error).

Fiel a synapse: identificadores (snake_case, ALLCAPS, backticks, comillas) → `lexical_exact` (sin multi-query); dos `?` o conector → `multi_hop`; default `semantic_fuzzy` (`query_category.py:100,129`). Prompt de variantes con `temperature=0.0, max_tokens=400` (`multi_query.py:150,200`).

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_expand.py
from memo.chat.expand import allows_multi_query, classify_query, expand_query


class _FakeChat:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list = []

    def chat(self, model, messages, options=None):
        self.calls.append((model, messages, options))
        return {"message": {"content": self._content}}


def test_classify() -> None:
    assert classify_query("qué hace `rrf_fuse` acá") == "lexical_exact"
    assert classify_query("dónde se define chat_ask_stream") == "lexical_exact"
    assert classify_query("¿quién vino? ¿y cuándo?") == "multi_hop"
    assert classify_query("qué comimos en el cumpleaños") == "semantic_fuzzy"


def test_gate() -> None:
    assert allows_multi_query("semantic_fuzzy") is True
    assert allows_multi_query("multi_hop") is True
    assert allows_multi_query("lexical_exact") is False


def test_expand_parses_variants() -> None:
    chat = _FakeChat('bla {"variants": ["variante uno", "variante dos", "tres"]} bla')
    out = expand_query(chat, "m", "pregunta original", n=2)
    assert out == ["variante uno", "variante dos"]
    assert chat.calls[0][2]["temperature"] == 0.0


def test_expand_malformed_returns_empty() -> None:
    assert expand_query(_FakeChat("no json"), "m", "q") == []

    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("mlx down")

    assert expand_query(_Boom(), "m", "q") == []
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/expand.py
"""Query category gate + LLM multi-query expansion (RRF-fused by the pipeline)."""

from __future__ import annotations

import json
import re
from typing import Any

_LEXICAL_IDENTIFIER_RE = re.compile(
    r"(`[^`]+`|\"[^\"]+\"|\b[a-z0-9]+(?:_[a-z0-9]+)+\b|\b[A-Z0-9]{3,}\b)"
)
_MULTI_HOP_RE = re.compile(
    r"\?.*\?|\by\s+(cu[aá]ndo|d[oó]nde|qui[eé]n|por\s+qu[eé])\b|\band\s+(when|where|who|why)\b",
    re.IGNORECASE | re.DOTALL,
)
_EXPAND_PROMPT = (
    "Sos un asistente de búsqueda. Reformulá la PREGUNTA en {n} variantes DIFERENTES "
    "— cambiá vocabulario, sinónimos, orden, nivel de formalidad, pero mantené la "
    "INTENCIÓN exacta. NO respondas. Cada variante en 1 línea, ≤ 200 caracteres. "
    'Devolvé SOLO JSON: {{"variants": ["...", "..."]}}.\nPREGUNTA: {q}'
)


def classify_query(q: str) -> str:
    if _LEXICAL_IDENTIFIER_RE.search(q or ""):
        return "lexical_exact"
    if _MULTI_HOP_RE.search(q or ""):
        return "multi_hop"
    return "semantic_fuzzy"


def allows_multi_query(category: str) -> bool:
    return category in {"semantic_fuzzy", "multi_hop"}


def _content_of(out: Any) -> str:
    if isinstance(out, dict):
        message = out.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(out.get("content", "") or out.get("response", ""))
    return str(out)


def expand_query(chat: Any, model: str, question: str, *, n: int = 2) -> list[str]:
    try:
        out = chat.chat(
            model,
            [{"role": "user", "content": _EXPAND_PROMPT.format(n=n, q=question)}],
            options={"temperature": 0.0, "max_tokens": 400},
        )
        content = _content_of(out)
        payload = content[content.index("{") : content.rindex("}") + 1]
        variants = json.loads(payload).get("variants", [])
        return [str(v).strip() for v in variants if str(v).strip()][:n]
    except Exception:
        return []
```

- [ ] **Step 4:** PASS. Verificar el shape real de `MLXChat.chat` con `git grep -n '"message"' src/memo/llm.py src/memo/memory/facade.py` — si el dict de retorno no es `{"message": {"content": ...}}`, ajustar `_content_of` (es defensivo, probablemente ya cubre).
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): query category gate and multi-query expansion"`

---

### Task 6: Feedback 👍👎 (`memo/chat/feedback.py`)

**Files:**
- Create: `src/memo/chat/feedback.py`
- Test: `tests/test_chat_feedback.py`

**Interfaces:**
- Produces:
  - `question_key(query: str) -> str` — sha256 hex[:16] de la query lowercased/strip.
  - `SourceVote` dataclass: `created_at: str, question_key: str, query: str, source_id: str, rating: str, query_embedding: list[float], schema: str = "memo.chat.source_vote.v1"`.
  - `ChatFeedback` dataclass: `feedback_id: str, created_at: str, chat_session_id: str, turn_id: str, query: str, answer: str, source_ids: list[str], rating: str, correction_text: str, schema: str = "memo.chat.feedback.v1"`.
  - `SourceVoteStore(root: Path)`: `.record(vote)`, `.load() -> list[SourceVote]`, `.latest_by_pair() -> dict[tuple[str, str], SourceVote]` (última gana por `(question_key, source_id)`).
  - `FeedbackStore(root: Path)`: `.append(fb: ChatFeedback) -> None`, `.load() -> list[ChatFeedback]`.
  - `filter_negative_sources(sources, latest, qkey) -> list[dict]`; `boost_positive_sources(sources, latest, qkey, *, factor) -> list[dict]` (factor clamp [1.0, 5.0], marca `source_vote_boost`, re-sortea por `score_of`); `boost_semantic(sources, query_vec, votes, *, threshold, factor) -> list[dict]` (coseno ≥ threshold contra embeddings de up-votes; saltea fuentes ya boosteadas; vectores de largo distinto se ignoran).
- Consumes: `score_of` de Task 3.

Archivos: `root/events.jsonl` y `root/source_votes.jsonl`, append-only, líneas corruptas se saltean. Fiel a synapse `feedback.py:295,384,469,516,593,567` (boost default 1.5, threshold 0.75, negativos solo exactos).

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_feedback.py
from pathlib import Path

from memo.chat.feedback import (
    SourceVote, SourceVoteStore, boost_positive_sources, boost_semantic,
    filter_negative_sources, question_key,
)


def _vote(qk: str, sid: str, rating: str, emb: list[float] | None = None) -> SourceVote:
    return SourceVote(
        created_at="2026-07-30T00:00:00", question_key=qk, query="q",
        source_id=sid, rating=rating, query_embedding=emb or [],
    )


def _src(sid: str, score: float) -> dict:
    return {"source": "memory", "id": sid, "title": sid, "score": score, "normalized_score": score, "snippet": "x"}


def test_question_key_stable() -> None:
    assert question_key("  Hola Mundo ") == question_key("hola mundo")
    assert len(question_key("x")) == 16


def test_store_roundtrip_and_latest_wins(tmp_path: Path) -> None:
    store = SourceVoteStore(tmp_path)
    store.record(_vote("k1", "s1", "up"))
    store.record(_vote("k1", "s1", "down"))
    (tmp_path / "source_votes.jsonl").open("a").write("not json\n")
    latest = SourceVoteStore(tmp_path).latest_by_pair()
    assert latest[("k1", "s1")].rating == "down"


def test_filter_negative_and_boost_positive() -> None:
    latest = {("k", "bad"): _vote("k", "bad", "down"), ("k", "good"): _vote("k", "good", "up")}
    sources = [_src("bad", 0.9), _src("good", 0.4), _src("meh", 0.5)]
    kept = filter_negative_sources(sources, latest, "k")
    assert [s["id"] for s in kept] == ["good", "meh"]
    boosted = boost_positive_sources(kept, latest, "k", factor=1.5)
    assert boosted[0]["id"] == "good"  # 0.4*1.5=0.6 > 0.5
    assert boosted[0]["source_vote_boost"] == 1.5


def test_boost_factor_clamped() -> None:
    latest = {("k", "a"): _vote("k", "a", "up")}
    out = boost_positive_sources([_src("a", 1.0)], latest, "k", factor=99.0)
    assert out[0]["source_vote_boost"] == 5.0


def test_semantic_boost_by_cosine() -> None:
    votes = [_vote("otra", "a", "up", emb=[1.0, 0.0]), _vote("otra", "b", "down", emb=[1.0, 0.0])]
    out = boost_semantic([_src("a", 0.4), _src("b", 0.4)], [1.0, 0.0], votes, threshold=0.75, factor=1.5)
    by_id = {s["id"]: s for s in out}
    assert by_id["a"].get("source_vote_boost") == 1.5  # up-vote similar
    assert "source_vote_boost" not in by_id["b"]       # down votes no generalizan
    far = boost_semantic([_src("a", 0.4)], [0.0, 1.0], votes, threshold=0.75, factor=1.5)
    assert "source_vote_boost" not in far[0]           # coseno 0 < 0.75
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/feedback.py
"""Chat feedback: append-only vote stores + retrieval boosts (exact + semantic)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memo.chat.dedup import score_of

_MIN_FACTOR, _MAX_FACTOR = 1.0, 5.0


def question_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SourceVote:
    created_at: str
    question_key: str
    query: str
    source_id: str
    rating: str  # "up" | "down"
    query_embedding: list[float] = field(default_factory=list)
    schema: str = "memo.chat.source_vote.v1"


@dataclass(frozen=True)
class ChatFeedback:
    feedback_id: str
    created_at: str
    chat_session_id: str
    turn_id: str
    query: str
    answer: str
    source_ids: list[str]
    rating: str
    correction_text: str = ""
    schema: str = "memo.chat.feedback.v1"


class _JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_dicts(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out


class SourceVoteStore(_JsonlStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "source_votes.jsonl")

    def record(self, vote: SourceVote) -> None:
        self._append(asdict(vote))

    def load(self) -> list[SourceVote]:
        fields = {f for f in SourceVote.__dataclass_fields__}
        return [SourceVote(**{k: v for k, v in d.items() if k in fields}) for d in self._load_dicts()]

    def latest_by_pair(self) -> dict[tuple[str, str], SourceVote]:
        latest: dict[tuple[str, str], SourceVote] = {}
        for vote in self.load():
            latest[(vote.question_key, vote.source_id)] = vote
        return latest


class FeedbackStore(_JsonlStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "events.jsonl")

    def append(self, fb: ChatFeedback) -> None:
        self._append(asdict(fb))

    def load(self) -> list[ChatFeedback]:
        fields = {f for f in ChatFeedback.__dataclass_fields__}
        return [ChatFeedback(**{k: v for k, v in d.items() if k in fields}) for d in self._load_dicts()]


def filter_negative_sources(
    sources: list[dict[str, Any]],
    latest: dict[tuple[str, str], SourceVote],
    qkey: str,
) -> list[dict[str, Any]]:
    out = []
    for s in sources:
        vote = latest.get((qkey, str(s.get("id"))))
        if vote is not None and vote.rating == "down":
            continue
        out.append(s)
    return out


def _boost_field(s: dict[str, Any]) -> str:
    for name in ("rerank_score", "normalized_score", "score"):
        if isinstance(s.get(name), (int, float)):
            return name
    return "score"


def boost_positive_sources(
    sources: list[dict[str, Any]],
    latest: dict[tuple[str, str], SourceVote],
    qkey: str,
    *,
    factor: float,
) -> list[dict[str, Any]]:
    factor = min(max(factor, _MIN_FACTOR), _MAX_FACTOR)
    out = []
    for s in sources:
        vote = latest.get((qkey, str(s.get("id"))))
        if vote is not None and vote.rating == "up":
            s = dict(s)
            name = _boost_field(s)
            s[name] = float(s.get(name) or 0.0) * factor
            s["source_vote_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def boost_semantic(
    sources: list[dict[str, Any]],
    query_vec: list[float],
    votes: list[SourceVote],
    *,
    threshold: float,
    factor: float,
) -> list[dict[str, Any]]:
    factor = min(max(factor, _MIN_FACTOR), _MAX_FACTOR)
    up_by_source: dict[str, list[list[float]]] = {}
    for v in votes:
        if v.rating == "up" and v.query_embedding:
            up_by_source.setdefault(v.source_id, []).append(v.query_embedding)
    out = []
    for s in sources:
        if "source_vote_boost" in s:
            out.append(s)
            continue
        embeddings = up_by_source.get(str(s.get("id")), [])
        if any(_cosine(query_vec, e) >= threshold for e in embeddings):
            s = dict(s)
            name = _boost_field(s)
            s[name] = float(s.get(name) or 0.0) * factor
            s["source_vote_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out
```

- [ ] **Step 4:** PASS.
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): feedback stores with exact and semantic vote boosts"`

---

### Task 7: Síntesis + relevance floor (`memo/chat/synthesis.py`)

**Files:**
- Create: `src/memo/chat/synthesis.py`
- Test: `tests/test_chat_synthesis.py`

**Interfaces:**
- Consumes: `score_of` (Task 3); chat backend con `chat_stream(model, messages, options) -> Iterator[str]` (`src/memo/llm.py:53`).
- Produces: `REFUSAL: str`; `filter_by_relevance(sources, *, floor: float) -> list[dict]`; `build_messages(question, sources, *, today: str) -> list[dict[str, str]]`; `synthesize_stream(chat, model, question, sources, *, max_tokens) -> Iterator[str]` (temperature 0.1).

Fiel a synapse: floor relativo `score >= top * floor` con precedencia `rerank > normalized > score`, nunca vacía el set, no-op con <2 fuentes, exime filas con `keep=True` (`synthesis.py:528,692,718`); contrato de rechazo con string exacto; header con fecha actual (`synthesis.py:1132`).

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_synthesis.py
from memo.chat.synthesis import REFUSAL, build_messages, filter_by_relevance, synthesize_stream


def _src(sid: str, norm: float, **kw) -> dict:
    return {"id": sid, "title": f"T{sid}", "snippet": f"cuerpo {sid}", "normalized_score": norm, **kw}


def test_floor_is_relative_to_top() -> None:
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.3), _src("c", 0.1)], floor=0.25)
    assert [s["id"] for s in kept] == ["a", "b"]  # 0.1 < 1.0*0.25


def test_floor_never_empties_and_keep_exempt() -> None:
    assert len(filter_by_relevance([_src("a", 1.0)], floor=0.25)) == 1  # <2 no-op
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.05, keep=True)], floor=0.25)
    assert {s["id"] for s in kept} == {"a", "b"}


def test_build_messages_contract() -> None:
    messages = build_messages("¿quién es Ana?", [_src("a", 1.0)], today="30/07/2026")
    assert messages[0]["role"] == "system"
    assert "EXCLUSIVAMENTE" in messages[0]["content"]
    assert "30/07/2026" in messages[0]["content"]
    assert REFUSAL in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "PREGUNTA: ¿quién es Ana?" in messages[1]["content"]
    assert "cuerpo a" in messages[1]["content"]


def test_synthesize_stream_passes_options() -> None:
    class _Fake:
        def chat_stream(self, model, messages, options=None):
            assert options["temperature"] == 0.1
            assert options["max_tokens"] == 1200
            yield "hola "
            yield "mundo"

    tokens = list(synthesize_stream(_Fake(), "m", "q", [_src("a", 1.0)], max_tokens=1200))
    assert "".join(tokens) == "hola mundo"
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/synthesis.py
"""Grounded synthesis over curated sources, with the relative relevance floor."""

from __future__ import annotations

from typing import Any, Iterator

from memo.chat.dedup import score_of

REFUSAL = "No encontré esa información en mis notas."


def filter_by_relevance(sources: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    if floor <= 0 or len(sources) < 2:
        return list(sources)
    top = max(score_of(s) for s in sources)
    if top <= 0:
        return list(sources)
    kept = [s for s in sources if s.get("keep") or score_of(s) >= top * floor]
    return kept if kept else [max(sources, key=score_of)]


def build_messages(
    question: str, sources: list[dict[str, Any]], *, today: str
) -> list[dict[str, str]]:
    header = (
        "Sos un asistente RAG de precisión alta. Respondés EXCLUSIVAMENTE con "
        "información que aparece en los SNIPPETS del mensaje del usuario.\n\n"
        f"Fecha actual: {today}. Usá esta fecha para calcular edades y tiempos exactos."
    )
    rules = (
        "Reglas:\n"
        "- Prosa clara; un párrafo por aspecto; sin marcadores [n] ni citas numeradas.\n"
        "- No agregues conocimiento externo a los SNIPPETS.\n"
        f'- Si los SNIPPETS no responden la pregunta, respondé exactamente: "{REFUSAL}"'
    )
    snippets = "\n\n".join(
        f"[{i + 1}] {s.get('title', '')}\n{s.get('snippet', '')}" for i, s in enumerate(sources)
    )
    return [
        {"role": "system", "content": f"{header}\n\n{rules}"},
        {"role": "user", "content": f"PREGUNTA: {question}\n\nSNIPPETS:\n{snippets}"},
    ]


def synthesize_stream(
    chat: Any,
    model: str,
    question: str,
    sources: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> Iterator[str]:
    from datetime import date

    messages = build_messages(question, sources, today=date.today().strftime("%d/%m/%Y"))
    yield from chat.chat_stream(
        model, messages, options={"temperature": 0.1, "max_tokens": max_tokens}
    )
```

- [ ] **Step 4:** PASS.
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): grounded synthesis with relative relevance floor"`

---

### Task 8: Fulldoc inline (`memo/chat/fulldoc.py`)

**Files:**
- Create: `src/memo/chat/fulldoc.py`
- Test: `tests/test_chat_fulldoc.py`

**Interfaces:**
- Consumes: `dedup_key`, `CHUNK_NUM`, `normalize_title` (Task 3); `memory.repo_get_file(repo: str, path: str, *, start=None, end=None) -> dict | None` (`src/memo/memory/repo_ops.py:113`).
- Produces: `dominant_doc_group(sources, *, min_share: float = 0.6, min_chunks: int = 2) -> list[dict] | None` (se evalúa sobre la lista fusionada PRE-dedup); `resolve_fulldoc(memory, members: list[dict]) -> dict | None` con shape `{"title": str, "text": str, "fulldoc_source": "repo"|"memory"}`.

Fiel a synapse: dispara cuando un doc posee ≥0.6 de los hits y ≥2 chunks (`docgroup`, gates FULLDOC_MIN_SHARE/MIN_CHUNKS); vault → archivo completo verbatim; nota de memoria → reensamble SOLO si están todos los chunks `(§n/N)` (`stream.py:1497-1562`).

- [ ] **Step 1: Test que falla**

```python
# tests/test_chat_fulldoc.py
from memo.chat.fulldoc import dominant_doc_group, resolve_fulldoc


def _chunk(n: int, total: int, path: str = "notes/x.md") -> dict:
    return {
        "source": "memory", "id": f"m{n}", "type": "note", "score": 1.0,
        "title": f"Doc X (§{n}/{total})", "snippet": f"parte {n}", "path": path,
    }


def test_dominant_requires_share_and_chunks() -> None:
    group = dominant_doc_group([_chunk(1, 2), _chunk(2, 2), {"source": "vault", "id": "v", "title": "otro", "score": 0.1, "snippet": "y", "path": "a.md"}])
    assert group is not None and len(group) == 2
    assert dominant_doc_group([_chunk(1, 2)]) is None  # <2 chunks
    others = [{"source": "vault", "id": f"v{i}", "title": f"t{i}", "score": 0.1, "snippet": "y", "path": f"{i}.md"} for i in range(4)]
    assert dominant_doc_group([_chunk(1, 2), _chunk(2, 2), *others]) is None  # share 2/6 < 0.6


def test_resolve_memory_note_requires_all_chunks() -> None:
    class _Mem:
        def repo_get_file(self, repo, path, *, start=None, end=None):
            raise AssertionError("no debe llamarse para notas de memoria")

    doc = resolve_fulldoc(_Mem(), [_chunk(2, 2), _chunk(1, 2)])
    assert doc == {"title": "doc x", "text": "parte 1\n\nparte 2", "fulldoc_source": "memory"}
    assert resolve_fulldoc(_Mem(), [_chunk(1, 3), _chunk(2, 3)]) is None  # falta §3


def test_resolve_vault_uses_repo_get_file() -> None:
    class _Mem:
        def repo_get_file(self, repo, path, *, start=None, end=None):
            assert (repo, path) == ("vault", "docs/plan.md")
            return {"text": "contenido completo"}

    members = [
        {"source": "vault", "id": "v1", "title": "plan (§1/2)", "score": 1.0, "snippet": "a",
         "path": "docs/plan.md", "repo_name": "vault"},
        {"source": "vault", "id": "v2", "title": "plan (§2/2)", "score": 0.9, "snippet": "b",
         "path": "docs/plan.md", "repo_name": "vault"},
    ]
    doc = resolve_fulldoc(_Mem(), members)
    assert doc is not None and doc["text"] == "contenido completo" and doc["fulldoc_source"] == "repo"
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/fulldoc.py
"""Fulldoc inline: when one doc dominates the hits, answer with the whole doc."""

from __future__ import annotations

from typing import Any

from memo.chat.dedup import CHUNK_NUM, dedup_key, normalize_title

_MIN_SHARE = 0.6
_MIN_CHUNKS = 2


def dominant_doc_group(
    sources: list[dict[str, Any]],
    *,
    min_share: float = _MIN_SHARE,
    min_chunks: int = _MIN_CHUNKS,
) -> list[dict[str, Any]] | None:
    if not sources:
        return None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for s in sources:
        groups.setdefault(dedup_key(s), []).append(s)
    members = max(groups.values(), key=len)
    if len(members) < min_chunks:
        return None
    if len(members) / len(sources) < min_share:
        return None
    return members


def _chunk_numbers(members: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    nums: set[int] = set()
    totals: set[int] = set()
    for m in members:
        match = CHUNK_NUM.search(str(m.get("title") or ""))
        if match:
            nums.add(int(match.group(1)))
            totals.add(int(match.group(2)))
    return nums, totals


def resolve_fulldoc(memory: Any, members: list[dict[str, Any]]) -> dict[str, Any] | None:
    head = members[0]
    if head.get("source") == "vault" and head.get("repo_name") and head.get("path"):
        doc = memory.repo_get_file(str(head["repo_name"]), str(head["path"]))
        if isinstance(doc, dict):
            text = str(doc.get("text") or doc.get("content") or "")
            if text:
                return {
                    "title": normalize_title(head.get("title")),
                    "text": text,
                    "fulldoc_source": "repo",
                }
        return None
    nums, totals = _chunk_numbers(members)
    if len(totals) != 1 or nums != set(range(1, next(iter(totals)) + 1)):
        return None
    ordered = sorted(
        (m for m in members if CHUNK_NUM.search(str(m.get("title") or ""))),
        key=lambda m: int(CHUNK_NUM.search(str(m["title"])).group(1)),  # type: ignore[union-attr]
    )
    text = "\n\n".join(str(m.get("snippet") or "") for m in ordered)
    return {"title": normalize_title(head.get("title")), "text": text, "fulldoc_source": "memory"}
```

- [ ] **Step 4:** PASS. Verificar las claves reales del dict de `repo_get_file`: `git grep -n "def get_file" src/memo/repo_intelligence.py src/memo/repo_index_search.py` y leer el return — si el campo de contenido no es `text`/`content`, agregar la clave real a `_content` en `resolve_fulldoc`.
- [ ] **Step 5:** `git add -A && git commit -m "feat(chat): fulldoc inline for dominant doc groups"`

---

### Task 9: Orquestador (`memo/chat/pipeline.py`)

**Files:**
- Create: `src/memo/chat/pipeline.py`
- Test: `tests/test_chat_pipeline.py`

**Interfaces:**
- Consumes: TODO lo anterior + `memory.search(query, *, limit, mode="hybrid") -> list[MemoryRecord]` (`src/memo/memory/search_ops.py:134`), `memory.repo_search(query, *, limit) -> list[RepoSearchHit]` (`repo_ops.py:86`), `memory._ensure_chat() -> ChatBackend` (`facade.py:254`), `memory.embedder.embed_query(q) -> list[float]` (`embed_base.py:49`), `memory.cfg.llm_model`, `memory.cfg.state_dir`, `boost_for` de `memo.retrieval_boost` (`retrieval_boost.py:126`).
- Produces: `chat_stream(memory, question: str, *, history: list[dict] | None = None, k: int | None = None) -> Iterator[dict]`. Eventos (shape UI de synapse — el contrato definitivo se verifica contra `web-chat/src/api.ts` en Task 11):
  - `{"type": "stage", "stage": "retrieval", "query": <memo_query>}`
  - `{"type": "context", "sources": [<source dict>...]}`
  - `{"type": "token", "text": str}` (N veces)
  - `{"type": "done", "answer": str, "sources": [...], "synthesis_source": str, "total_ms": int}`
  - `{"type": "error", "message": str, "answer_partial": str}` (en fallo de síntesis)
- Adaptadores internos: `_record_to_source(r: MemoryRecord) -> dict` (usa `r.id, r.title, r.type, r.score, r.body[:700] → snippet, r.path`; `source="memory"`); `_hit_to_source(h: RepoSearchHit) -> dict` (usa `h.id, h.path → title, h.score, h.text → snippet, h.repo_name, h.locator`; `source="vault"`).

Orden del pipeline (calcado del fast path de synapse, `stream.py`): rewrite → search+repo_search en paralelo → multi_query (gateado) → RRF → normalize → title_boost → **dominant sobre la lista PRE-dedup** → dedup → feedback (filter/boost/semantic) → `context` → fulldoc o síntesis (floor adentro del head).

- [ ] **Step 1: Test que falla** (fake Memory completo, sin MLX ni DB)

```python
# tests/test_chat_pipeline.py
from types import SimpleNamespace

from memo.chat.pipeline import chat_stream


class _FakeRecord(SimpleNamespace):
    pass


class _FakeChatBackend:
    def chat(self, model, messages, options=None):  # multi_query expansion
        return {"message": {"content": '{"variants": []}'}}

    def chat_stream(self, model, messages, options=None):
        yield "respuesta "
        yield "sintetizada"


class _FakeEmbedder:
    def embed_query(self, q):
        return [1.0, 0.0]


class _FakeMemory:
    def __init__(self, tmp_path):
        self.cfg = SimpleNamespace(llm_model="fake-model", state_dir=tmp_path)
        self.embedder = _FakeEmbedder()

    def search(self, query, *, limit=None, mode="hybrid", **kw):
        return [
            _FakeRecord(id="m1", title="Nota uno", type="note", score=0.9,
                        body="cuerpo de la nota uno", path="notes/uno.md"),
        ]

    def repo_search(self, query, *, limit=10, **kw):
        return [
            SimpleNamespace(id="r1", repo_name="vault", path="docs/dos.md", score=0.7,
                            text="texto del vault", locator="repo:vault:docs/dos.md:1-10@abcd1234"),
        ]

    def repo_get_file(self, repo, path, *, start=None, end=None):
        return None

    def _ensure_chat(self):
        return _FakeChatBackend()


def test_event_sequence_and_shapes(tmp_path) -> None:
    events = list(chat_stream(_FakeMemory(tmp_path), "qué sabés de la nota uno?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stage"
    assert "context" in kinds and "token" in kinds
    assert kinds[-1] == "done"
    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids
    for s in context["sources"]:
        assert {"source", "id", "title", "score", "snippet"} <= set(s)
        assert "normalized_score" in s
    done = events[-1]
    assert done["answer"] == "respuesta sintetizada"
    assert done["total_ms"] >= 0
    assert done["synthesis_source"] == "memo.chat"


def test_synthesis_error_yields_error_event(tmp_path) -> None:
    class _Boom(_FakeChatBackend):
        def chat_stream(self, model, messages, options=None):
            yield "parcial"
            raise RuntimeError("mlx died")

    mem = _FakeMemory(tmp_path)
    mem._ensure_chat = lambda: _Boom()  # type: ignore[method-assign]
    events = list(chat_stream(mem, "pregunta simple"))
    assert events[-1]["type"] == "error"
    assert events[-1]["answer_partial"] == "parcial"
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/chat/pipeline.py
"""Chat pipeline orchestrator: retrieval → quality stages → synthesis, as events."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

from memo.chat.config import ChatConfig
from memo.chat.dedup import collapse_near_duplicates, score_of
from memo.chat.expand import allows_multi_query, classify_query, expand_query
from memo.chat.feedback import (
    SourceVoteStore,
    boost_positive_sources,
    boost_semantic,
    filter_negative_sources,
    question_key,
)
from memo.chat.fulldoc import dominant_doc_group, resolve_fulldoc
from memo.chat.fusion import normalize_scores, rrf_fuse
from memo.chat.rewrite import rewrite_query
from memo.chat.synthesis import filter_by_relevance, synthesize_stream
from memo.retrieval_boost import boost_for

_SNIPPET_CHARS = 700


def _record_to_source(r: Any) -> dict[str, Any]:
    body = str(getattr(r, "body", "") or "")
    return {
        "source": "memory",
        "id": str(getattr(r, "id", "")),
        "title": str(getattr(r, "title", "")),
        "type": str(getattr(r, "type", "")),
        "score": float(getattr(r, "score", None) or 0.0),
        "snippet": body[:_SNIPPET_CHARS],
        "path": str(getattr(r, "path", "") or ""),
    }


def _hit_to_source(h: Any) -> dict[str, Any]:
    return {
        "source": "vault",
        "id": str(getattr(h, "id", "")),
        "title": str(getattr(h, "path", "") or ""),
        "type": "repo",
        "score": float(getattr(h, "score", None) or 0.0),
        "snippet": str(getattr(h, "text", "") or "")[:_SNIPPET_CHARS],
        "path": str(getattr(h, "path", "") or ""),
        "repo_name": str(getattr(h, "repo_name", "") or ""),
        "locator": str(getattr(h, "locator", "") or ""),
    }


def _apply_title_boost(sources: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    out = []
    for s in sources:
        s = dict(s)
        factor = boost_for(
            query=query,
            filename=str(s.get("path") or ""),
            title=str(s.get("title") or ""),
        )
        if factor > 1.0 and isinstance(s.get("normalized_score"), (int, float)):
            s["normalized_score"] = round(float(s["normalized_score"]) * factor, 6)
            s["filename_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out


def chat_stream(
    memory: Any,
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    k: int | None = None,
) -> Iterator[dict[str, Any]]:
    cfg = ChatConfig.load(memory.cfg.state_dir)
    base_k = k or cfg.base_k
    t0 = time.monotonic()

    memo_query = rewrite_query(question, history)
    yield {"type": "stage", "stage": "retrieval", "query": memo_query}

    with ThreadPoolExecutor(max_workers=2) as pool:
        mem_future = pool.submit(memory.search, memo_query, limit=base_k, mode="hybrid")
        vault_future = pool.submit(memory.repo_search, memo_query, limit=base_k)
        mem_sources = [_record_to_source(r) for r in (mem_future.result() or [])]
        vault_sources = [_hit_to_source(h) for h in (vault_future.result() or [])]

    rankings = [mem_sources, vault_sources]
    if cfg.multi_query and allows_multi_query(classify_query(memo_query)):
        variants = expand_query(
            memory._ensure_chat(), memory.cfg.llm_model, memo_query, n=cfg.multi_query_n
        )
        for variant in variants:
            hits = memory.search(variant, limit=base_k, mode="hybrid") or []
            rankings.append([_record_to_source(r) for r in hits])

    fused = normalize_scores(rrf_fuse(rankings, limit=base_k))
    fused = _apply_title_boost(fused, memo_query)
    dominant = dominant_doc_group(fused) if cfg.fulldoc else None

    sources = collapse_near_duplicates(fused)
    store = SourceVoteStore(cfg.feedback_dir)
    latest = store.latest_by_pair()
    qkey = question_key(memo_query)
    sources = filter_negative_sources(sources, latest, qkey)
    sources = boost_positive_sources(sources, latest, qkey, factor=cfg.vote_boost)
    try:
        query_vec = memory.embedder.embed_query(memo_query)
    except Exception:
        query_vec = []
    if query_vec:
        sources = boost_semantic(
            sources, query_vec, store.load(),
            threshold=cfg.semantic_threshold, factor=cfg.vote_boost,
        )

    yield {"type": "context", "sources": sources[:base_k]}

    def _done(answer: str, synthesis_source: str) -> dict[str, Any]:
        return {
            "type": "done",
            "answer": answer,
            "sources": sources[:base_k],
            "synthesis_source": synthesis_source,
            "total_ms": int((time.monotonic() - t0) * 1000),
        }

    if dominant:
        doc = resolve_fulldoc(memory, dominant)
        if doc:
            yield {"type": "token", "text": doc["text"]}
            yield _done(doc["text"], f"memo.fulldoc.{doc['fulldoc_source']}")
            return

    head = filter_by_relevance(sources, floor=cfg.relevance_floor)[: cfg.synth_head]
    parts: list[str] = []
    try:
        for token in synthesize_stream(
            memory._ensure_chat(), memory.cfg.llm_model, question, head,
            max_tokens=cfg.answer_max_tokens,
        ):
            parts.append(token)
            yield {"type": "token", "text": token}
    except Exception:
        yield {"type": "error", "message": "synthesis failed", "answer_partial": "".join(parts)}
        return
    yield _done("".join(parts), "memo.chat")
```

- [ ] **Step 4:** PASS ambos tests.
- [ ] **Step 5:** `uv run --no-sync pytest tests/test_chat_*.py -q` — toda la suite chat en verde.
- [ ] **Step 6:** `git add -A && git commit -m "feat(chat): pipeline orchestrator emitting UI-shaped events"`

---

### Task 10: Sesiones + superficie HTTP (`memo/chat/sessions.py`, `memo/chat/http.py`, `memo chat serve`)

**Files:**
- Create: `src/memo/chat/sessions.py`
- Create: `src/memo/chat/http.py`
- Modify: `src/memo/cli_chat.py` (agregar subcomando `serve` al `chat_group` existente, línea 33)
- Test: `tests/test_chat_sessions.py`, `tests/test_chat_http.py`

**Interfaces:**
- Consumes: `chat_stream` (Task 9), `FeedbackStore`/`SourceVoteStore`/`ChatFeedback`/`SourceVote`/`question_key` (Task 6), `ChatConfig` (Task 1).
- Produces:
  - `SessionStore(root: Path)`: `.append_turn(session_id, role, text) -> None`; `.list_sessions(limit=50) -> list[dict]` (`{"id", "updated", "first_query", "turns"}` orden updated desc); `.get(session_id) -> list[dict]` (turnos `{"role", "text", "ts"}`); `.delete(session_id) -> bool`; `.delete_all() -> int`; `.recent_queries(limit=8) -> list[str]`. **Session ids validados con `^[A-Za-z0-9_-]{1,64}$`** (guard de path traversal); id inválido → `ValueError`.
  - `build_app(memory) -> FastAPI` en `http.py` (import de fastapi ADENTRO de la función). Rutas (contrato de `web-chat/src/api.ts`):
    - `POST /api/ask/stream` — body `{"q": str, "history": list, "k": int|None, "chat_session_id": str|None}` → `StreamingResponse` media_type `text/event-stream`, frames `data: {json}\n\n` con los eventos del pipeline; al terminar persiste user+assistant turn en `SessionStore`.
    - `POST /api/ask` — mismo body, consume el stream y devuelve el evento `done` como JSON.
    - `POST /api/feedback` — body `{trace_id?, chat_session_id, turn_id, query, answer, sources, rating, correction_text?}` → append `ChatFeedback` → `{"ok": true, "feedback_id": str}`.
    - `POST /api/feedback/source` — body `{source_id, query, rating}` → `SourceVote` con `query_embedding = memory.embedder.embed_query(query)` (en fallo del embedder: `[]`) → `{"ok": true}`.
    - `GET /api/sessions?limit=`, `GET /api/sessions/{id}`, `POST /api/sessions/delete` `{session_id}`, `POST /api/sessions/delete-all`, `GET /api/suggestions?limit=` (→ `{"suggestions": recent_queries()}`).
    - `POST /api/memory/delete` y `POST /api/insight/capture` → `501 {"error": "deferred to plan 2"}`.
    - Static: si se pasa `dist`, montar SPA con fallback a `index.html` para rutas no-`/api`.
  - CLI: `memo chat serve --host 127.0.0.1 --port 8765 --dist <path|None>`.

- [ ] **Step 1: Tests que fallan**

```python
# tests/test_chat_sessions.py
import pytest

from memo.chat.sessions import SessionStore


def test_roundtrip_and_listing(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "hola")
    store.append_turn("s1", "assistant", "respuesta")
    store.append_turn("s2", "user", "otra consulta")
    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0]["id"] == "s2"  # más reciente primero
    turns = store.get("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert store.recent_queries() == ["otra consulta", "hola"]


def test_delete(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "x")
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    store.append_turn("a", "user", "1")
    store.append_turn("b", "user", "2")
    assert store.delete_all() == 2


def test_invalid_session_id_rejected(tmp_path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.append_turn("../evil", "user", "x")
    with pytest.raises(ValueError):
        store.get("a/b")
```

```python
# tests/test_chat_http.py
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memo.chat.http import build_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.test_chat_pipeline import _FakeMemory

    memory = _FakeMemory(tmp_path)
    app = build_app(memory)
    return TestClient(app)


def test_ask_stream_sse(client) -> None:
    with client.stream("POST", "/api/ask/stream", json={"q": "hola", "history": []}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())
    frames = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    kinds = [f["type"] for f in frames]
    assert "context" in kinds and "done" in kinds


def test_ask_non_stream(client) -> None:
    resp = client.post("/api/ask", json={"q": "hola"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "done"


def test_feedback_source_roundtrip(client) -> None:
    resp = client.post("/api/feedback/source", json={"source_id": "m1", "query": "hola", "rating": "up"})
    assert resp.status_code == 200 and resp.json()["ok"] is True


def test_deferred_endpoints_501(client) -> None:
    assert client.post("/api/memory/delete", json={}).status_code == 501
    assert client.post("/api/insight/capture", json={}).status_code == 501


def test_sessions_endpoints(client) -> None:
    client.post("/api/ask", json={"q": "hola", "chat_session_id": "s1"})
    sessions = client.get("/api/sessions").json()["sessions"]
    assert any(s["id"] == "s1" for s in sessions)
    assert client.post("/api/sessions/delete", json={"session_id": "s1"}).json()["ok"] is True
```

- [ ] **Step 2:** FAIL. Nota: si `fastapi` no está en el venv de dev, correr `uv sync --extra http --extra dev` (o el equivalente del repo) para tenerlo en tests; `importorskip` cubre CI sin el extra.
- [ ] **Step 3: Implementación de `sessions.py`**

```python
# src/memo/chat/sessions.py
"""Per-session chat history as one JSONL file per session id."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.match(session_id or ""):
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"role": role, "text": text, "ts": time.time()}, ensure_ascii=False) + "\n")

    def get(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        turns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return turns

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows = []
        for path in self.root.glob("*.jsonl"):
            turns = self.get(path.stem)
            first_user = next((t["text"] for t in turns if t.get("role") == "user"), "")
            rows.append({
                "id": path.stem,
                "updated": path.stat().st_mtime,
                "first_query": first_user,
                "turns": len(turns),
            })
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return rows[:limit]

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_all(self) -> int:
        count = 0
        if self.root.exists():
            for path in self.root.glob("*.jsonl"):
                path.unlink()
                count += 1
        return count

    def recent_queries(self, limit: int = 8) -> list[str]:
        queries: list[str] = []
        for row in self.list_sessions(limit=limit * 3):
            for turn in reversed(self.get(row["id"])):
                if turn.get("role") == "user" and turn.get("text"):
                    if turn["text"] not in queries:
                        queries.append(turn["text"])
                    break
        return queries[:limit]
```

- [ ] **Step 4: Implementación de `http.py`**

```python
# src/memo/chat/http.py
"""FastAPI surface for the chat UI. Import-safe without the [http] extra."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def build_app(memory: Any, *, dist: Path | None = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    from memo.chat.config import ChatConfig
    from memo.chat.feedback import (
        ChatFeedback, FeedbackStore, SourceVote, SourceVoteStore, question_key,
    )
    from memo.chat.pipeline import chat_stream
    from memo.chat.sessions import SessionStore

    cfg = ChatConfig.load(memory.cfg.state_dir)
    sessions = SessionStore(cfg.sessions_dir)
    app = FastAPI(title="memo chat", docs_url=None, redoc_url=None)

    def _sse(events: Any) -> Any:
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def _run(body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        question = str(body.get("q") or "").strip()
        session_id = body.get("chat_session_id") or None
        history = body.get("history") or (sessions_history(session_id) if session_id else None)
        events = list(
            chat_stream(memory, question, history=history, k=body.get("k") or None)
        )
        if session_id:
            done = next((e for e in events if e.get("type") == "done"), None)
            sessions.append_turn(session_id, "user", question)
            if done:
                sessions.append_turn(session_id, "assistant", str(done.get("answer", "")))
        return events, session_id

    def sessions_history(session_id: str | None) -> list[dict[str, str]] | None:
        if not session_id:
            return None
        try:
            turns = sessions.get(session_id)
        except ValueError:
            return None
        return [{"role": t.get("role", ""), "content": t.get("text", "")} for t in turns][-12:]

    @app.post("/api/ask/stream")
    async def ask_stream(request: Request) -> Any:
        body = await request.json()
        question = str(body.get("q") or "").strip()
        if not question:
            return JSONResponse({"error": "q required"}, status_code=400)
        session_id = body.get("chat_session_id") or None
        history = body.get("history") or sessions_history(session_id)

        def _generate() -> Any:
            answer = ""
            for event in chat_stream(memory, question, history=history, k=body.get("k") or None):
                if event.get("type") == "done":
                    answer = str(event.get("answer", ""))
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if session_id:
                try:
                    sessions.append_turn(session_id, "user", question)
                    sessions.append_turn(session_id, "assistant", answer)
                except ValueError:
                    pass

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/ask")
    async def ask(request: Request) -> Any:
        body = await request.json()
        if not str(body.get("q") or "").strip():
            return JSONResponse({"error": "q required"}, status_code=400)
        events, _ = _run(body)
        done = next((e for e in reversed(events) if e.get("type") in {"done", "error"}), None)
        return JSONResponse(done or {"type": "error", "message": "no events"})

    @app.post("/api/feedback")
    async def feedback(request: Request) -> Any:
        body = await request.json()
        fb = ChatFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            chat_session_id=str(body.get("chat_session_id") or ""),
            turn_id=str(body.get("turn_id") or ""),
            query=str(body.get("query") or ""),
            answer=str(body.get("answer") or ""),
            source_ids=[str(s.get("id")) for s in body.get("sources") or [] if isinstance(s, dict)],
            rating=str(body.get("rating") or ""),
            correction_text=str(body.get("correction_text") or ""),
        )
        FeedbackStore(cfg.feedback_dir).append(fb)
        return {"ok": True, "feedback_id": fb.feedback_id}

    @app.post("/api/feedback/source")
    async def feedback_source(request: Request) -> Any:
        body = await request.json()
        query = str(body.get("query") or "")
        try:
            embedding = memory.embedder.embed_query(query) if query else []
        except Exception:
            embedding = []
        vote = SourceVote(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            question_key=question_key(query),
            query=query,
            source_id=str(body.get("source_id") or ""),
            rating=str(body.get("rating") or ""),
            query_embedding=list(embedding),
        )
        SourceVoteStore(cfg.feedback_dir).record(vote)
        return {"ok": True}

    @app.get("/api/sessions")
    async def list_sessions(limit: int = 50) -> Any:
        return {"sessions": sessions.list_sessions(limit=limit)}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> Any:
        try:
            return {"id": session_id, "turns": sessions.get(session_id)}
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)

    @app.post("/api/sessions/delete")
    async def delete_session(request: Request) -> Any:
        body = await request.json()
        try:
            return {"ok": sessions.delete(str(body.get("session_id") or ""))}
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)

    @app.post("/api/sessions/delete-all")
    async def delete_all_sessions() -> Any:
        return {"ok": True, "deleted": sessions.delete_all()}

    @app.get("/api/suggestions")
    async def suggestions(limit: int = 8) -> Any:
        return {"suggestions": sessions.recent_queries(limit=limit)}

    @app.post("/api/memory/delete")
    async def memory_delete() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)

    @app.post("/api/insight/capture")
    async def insight_capture() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)

    if dist is not None and (dist / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str) -> Any:
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
```

- [ ] **Step 5: CLI `memo chat serve`** — en `src/memo/cli_chat.py`, debajo del comando `ask`:

```python
@chat_group.command(name="serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--dist", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Directorio dist de la SPA web-chat (opcional).")
def chat_serve(host: str, port: int, dist: Path | None) -> None:
    """Serve the chat UI + API over HTTP (requires the [http] extra)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "chat serve requiere el extra http: uv tool install 'mlx-memo[http]'"
        ) from exc
    from memo.chat.http import build_app

    memory = _build_memory()
    uvicorn.run(build_app(memory, dist=dist), host=host, port=port, log_level="info")
```

Antes de escribirlo, mirar cómo `chat_ask` (`cli_chat.py:39-84`) construye su `Memory` y factorizar/reusar esa construcción como `_build_memory()` (si `chat_ask` usa `Memory(load_config())` directo, definir `_build_memory` con ese mismo código y que ambos lo llamen). Importar `Path` de `pathlib` si no está.

- [ ] **Step 6:** `uv run --no-sync pytest tests/test_chat_sessions.py tests/test_chat_http.py -v` → PASS.
- [ ] **Step 7:** `uv run --no-sync ruff check src/memo/chat tests/ && uv run --no-sync mypy src/memo/chat` → limpio.
- [ ] **Step 8:** `git add -A && git commit -m "feat(chat): HTTP surface, session store and memo chat serve"`

---

### Task 11: UI web-chat

**Files:**
- Create: `web-chat/` (copia del archivado, sin `node_modules`/`dist`)
- Modify (solo si el contrato difiere): `src/memo/chat/pipeline.py` / `src/memo/chat/http.py`

**Interfaces:**
- Consumes: los endpoints de Task 10. **`web-chat/src/api.ts` es la autoridad del contrato** — ante cualquier diferencia de shape, se ajusta el lado Python, no la UI.

- [ ] **Step 1: Copiar**

```bash
cd ~/repos/memo
rsync -a --exclude node_modules --exclude dist ~/repos/_archived/synapse/web-chat/ web-chat/
```

- [ ] **Step 2: Verificar contrato.** Leer `web-chat/src/api.ts` completo. Chequear contra la implementación: paths (`/api/ask/stream`, `/api/feedback`, `/api/feedback/source`, `/api/sessions*`, `/api/suggestions`), shape del body, y la unión de eventos SSE (`context|token|stage|done|insight_proposal|error`) con sus campos (`token.text`, `done.answer`, `done.sources`, campos de cada source que la UI renderiza: título, snippet, score, id). Cualquier campo que la UI espere y el pipeline no emita → agregarlo al evento en `pipeline.py` (con test actualizado en `tests/test_chat_pipeline.py`). `insight_proposal` nunca se emite en v1 (la UI simplemente no muestra ese afford).
- [ ] **Step 3: Build**

```bash
cd web-chat && npm ci && npm run build && cd ..
ls web-chat/dist/index.html
```

- [ ] **Step 4:** Agregar a `.gitignore` del repo: `web-chat/node_modules/` y `web-chat/dist/` (verificar antes si el `.gitignore` ya tiene patrones node; seguir el estilo existente).
- [ ] **Step 5: Smoke manual (requiere MLX local + índice memo real):**

```bash
uv run --no-sync memo chat serve --port 8765 --dist web-chat/dist &
sleep 3
curl -s http://127.0.0.1:8765/ | head -5                    # index.html
curl -sN -X POST http://127.0.0.1:8765/api/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"q": "qué sabés de memo?"}' | head -20               # frames data: {...}
kill %1
```

Verificar en los frames: `stage` → `context` con sources → `token`s → `done`.
- [ ] **Step 6:** `git add web-chat .gitignore && git commit -m "feat(chat): vendor web-chat SPA from archived synapse"`

---

### Task 12: Gate de regresión (`memo eval chat`)

**Files:**
- Create: `eval/chat_regression_corpus.json` (copiado del archivado)
- Create: `src/memo/eval_chat.py`
- Modify: `src/memo/cli_eval.py` (nuevo subcomando en `eval_group`, línea 85)
- Test: `tests/test_eval_chat.py`

**Interfaces:**
- Consumes: `chat_stream` (Task 9), `REFUSAL` (Task 7).
- Produces: `apply_checks(query: dict, done: dict, total_ms: int) -> dict` (pura, en `eval_chat.py`) con shape `{"id", "passed": bool, "checks": [{"check", "passed"}], "total_ms"}`; CLI `memo eval chat [--corpus PATH] [--only ID]` que imprime tabla rich y sale con código 1 si algún query falla.

Formato del corpus (schema synapse `synapse.eval_chat.query.v1`, se preserva): `{"queries": [{"id", "question", "expected_source_ids": [...], "category", "checks": {"require_substrings", "forbid_substrings", "forbid_refusal", "min_sources", ...}}]}`. Se evalúan SOLO los checks declarativos; las métricas ragas/judge de synapse no se portan.

- [ ] **Step 1: Copiar corpus**

```bash
cd ~/repos/memo && mkdir -p eval
cp ~/repos/_archived/synapse/eval/regression_corpus.json eval/chat_regression_corpus.json
python3 -c "import json; d=json.load(open('eval/chat_regression_corpus.json')); print(len(d['queries']), 'queries')"
```

- [ ] **Step 2: Test que falla**

```python
# tests/test_eval_chat.py
from memo.eval_chat import apply_checks


def _done(answer: str, ids: list[str]) -> dict:
    return {"type": "done", "answer": answer, "sources": [{"id": i} for i in ids]}


def test_require_and_forbid_substrings() -> None:
    query = {"id": "q1", "checks": {"require_substrings": ["Avature"], "forbid_substrings": ["lambda"]}}
    ok = apply_checks(query, _done("Avature es una empresa", ["s1"]), 100)
    assert ok["passed"] is True
    bad = apply_checks(query, _done("usa lambda", ["s1"]), 100)
    assert bad["passed"] is False


def test_forbid_refusal_and_min_sources() -> None:
    from memo.chat.synthesis import REFUSAL

    query = {"id": "q2", "checks": {"forbid_refusal": True, "min_sources": 2}}
    assert apply_checks(query, _done(REFUSAL, ["a", "b"]), 10)["passed"] is False
    assert apply_checks(query, _done("respuesta", ["a"]), 10)["passed"] is False
    assert apply_checks(query, _done("respuesta", ["a", "b"]), 10)["passed"] is True


def test_expected_source_hit() -> None:
    query = {"id": "q3", "expected_source_ids": ["dev-PublicCloud"], "checks": {}}
    assert apply_checks(query, _done("x", ["dev-PublicCloudInfrastructure"]), 10)["passed"] is True
    assert apply_checks(query, _done("x", ["otro"]), 10)["passed"] is False
```

- [ ] **Step 3:** FAIL.
- [ ] **Step 4: Implementación `src/memo/eval_chat.py`**

```python
# src/memo/eval_chat.py
"""Declarative regression checks for the chat pipeline (corpus rescued from synapse)."""

from __future__ import annotations

from typing import Any

from memo.chat.synthesis import REFUSAL


def apply_checks(query: dict[str, Any], done: dict[str, Any], total_ms: int) -> dict[str, Any]:
    checks_spec = query.get("checks") or {}
    answer = str(done.get("answer") or "")
    answer_lower = answer.lower()
    source_ids = [str(s.get("id") or "") for s in done.get("sources") or []]
    results: list[dict[str, Any]] = []

    for sub in checks_spec.get("require_substrings") or []:
        results.append({"check": f"require:{sub}", "passed": str(sub).lower() in answer_lower})
    for sub in checks_spec.get("forbid_substrings") or []:
        results.append({"check": f"forbid:{sub}", "passed": str(sub).lower() not in answer_lower})
    if checks_spec.get("forbid_refusal"):
        results.append({"check": "forbid_refusal", "passed": REFUSAL not in answer})
    min_sources = checks_spec.get("min_sources")
    if isinstance(min_sources, int):
        results.append({"check": f"min_sources:{min_sources}", "passed": len(source_ids) >= min_sources})
    expected = query.get("expected_source_ids") or []
    if expected:
        hit = any(e in sid or sid in e for e in map(str, expected) for sid in source_ids if sid)
        results.append({"check": "expected_source_hit", "passed": hit})

    return {
        "id": str(query.get("id") or "?"),
        "passed": all(r["passed"] for r in results) if results else True,
        "checks": results,
        "total_ms": total_ms,
    }
```

- [ ] **Step 5:** PASS el test.
- [ ] **Step 6: Subcomando CLI** — en `src/memo/cli_eval.py`, siguiendo el patrón de `eval_recall_cmd` (`cli_eval.py:283`):

```python
@eval_group.command(name="chat")
@click.option("--corpus", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/chat_regression_corpus.json"), show_default=True)
@click.option("--only", default=None, help="Correr solo el query con este id.")
def eval_chat_cmd(corpus: Path, only: str | None) -> None:
    """Regression gate: run the corpus through the chat pipeline and check outputs."""
    import json as _json
    import time as _time

    from rich.console import Console
    from rich.table import Table

    from memo.chat.pipeline import chat_stream
    from memo.eval_chat import apply_checks

    memory = _build_memory()  # misma construcción que usa eval_recall_cmd; reusar su helper
    data = _json.loads(corpus.read_text(encoding="utf-8"))
    queries = [q for q in data.get("queries", []) if not only or q.get("id") == only]
    rows = []
    for query in queries:
        t0 = _time.monotonic()
        events = list(chat_stream(memory, str(query.get("question") or "")))
        done = next((e for e in reversed(events) if e.get("type") in {"done", "error"}), {})
        rows.append(apply_checks(query, done, int((_time.monotonic() - t0) * 1000)))

    table = Table(title=f"eval chat — {corpus.name}")
    table.add_column("id"); table.add_column("passed"); table.add_column("ms", justify="right"); table.add_column("failed checks")
    for row in rows:
        failed = ", ".join(c["check"] for c in row["checks"] if not c["passed"])
        table.add_row(row["id"], "✅" if row["passed"] else "❌", str(row["total_ms"]), failed)
    Console(force_terminal=True).print(table)
    latencies = sorted(r["total_ms"] for r in rows) or [0]
    Console(force_terminal=True).print(
        f"p50={latencies[len(latencies) // 2]}ms p95={latencies[int(len(latencies) * 0.95) - 1]}ms"
    )
    if any(not r["passed"] for r in rows):
        raise SystemExit(1)
```

Antes de escribirlo, ver cómo `eval_recall_cmd` construye `Memory` y usar exactamente ese mecanismo (si es inline, extraer `_build_memory()` local al módulo).

- [ ] **Step 7: Gate real (requiere MLX + índice):** `uv run --no-sync memo eval chat` → tabla; anotar el resultado. Los queries cuyos `expected_source_ids` referencien memorias que ya no existen en el índice se marcan y se decide por query: actualizar el id esperado o borrar el query del corpus (documentar en el commit). **Criterio de aceptación del plan: todos los queries restantes en verde y p50 ≤ 8000ms warm.**
- [ ] **Step 8:** `git add -A && git commit -m "feat(eval): chat regression gate with rescued corpus"`

---

### Task 13: Migración de estado desde el backup

**Files:**
- Create: `scripts/migrate_synapse_chat_state.py`
- Test: `tests/test_migrate_chat_state.py`

**Interfaces:**
- Consumes: formato de stores de Task 6.
- Produces: script idempotente `python scripts/migrate_synapse_chat_state.py [--backup DIR] [--state DIR]`. Fuente default: `~/.memo-daemon-backups/20260730T213401-synapse-final/dot-synapse/state/feedback/` (`events.jsonl`, `source_votes.jsonl`). Destino: `<state>/chat/feedback/`. Los embeddings de los votos synapse son compatibles (mismo embedder memo 4B/2560 vía daemon). Función pura exportada: `migrate_feedback(src_dir: Path, dst_dir: Path) -> dict` → `{"events": int, "votes": int, "skipped": int}`.

Mapeo de campos: los jsonl de synapse usan schemas `synapse.chat_feedback.v1` / `synapse.source_vote.v1` con los mismos campos que los dataclasses de Task 6 (más extras que se ignoran); al migrar se reescribe `schema` al valor memo y se descartan claves desconocidas. Línea corrupta → skip contado.

- [ ] **Step 1: Test que falla**

```python
# tests/test_migrate_chat_state.py
import json
from pathlib import Path

from scripts.migrate_synapse_chat_state import migrate_feedback


def _write(path: Path, lines: list[dict | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for line in lines:
            fh.write((line if isinstance(line, str) else json.dumps(line)) + "\n")


def test_migrates_and_remaps_schema(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src / "source_votes.jsonl", [
        {"created_at": "t", "question_key": "k", "query": "q", "source_id": "s",
         "rating": "up", "query_embedding": [0.1], "schema": "synapse.source_vote.v1",
         "extra_synapse_field": 1},
        "corrupt line",
    ])
    _write(src / "events.jsonl", [
        {"feedback_id": "f1", "created_at": "t", "chat_session_id": "s", "turn_id": "1",
         "query": "q", "answer": "a", "source_ids": ["x"], "rating": "up",
         "schema": "synapse.chat_feedback.v1", "trace_id": "ignored"},
    ])
    stats = migrate_feedback(src, dst)
    assert stats == {"events": 1, "votes": 1, "skipped": 1}
    vote = json.loads((dst / "source_votes.jsonl").read_text().splitlines()[0])
    assert vote["schema"] == "memo.chat.source_vote.v1"
    assert "extra_synapse_field" not in vote
    # idempotente: segunda corrida no duplica
    stats2 = migrate_feedback(src, dst)
    assert stats2["events"] == 0 and stats2["votes"] == 0
```

- [ ] **Step 2:** FAIL (crear `scripts/__init__.py` vacío si el import lo requiere; si `scripts/` no es importable con `--import-mode=importlib`, mover la lógica a `src/memo/chat/migrate.py` y dejar `scripts/migrate_synapse_chat_state.py` como wrapper de 5 líneas que la llama — decidirlo según pase el import del test).
- [ ] **Step 3: Implementación**

```python
# scripts/migrate_synapse_chat_state.py
"""One-off: migrate synapse chat feedback signals from the final backup into memo state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_BACKUP = Path.home() / ".memo-daemon-backups" / "20260730T213401-synapse-final" / "dot-synapse" / "state" / "feedback"
_VOTE_FIELDS = {"created_at", "question_key", "query", "source_id", "rating", "query_embedding"}
_EVENT_FIELDS = {"feedback_id", "created_at", "chat_session_id", "turn_id", "query", "answer", "source_ids", "rating", "correction_text"}


def _migrate_file(src: Path, dst: Path, fields: set[str], schema: str) -> tuple[int, int]:
    if not src.exists():
        return 0, 0
    existing: set[str] = set()
    if dst.exists():
        existing = set(dst.read_text(encoding="utf-8").splitlines())
    migrated, skipped = 0, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as out:
        for line in src.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            clean = {k: v for k, v in record.items() if k in fields}
            clean["schema"] = schema
            serialized = json.dumps(clean, ensure_ascii=False)
            if serialized in existing:
                continue
            out.write(serialized + "\n")
            existing.add(serialized)
            migrated += 1
    return migrated, skipped


def migrate_feedback(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    votes, vote_skipped = _migrate_file(
        src_dir / "source_votes.jsonl", dst_dir / "source_votes.jsonl",
        _VOTE_FIELDS, "memo.chat.source_vote.v1",
    )
    events, event_skipped = _migrate_file(
        src_dir / "events.jsonl", dst_dir / "events.jsonl",
        _EVENT_FIELDS, "memo.chat.feedback.v1",
    )
    return {"events": events, "votes": votes, "skipped": vote_skipped + event_skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, default=_DEFAULT_BACKUP)
    parser.add_argument("--state", type=Path, default=Path.home() / ".local" / "share" / "memo")
    args = parser.parse_args()
    stats = migrate_feedback(args.backup, args.state / "chat" / "feedback")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** PASS.
- [ ] **Step 5: Correr en serio:** `python3 scripts/migrate_synapse_chat_state.py` → anotar counts; verificar `wc -l ~/.local/share/memo/chat/feedback/*.jsonl` contra el backup.
- [ ] **Step 6:** `git add -A && git commit -m "feat(chat): one-off state migration from synapse backup"`

---

### Task 14: `memo ops install|uninstall|status`

**Files:**
- Create: `src/memo/ops_launchd.py`
- Modify: `src/memo/cli_ops.py` (agregar comandos al `ops_group` existente, `cli_ops.py:32`)
- Test: `tests/test_ops_launchd.py`

**Interfaces:**
- Consumes: patrón programático de plist ya existente en `src/memo/watcher.py:183` (`render_plist`) y bootout/bootstrap en `src/memo/runtime/daemon.py:90-133` — leerlos antes de implementar y seguir su estilo.
- Produces (en `ops_launchd.py`): `render_chat_plist(memo_bin: str, home: str, *, port: int = 8765, dist: str | None = None) -> str`; `parse_launchctl_list(output: str) -> list[dict]` (`{"label", "pid": int | None, "last_exit": int}` solo labels `com.memo.*`); `install_chat() / uninstall_chat()` (subprocess a `launchctl`). CLI: `memo ops install chat [--port] [--dist]`, `memo ops uninstall chat`, `memo ops status`.

- [ ] **Step 1: Test que falla** (solo funciones puras; NADA de launchctl en tests)

```python
# tests/test_ops_launchd.py
from memo.ops_launchd import parse_launchctl_list, render_chat_plist


def test_render_chat_plist_contents() -> None:
    plist = render_chat_plist("/usr/local/bin/memo", "/Users/tester", port=8765, dist="/x/dist")
    assert "<key>Label</key>" in plist and "com.memo.chat" in plist
    assert "/usr/local/bin/memo" in plist
    assert "serve" in plist and "8765" in plist and "/x/dist" in plist
    assert "/Users/tester/Library/Logs/memo/chat.log" in plist
    assert "KeepAlive" in plist


def test_render_without_dist_omits_flag() -> None:
    plist = render_chat_plist("/bin/memo", "/Users/t", port=8765, dist=None)
    assert "--dist" not in plist


def test_parse_launchctl_list() -> None:
    raw = "PID\tStatus\tLabel\n50864\t0\tcom.memo.recall-daemon\n-\t0\tcom.memo.nightly\n123\t0\tcom.other.thing\n"
    rows = parse_launchctl_list(raw)
    labels = {r["label"] for r in rows}
    assert labels == {"com.memo.recall-daemon", "com.memo.nightly"}
    recall = next(r for r in rows if r["label"] == "com.memo.recall-daemon")
    assert recall["pid"] == 50864 and recall["last_exit"] == 0
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Implementación**

```python
# src/memo/ops_launchd.py
"""Launchd install/uninstall/status for memo-owned agents (chat service)."""

from __future__ import annotations

import subprocess
from pathlib import Path

CHAT_LABEL = "com.memo.chat"


def render_chat_plist(
    memo_bin: str, home: str, *, port: int = 8765, dist: str | None = None
) -> str:
    args = [memo_bin, "chat", "serve", "--host", "127.0.0.1", "--port", str(port)]
    if dist:
        args += ["--dist", dist]
    args_xml = "\n".join(f"      <string>{a}</string>" for a in args)
    log = f"{home}/Library/Logs/memo/chat.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>{CHAT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>{home}/.local/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
  </dict>
</plist>
"""


def parse_launchctl_list(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2].startswith("com.memo."):
            continue
        pid_raw, exit_raw, label = parts
        rows.append({
            "label": label.strip(),
            "pid": int(pid_raw) if pid_raw.strip().isdigit() else None,
            "last_exit": int(exit_raw) if exit_raw.strip().lstrip("-").isdigit() else 0,
        })
    return rows


def _plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{CHAT_LABEL}.plist"


def install_chat(memo_bin: str, home: Path, *, port: int = 8765, dist: str | None = None) -> Path:
    (home / "Library" / "Logs" / "memo").mkdir(parents=True, exist_ok=True)
    path = _plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_chat_plist(memo_bin, str(home), port=port, dist=dist), encoding="utf-8")
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)], capture_output=True, check=True)
    return path


def uninstall_chat(home: Path) -> bool:
    path = _plist_path(home)
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True, check=False)
    if path.exists():
        path.unlink()
        return True
    return False
```

- [ ] **Step 4: CLI** — en `src/memo/cli_ops.py`, siguiendo el estilo de los comandos existentes (`gc-vault-orphans` en línea 50):

```python
@ops_group.command(name="install")
@click.argument("service", type=click.Choice(["chat"]))
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--dist", default=None, help="Directorio dist de la SPA (opcional).")
def ops_install(service: str, port: int, dist: str | None) -> None:
    """Install a memo launchd agent (currently: chat)."""
    import shutil
    from pathlib import Path

    from memo.ops_launchd import install_chat

    memo_bin = shutil.which("memo")
    if not memo_bin:
        raise click.ClickException("no encuentro el binario `memo` en PATH")
    path = install_chat(memo_bin, Path.home(), port=port, dist=dist)
    click.echo(f"installed {path}")


@ops_group.command(name="uninstall")
@click.argument("service", type=click.Choice(["chat"]))
def ops_uninstall(service: str) -> None:
    """Uninstall a memo launchd agent."""
    from pathlib import Path

    from memo.ops_launchd import uninstall_chat

    click.echo("removed" if uninstall_chat(Path.home()) else "not installed")


@ops_group.command(name="status")
def ops_status() -> None:
    """Show all com.memo.* launchd agents."""
    import subprocess

    from memo.ops_launchd import parse_launchctl_list

    out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, check=True).stdout
    for row in parse_launchctl_list(out):
        state = f"pid {row['pid']}" if row["pid"] else f"exit {row['last_exit']}"
        click.echo(f"{row['label']}\t{state}")
```

- [ ] **Step 5:** PASS tests + `uv run --no-sync memo ops status` lista la fleet actual (recall-daemon, nightly, vault-ingest, dream, watch).
- [ ] **Step 6:** `git add -A && git commit -m "feat(ops): launchd install/uninstall/status for memo chat"`

---

### Task 15: Integración final, gate y docs

**Files:**
- Modify: `README.md` (inventario de comandos: `memo chat serve`, `memo eval chat`, `memo ops install|uninstall|status`)
- Modify: `CLAUDE.md` del repo (sección chat: puerto 8765, knobs `MEMO_CHAT_*`, label `com.memo.chat`)
- Modify: `~/CLAUDE.md` (agregar `com.memo.chat` a la lista de launchd agents) — fuera del repo, editar directo

- [ ] **Step 1: Suite completa:** `uv run --no-sync pytest tests/ -q` → verde. `ruff` + `mypy` limpios.
- [ ] **Step 2: Gate de regresión:** `uv run --no-sync memo eval chat` → todos en verde, p50 ≤ 8000ms warm (criterio del spec). Si falla por contenido (no por bug), iterar knobs/corpus según Task 12 Step 7.
- [ ] **Step 3: Instalar en serio:** `memo ops install chat --dist ~/repos/memo/web-chat/dist` → `memo ops status` muestra `com.memo.chat` con pid; abrir `http://127.0.0.1:8765` y smoke manual: pregunta, chips de fuentes, voto 👍 en una fuente, repetir la pregunta y verificar que la fuente votada sube (boost aplicado), follow-up "resumime eso".
- [ ] **Step 4: Verificar convivencia de memoria:** con el chat warm (post primera síntesis), `memory_pressure` o Activity Monitor: memo-daemon (modelo de `MEMO_LLM_MODEL`) + recall-daemon (4B) dentro del budget de 36 GB. Si no entra, documentar y bajar `MEMO_LLM_MODEL` a un modelo menor en el plist (riesgo #1 del spec).
- [ ] **Step 5: Docs** (README + CLAUde.md del repo + ~/CLAUDE.md como arriba).
- [ ] **Step 6:** `git add -A && git commit -m "docs: memo chat surface (serve, eval, ops)"`
- [ ] **Step 7: PR:** `git push -u origin feat/memo-chat && gh pr create --title "feat: native memo chat (synapse chat rescue)" --body "Implements docs/SPECS/2026-07-30-memo-chat-design.md ..."` — cuerpo con resumen por task y el resultado del gate (tabla de eval chat + p50/p95).

---

## Self-review (hecho al escribir el plan)

1. **Cobertura del spec:** pipeline (Tasks 2-9) ✔; UI + :8765 (10-11) ✔; `MEMO_CHAT_*` defaults prod (1) ✔; feedback exacto+semántico (6) ✔; eval gate + corpus (12) ✔; migración de señales (13) ✔; ops install (14) ✔; riesgo 30B verificado (15.4) ✔. Diferidos declarados en Global Constraints (learning layer → plan 2) ✔. El spec nombra el servicio "memo-daemon"; el plan lo materializa como `memo chat serve` con label `com.memo.chat` (mismo rol, nombre más preciso).
2. **Placeholders:** ninguno — todo step tiene código o comando concreto; los dos puntos con incertidumbre real (shape de retorno de `MLXChat.chat`, claves de `repo_get_file`) tienen paso de verificación explícito con comando y fallback defensivo ya escrito.
3. **Consistencia de tipos:** `score_of`/`dedup_key`/`CHUNK_NUM` viven en `dedup.py` y todos los consumidores importan de ahí; el "source dict" está definido en Task 2 y los adaptadores en Task 9 lo producen con esas claves; `ChatConfig` (Task 1) es el único origen de knobs.
