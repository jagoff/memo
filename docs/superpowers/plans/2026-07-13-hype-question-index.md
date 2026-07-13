# HyPE — índice pregunta-space (Fase B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cada memoria durable gana 2-3 vectores de "preguntas que esta memoria responde" (generados de noche por el LLM local); el retrieval hace max-fold `max(doc_sim, max(question_sims))` y **agrega candidatos** que el vector del doc solo no trae — ataca el bucket más débil medido (`multi_session_synthesis` 0.493, cuyo gap probado es candidate generation, no ranking).

**Architecture:** Sidecar vec0 `hype_vec` + tabla `hype_questions` en el MISMO `memvec.db` (patrón `source_feedback_vec` para el DDL vec0; patrón `FactEdgeStore` para la clase con conexiones thread-local propias). Generación = pase dream nocturno (`dream_hype.py`, watermark por `body_hash`, backlog priorizado por ROI, cap por noche) — separado del fold de lectura por DOS flags: `MEMO_DREAM_HYPE_ENABLED` (construye el índice "dark") y `MEMO_HYPE_ENABLED` (fold en el read path). Eso permite construir el índice sin tocar recall y flipear la lectura SOLO si el triple gate pasa. Instrumento de medición: config **K** nueva en `eval_recall` (vec + hype ON) para A/B contra la config A baseline.

**Tech Stack:** Python 3.11+, sqlite-vec (vec0), MLX vía `chat_with_timeout`/`embed_query` (imports diferidos), pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-roadmap-gamechangers-design.md` §1. Antecedente: `docs/eval/capability-baseline-and-levers.md` prescribe multi-vector retrieval para multi_session_synthesis; `docs/superpowers/specs/2026-07-10-memo-ecosystem-resurvey-design.md:112` lo specea (pi=3/M). NO confundir con `MEMO_DREAM_HYDE_TUNE_ENABLED` (`cli_dream.py:449`, HyDE query-side existente): HyPE es ingest-side.

## Global Constraints

- Flags default-OFF en `src/memo/flags_ingest.py`: `MEMO_HYPE_ENABLED` (bool, False — fold de lectura), `MEMO_DREAM_HYPE_ENABLED` (bool, False — pase generador), `MEMO_HYPE_QUESTIONS_PER_MEMORY` (int, 3, min 1 max 5), `MEMO_HYPE_NIGHT_CAP` (int, 400, min 1), `MEMO_HYPE_FOLD_POOL` (int, 30, min 5 — k del knn de preguntas).
- Invariantes MLX: preguntas se embeben con `embedder.embed_query(q)` UNA por vez (str — nunca lista a embed_query, memoria [73f9ef0e]); `embed()` solo con `Sequence[str]`; imports mlx/LLM diferidos; dims del índice = `cfg.embedder_dims`. Decisión de espacio: preguntas CON prefijo de query (simetría query↔pregunta); la variante sin prefijo se decide por eval, no acá.
- `run_hype_pass` NUNCA lanza (contrato `run_profile_pass`); errores → `receipt["errors"]`.
- El fold NO corre si el flag está OFF — cero costo para quien no lo activa; el hook de recall no gana latencia sin medirla (gate c).
- `hype_vec`/`hype_questions` son DERIVADAS y rebuildables; `reindex --rebuild` NO las trunca (solo meta/vec/fts, verificado `queries.py:774-813`) — el pase nocturno re-deriva por watermark y poda huérfanas.
- Working tree compartido: `git add` paths explícitos SIEMPRE. Lint/mypy solo archivos propios. Tests aislados (tmp_cfg / fakes dims=4 con `MEMO_EMBEDDER_DIMS` pineado al stub).
- Comandos: `uv run --no-sync pytest tests/<file> -v` · `ruff check <propios>` · `mypy <propios>`.
- **REGLA DE FLIP (no negociable):** este plan construye y MIDE. `MEMO_HYPE_ENABLED` no se enciende por defecto en ninguna parte; el flip es decisión de Fer con los números del gate en la mano.

## File Structure

| Archivo | Responsabilidad |
|---|---|
| Create `src/memo/store/hype_store.py` | `HypeStore`: DDL idempotente (vec0 + tabla texto), upsert/replace por memoria, knn de preguntas, poda de huérfanas, stats |
| Create `src/memo/dream_hype.py` | Pase nocturno: selección por watermark `body_hash` + prioridad ROI + cap, `_llm_questions`, `run_hype_pass` |
| Create `src/memo/hype_fold.py` | Fold puro de lectura: merge de hits doc + candidatos pregunta-space |
| Modify `src/memo/flags_ingest.py` | 5 FlagSpecs |
| Modify `src/memo/cli_dream.py` | Bloque gateado (después del bloque chronicle, `cli_dream.py:~584`) + subcomando `memo dream hype` |
| Modify `src/memo/memory/search_ops.py` (o el punto único de candidatos que Task 5 verifica) | Wiring del fold detrás de `MEMO_HYPE_ENABLED` |
| Modify `src/memo/eval_recall.py` | Config **K** (vec + MEMO_HYPE_ENABLED=1) |
| Create `src/memo/cli_hype.py` + Modify `src/memo/cli.py` | `memo hype status` (cobertura del índice) |
| Tests | `tests/test_hype_store.py`, `tests/test_dream_hype.py`, `tests/test_hype_fold.py`, `tests/test_cli_hype.py` |

## Datos verificados que las tasks consumen

- DDL vec0 de referencia (`store/schema.py:535-567`): `CREATE VIRTUAL TABLE ... USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine, ...)`; score = `1.0 - distance` (`queries.py:757-772`).
- Patrón sidecar (`store/fact_edge_store.py:94-143`): clase con `threading.local()` + `_ensure_schema()` en init; con `MEMO_SINGLE_DB` los sidecars colapsan en `db_path` — HypeStore vive DIRECTO en `cfg.db_path` (memvec.db) siempre, así el knn comparte archivo con `vec`. La conexión debe replicar el bootstrap de sqlite-vec de `VecStore` (enable_load_extension + load — copiar de `store/connection.py`/`queries.py`).
- Iteración de memorias: `mem.store.all_ids()` (`store/signal_queries.py`) + `mem.store.get(id)` → row con `type`, `body_hash` (`schema.py:23-33`), `title`; body vía `mem.store.get_fts_body(id)`. Durables = `tiers.DURABLE_TYPES` (`tiers.py:51-65`).
- ROI para priorizar: `outcome.compute_utilities(state_dir)` → `by_prefix[{8char}: {"utility": 0-1}]` (`outcome.py:36-113`).
- LLM: `chat_with_timeout(chat, timeout=30, model=mem.cfg.helper_model, messages=[...], options={"temperature": 0.0, "max_tokens": 256, "thinking": False})` (`record.py:296`, patrón `_run_compress` en loop, `cli_dream_passes.py:209-221`).
- Embeddings: `mem.embedder.embed_query(q: str) -> list[float]` (prefijo asimétrico incluido, cache de queries integrado).
- Eval: configs A-J en `eval_recall.py:237-289` (J = overlay `MEMO_HYDE_ENABLED=1` — plantilla exacta para K con `MEMO_HYPE_ENABLED=1`); perfiles en `eval_recall.py:295-307`; baseline en `state_dir/eval/recall_baseline.json`; `harvest_labels` en `eval_recall.py:1100`; bench per-bucket `memo eval bench run --dataset longmemeval_oracle` (`cli_eval_bench.py:85-97`); latencia por config `latency_ms_p50` (`eval_recall.py:458`).
- Bloque dream de plantilla: chronicle en `cli_dream.py:564-584`; subcomando standalone: `dream chronicle` al final de cli_dream.py.
- Test con LLM stub: `tests/test_dream_consolidate.py:76-106` (monkeypatch de `_llm_synthesize`); para stores: fake embedder dims=4.

---

### Task 1: Flags

**Files:** Modify `src/memo/flags_ingest.py` · Test `tests/test_hype_store.py` (nuevo, arranca con el test de flags)

**Interfaces:** Produces los 5 flags listados en Global Constraints, resolubles vía `flag_bool`/`flag_int`.

- [ ] Test primero (patrón exacto de `test_chronicle_flags_registered_default_off` en `tests/test_dream_chronicle.py`): assert los 5 en `REGISTRY`, bools default False, ints con default 3/400/30. RED → implementar los `_spec(...)` (calcar `MEMO_INGEST_MIN_CHARS` en `flags_ingest.py:8` para ints con min/max) → GREEN → `ruff` → commit `feat(hype): flags (default off)` staging solo esos 2 archivos.

### Task 2: HypeStore

**Files:** Create `src/memo/store/hype_store.py` · Test `tests/test_hype_store.py`

**Interfaces — Produces:**
```python
class HypeStore:
    def __init__(self, db_path: Path | str, dims: int) -> None: ...
    def replace_for_memory(self, memory_id: str, body_hash: str, model: str,
                           questions: list[tuple[str, list[float]]]) -> int:
        """Delete old rows for memory_id, insert (question_text, embedding) rows.
        question_id = sha256(f"{memory_id}:{text}").hexdigest()[:32]. Returns inserted count."""
    def body_hash_for(self, memory_id: str) -> str | None: ...
    def knn(self, embedding: list[float], k: int) -> list[dict]:
        """[{memory_id, question, score}] score = 1.0 - distance, mejor pregunta por memoria
        (GROUP BY memory_id, MAX score en Python tras el knn crudo)."""
    def prune_orphans(self, live_ids: set[str]) -> int: ...
    def stats(self) -> dict:  # {memories: int, questions: int}
    def close(self) -> None: ...
```
DDL en `_ensure_schema` (idempotente, en init):
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS hype_vec USING vec0(
    question_id TEXT PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine);
CREATE TABLE IF NOT EXISTS hype_questions (
    question_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, question TEXT NOT NULL,
    body_hash TEXT NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_hype_mem ON hype_questions(memory_id);
```
(el knn joinea `hype_vec` con `hype_questions` por `question_id` para recuperar `memory_id` — vec0 no lleva la columna memory_id para no duplicar metadata mutable). Conexión: thread-local + bootstrap sqlite-vec copiado del connect de `VecStore`. Escrituras en transacción (`BEGIN IMMEDIATE`, patrón `_tx()`).

- [ ] Tests TDD (dims=4, embeddings sintéticos, tmp db path): replace inserta y reemplaza (idempotente por body_hash igual); knn devuelve mejor-por-memoria ordenado por score; prune borra memory_ids fuera de live_ids (vec + texto); stats cuenta. RED → implementar → GREEN → ruff/mypy → commit `feat(hype): HypeStore sidecar (vec0 + question text)`.

### Task 3: dream_hype — selección + generación LLM

**Files:** Create `src/memo/dream_hype.py` · Test `tests/test_dream_hype.py`

**Interfaces — Produces:**
```python
def select_backlog(mem, store: HypeStore, *, cap: int) -> list[dict]:
    """Memorias durables (tiers.DURABLE_TYPES) cuyo body_hash difiere del guardado
    en hype_questions (o sin filas). Ordenadas por utility ROI desc
    (outcome.compute_utilities by_prefix; sin datos → 0.5 neutral). Máximo cap."""
def _llm_questions(mem, title: str, body: str, *, n: int) -> list[str]:
    """UNA llamada chat_with_timeout (timeout=30, helper_model, temp 0, thinking False).
    Prompt _SYS: 'reply with a JSON array of N short questions this note answers,
    in the note's language; questions only a reader of THIS note could answer;
    no generic questions'. Parse JSON array; filtra len<12 o >200 chars; dedup; cap n.
    None/[] si timeout o parse falla."""
def run_hype_pass(cfg, mem, *, questions_per_memory=3, night_cap=400, dry_run=False) -> dict:
    """Nunca lanza. {status: done|skipped|error, generated: int, memories: int,
    pruned: int, backlog_remaining: int}. skipped si backlog vacío.
    Por memoria: _llm_questions → embed_query POR PREGUNTA (str) → store.replace_for_memory.
    Fallo de una memoria no aborta el pase (continue + contador errors_items).
    Al final: prune_orphans(set(all_ids durables)). dry_run: computa backlog, no genera."""
```

- [ ] Tests TDD con LLM stubbeado (`monkeypatch.setattr(dh, "_llm_questions", ...)`, patrón `test_dream_consolidate.py:76`) y embedder fake dims=4 + `MEMO_EMBEDDER_DIMS=4` pineado: backlog filtra por body_hash y respeta cap y orden ROI (utilities monkeypatcheadas); pass genera y guarda; watermark hace el 2do run skipped; LLM None → item salteado sin abortar; never-raises. Commit `feat(hype): nightly question generation pass`.

### Task 4: wiring dream + subcomando

**Files:** Modify `src/memo/cli_dream.py` · Test `tests/test_dream_hype.py`

- [ ] Bloque gateado `if flag_bool("MEMO_DREAM_HYPE_ENABLED"):` inmediatamente DESPUÉS del bloque chronicle (`cli_dream.py:~584`), espejo EXACTO del estilo (progress → try → `receipt["hype"] = run_hype_pass(...)` con `questions_per_memory`/`night_cap` desde flags → status error a `receipt["errors"]` → except append + warn). Subcomando `@dream_cmd.command(name="hype")` con `--dry-run`/`--json` calcado de `dream chronicle` (incluye `from memo.flags import flag_bool, flag_int` local — lección C7: el archivo usa imports por-función). Test CliRunner del subcomando con `run_hype_pass` monkeypatcheado (patrón `test_dream_chronicle_subcommand_json`). Commit `feat(hype): wire nightly pass + memo dream hype subcommand`.

### Task 5: fold de lectura (candidate generation)

**Files:** Create `src/memo/hype_fold.py` · Modify el punto único de candidatos · Test `tests/test_hype_fold.py`

**Interfaces — Produces:**
```python
def hype_fold(doc_hits: list[dict], query_embedding: list[float], store: HypeStore,
              fetch_meta: Callable[[str], dict | None], *, pool: int, limit: int) -> list[dict]:
    """knn(embedding, k=pool) sobre preguntas → best score por memoria.
    - memoria YA en doc_hits: score = max(score_doc, score_pregunta); marca extra hype=True.
    - memoria NUEVA (el doc no la trajo): fetch_meta(id) → si existe y no borrada,
      append como hit con score=score_pregunta y hype=True.  ← la ganancia real
    Reordena por score desc, corta a limit. Puro, sin flags adentro."""
```
**Paso de verificación previo (obligatorio, dentro de la task):** grep/lectura para identificar EL punto único donde el retrieval vec arma candidatos que atraviesan (a) el recall hook daemon (`recall_logic`), (b) el hook subprocess (`cli_recall_hook`) y (c) `Memory.search` (que es lo que `eval_recall.run_config` ejercita — sin esto el gate no puede medir el fold). Ese punto está en `memory/search_ops.py` (o donde `VecStore.search` se invoca para el modo vec). El wiring va AHÍ, gateado: `if flag_bool("MEMO_HYPE_ENABLED"):` → construir `HypeStore(cfg.db_path, cfg.embedder_dims)` lazy (cachear en el facade) → `hits = hype_fold(hits, query_emb, store, mem.store.get, pool=flag, limit=limit)`. Si los tres paths NO comparten un punto único, el implementer reporta BLOCKED con el mapa de call sites encontrado (no parchear los 3 a mano sin diseño).

- [ ] Tests TDD del fold puro (fakes): max-fold sube score de hit existente; candidato nuevo aparece con meta fetcheada; memoria borrada (fetch_meta None) no entra; respeta limit; pool corto no explota. Test de wiring: `Memory.search` con flag ON y HypeStore poblado devuelve el candidato pregunta-space (embedder stub dims=4). Test flag OFF = comportamiento idéntico byte a byte (mismos ids/orden). Commit `feat(hype): read-path max-fold behind MEMO_HYPE_ENABLED`.

### Task 6: instrumento de medición — eval config K

**Files:** Modify `src/memo/eval_recall.py` · Test: el existente de configs si lo hay + smoke

- [ ] Agregar config `K` = base vec (como A) + overlay `MEMO_HYPE_ENABLED=1` — calcada EXACTO de cómo J overlaya `MEMO_HYDE_ENABLED=1` (`eval_recall.py:237-289`), y sumarla al profile `expensive` (o un profile nuevo `hype` si `expensive` está acoplado a J — decisión del implementer con disclosure). Verificar que `--config K` corre y que con el índice vacío K == A (fold sin filas = no-op). Commit `feat(hype): eval config K (vec + hype fold) for A/B gating`.

### Task 7: `memo hype status`

**Files:** Create `src/memo/cli_hype.py` · Modify `src/memo/cli.py` (import ORDENADO alfabéticamente — lección C8/I001 — + add_command) · Test `tests/test_cli_hype.py`

- [ ] `memo hype status [--json]`: `HypeStore.stats()` + total de durables (cobertura %) + backlog pendiente (`select_backlog` en dry). Patrón `cli_related.py`. Tests CliRunner (env aislado). Commit `feat(hype): memo hype status coverage command`.

### Task 8: gate final + medición dark (SIN flip)

**Files:** verificación; sin código salvo fixes.

- [ ] 1. Suites dirigidas (4 archivos de test nuevos) + `ruff` + `mypy` en archivos propios; suite completa `pytest tests/ -q` (fails pre-existentes conocidos de config-leak NO cuentan).
- [ ] 2. `memo config validate` limpio con los flags nuevos.
- [ ] 3. **Construcción dark en esta máquina**: `uv run --no-sync memo dream hype --json` (cap por defecto 400; correr 2-3 veces si hace falta para cubrir buena parte del corpus; NUNCA concurrente con otra carga MLX — exit-144). Reportar `memo hype status`.
- [ ] 4. **Medición (reportar, NO flipear):** `memo eval recall --labels eval/regression_labels.json --k 5 --force --config A --config K --json` → tabla A vs K (prec@5, noise@5, latency_ms_p50). Si hay labels harvested (`harvest_labels`), correr también sobre esas. Bench per-bucket si el dataset local está disponible (`memo eval bench run ... --retrieval-only`); si no, documentar cómo correrlo.
- [ ] 5. Reporte final con los números y la recomendación flip/no-flip. `MEMO_HYPE_ENABLED` queda OFF. Commit solo si hubo fixes.

## Medición y criterio de flip (del spec — decisión de Fer, fuera de este plan)

Triple gate: (a) `eval recall --gate` sin regresión (prec@5 ≥ baseline 0.812, noise 0.0); (b) mejora en config K vs A sobre labels curados + harvested, y en bench `multi_session_synthesis` si corrible; (c) `latency_ms_p50` de K dentro de presupuesto (hook <5s con margen). Si (b) no mejora, la feature queda dark — sin excepción.
