# memo dream — 4 mejoras (2026-06-18)

Inspiradas en el análisis de Claude Dream (Anthropic). Cada mejora es opt-out
con flag, arquitectura existente intacta.

## Mejora 1 — Phase 0: Signal Gather

**Problema.** `dream run` nunca procesa transcripts nuevos. `memo mine-history`
existe pero está desconectado del pipeline.

**Diseño.**

- `dream run` calcula `since_days` = días desde el último dream run (archivo
  `state_dir/dream/.last_run_ts`). Si es el primer run: 7 días.
- Llama a `mine_transcripts(since_days=since_days, max_transcripts=20)` de
  `memo.transcript_miner`. La función ya tiene watermark incremental (lines_processed
  por archivo) → re-runs son idempotentes.
- Resultado se añade al receipt: `signal_gathered: {memorias_saved, files_processed, skipped}`.
- Flag CLI: `--skip-signal-gather`. También se salta en `--dry-run` (ningún transcript
  se minea en dry-run).
- Numeración de pasos: pasa a ser paso 0 explícito. Los pasos 1-6 existentes no cambian.

**Archivos.** `cli_dream.py` únicamente — wiring de `mine_transcripts`.

---

## Mejora 2 — Date normalization en synthesis

**Problema.** Cuerpos de síntesis guardan fechas relativas ("ayer decidimos…"),
que se vuelven ambiguas o falsas con el tiempo.

**Diseño.**

Dos capas independientes:

1. **Instrucción en prompt.** Añadir a `_SYNTHESIS_SYSTEM_PROMPT` en
   `memory/prompts.py`: "Si el insight contiene referencias temporales relativas
   (ayer, la semana pasada, hace N días, yesterday, last week, N days ago),
   convértelas a fechas ISO absolutas en el body."

2. **Post-process Python** en `consolidate_ops.py`, función
   `_normalize_relative_dates(text: str, ref_date: date) -> str`. Corre sobre
   el `body` devuelto por el LLM antes de llamar a `save()`. Regex cubre:

   | Patrón | Resultado |
   |--------|-----------|
   | `ayer` / `yesterday` | `(YYYY-MM-DD)` |
   | `hoy` / `today` | `(YYYY-MM-DD)` |
   | `anteayer` | `(YYYY-MM-DD)` |
   | `la semana pasada` / `last week` | `(semana del YYYY-MM-DD)` |
   | `el mes pasado` / `last month` | `(YYYY-MM)` |
   | `hace N días` / `N days ago` | `(YYYY-MM-DD)` |

   La función nunca falla: errores de regex → texto original sin cambio.

**Archivos.** `memory/prompts.py`, `memory/consolidate_ops.py`.

---

## Mejora 3 — Quality-floor prune

**Problema.** ROI decay (`×0.98`) floors en 0.1 pero nunca elimina. Memorias
nunca accedidas y con roi bajo se acumulan indefinidamente.

**Diseño.**

- Nuevo método `prune_floor_candidates(roi_floor, min_age_days, exclude_types)` en
  `store/signal_queries.py`:

  ```sql
  SELECT m.id FROM meta m
  LEFT JOIN memory_health h ON h.id = m.id
  LEFT JOIN access a ON a.id = m.id
  WHERE COALESCE(h.roi_score, 1.0) < :roi_floor
    AND COALESCE(a.access_count, 0) = 0
    AND m.updated < datetime('now', '-' || :min_age_days || ' days')
    AND m.type NOT IN (:exclude_types)
  ```

- En `dream_run`, paso después de ROI decay: obtiene candidatos →
  `lifecycle.archive_memoria(id)` por cada uno. **Archive, no hard-delete**
  (reversible, coherente con filosofía de memo).
- Defaults: `roi_floor=0.15`, `min_age_days=90`.
- Nuevos flags en `flags_misc.py`:
  - `MEMO_DREAM_PRUNE_FLOOR` (float, default 0.15)
  - `MEMO_DREAM_PRUNE_MIN_AGE_DAYS` (int, default 90)
- Tipos siempre excluidos: `{"synthesis", "reference"}`.
- Flag CLI: `--skip-prune-floor`.
- Receipt: `pruned_floor: [{id, roi_score, days_old}]`.

**Archivos.** `store/signal_queries.py`, `flags_misc.py`, `cli_dream.py`.

---

## Mejora 4 — Orientation summary

**Problema.** `dream run` muta sin haber inventariado el estado. No hay
visibilidad de lo que existe antes de que el pipeline toque nada.

**Diseño.**

- Primer paso de `dream run`, read-only, antes de cualquier mutación.
- Queries contra `meta`, `memory_health`, `access`, `contradict`:

  | Métrica | Query |
  |---------|-------|
  | Total memorias | `SELECT COUNT(*) FROM meta WHERE type != 'reference'` |
  | Por tipo | `SELECT type, COUNT(*) FROM meta GROUP BY type` |
  | Con roi_score < 0.3 | JOIN `memory_health` |
  | Stale candidates (>365d sin acceso) | JOIN `access` |
  | Contradicciones abiertas | `contradict_pairs` WHERE status='open' |
  | Sin entities indexadas | `meta` LEFT JOIN `entity_mentions` WHERE entity_mentions.id IS NULL |

- Output: Rich panel `[bold cyan]Inventario pre-dream[/bold cyan]` con tabla 2 cols.
- También almacenado en receipt como `orientation: {total, by_type, low_roi, stale_candidates, open_contradictions, unindexed_entities}`.
- Flag CLI: `--skip-orientation` (silencia el panel; orientation sigue en receipt).

**Archivos.** `cli_dream.py` únicamente (queries directas sobre `mem.store._conn`).

---

## Resumen de cambios

| Archivo | Cambio |
|---------|--------|
| `src/memo/cli_dream.py` | +4 pasos, +4 flags CLI, wiring signal-gather, prune-floor, orientation |
| `src/memo/memory/prompts.py` | Instrucción date-norm en `_SYNTHESIS_SYSTEM_PROMPT` |
| `src/memo/memory/consolidate_ops.py` | `_normalize_relative_dates()` + llamada pre-save |
| `src/memo/store/signal_queries.py` | `prune_floor_candidates()` |
| `src/memo/flags_misc.py` | `MEMO_DREAM_PRUNE_FLOOR`, `MEMO_DREAM_PRUNE_MIN_AGE_DAYS` |

## No modificado

- `lifecycle.py` — `archive_memoria()` se usa tal cual
- `transcript_miner.py` — `mine_transcripts()` se usa tal cual
- Receipt schema es backwards-compatible (sólo se agregan claves)
- Pasos 1-6 existentes intactos en número y comportamiento

## Tests requeridos

- `test_dream_signal_gather`: mock `mine_transcripts`, verifica key en receipt
- `test_normalize_relative_dates`: unit test fecha ref fija, todos los patrones
- `test_prune_floor_candidates`: DB fixtures con roi < floor y age > threshold
- `test_dream_prune_floor_in_pipeline`: integration, verifica archival en dry-run=False
- `test_dream_orientation`: verifica receipt.orientation keys presentes
