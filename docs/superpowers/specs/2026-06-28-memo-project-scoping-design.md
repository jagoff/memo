# Design — Guardar memorias por proyecto + relevancia 3-tiers

- **Fecha:** 2026-06-28
- **Estado:** Aprobado (brainstorm), pendiente de plan de implementación
- **Repo:** memo (`~/repos/memo`)

## Problema

Hoy memo guarda todas las memorias sueltas (flat) en `cfg.memory_dir`. El usuario
quiere:

1. **Guardar por proyecto** — los `.md` agrupados en carpetas por proyecto (organización).
2. **Consultar como si estuvieran en la misma carpeta** — la búsqueda/recall sigue siendo
   un pool global (no un filtro duro al proyecto actual).
3. **Darle relevancia** — las memorias del proyecto actual deben pesar más en el ranking.

## Qué ya existe (no se reconstruye)

- **Tag de proyecto:** `current_project_tag(cwd)` deriva `project:<slug>` del git toplevel.
  `save` lo auto-agrega (`auto_project=True`, gated por `MEMO_AUTO_PROJECT_TAG`).
  Fuente: `src/memo/project.py`, `src/memo/memory/write_ops.py:318`.
- **Boost de proyecto en recall:** `_apply_project_boost(hits, project_tag, 0.15)` suma
  `+0.15` a los hits cuyo `tags` contiene el `project:` actual; on por default
  (`MEMO_RECALL_PROJECT_BOOST`). Fuente: `src/memo/recall_logic.py:129`,
  `src/memo/cli_recall_hook.py:210`.
- **Índice global y recursivo:** reindex/ingest globean `rglob("*.md")`, así que los
  folders en disco son transparentes para el índice → el pool de búsqueda ya es global.
  Fuente: `src/memo/memory/maintain_ops.py:243`, `src/memo/cli_ingest.py:227`.

**Conclusión de diseño:** la señal de proyecto para relevancia viene del **tag**, no de la
carpeta. Mover archivos a carpetas no cambia el ranking por sí solo. Por eso storage y
ranking se diseñan como dos componentes derivados de la misma fuente de verdad (el tag).

## Decisiones (del brainstorm)

1. Objetivo: **ambas** — carpetas por proyecto **y** relevancia más fuerte.
2. Ranking: **boost fuerte soft** (no filtro duro) — el proyecto actual sube, pero una
   global/otro-proyecto mucho más similar todavía puede ganar.
3. Memorias globales: **3 tiers soft** — proyecto-actual > global/transversal > otros-proyectos.
4. Fuente de verdad: **el tag manda; la carpeta y el tier de ranking derivan de él**
   (consistente con "markdown frontmatter = verdad, sqlite = índice derivado").

## Arquitectura

El tag `project:<slug>` en el frontmatter es la **única fuente de verdad**. De él derivan:

- **La carpeta** en disco (organización humana).
- **El tier de ranking** (relevancia en recall).

El índice sqlite no cambia (ya es global + recursivo) → "misma carpeta" sale gratis.

```
tag project:<slug>  ──┬──►  carpeta  memory_dir/<slug>/  (derivada)
   (frontmatter,      │
    fuente de verdad) └──►  tier de ranking en recall   (derivado)

índice sqlite  ──►  pool de búsqueda GLOBAL (rglob recursivo, sin cambios)
```

## Componente 1 — Storage layout

```
memory_dir/
  memo/            ← project:memo
  synapse/         ← project:synapse
  _global/         ← sin proyecto (--no-project-tag, o save fuera de un repo)
    2026-06-28-preferencia-espanol.md
```

- En `save`, el constructor de path (`_make_rel_path`, `write_ops.py:~794`) antepone el
  bucket: `rel_path = f"{bucket}/{date}-{slug}.md"`, donde `bucket` = slug del tag de
  proyecto (sin el prefijo `project:`) o `_global`.
- La verificación de unicidad de nombre pasa a ser **por-bucket**.
- El proyecto se fija al **crear** (git toplevel donde se guardó). El archivo **nunca se
  mueve** en edits posteriores — `update` ya mantiene el path estable. Cero churn.
- `_global` contiene: saves con `--no-project-tag` y saves fuera de cualquier repo.
- **Flag:** `MEMO_STORE_BY_PROJECT` (default **on**). Como el índice es recursivo, los
  layouts flat y foldered **conviven** durante la transición — no hace falta migrar de golpe.

## Componente 2 — Ranking 3-tiers soft

Se extiende `_apply_project_boost` → `_apply_project_tiers(hits, current_tag, weights)`:

| tier | qué | boost (default) | flag |
|---|---|---|---|
| 1 | proyecto actual (`current_tag` en `tags`) | `+0.25` | `MEMO_RECALL_PROJECT_BOOST` (sube de 0.15) |
| 2 | global/transversal: sin `project:` tag **o** `type ∈ {preference, feedback}` | `+0.10` | `MEMO_RECALL_GLOBAL_BOOST` (nuevo) |
| 3 | otros proyectos | `+0` (baseline) | — |

- **Precedencia de tiers** (un hit puede matchear varios): `preference`/`feedback` →
  **tier-2 siempre** (gana sobre tier-1, incluso con el tag del proyecto actual);
  si no, tier-1 cuando tiene el `current_tag`; si no, tier-3. Así una preferencia del
  proyecto actual sigue tratándose como transversal (+0.10), no como nota local (+0.25).
- Aditivo y **soft**: una global/otro-proyecto mucho más similar todavía gana
  (fiel a "misma carpeta con relevancia").
- Se aplica sobre el pool ya sobre-fetcheado (`search_k = top_k * 3` cuando hay
  `project_tag`; se mantiene/ensancha para garantizar que las globales entren al pool
  antes del boost).
- **`preference`/`feedback` van siempre a tier-2** (transversales) aunque tengan tag de
  proyecto — son cross-cutting (ej. "responder en español" debe surgir en cualquier
  proyecto). Confirmado en brainstorm.
- Todos los pesos vía flags → `memo config validate` los cubre.

Puntos de aplicación: `recall_logic.py` (función de tiers) consumida por
`cli_recall_hook.py` y el path de recall del warm daemon (`recall_server`/socket), para que
el hook y el socket compartan la misma lógica.

## Componente 3 — Migración

- Se extiende `memo migrate` con `--bucket-by-project`: por cada `.md` en la raíz de
  `memory_dir`, lee su tag `project:` y lo **mueve** a `<bucket>/`; sin tag → `_global/`.
  Luego `reindex` (los `path` cambiaron).
- **No destructivo:** es un move dentro de `memory_dir`; el markdown se preserva;
  `memvec.db` nunca se dropea (reindex actualiza `path` por `id` estable).
- **Idempotente:** archivos ya en un bucket se saltean.
- **Reversible:** `--rollback` (o mover de vuelta a la raíz).
- `[[id]]` wikilinks no se afectan; links por path de Obsidian (raros) cambiarían — se
  documenta en el output del comando.

## Invariantes y edge cases

- **Markdown = verdad (intacto):** el tag manda; carpeta + índice derivan. Un move a mano
  en Obsidian a otra carpeta **no** cambia el proyecto (gana el tag en reindex) — reindex
  no pelea con el usuario porque el proyecto sale del tag, no del path.
- **Path estable:** proyecto fijo al crear → sin churn en updates.
- **Colisión de slug:** dos repos con el mismo basename → mismo bucket. Limitación
  conocida y aceptable; mejora futura: desambiguar por directorio padre. YAGNI ahora.
- **Vault mode** (`MEMO_MEMORIES_IN_VAULT`): buckets bajo `<vault>/AI/memory/<proyecto>/`;
  ingest ya excluye `AI/`. Sin cambios adicionales.
- **Sync git:** los moves son renames en git → se sincronizan normal.
- **Reindex/rebuild:** sin cambios (recursivo). `_global` y `<proyecto>` ambos se globean.

## Archivos afectados

- `src/memo/memory/write_ops.py` — `_make_rel_path` antepone el bucket; unicidad por-bucket.
- `src/memo/recall_logic.py` — `_apply_project_tiers` (reemplaza/extiende `_apply_project_boost`).
- `src/memo/cli_recall_hook.py` — consumir la nueva función de tiers.
- `src/memo/recall_server.py` / socket de recall — misma lógica de tiers para el warm daemon.
- `src/memo/flags_recall.py` — `MEMO_RECALL_GLOBAL_BOOST` (nuevo), `MEMO_STORE_BY_PROJECT`
  (nuevo); ajustar default de `MEMO_RECALL_PROJECT_BOOST`.
- `src/memo/runtime/migrate.py` (o el `cli_*` de migrate) — `--bucket-by-project`.
- Tests (abajo).

## Testing

- **Unit (storage):** derivación de bucket (proyecto / global / sin-tag); `_make_rel_path`
  pone el archivo en el bucket correcto; unicidad de nombre por-bucket.
- **Unit (ranking):** ordering 3-tiers — proyecto-actual gana a igual-similitud de otro
  proyecto; global/`preference` se mantiene a flote; una global/otro **mucho** más similar
  igual gana (soft, no duro).
- **Migración:** corpus flat → bucketed; sin-tag → `_global`; idempotente; reindex
  actualiza paths; count preservado round-trip; rollback restaura.
- **Gate de regresión:** `memo eval recall --labels eval/regression_labels.json --k 5`
  para confirmar que el cambio de ranking **no** baja precisión / sube ruido (disciplina
  retrieval-regression de memo).
- **Aislamiento:** `tmp_cfg` / `Config` aislado; nunca el vault real (ver `tests/conftest.py`).

## Flags nuevos / cambiados (resumen)

| flag | default | rol |
|---|---|---|
| `MEMO_STORE_BY_PROJECT` | on | guardar nuevos `.md` en carpeta por proyecto |
| `MEMO_RECALL_PROJECT_BOOST` | 0.25 (sube de 0.15) | boost tier-1 (proyecto actual) |
| `MEMO_RECALL_GLOBAL_BOOST` | 0.10 (nuevo) | boost tier-2 (global/transversal) |

## Fuera de alcance (YAGNI)

- Desambiguación de slugs en colisión (basename repetido).
- Re-proyectar una memoria moviendo el archivo en Obsidian (sería "carpeta manda",
  rechazado — choca con markdown-as-truth).
- Filtro duro por proyecto / scoping exclusivo (se eligió pool global soft).
