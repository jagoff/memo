---
name: memo
description: "Router para memo — el MCP local de memoria persistente backed by Obsidian (markdown plano + sqlite-vec + MLX, 100% local, zero Ollama). Trigger ÚNICO: `/memo`. Sin argumentos → smart capture: destila el insight accionable del turno actual (decisión / fix / discovery / preferencia), arma body en formato CLAUDE.md, auto-deriva title/type/tags y guarda **directo sin gate** (output post-hoc para auditar; si guardó basura, `/memo delete <id>`). Usá cuando el user tipea `/memo <query>` (search semántico — default), `/memo` solo (smart capture), `/memo list [n]` (últimas n por updated, default 20), `/memo save <texto>` (guarda lo que dice el user con auto-derivación de title/type/tags), `/memo get <id|prefix>` (acepta prefix git-style ≥4 chars), `/memo update <id|prefix> [--title X] [--type T] [--tag a --tag b]` (replace metadata o body), `/memo delete <id|prefix>` (PIDE confirmación), `/memo stats` (totales + paths + modelos activos), `/memo reindex` (re-absorbe edits hechos directo en Obsidian), `/memo doctor [--gc [--fix]]` (self-check + orphan detect). Siempre routeá al MCP tool `mcp__memo__memory_*` correspondiente, NUNCA escribas el .md a mano (el MCP maneja frontmatter + sqlite-vec index)."
argument-hint: "(vacío = smart capture) | <query> | list [n] | save <text> | get <id|prefix> | update <id|prefix> [flags] | delete <id|prefix> | stats | reindex | doctor [--gc] [--fix]"
---

# /memo — router para memo MCP

`memo` (path local: [`/Users/fer/repositories/memo/`](file:///Users/fer/repositories/memo/)) es un MCP local que da memoria persistente backed by Obsidian. Stack 100% local-first, **zero Ollama, zero cloud**:

- LLM: [`mlx-lm`](https://github.com/ml-explore/mlx-lm) loadeando Qwen2.5-Instruct quantizado, in-process en Apple Silicon Metal.
- Embedder: [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ), 1024-dim, last-token pooling, L2-normalised.
- Vector store: [`sqlite-vec`](https://github.com/asg017/sqlite-vec), single file en `~/.local/share/memo/memvec.db`.
- Storage of record: `.md` planos en `<vault>/99-obsidian/99-AI/memory/`.

El user te invocó con `/memo $ARGUMENTS`. Tu trabajo es parsear `$ARGUMENTS` y rutear al MCP tool correspondiente.

## Routing — primera palabra de `$ARGUMENTS`

| Primera palabra | Acción | MCP tool / sección |
|---|---|---|
| (vacío) | **smart capture** — destila insight del turno y guarda directo | "Smart capture" |
| `ask` + pregunta | **RAG**: respuesta sintetizada con citas `[id]` | `mcp__memo__memory_ask` |
| `list` (opcional `<n>`) | últimas `n` por `updated` desc (default 20) | `mcp__memo__memory_list` |
| `save` + texto | guardar con auto-derivación de title/type/tags | "Save flow" |
| `get` + `<id\|prefix>` | mostrar memoria completa | `mcp__memo__memory_get` |
| `update` + `<id\|prefix>` + flags/texto | patchear metadata o body | "Update flow" |
| `delete` + `<id\|prefix>` | borrar (CON confirmación) | "Delete flow" |
| `stats` | totales + paths + modelos | `mcp__memo__memory_stats` |
| `reindex` | re-absorbe edits hechos en Obsidian | `mcp__memo__memory_reindex` |
| `consolidate` [threshold] | clusters near-duplicate + propuestas de merge | `mcp__memo__memory_consolidate` |
| `entities` [type] | top entidades del knowledge graph | `mcp__memo__memory_entities` |
| `entity` `<name>` | memorias que mencionan una entidad | `mcp__memo__memory_entity` |
| `extract-entities` [`--all`\|`--id X`] | poblar el graph (Qwen2.5-3B sobre cuerpos) | `mcp__memo__memory_extract_entities` |
| `lint` | memorias con problemas (legacy_extra / few_tags / body_skinny / untitled) | `mcp__memo__memory_lint` |
| `history` [op] [id] | audit log de save/update/delete | `mcp__memo__memory_history` |
| `doctor` [`--gc`] [`--fix`] | self-check + orphan detect | shell: `memo doctor` |
| `tui` [`--refresh N`] | live terminal dashboard (corpus, saves, recalls, MLX state, sparklines) | shell: `memo tui` |
| `watch` | foreground file-watcher (auto-reindex on `.md` edit) | shell: `memo watch` |
| `install-watcher` / `uninstall-watcher` | launchd plist daemon for the watcher | shell: `memo install-watcher` / `memo uninstall-watcher` |
| `mine-history` [`--since N`] | backfill memorias from past Claude Code transcripts | shell: `memo mine-history` |
| cualquier otra cosa | **search semántico** | `mcp__memo__memory_search({query: $ARGUMENTS, limit: 5, body_chars: 280})` |

### Tool naming

El servidor MCP se registra como `memo` (lowercase, hyphenated). Las herramientas que vas a llamar son:

```
mcp__memo__memory_save
mcp__memo__memory_search
mcp__memo__memory_list
mcp__memo__memory_get
mcp__memo__memory_update
mcp__memo__memory_reindex
mcp__memo__memory_delete
mcp__memo__memory_stats
```

Si NO ves estas herramientas en tu sesión actual, el server no está registrado. Avisale al user que corra:

```bash
claude mcp add memo -s user memo-mcp
```

…y reinicie Claude Code.

## ID — full UUID4 hex o prefix ≥4 chars

`memo` acepta el id completo (32 chars) o un prefix git-style. Si el prefix matchea ≥2 records, el MCP devuelve:

```json
{"error": "ambiguous", "prefix": "aaaa", "matches": ["aaaaaaaa1111…", "aaaaaaaa2222…"]}
```

Cuando ves esa shape, mostrá los matches al user en un mensaje corto y pedí prefix más largo. No reintentar con guess.

## Search — comportamiento del default

`memory_search` corta el `body` a 280 chars y agrega `body_truncated: true`. **Si el user pidió leer el contenido completo** (ej. `/memo get` o "abrime esa memoria"), llamá a `memory_get(id)` para traer el body sin truncar. Para ranking + decidir relevancia, el snippet alcanza.

Output sugerido para search (5 resultados, 1 línea cada uno):

```
1. 0.873 · decision · MLX migration cierre formal · 2026-04-29
   Decidí migrar obsidian-rag entero a MLX. Replaza Ollama nomic + bge-m3...
2. 0.812 · note · Sistema mem-vault — qué es y dónde vive · 2026-04-28
   ...
```

Si `len(hits) == 0`: avisar `no hay matches para <query>. ¿Querés guardar esto como memoria nueva? /memo save <texto>`.

## Smart capture — `/memo` sin args

**Trigger**: el user invocó `/memo` sin argumentos. Asumí que querés destilar lo que se charló en el turno actual y guardarlo.

### Algoritmo

1. **Mirá hacia atrás los últimos ~8-10 turnos** de la conversación. Identificá **uno** de estos artefactos accionables:
   - Decisión arquitectónica con tradeoff explícito.
   - Bug fix con root cause documentada.
   - Discovery del codebase (gotcha, invariante, comportamiento no-obvio).
   - Convención nueva o workflow operativo descubierto.
   - Preferencia / feedback explícito del user sobre cómo trabajar.
   - Setup operativo nuevo (env var, plist, config persistente).
   - Performance finding con números reales.

2. **Si NO hay insight clarito** (charla casual, lookup que no llegó a conclusión, sesión exploratoria sin cerrar):
   ```
   No detecté un insight accionable. ¿Qué guardo?
   (pasame el contenido o usá `/memo save <texto>` con el body que quieras)
   ```
   No inventes algo — guardar ruido degrada el recall futuro.

3. **Si hay insight**, armá un body en formato CLAUDE.md (secciones `## Contexto`, `## Decisión` o `## Causa raíz`, `## Solución` o `## Detalle`, `## Aprendido el YYYY-MM-DD` con la fecha de hoy + commit SHA si lo tenés a mano). El body lo escribís VOS — destilación, no copia textual.

4. **Auto-derivá title/type/tags**:
   - **title**: primera línea descriptiva, máx 80 chars, sin fecha al frente.
   - **type**: heurística por contenido —
     - mention de "decidí / decidimos / opto por / vamos con" → `decision`
     - mention de "fix / bug / arreglo / root cause" → `bug`
     - "aprendí que / descubrí que / gotcha" → `fact`
     - "prefiero / por favor no hagas / siempre hacé" → `preference`
     - feedback recibido del user sobre tu trabajo → `feedback`
     - sino → `note`
   - **tags**: ≥3 tags. Mix de:
     - **project tag**: derivado del cwd (ej. `memo`, `obsidian-rag`).
     - **domain tags**: tema (ej. `mlx`, `embedder`, `sqlite-vec`).
     - **technique tag**: si aplica (ej. `migration`, `perf`, `setup`).
   - Lower-case + de-dup automático lo hace el MCP.

5. **Guardá directo, sin preview ni gate**. Llamá:
   ```
   mcp__memo__memory_save({
     content: <body>,
     title: <title-derivado>,
     type: <type-derivado>,
     tags: [<tags-derivados>]
   })
   ```

6. **Output post-save (informativo, NO de gate)**:
   ```
   📚 Guardé memoria: `<id-corto>` (<type>, [<a>, <b>, <c>])

   Title: <título>
   Body (primeras ~5 líneas):
   <line-1>
   <line-2>
   <line-3>
   <line-4>
   <line-5>
   ```
   Mostrá las primeras ~5 líneas del body. Si body <5 líneas, mostralo entero. El criterio del user es "fricción cero — si guardé basura, la borro con `/memo delete <id>`".

### Cuándo NO disparar smart capture

- La conversación tiene <3 turnos (no hay material aún para destilar).
- El user dijo explícitamente "no guardes" en la conversación reciente.

## Save flow — `/memo save <texto>`

El user pasa el contenido textual. Vos auto-derivás title/type/tags con la misma lógica de smart capture (paso 4) y llamás `memory_save`. Output post-save igual al de smart capture.

**Override manual**: si el user pasó flags inline, respetalos:

```
/memo save --type=decision --tags=mlx,perf "Body del memo aquí"
```

→ usar esos valores como override de la auto-derivación.

## Update flow — `/memo update <id|prefix> [flags] [body]`

Patches:
- `--title X` cambia el título.
- `--type T` cambia el tipo (uno de `decision`, `fact`, `bug`, `feedback`, `preference`, `note`, `manual`).
- `--tag a --tag b` REEMPLAZA la lista de tags (no merge — pasale los tags finales que querés).
- Cualquier texto restante se interpreta como nuevo `content` (replace, no append).

Llamada:

```
mcp__memo__memory_update({
  id: "<resolved>",
  title: "...",  // opcional
  type: "...",   // opcional
  tags: [...],   // opcional, si presente reemplaza
  content: "..." // opcional, si presente re-embeds
})
```

Si la tool devuelve `{"error": "ambiguous", ...}`, mostrar matches y pedir prefix más largo.

## Delete flow — `/memo delete <id|prefix>`

**SIEMPRE pedí confirmación** antes de llamar `memory_delete`:

```
¿Borrar memoria `<id-corto>`? Title: "<título>". Tags: [<a>, <b>]. (sí/no)
```

Sólo después del "sí" explícito llamás `mcp__memo__memory_delete({id: "<resolved>"})`. Si devuelve `{"deleted": true}` informá; si devuelve `ambiguous` shape, surfaceá los matches.

El delete es destructivo (borra `.md` + entry sqlite-vec). No hay undo automático — el user puede recuperar el `.md` desde el git de iCloud si lo necesita, pero la entry vec se va.

## Doctor — `/memo doctor [--gc] [--fix]`

No hay MCP tool para esto — corré el shell:

```bash
memo doctor          # verificación básica
memo doctor --gc     # detectar orphans
memo doctor --gc --fix  # dropear orphan store rows (.md NUNCA se borran auto)
```

Mostrá el output literal al user.

## Reindex — `/memo reindex`

Llamá `mcp__memo__memory_reindex()`. La tool devuelve `{"checked", "reindexed", "added", "skipped"}`. Mostrá el resultado en una línea:

```
checked 142 · reindexed 3 · added 1 · skipped 0
```

Útil después de que el user editó memorias directamente desde Obsidian (cambia el body → nuevo body_hash → reindex re-embeb), o tras restaurar archivos `.md` desde un backup (sin entry en el store).

## Stats — `/memo stats`

Llamá `mcp__memo__memory_stats()` y mostrá el dict literal:

```
total          142
vault_path     /Users/fer/.../Notes
memory_dir     /Users/fer/.../Notes/99-obsidian/99-AI/memory
db_path        /Users/fer/.local/share/memo/memvec.db
embedder_model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

## Diferencia con `/mv`

`/mv` (alias `/mem_vault`, `/memory`) usa el MCP `mem-vault` (mem0 + Qdrant + Ollama). `/memo` usa el MCP `memo` (sqlite-vec + MLX). Comparten storage layout (mismo `99-AI/memory/` subfolder) y schema de frontmatter, **pero NO comparten el index** — cada uno tiene su propio sqlite/qdrant. Si el user quiere migrar memorias de uno al otro, hay que correr `memo reindex` apuntando al mismo vault path para que memo absorba los `.md` que mem-vault dejó ahí.

No mezcles `/mv` y `/memo` en la misma operación — son dos sistemas paralelos. El user elige cuál usar; vos no decidís por él.

## Errores comunes

- **`memo-mcp: command not found`** → el package no está instalado. Correr `cd ~/repositories/memo && uv pip install -e .`.
- **`Vault path does not exist`** → la vault de Obsidian no está donde el config la espera. Setear `MEMO_VAULT_PATH=...` (default = iCloud Mobile Documents).
- **MCP tools no aparecen en la sesión** → el server no fue registrado o Claude Code no se reinició. Correr `claude mcp add memo -s user memo-mcp` y reabrir Claude Code.
- **`Embedder produced dim=X but config expects 1024`** → swap de modelo embedder sin actualizar `MEMO_EMBEDDER_DIMS`. Sólo aplica si el user cambió el modelo manualmente.
