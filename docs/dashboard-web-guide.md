# Guía exhaustiva del dashboard web de memo (`http://127.0.0.1:8787/`)

> Guía de mantenimiento y modificación. Para CADA número que muestra el dashboard:
> de dónde sale (archivo/DB/log), qué función lo calcula (`archivo:línea`), por qué
> campo JSON viaja, qué función JS lo dibuja, y qué constante/flag hay que tocar
> para afinarlo. Todas las referencias `archivo:línea` apuntan al repo `memo`
> (`/Users/fer/repos/memo`). Si el código cambió, verificá contra la fuente.

---

## 0. TL;DR — el dashboard web son DOS archivos

| Archivo | Rol | Líneas |
|---|---|---|
| `src/memo/cli_dashboard.py` | Servidor HTTP + comando `memo dashboard`. Sirve `/` (snapshot estático) y `/api/data.json` (poll en vivo). Solo localhost. | 97 |
| `web/build.py` | **El cerebro.** `collect_data()` (orquestador) + ~22 colectores + la plantilla front-end `_HTML_TEMPLATE` (HTML/CSS/JS que dibuja cada panel desde el JSON). | 1432 |

Capa de cómputo que `web/build.py` importa (acá viven las fórmulas y los umbrales):

| Archivo | Rol |
|---|---|
| `src/memo/dashboard.py` | Fachada: re-exporta todo desde los módulos de abajo (no tiene lógica). |
| `src/memo/dashboard_metrics.py` | **Las métricas y TODOS los umbrales** (`recall_health`, `grounded_rate`, `verdict`, `consult_breakdown`, `reask_stats`, constantes `*_SCORE`, `VERDICT_*`, `EXPECTED_CONSUMERS`). |
| `src/memo/dashboard_logs.py` | Lectura/escritura de los logs JSONL en `state_dir` (rutas + caps de rotación). |
| `src/memo/cli_roi.py` | `compute_roi` (tiempo/tokens ahorrados; flags `MEMO_ROI_*`). |
| `src/memo/outcome.py` | `detect_gaps` (panel "¿qué le falta saber a memo?"). |
| `src/memo/cli_diag.py` | `_db_health_report` / `_profile_status_report` (datos de los pilares y del doctor). |
| `src/memo/dashboard_panels.py` | `_fetch_memflow_utility` (shell-out a `memflow utility`). |

**Cadena mental para cualquier número:**
`fuente (sqlite / .md / log JSONL) → colector en build.py → campo del JSON → función JS en _HTML_TEMPLATE → nodo del DOM`.

---

## 1. Arquitectura y flujo de datos

### 1.1 Dos modos de la MISMA página

`web/health.html` (328 KB, generado por `web/build.py build`) es un **snapshot estático** que se
abre con `file://`. El comando `memo dashboard` sirve **la misma plantilla** sobre `http://`
y le agrega el endpoint `/api/data.json` que la página pollea, así los números se refrescan
en el lugar sin re-correr el build (`cli_dashboard.py:1-11`).

La página detecta el modo sola (`build.py:1393`):
- `file://` → badge azul "snapshot · usá `memo dashboard` para tiempo real". No pollea.
- `http://` → badge verde "en vivo", `setInterval(poll, refresh_interval_s*1000)` (`build.py:1393-1407`).

### 1.2 La optimización cara-vs-barata (clave)

`collect_data(cfg, *, include_projection=True, limit=1500)` (`build.py:769`) tiene UN solo
paso caro: leer **todos** los vectores de sqlite-vec y proyectarlos a 3-D por PCA/UMAP.

- **Carga inicial de la página** (`cli_dashboard.py:129`): `include_projection=True` → calcula la
  proyección una vez y la incrusta en el HTML del snapshot.
- **Cada poll** (`cli_dashboard.py:67`): `include_projection=False` → se saltea la proyección y solo
  hace un `SELECT COUNT(*) FROM vec` barato (`build.py:824-834`). Por eso el refresco no pesa.

> Consecuencia práctica: el panel 3-D **no** existe en la plantilla web actual (ver §4.10);
> la proyección se calcula y viaja en el JSON pero ningún panel la dibuja todavía.

### 1.3 Ciclo de un request

```
GET /                → shell_html (snapshot construido al arrancar, con proyección)   cli_dashboard.py:62
GET /api/data.json   → collect_data(include_projection=False) + refresh_interval_s     cli_dashboard.py:65-75
                       (si falla: 500 con {"error": "..."} y el daemon sigue vivo)
GET (cualquier otro) → 404 "not found"                                                 cli_dashboard.py:76
```

---

## 2. El servidor — `src/memo/cli_dashboard.py`

### 2.1 Comando `memo dashboard` (`dashboard_cmd`, línea 102)

| Flag | Default | Qué hace | Línea |
|---|---|---|---|
| `--port` | `None` → `MEMO_DASHBOARD_PORT` o `8787` | Puerto de bind (solo `127.0.0.1`). | 103, 114 |
| `--interval` | `5` (mín 1) | Segundos entre polls; se inyecta como `refresh_interval_s`. | 104, 115 |
| `--open/--no-open` | `--open` | Abrir el navegador al arrancar. | 105 |
| `--background`, `-b` | off | Re-ejecuta el comando desacoplado (sobrevive a cerrar la terminal). | 106 |
| `--foreground-only` | off (oculto) | Interno: sirve sin re-spawnear (lo usa `--background`). | 107 |

**Resolución del puerto** (`cli_dashboard.py:114`):
```python
resolved_port = port if port is not None else (flag_int("MEMO_DASHBOARD_PORT") or 8787)
```
Para cambiar el puerto por defecto: `export MEMO_DASHBOARD_PORT=9000` (flag definido en
`flags_misc.py:44`, default `8787`).

### 2.2 Rutas (`_make_handler`, línea 48)

- `do_GET` parte el path en `?` y enruta (`cli_dashboard.py:60-77`).
- `/` y `/index.html` → `shell_html` (el snapshot ya renderizado al arrancar, `:62-64`).
- `/api/data.json` → `collect_data(cfg, include_projection=False)`, le agrega
  `data["refresh_interval_s"] = interval`, serializa con `json.dumps(..., default=str)` (`:65-75`).
  Si `collect_data` tira excepción, responde `500` con `{"error": str(exc)}` y **no** mata el
  servidor (`:70-73`) — un poll malo no tumba el daemon.
- Todo lo demás → `404` (`:76`).
- `log_message` está silenciado (`:50`) — no loguea por request.

### 2.3 Arranque y background

- Foreground (`:128-144`): imprime "Building dashboard…", llama `collect_data(include_projection=True)`,
  `shell_html = builder._render_html(data)`, levanta `ThreadingHTTPServer(("127.0.0.1", port), handler)`,
  `serve_forever()`. `Ctrl-C` → shutdown limpio.
- Background (`_spawn_background`, línea 81): `subprocess.Popen([...,"dashboard","--no-open",
  "--foreground-only","--port",...,"--interval",...], start_new_session=True)`. Log a
  `cfg.state_dir / "dashboard.log"` (`:119`). Imprime el PID y `stop: kill <pid>`.

### 2.4 Cómo se carga el builder (`_load_builder`, línea 33)

`web/build.py` **no** es parte del paquete `memo` — es un script suelto. El servidor lo importa
metiendo `web/` en `sys.path` y haciendo `import build` (`:36-44`). Si falta (instalación rota,
no editable) tira `MemoError`. Por eso esto solo funciona en un checkout/editable install, no
desde un wheel pelado.

---

## 3. El pipeline de recolección — `collect_data` (`web/build.py:769`)

Arma un dict JSON-serializable con **todo**. Orden de operaciones:

1. **`doctor`** (`:776-790`) — `schema`, `runtime` (`_runtime_install_report`), `storage`
   (data_dir/vault_path), `profile` (`_profile_status_report`), `imports` (`_imports_probe`:
   sqlite_vec + mlx), `db` (`_db_health_report`).
2. **`drift`** = `_body_hash_drift(cfg)` (`:792`).
3. **`recall_log`** = `read_recall_log(state_dir, limit=200)` (`:793`).
4. **`recall_health_data`** = `recall_health(state_dir, limit=500)` (`:794`).
5. **`history`** = `_history_recent(cfg, limit=50)` (`:795`).
6. **`contradictions`** = `_contradictions_stats(cfg)` (`:796`).
7. **Paso caro condicional** (`:803-834`): si `include_projection`, lee vectores y PCA; si no,
   `COUNT(*) FROM vec` barato.
8. **`pillars`** (`:836-842`): vector_db, embedder, recall, corpus.
9. **`growth`** = `_growth_by_day(history, days=30)` (`:844`).
10. Ensambla **`data`** (`:846-889`) y lo devuelve.

### 3.1 Forma del JSON (`data`, `build.py:846-889`) — verificada contra el endpoint vivo

Claves de primer nivel: `generated_at`, `memo_version`, `pillars`, `projection`, `type_palette`,
`type_counts`, `growth`, `history`, `recall_log`, `recall_util`, `bail_breakdown`, `usefulness`,
`doctor_raw`, `contradictions`, `verdict`, `memflow_util`, `gerencial`, `gaps`, `sync`
(+ `refresh_interval_s` que agrega el servidor en el poll).

> **Importante para modificar:** el JSON es MÁS RICO que lo que se dibuja. Hoy la plantilla web
> solo consume `verdict`, `gerencial`, `usefulness`, `gaps`, `pillars`, `sync`, `memo_version`,
> `generated_at` y `refresh_interval_s`. Todo lo demás (`projection`, `type_counts`, `growth`,
> `history`, `recall_log`, `recall_util`, `bail_breakdown`, `doctor_raw`, `contradictions`,
> `memflow_util`) **ya está disponible** para un panel nuevo sin tocar el backend.

---

## 4. Referencia panel por panel (lo que SÍ se dibuja)

La plantilla está en `build.py:940-1415`. El `<script>` (`:1156-1411`) parsea el payload, define
`render(DATA)` (`:1182-1385`) y arranca el poll. Estructura del DOM en `:1077-1153`.

Para cada panel: **DOM → función/línea JS → campo JSON → colector (build.py) → fuente real**.

### 4.1 Cabecera y footer

| Dato | DOM | JS | Campo | Origen |
|---|---|---|---|---|
| Fecha del dato | `#meta-stamp` | `:1187` | `DATA.generated_at` | `datetime.now(UTC)` en `collect_data` (`:847`) |
| Versión memo | `#memo-version` | `:1188` | `DATA.memo_version` | `doctor.runtime.memo_version` = `_runtime_install_report()` (`:848`) |
| Estado sistema | `#sys-status` | `:1368-1375` | `DATA.pillars[].status` | cuenta `red`/`yellow` de los 4 pilares |
| Chip GitHub sync | `#sync-status` | `:1377-1384` | `DATA.sync.{state,label}` | `_sync_health(cfg)` (`build.py:641`) |
| Badge en vivo | `#live-badge` | `:1390-1410` | `refresh_interval_s` + `location.protocol` | servidor / navegador |

### 4.2 VEREDICTO (hero) — "¿está funcionando como memoria?"

- DOM: `#verdict-panel` (`:1079-1089`). JS: `:1190-1204`.
- Campo: `DATA.verdict` → `verdict(state_dir, limit=500)` (`dashboard_metrics.py:520`).
- Qué muestra:
  - `status` ∈ {`ok`,`weak`,`unmeasured`,`unused`} elige glyph/color/texto del mapa
    `VERDICT_COPY` (`build.py:1171-1180`).
  - `label` (ej. "✅ ÚTIL") se limpia del emoji con regex (`:1197`).
  - Stat derecho: `consults_sampled / consults_total` (`:1200-1203`).
- **Lógica del veredicto** (`dashboard_metrics.py:541-551`):
  1. `consults_sampled < VERDICT_MIN_CONSULTS (20)` → **unused** "❌ NO SE USA".
  2. `measured_turns < 1` **o** `measurement_coverage < 0.05` → **unmeasured** "⚠️ SE LEE PERO NO SE MIDE".
  3. `grounded_rate < VERDICT_MIN_GROUNDED (0.10)` → **weak** "⚠️ SE LEE PERO NO AYUDA".
  4. Si no → **ok** "✅ ÚTIL".
- `consults_sampled` = `recall_health.sampled` (muestra ambiente real, no el flood de evals).
  `consults_total` = suma de `daily_trend.json` (acumulado histórico).

### 4.3 EMBUDO — "De cada pregunta, ¿cuánto aporta memo?"

- DOM: `#funnel` (`:1092-1096`). JS: `:1206-1217`.
- Campo: `DATA.gerencial.funnel` (lista de 3 etapas, `build.py:698-717`).
- Las 3 etapas (label + sub + value):
  1. **Consultas totales** = `consults_total` (acumulado en `daily_trend.json`).
  2. **Consultas analizadas** = `sampled` (`recall_health.sampled`).
  3. **Activaciones históricas** = `activated_total` (suma de `activado` en `daily_trend.json`).

### 4.4 KPIs — 8 tarjetas (`build.py:1219-1258`)

Construidas en el array `kpis` (`:1230-1248`), dibujadas en `:1249-1258`. Todas leen de
`DATA.gerencial` (`G`). Colores por umbral inline en JS:

| # | Tarjeta | Campo JSON | Origen del número | Color (umbral JS) |
|---|---|---|---|---|
| 1 | Consultas totales | `G.consults_total` | suma `consultas` de `daily_trend.json` (`_gerencial:692`) | azul fijo |
| 2 | Consultas analizadas | `G.consults_sampled` | `recall_health.sampled` | amarillo fijo |
| 3 | Activación histórica | `G.activation_rate_total` (`activated_total/consults_total`, `:740`) | `daily_trend.json` | `≥0.7` verde, si no amarillo (`:1223`) |
| 4 | Hit rate de la muestra | `G.hit_rate` | `recall_health.hit_rate` (with_hits/fired) | verde fijo |
| 5 | Hechos reutilizados | `G.grounded_rate` | `grounded_rate()` (`metrics:174`) | `≥0.1` verde, si no amarillo (`:1224`) |
| 6 | Sus datos se usaron | `G.used_rate` (+`used_grounded`/`used_total`) | `answer_rate_knowledge` o `answer_rate` (`_gerencial:720`) | `≥0.6` verde / `≥0.35` amarillo / rojo (`:1221`) |
| 7 | Cobertura de medición | `G.measurement_coverage` (+`measured_turns`/`surfaced_turns`) | `grounded_rate()` (`metrics:238`) | amarillo fijo |
| 8 | Ahorro neto de tokens hoy | `G.tokens_net_today_human` | `_token_savings.today_net` (`build.py:630`) | `≥0` azul, si no rojo (`:1246`) |

`asPct(x)` = `(x*100).toFixed(0)+"%"` (`:1159`). `fmtTok` formatea k/M (`:1226`).

### 4.5 AHORRO DE TOKENS (detalle) (`build.py:1101-1124` DOM, `:1260-1293` JS)

- Campo: `DATA.gerencial.token_detail` = `_token_savings(state_dir, days=14)` (`build.py:531`).
- **Número grande** `#tok-total` = `td.total` (BRUTO = grounded + reask), formateado (`:1267`).
- **Línea de supuestos** `#tok-assump` (`:1268-1271`):
  `<tok_grounded> tok/hecho · <tok_reask> tok/repregunta · −<context_tokens> contexto → neto <net>`
  (+ `~<avg_answer_tokens> tok/respuesta medido` si existe).
- **Barra de composición** (verde=grounded, azul=reask): anchos `gTok/tTot` y el resto (`:1272-1278`).
- **Gráfico diario** `#token-trend` (Plotly bar, `:1280-1293`): `x=daily[].date`, `y=daily[].net_tokens`,
  hover muestra `grounded` (hechos). Solo barras de **net_tokens** (verde `#2ee6a6`).
- **Cómo se calcula** (`_token_savings`, `build.py:531-638`):
  - `tok_grounded = flag_int("MEMO_ROI_TOKENS_PER_GROUNDED") or 350` (`:547`).
  - `tok_reask = flag_int("MEMO_ROI_TOKENS_PER_REASK") or 900` (`:548`).
  - Cuenta por día las filas de `grounding.log` con `used_score ≥ GROUNDED_SCORE (0.6)`, deduplicadas
    por `(session_id, turn, recall_id)` (`:555-570`).
  - Costo de contexto por día desde `context_cost.log` (`tokens_est = (chars+3)//4`) (`:572-582`).
  - `daily[].net_tokens = max(0, grounded*tok_grounded − context_tokens)` (`:601`) — piso en 0 a
    propósito (un día sin ahorro medido es "ahorró 0", no "te costó tokens", ver comentario `:597-600`).
  - `total = grounded_tokens + reask_tokens`; `net = max(0, total − context_tokens)` (`:619,632`).
  - `reask_avoided` viene de `reask_stats` (`:606-610`).

### 4.6 ¿QUIÉN USA MEMO? (`build.py:1126-1132` DOM, `:1295-1325` JS)

- Campo: `DATA.usefulness` = `_usefulness(cfg)` (`build.py:464`), que envuelve
  `consult_breakdown(state_dir, limit=500)` (`metrics:442`) y `reask_stats(state_dir, limit=500)`.
- **Tabla de consumidores** `#tools` (`:1298-1315`): ordenada por `consults` desc. Por cada tool:
  - Barra proporcional a `consults/maxC` (`:1307`).
  - Estado: `consults < 5` → "uso esporádico" (amarillo); si no "uso activo" (verde) (`:1303-1305`).
  - Sub-línea: `usa <grounded_rate>%` o `hit <hit_rate>%` + `ago(last_seen)` (`:1306`).
  - `helping` = `grounded_rate ≥ 0.10` (`:1301`) — hoy no cambia el color pero está calculado.
- **Callout silencioso** `#silent-callout` (`:1316-1325`): `DATA.usefulness.silent` = consumidores de
  `EXPECTED_CONSUMERS` que no aparecen ni en la ventana ni en `consumer_last_seen.json` (< 30 días)
  (`metrics:504-516`). Si vacío: vacío (sin mensaje).
- **De dónde sale `consult_breakdown`** (`metrics:442-517`): deduplica `recall.log`, agrupa por
  `consumer_label(row)` (client → source → via→claude-code/mcp/bail, `metrics:425-439`), cuenta
  consults/fired/with_hits/strong, calcula `hit_rate`, `grounded_rate` (cruzando con
  `grounding.log`), `last_seen`.

### 4.7 VACÍOS DE CONOCIMIENTO (`build.py:1134-1139` DOM, `:1327-1345` JS)

- Campo: `DATA.gaps` = `_gaps(cfg, top=12)` → `detect_gaps(cfg.state_dir)[:12]` (`build.py:660`,
  `outcome.py:185`).
- Cada fila: `count ×` + `prompt` (90 chars) + `reasons` (`:1335-1343`). Si vacío:
  "✅ Sin vacíos detectados…".
- **Qué cuenta como vacío** (`detect_gaps`, `outcome.py`): prompt **de conocimiento**
  (`_is_knowledge_prompt`), **ambiente** y de **sesión real**, cuyo recall: (a) bailó por
  `min_sim`/"no hits" → "sin coincidencias"; (b) 0 hits → "0 resultados"; (c) mostró algo pero el
  turno fue medido y no se usó → "encontró algo pero no se usó". Clusteriza por Jaccard
  (`sim_threshold=0.6`) y ordena por frecuencia/recencia.

### 4.8 USO EN EL TIEMPO (tendencia) (`build.py:1141-1146` DOM, `:1347-1366` JS)

- Campo: `DATA.gerencial.trend` = `_consult_trend(state_dir, days=14)` (`build.py:487`).
- Plotly **barras apiladas** `#trend` (`:1351-1365`): verde "consultadas" = `activado`; gris
  "omitidas" = `max(0, consultas − activado)`.
- **De dónde sale** (`_consult_trend`): primario = `daily_trend.json` (acumulador persistente,
  `dashboard_logs.py:139`); además fusiona el `recall.log` de HOY por si el JSON va atrasado
  (`activado` = `via ∈ {daemon, subprocess}`) (`build.py:497-518`).

### 4.9 GitHub sync chip (footer) (`build.py:1377-1384` JS)

- Campo: `DATA.sync` = `_sync_health(cfg)` (`build.py:641`), que llama `sync_status(cfg)`
  (`sync_git.py`). Estados: `off` (data_dir no es repo git), `bad` (commits varados), `warn`
  (ahead/dirty), `ok` (al día). El color sale del mapa `sc` (`:1381`).

### 4.10 Lo que NO se dibuja todavía (pero viaja en el JSON)

`projection` (PCA 3-D, `build.py:809-817`), `type_counts` + `type_palette` (`_TYPE_COLORS`,
`build.py:448-456`), `growth` (saves/día 30d, `_growth_by_day`), `history` (50 últimos eventos),
`recall_log` (20 últimos), `recall_util` (recall_health completo, `build.py:856-879`),
`bail_breakdown` (`metrics:54`), `contradictions` (`_contradictions_stats`), `doctor_raw`,
`memflow_util` (`_fetch_memflow_utility`: `memflow utility --since-days 7 --json`, timeout 8s).
**Para agregar un panel basta con dibujar uno de estos campos** (ver §7.2).

---

## 5. Las fuentes de datos (rutas exactas)

Todo cuelga de `cfg.state_dir` (default `_DEFAULT_STATE_DIR`, override `MEMO_STATE_DIR`;
`config.py:201`) salvo los `.md` que cuelgan de `cfg.memory_dir`.

### 5.1 Bases sqlite (`config.py`)

| Path (propiedad Config) | Default | Usado por |
|---|---|---|
| `cfg.db_path` | `state_dir/memvec.db` (`:396`) | `_read_vectors`, `_body_hash_drift`, pilar vector/corpus |
| `cfg.history_db` | `state_dir/history.db`† (`:401`) | `_history_recent`, `_growth_by_day` |
| `cfg.contradictions_db` | `state_dir/contradictions.db`† (`:419`) | `_contradictions_stats` |
| `cfg.memory_dir` | derivado de `data_dir` (`:377`) | `_body_hash_drift` (lee los `.md`) |

† Con `MEMO_SINGLE_DB=1` estas colapsan sobre `memvec.db` (`config.py:404,422`).

Consultas SQL concretas:
- Vectores: `SELECT vec.id, vec.embedding, meta.title, meta.type, meta.tags, meta.created,
  meta.updated FROM vec JOIN meta ON meta.id = vec.id ORDER BY meta.updated DESC LIMIT N`
  (`build.py:85-91`). El blob se desempaqueta con `struct.unpack("<Nf")` (`:99-100`).
- History: `SELECT ts, op, record_id, title, type FROM events ORDER BY ts DESC LIMIT ?`
  (`build.py:198-202`).
- Contradicciones: `SELECT status, count(*) FROM pairs GROUP BY status` (`build.py:236-238`).
- Drift: `SELECT id, path, body_hash FROM meta` + hash de cada `.md` con `_sha256_short` (16 hex,
  `build.py:54-55,163-188`).
- Conteo de la DB / pilares: `_db_health_report` (`cli_diag.py:163`) saca `records` (count de `meta`),
  `vec_dims` (`_sqlite_vec_dims`), `expected_dims = cfg.embedder_dims`, `size_bytes`,
  `integrity_check`, `latest_memory_update`.

### 5.2 Logs JSONL en `state_dir` (`dashboard_logs.py`)

| Archivo | Ruta (fn) | Cap líneas / tamaño | Lector | Alimenta |
|---|---|---|---|---|
| `recall.log` | `recall_log_path` (`:12`) | 200 / 200 KB (`:113`) | `read_recall_log` | tendencia, consumidores, gaps, reask |
| `recall_hook.log` | `recall_hook_log_path` (`:16`) | 2000 / 2 MB (`:115`) | `read_recall_hook_log` | `recall_health`/`grounded_rate` (ambiente) |
| `grounding.log` | `grounding_log_path` (`:230`) | 1000 / 200 KB (`:270`) | `read_grounding_log` | hechos usados, ahorro de tokens |
| `context_cost.log` | `context_cost_log_path` (`:20`) | 1000 / 400 KB (`:202-207`) | `read_context_cost_log` | costo de contexto (neto) |
| `usage.log` | `usage_log_path` (`:214`) | 500 / 100 KB (`:223`) | `read_usage_log` | `referenced_rate` |
| `grounding_diag.log` | `grounding_diag_log_path` (`:277`) | 500 / 100 KB (`:297-302`) | `read_grounding_diag_log` | diagnóstico del veredicto |
| `daily_trend.json` | `state_dir/daily_trend.json` (`:126`) | acumulador `{día:{consultas,activado}}` | `read_daily_trend` | totales históricos, tendencia |
| `consumer_last_seen.json` | `state_dir/consumer_last_seen.json` (`:150`) | `{consumer: ts}` | `read_consumer_last_seen` | detección de "silenciosos" |

Rotación: `_write_jsonl_entry` (`dashboard_logs.py:24-39`) trunca a las últimas `cap` líneas cuando
el archivo supera `size_limit`.

---

## 6. Las métricas en detalle (`dashboard_metrics.py`)

### 6.1 `recall_health(state_dir, limit=200)` (`:296`)

Cuenta el embudo de recall **ambiente** (tu uso real), no las búsquedas explícitas de un agente:
- Fuente de verdad = `recall_hook.log` filtrado por `is_ambient_recall` (`via ∈ {daemon, subprocess,
  bail, daemon_error}`, `:378-382`), deduplicado con `dedup_double_fire` (ventana 15 s, `:95`).
- Descarta sesiones de eval throwaway con `filter_real_sessions` (≥2 turnos, `:385-422`).
- `fired` = `via ∈ {daemon, subprocess}`; `bailed` = `via == bail`.
- `hit_rate = with_hits/fired`; `strong_hit_rate = (score > STRONG_SCORE 0.85)/fired`.
- Delega a `grounded_rate()` y `referenced_rate()` para lo de "se usó".

### 6.2 `grounded_rate(state_dir)` (`:174`) — el corazón del "se usó"

- Fusiona `recall_hook.log` + `recall.log`, dedup por `(session_id, turn)` (`:191-201`).
- `surfaced_by_turn` = qué memorias se mostraron por turno (`:202-214`).
- `scored_turns` / `grounded_keys` = turnos que el detector de grounding midió y los `(sid,turn,rid)`
  que `grounding_used()` marca como usados (`:217-231`).
- **Denominador honesto**: solo turnos efectivamente medidos cuentan (`measured`, `:235`); los no
  medidos se excluyen, no se cuentan como miss.
- `grounded_rate = grounded / measured_surfaced`; `answer_rate = answers_grounded / measured_turns`;
  `answer_rate_knowledge` = igual pero solo sobre prompts de conocimiento (`_is_knowledge_prompt`,
  `:162-171`). `measurement_coverage = measured_turns / surfaced_turns` (`:238`).
- **`grounding_used(row)`** (`:86-92`): `used_score ≥ USED_SCORE_STRONG (0.8)` **o**
  `specific_score ≥ SPECIFIC_MARGIN (0.06)` **o** hay `downstream_action`.

### 6.3 `consult_breakdown` (`:442`), `verdict` (`:520`), `reask_stats` (`:596`)

- `consult_breakdown`: por-consumidor sobre `recall.log` (ver §4.6). "Silencioso" = falta < 30 días
  (`:513`).
- `verdict`: combina volumen (`consults_sampled`) + outcome (`grounded_rate`) en un estado (ver §4.2).
- `reask_stats(window_turns=4, sim_threshold=0.6)`: cuenta repreguntas evitadas — turnos grounded
  cuyo prompt NO recurre con Jaccard ≥ 0.6 dentro de 4 turnos (`:596-640`). Tokeniza con
  `_reask_tokens` (regex `[a-z0-9]{3,}`, stopwords ES/EN, `:580-585`).

---

## 7. Recetas de modificación

### 7.1 Afinar un número (cambiar un umbral)

La mayoría son **constantes en `dashboard_metrics.py`** o **flags `MEMO_*`**. Ver tabla §8.
Flujo: editar la constante/flag → reiniciar `memo dashboard` (o esperar el próximo poll si es flag
leído en cada `collect_data`) → `curl -s http://127.0.0.1:8787/api/data.json | jq` para verificar.

Ejemplos:
- "El veredicto pide demasiadas consultas" → bajar `VERDICT_MIN_CONSULTS` (`metrics:31`).
- "Cuenta como 'usado' cosas que no lo son" → subir `USED_SCORE_STRONG` (`metrics:23`).
- "El ahorro de tokens infla/desinfla" → `export MEMO_ROI_TOKENS_PER_GROUNDED=...` /
  `MEMO_ROI_TOKENS_PER_REASK=...` (defaults 350 / 900, `flags_misc.py:354,364`).
- "Color del KPI used_rate" → umbrales JS inline `0.6 / 0.35` (`build.py:1221-1222`).

### 7.2 Agregar un panel nuevo

Si usa un campo que **ya viaja** (§4.10): solo front-end.
1. Agregar el `<section class="panel">` con un `<div id="mi-panel">` en `build.py:1077-1147`.
2. Dentro de `render(DATA)` (`:1182-1385`) leer `DATA.<campo>` y poblar el DOM (copiar el patrón de
   `#funnel`/`#tools`).
3. Reconstruir el snapshot estático: `python web/build.py` (regenera `web/health.html`) y/o reiniciar
   `memo dashboard`.

Si necesita un dato **nuevo**: agregar un colector en `build.py`, sumar su clave al dict `data`
(`:846-889`), y recién ahí dibujarlo. Si es un número por-día caro, respetar el patrón
`include_projection` (no meterlo en el path del poll si es lento).

### 7.3 Cambiar el intervalo de refresco o el puerto

- Intervalo: `memo dashboard --interval 10`. Viaja como `refresh_interval_s` y lo usa el `setInterval`
  (`build.py:1394,1406`).
- Puerto: `memo dashboard --port 9000` o `export MEMO_DASHBOARD_PORT=9000`.

### 7.4 Editar estilos / colores

Tokens CSS en `:root` (`build.py:948-962`): `--green #2ee6a6`, `--yellow #fbbf24`, `--red #fb7185`,
`--blue #5b9dff`, etc. Colores de tipo de memoria en `_TYPE_COLORS` (`build.py:448-456`).
Los colores de Plotly están hardcodeados en JS (`#2ee6a6`, `#3a4a68`, grids `#243049`).

### 7.5 Regenerar el snapshot estático

```bash
python web/build.py            # escribe web/health.html (con proyección 3-D)
python web/build.py --open     # y lo abre en el navegador
python web/build.py --limit 500   # menos memorias a proyectar (más rápido)
```
`build()` está en `build.py:893-906`; CLI en `main()` (`:1418-1428`).

---

## 8. Tabla de TODOS los números afinables

### 8.1 Constantes en `dashboard_metrics.py`

| Constante | Valor | Línea | Efecto |
|---|---|---|---|
| `STRONG_SCORE` | `0.85` | 15 | Umbral de "hit fuerte" (`strong_hit_rate`, breakdown). |
| `GROUNDED_SCORE` | `0.6` | 17 | Barra para que una fila de `grounding.log` cuente en reask/ahorro diario. |
| `USED_SCORE_STRONG` | `0.8` | 23 | "La respuesta USÓ la memoria" (grounded_rate, KPI 5/6). |
| `SPECIFIC_MARGIN` | `0.06` | 28 | Recuperación de paráfrasis: cuánto más matchea la respuesta que la pregunta. |
| `VERDICT_MIN_CONSULTS` | `20` | 31 | Menos que esto → veredicto "NO SE USA". |
| `VERDICT_MIN_GROUNDED` | `0.10` | 32 | Menos que esto (con medición) → "SE LEE PERO NO AYUDA". |
| `VERDICT_MIN_MEASURED_TURNS` | `1` | 37 | Mínimo de turnos medidos para salir de "unmeasured". |
| `VERDICT_MIN_MEASUREMENT_COVERAGE` | `0.05` | 40 | Cobertura mínima para juzgar utilidad. |
| `EXPECTED_CONSUMERS` | claude-code, synapse, memflow, codex, blackbox | 45 | Quiénes se marcan "silenciosos" si faltan. |
| `dedup_double_fire(window_s)` | `15.0` | 95 | Ventana para colapsar doble-disparo del mismo prompt. |
| silent cutoff | `30 días` | 513 | Antigüedad máxima para no marcar un consumidor como silencioso. |
| `reask_stats(window_turns, sim_threshold)` | `4`, `0.6` | 596 | Ventana y similitud para contar una repregunta. |

### 8.2 Constantes / límites en `web/build.py`

| Qué | Valor | Línea | Efecto |
|---|---|---|---|
| `collect_data(limit)` | `1500` | 769 | Máx. memorias leídas para la proyección 3-D. |
| `recall_log` limit (payload) | `200` | 793 | Cuántas filas de recall.log se leen. |
| `recall_health` / `consult_breakdown` / `reask` limit | `500` | 794, 471, 475 | Tamaño de muestra reciente. |
| `_history_recent` limit | `50` | 795 | Eventos de history para growth/corpus. |
| `_consult_trend(days, limit)` | `14`, `1000` | 487 | Ventana de la tendencia. |
| `_token_savings(days)` | `14` | 531 | Ventana del gráfico de ahorro. |
| `_growth_by_day(days)` | `30` | 844 | Ventana del growth (no dibujado hoy). |
| `_gaps(top)` | `12` | 660 | Máx. de vacíos mostrados. |
| `history[:20]`, `recall_log[:20]` | `20` | 854-855 | Recortes del payload. |
| `_TYPE_COLORS` | paleta | 448-456 | Colores por tipo de memoria. |
| tokens defaults | `350` / `900` | 547-548 | Fallback si los flags no están seteados. |

### 8.3 Umbrales inline en el JS (`web/build.py`, dentro de `_HTML_TEMPLATE`)

| Qué | Valor | Línea | Efecto |
|---|---|---|---|
| Color `used_rate` | `≥0.6` verde / `≥0.35` amarillo / rojo | 1221-1222 | KPI 6. |
| Color `coverage_rate` | `≥0.7` verde | 1223 | KPI 3. |
| Color `grounded_rate` | `≥0.1` verde | 1224-1225 | KPI 5. |
| `helping` consumidor | `grounded_rate ≥ 0.10` | 1301 | Etiqueta de la tabla de tools. |
| "uso esporádico" | `consults < 5` | 1303 | Estado del consumidor. |
| Default refresh | `5` s | 1394 | Si falta `refresh_interval_s`. |

### 8.4 Flags `MEMO_*` (`flags_misc.py`)

| Flag | Default | Línea | Efecto |
|---|---|---|---|
| `MEMO_DASHBOARD_PORT` | `8787` | 44 | Puerto del servidor. |
| `MEMO_ROI_TOKENS_PER_GROUNDED` | `350` | 354 | Tokens estimados ahorrados por hecho reutilizado. |
| `MEMO_ROI_TOKENS_PER_REASK` | `900` | 364 | Tokens estimados por repregunta evitada. |
| `MEMO_ROI_SECS_PER_GROUNDED` | `30` | 278 | Segundos ahorrados por hecho (panel ROI/TUI). |
| `MEMO_ROI_SECS_PER_REASK` | `120` | 286 | Segundos por repregunta (panel ROI/TUI). |
| `MEMO_SINGLE_DB` | off | — | Colapsa history/contradictions sobre `memvec.db`. |
| `MEMO_STATE_DIR` | `_DEFAULT_STATE_DIR` | `config.py:201` | Dónde viven DB + logs. |

---

## 9. Verificación

```bash
# 1. ¿Está vivo y qué sirve?
lsof -nP -iTCP:8787 -sTCP:LISTEN
curl -s http://127.0.0.1:8787/api/data.json | jq 'keys'
curl -s http://127.0.0.1:8787/api/data.json | jq '.gerencial | keys'
curl -s http://127.0.0.1:8787/api/data.json | jq '.verdict.status, [.pillars[] | .label+"="+.status]'

# 2. Reconstruir el snapshot estático tras tocar la plantilla
python web/build.py            # → web/health.html

# 3. Reiniciar el servidor (foreground)
memo dashboard --no-open --port 8787 --interval 5
#   o en background:
memo dashboard --background

# 4. Tests del dashboard
uv run --no-sync pytest tests/test_dashboard.py tests/test_dashboard_build.py -v
```

**Checks rápidos al modificar:**
- ¿Tocaste un colector? → `curl .../api/data.json | jq '.<campo>'` debe reflejar el cambio.
- ¿Tocaste la plantilla? → regenerá `web/health.html` Y reiniciá `memo dashboard` (el snapshot
  inicial se construye al arrancar, `cli_dashboard.py:129-131`).
- ¿Un poll responde 500? → `{"error": ...}` en el body (`cli_dashboard.py:71`); el colector que rompió
  está en el traceback de `cfg.state_dir/dashboard.log` si corre en background.

---

## 10. Mapa de un vistazo (panel → función → fuente)

| Panel | Campo JSON | Colector (build.py) | Fuente real |
|---|---|---|---|
| Veredicto | `verdict` | `verdict()` (metrics:520) | recall_hook.log + grounding.log + daily_trend.json |
| Embudo | `gerencial.funnel` | `_gerencial` (671) | daily_trend.json + recall_health |
| KPIs (8) | `gerencial.*` | `_gerencial` (671) | recall_health + token_savings |
| Ahorro tokens | `gerencial.token_detail` | `_token_savings` (531) | grounding.log + context_cost.log + flags ROI |
| Quién usa memo | `usefulness` | `_usefulness` (464) | recall.log + grounding.log + consumer_last_seen.json |
| Vacíos | `gaps` | `_gaps`→`detect_gaps` (660 / outcome.py:185) | recall.log + grounding.log |
| Tendencia | `gerencial.trend` | `_consult_trend` (487) | daily_trend.json + recall.log |
| Sys-status | `pillars` | `_pillar_*` (251-443) | doctor (`_db_health_report`, profile, imports) + drift |
| Sync chip | `sync` | `_sync_health` (641) | `sync_status` (sync_git.py) |
| (sin dibujar) | `projection`, `growth`, `type_counts`, `recall_util`, `history`, `recall_log`, `contradictions`, `memflow_util`, `doctor_raw` | varios | memvec.db, history.db, contradictions.db, memflow |
