# Roadmap gamechangers — design doc

- **Fecha**: 2026-07-13
- **Estado**: aprobado por Fer (alcance: roadmap completo por fases)
- **Método**: workflow ultracode `memo-gamechanger-scan` (run `wf_38c809eb-0d1`, 36 agentes): 5 lectores (código, docs, memorias+grafo, señales vivas, ecosistema) → 5 lentes de ideación (20 ideas) → dedup (12) → panel adversarial por idea (auditor de implementabilidad que leyó el código real + escéptico de impacto) → síntesis.
- **Veredicto general**: ninguna idea sobrevivió el panel como "gamechanger" literal; las 6 finalistas son **incremental-fuerte**, todas implementables hoy, ninguna re-litiga decisiones cerradas (graph re-rank, flip hybrid, reranker batching, version-file update, cognition layer v3.3.0).

## TL;DR

| # | Idea | Score (impl/impacto) | Esfuerzo | Veredicto en una línea |
|---|------|------|----------|------------------------|
| 1 | **HyPE** (índice pregunta-space en dream) | 8/6 | 1.5-2.5 sem | La palanca de retrieval más grande que no re-litiga nada; ataca el bucket más débil medido (multi_session_synthesis 0.493) con cero latencia en el hook |
| 2 | **Chronicle** (diario nocturno en el vault) | 8/6 | 4-8 días | Mejor ratio valor/esfuerzo del lote; el mejor asset de demo del roadmap; patrón `dream_profile` ya probado |
| 3 | **`memo onboard`** (mitad buena del funnel) | 8/6 | ~1 sem | Quick-win: backfill Day-0 + hook + import en un wizard; el bootstrap Node MCPB queda como apuesta barata secundaria |
| 4 | **Gateway** (proxy OpenAI-compat) | 8/6 | 2-3 sem | Sólida pero demanda flaca en datos propios (agentes target con 0 consults porque casi no se usan); scope v1 a opencode+hermes; **diferida** |
| 5 | **Total Recall** (tier verbatim) | 8/6 | 2-3 sem | Bien encajada pero espacio comoditizado (claude-mem); si se hace, arrancar FTS5-only |
| 6 | **Recall RCT** (`memo proof`) | 8/5 | 1-2 sem + espera | Recortar a "modo campaña": a 5% de holdout nunca junta n; medio pipeline ya existe (`ablation_stats`) |

---

## 1. HyPE — índice en espacio-pregunta generado en dream

**Qué es.** Cada memoria durable gana 2-3 vectores de "preguntas que esta memoria responde", generados de noche por el LLM local; el recall hace max-fold `max(doc_sim, max(question_sims))` y encuentra memorias por lo que responden, no por las palabras que contienen.

**Por qué importa.** Único finalista que ataca directo el techo de calidad de retrieval (mismatch pregunta↔afirmación) sin tocar nada cerrado: no re-litiga el default vec, ni el grafo (medido negativo), ni el reranker. Cero costo en el hook: generación nocturna; el knn extra sobre sqlite-vec son milisegundos. `docs/eval/capability-baseline-and-levers.md` prescribe exactamente esta familia para el bucket más débil. Honestidad: el claim de moat es falso (mem0 podría copiarlo en un sprint) y el prefijo asimétrico de Qwen3 ya cubre parte del gap — el headroom real se sabe recién con el eval.

**Plan** (flags default-OFF en `flags_ingest.py`: `MEMO_HYPE_ENABLED`, `MEMO_HYPE_QUESTIONS_PER_MEMORY=3`, `MEMO_HYPE_FOLD=max`):

1. **Storage**: tabla sidecar `hype_vec` en `store/schema.py` — patrón `fact_edge_store.py` (derivada del markdown, rebuildable, re-join por id estable, compatible con `reindex --rebuild` con regen lazy). Cache content-addressed estilo `repo_embedding_cache` (`schema.py:103`), keyed `model+dims+sha256(pregunta)`.
2. **Pase dream**: `dream_hype.py` nuevo + `_run_hype` en `cli_dream_passes.py` (receipt/errors gratis). Solo tier durable (`tiers.py`, excluye reference), watermark por `body_hash` (incremental), generación vía `llm.py` (import diferido), dedup + cap por memoria, backlog priorizado por `roi_score` (`outcome.py`) para cubrir el corpus en 2-3 noches. Nunca 2 cold-loads MLX concurrentes (exit-144).
3. **Read path**: knn sobre `hype_vec` en `store/queries.py`. Atención: el fold en una sola pasada (UNION-ALL de dos knn + MAX con GROUP BY por parent id) es código nuevo — `SEARCH_CHUNK_PARENT` NO es precedente (es mapeo post-search chunk→parent, no GROUP BY en vec0). Debe preservar la semántica de filtros type/tag/date de `VecStore.search` (`queries.py:682`). En `rank_hits` (`recall_logic.py`) el fold entra ANTES de boosts.
4. **Decisión de espacio**: embebe preguntas con `embedder.embed_query` (CON prefijo, tabla separada, query↔pregunta simétrico) pero el eval A/B-ea también la variante sin prefijo — el gate decide, no la intuición.
5. **Medición del hook**: p95 con hype ON antes de flipear (índice ×4 filas ≈ +100-300ms warm en 2560 dims — dentro del budget de 5s, pero se mide, no se asume).

**Esfuerzo**: 1.5-2.5 semanas. **Riesgos**: calidad de preguntas del LLM local (riesgo empírico real — si no mueve el gate, shippea dark); fold SQL sobreviviendo migración de dims; primera noche = backlog 4.6k×3 (batchear).

**Cómo se mide.** Triple gate, en orden: (a) `memo eval recall --gate` — prec@K sube, noise@K queda en 0.0 sobre TODO el regression set; (b) el gate curado solo NO alcanza (saturado en prec@5=0.2 máximo teórico): condición de flip = mejora en bench per-bucket (`memo eval bench`, específicamente multi_session_synthesis) + labels harvested; (c) p95 del hook con hype ON. Si (b) no mejora, no se enciende — sin excepción.

---

## 2. Chronicle — diario de ingeniería nocturno

**Qué es.** Pase de dream que cada noche redacta la crónica del día en markdown humano dentro del vault (`<vault>/<SYSTEM_DIR>/AI/chronicle/YYYY-MM-DD.md`): qué se trabajó, decidió, contradijo, cuántos tokens ahorró memo — con provenance linkeado a ids.

**Por qué importa.** Superficie de consumo humano sobre el exhaust que solo memo tiene (episodios + uso real + ledger + contradicciones). El asset de demo más mostrable del roadmap ("una página Obsidian que se escribe sola" > prec@5). Honestidad: `dream_profile.py` ya es exactamente este patrón (dream pass → markdown humano con provenance, default-OFF) — Chronicle es la misma plantilla en eje temporal, no categoría nueva. No mejora recall ni ahorra tokens: es reporting.

**Plan** (flags en `flags_misc.py`: `MEMO_DREAM_CHRONICLE_ENABLED`, `MEMO_CHRONICLE_WEEKLY`, default-OFF):

1. `dream_chronicle.py` + `_run_chronicle` en `cli_dream_passes.py` (patrón idéntico a anticipate/consolidate/profile en `cli_dream.py`).
2. Readers ya shipped: `EpisodeStore.recent()` (`episode_store.py:159`) por proyecto/día, `grounding.log` (qué se USÓ), `token_ledger.py` (`grounded_by_day`/`summarize`), receipt previo (`state_dir/dream/last.json`), `identity.py`.
3. Un call a `llm.py` con prompt en `memory/prompts.py`: 300-500 palabras, cada afirmación cita id de memoria/episodio; lo no-citable se descarta (patrón "never fabricates" de anticipate). `dream_consolidate._llm_synthesize` es la plantilla directa del call.
4. Escribe en `AI/chronicle/` — ya excluido del ingest reference (no se double-indexa); hand-edit humano gana.
5. `memo chronicle [--week|--month]` agregador on-demand. Umbral mínimo de señal para días vacíos.
6. **Recortes obligados**: (a) provenance a commits de repos de trabajo NO tiene lector — `identity.py` solo atribuye commits del repo memo-sync; recortar ese claim o presupuestar un reader git aparte; (b) fallback sin `vault_path` definido (data_dir/chronicle/ o skip).

**Esfuerzo**: 4-8 días. **Riesgos**: la prosa del LLM local ES el producto — un diario aburrido o alucinado lo mata en semana 1; días vacíos; contenido generado en el vault del humano (carpeta propia + default-OFF = borrar trivial).

**Cómo se mide.** Sin gate automático posible para prosa → gate humano explícito: Fer lo lee 2 semanas antes de anunciar nada. Métrica proxy: % de afirmaciones con id citado verificable (target 100%; test con LLM stubbeado valida el filtro). Arrancar por `--week`: el semanal tolera prosa mediocre mejor que el diario.

---

## 3. `memo onboard` (+ bootstrap Node MCPB como apuesta secundaria)

**Qué es.** Wizard 4 pasos que colapsa el time-to-value a ~5 minutos: hook + backfill de transcripts ya en disco + import + briefing "3 cosas que ya sé de vos". La variante Node del .mcpb (bootstrap.js que auto-instala uv+mlx-memo) desbloquea la submission MCPB ya enviada.

**Por qué importa (y por qué recortado).** Ataca el eje correcto — distribución, donde memo es más débil — y el backfill Day-0 mina algo que solo una tool local tiene (transcripts en disco; mem0/Zep arrancan vacíos sí o sí). Mismatch del claim gamechanger: los no-devs de Claude Desktop que el shim Node desbloquea NO tienen `~/.claude/projects` que minar — para ellos memo instala vacío. La demo que convierte es para devs, que ya tienen uv. Por eso: **onboard = quick-win prioritario; bootstrap Node = apuesta barata sin expectativa de mover la aguja sola**.

**Plan**:

1. **Onboard** (~1 sem): `cli_onboard.py` nuevo registrado en `cli.py`, wizard estilo `memo sync setup`: (1) `install-recall-hook` + shims (`cli_hooks.py`, idempotente); (2) backfill vía mine-history de `cli_transcripts.py` (ya resumable por cursores, `--dry-run` de estimación, `MEMO_ONBOARD_BACKFILL_DAYS=90` en `flags_capture.py`, progreso rich); el redact existente (`MEMO_REDACT_SECRETS` default-ON en `capture_core.py`) cubre porque funnel-ea por el mismo extractor; bonus verificado: `history_importers.py` ya tiene `iter_codex/opencode/chatgpt/claude_export` — la extensión multi-agente no necesita parsers nuevos; (3) oferta `cli_import.py` (chatgpt-export/whatsapp); (4) briefing (`briefing.py`) sesgado a memorias >30d. `MEMO_NONINTERACTIVE` respetado.
2. **MCPB Node** (4-7 días, después): variante en `release_mcpb.py` con `bootstrap.js` de cero deps npm: localiza/descarga uv, `uv tool install mlx-memo==<pin del manifest>`, responde `initialize` con progreso de descarga del modelo (~600MB) reusando `runtime/install.py` + `report.py`; validación con `memo doctor --strict-runtime`; smoke en CI (macos-smoke existe); test que diffea ambos manifests contra pyproject para no driftar. La única ingeniería nueva real es bootstrap.js (responder initialize durante install + proxy stdio al child: replay de handshake, buffering — fiddly).

**Esfuerzo**: ~1 sem onboard + 4-7 días MCPB. **Riesgos**: backfill de meses = muchas pasadas del helper-LLM (cap 90d + resumable + throttled, 1 solo cold-load MLX); calidad de memorias viejas menor (dedup por embedding amortigua); UX de Claude Desktop durante un connect de minutos no verificable desde el repo; latencia de review del Directory es externa.

**Cómo se mide.** Métrica funnel Day-0 explícita: memorias creadas en día 0 + tiempo-hasta-primer-grounded-recall (ambos observables en `grounding.log`/ledger). Backfill: el corpus minado pasa por `memo eval recall --gate` — si mete ruido (noise@K sube), se baja el cap o se filtra por tipo antes de shippear.

---

## 4. memo gateway — proxy OpenAI-compat con recall ambiente (DIFERIDA)

**Qué es.** Proxy localhost `/v1/chat/completions` que inyecta el bloque de recall en el request de cualquier agente OpenAI-compat y, viendo la respuesta completa, extiende grounding/ROI medido más allá de Claude Code.

**Por qué está diferida.** "Memoria de la máquina, no de Claude Code" — y la telemetría grounded-by-agent en el wire es genuinamente única (un cloud memory nunca ve la respuesta). Pero los datos propios desinflan la urgencia: synapse 410 consults (ver hallazgo lateral: casi todos ruido) + claude-code 55; codex/opencode/gemini/devin tienen CERO consults probablemente porque casi no se usan. El concepto ya existe in-repo: `integrations/wrap.py` hace exactamente esto in-process (inyección pre-call vía socket warm + marcador `_RECALL_HEADER` anti-double-inject + capture post-call) — el gateway es wrap.py levantado al cable.

**Plan (cuando haya demanda)** (flags en `flags_misc.py`: `MEMO_GATEWAY_ENABLED`, `MEMO_GATEWAY_PORT`, `MEMO_GATEWAY_INJECT_BUDGET`, default-OFF):

1. **Scope v1 recortado**: opencode + hermes (OpenAI-compat puro con base_url configurable). Codex ya migró a `/v1/responses`, gemini-cli no es OpenAI-compat, Claude Code es protocolo Anthropic — fuera de v1.
2. `cli_gateway.py` + paquete `src/memo/gateway/` (patrón `server_http.py`; fastapi/httpx/uvicorn ya en uv.lock, cero deps nuevas). Identificación por header `X-Memo-Client` o token por-agente (patrón `secret_store.py`).
3. Recall vía `recall_client.py` → `op:search` de `recall_socket.py` con `client=` (atribución gratis, p50 warm ~630ms). Inyección solo en el primer user-message nuevo (dedup por hash) + reuso del marcador de wrap.py.
4. Forward con passthrough SSE + auth/API-key passthrough al upstream. Config `memo config set gateway.upstream` — necesita mapa per-provider, no un solo upstream (opencode rutea varios providers), o una instancia por upstream.
5. Grounding al cerrar la respuesta: matcher de `grounding.py` contra ids inyectados, fold en `token_ledger.py`/`outcome.py` con `agent=<client>`. **Trabajo real escondido**: el ledger tiene split duro "grounded=claude-code / consults=el resto, sin double-count" — hay que rediseñar esa semántica, no bypassearla. El tee SSE que parsea deltas OpenAI para grounding es la parte dura conocida (wrap.py la pateó a v2).
6. Preset en `runtime/agent_presets.py` que escribe base_url (precedente: upsert idempotente del username badge) + plist opcional vía `runtime/daemon.py`.
7. **Nota de frontera obligatoria**: un proxy huele a routing (turf de synapse). Defendible como read-path injection (memo ya posee el plumbing per-agent: wrap.py, agent_presets, codex.py), pero antes de arrancar, resolver en un design doc de una página si el gateway pertenece a synapse con memo solo como fuente del bloque.

**Esfuerzo**: 2-3 semanas. **Riesgos**: fidelidad SSE (lo primero que se rompe); MITM de tráfico de agentes rompe con cada drift de protocolo (failure mode "mi agente se rompió"); API keys fluyendo por el proxy; falsos positivos del matcher sobre modelos distintos.

**Cómo se mide.** (a) Latencia agregada solo en primer turno, contra el p50 630ms del socket; (b) `memo eval tokens` + panel by-agent: consults y grounded_rate por agente ANTES/DESPUÉS de encender el gateway por agente; (c) umbrales de `grounding.py` validados sobre respuestas de cada modelo antes de publicar números. Criterio de éxito honesto: si tras 2 semanas los agentes proxy-ados siguen sin consults productivos, el problema era uso, no transporte — y se archiva sin drama.

---

## 5. Total Recall — tier episódico verbatim (condicional a demanda)

**Qué es.** Indexar a nivel turno los transcripts JSONL ya en disco (~812MB, 2082 archivos, ≥244k líneas medidos) en un sidecar rebuildable, buscable on-demand: "qué dijimos EXACTAMENTE cuando arreglamos X".

**Por qué con descuento.** El gap es más chico de lo que vende el pitch: `episode_store.py` + `memo resume` ya resuelven "encontrar la sesión donde arreglamos X" semánticamente; falta la última milla turn-level, y para lo verbatim exacto `rg` sobre los JSONL es sustituto gratis. claude-mem ya posee este espacio siendo local/gratis — feature-parity, no categoría. Uso real esperado: ocasional.

**Plan** (flags en `flags_ingest.py`: `MEMO_VERBATIM_INDEX`, `MEMO_VERBATIM_MAX_DAYS`, `MEMO_VERBATIM_TRANSCRIPT_DIRS`, default-OFF) — recorte grande de la auditoría: **fase 1 es FTS5-only, sin embeddings**:

1. **Fase 1 (léxica, ~80% del valor, ~1 sem)**: `store/turn_store.py` siguiendo el patrón de `store/episode_store.py` (derivado, rebuildable, thread-local conns, fold `MEMO_SINGLE_DB`), tabla turns(session_id, turn_idx, role, ts, text, sha256) + FTS5 reusando infra en repo (`bm25_queries.py` / `store/tantivy_index.py`). Sin backfill MLX. Ingesta incremental con el watermark per-session de capture-tick.
   - **Pre-work obligatorio**: `_parse_transcript` NO trae timestamps — hace falta una variante chica (~15 líneas, el JSONL trae timestamp). Decidir la política de tool_result ANTES de congelar schema: hoy `_extract_text` lo capea a proyección de 300 chars tras `MEMO_CAPTURE_TOOL_EVIDENCE` — el "comando exacto / error exacto" vive ahí, y capturarlo full multiplica volumen 2-5×.
2. `redact.redact_secrets()` SIEMPRE pre-index; DB en state_dir (fuera de git-sync por construcción, verificado: sync solo toca data_dir).
3. Superficie: `memo verbatim <query> [--session --since]` en `cli_retrieve.py` + `server_verbatim.py` con register (verbo de búsqueda, pasa `test_architecture_boundaries` — verificado). `tiers.py` NO cambia; jamás entra al recall hook.
4. Posicionamiento de un párrafo vs EpisodeStore/memo_episodes_search (misma fuente, otra granularidad).
5. **Fase 2 (embeddings, solo si fase 1 lo justifica)**: pase nocturno `_run_verbatim_index` en `cli_dream_passes.py`, cache content-addressed, retención por `MEMO_VERBATIM_MAX_DAYS`, backfill throttled (1 cold-load). Ojo: knn brute-force sobre 200-300k vectores 1024-dim ≈ 1s+/query — aceptable on-demand, medirlo.

**Esfuerzo**: fase 1 ~1 semana; fase 2 suma 1-2 más. **Riesgos**: volumen; secretos en transcripts (redact + no-sync); scope-creep hacia auto-recall (prohibido: mata el budget 5s y reabre el ruido que dedup-collapse cerró).

**Cómo se mide.** Fase 1: contador de uso real de `memo verbatim`/`memo_verbatim_search` (atribución vía source= existente) — **fase 2 solo se aprueba si el uso léxico lo justifica** (criterio: N queries/semana durante un mes, N definido por Fer antes de arrancar). Recall del hook intocado: `memo eval recall --gate` verde por construcción + p95 del hook sin cambio.

---

## 6. Recall RCT — holdout causal + `memo proof` (modo campaña, post-HyPE)

**Qué es.** % muestreado de SESIONES corre recall en shadow (computa qué habría inyectado, no inyecta, loguea); `memo proof` compara re-ask rate y tokens medidos entre brazos con CI honesto.

**Por qué va última.** La dirección es correcta: el "1.08M tokens saved" es estimación por flag y la data observacional lo contradice (grounded gasta MÁS: Δ−2287, confundido con dificultad de tarea). Pero: (a) medio pipeline ya existe — `MEMO_RECALL_DISABLE` stampa `via="disabled"`, `ablation_stats()` en `dashboard_metrics.py:728` ya computa cohortes on/off con re-ask rate, surfaceado en `memo roi`/`memo tokens`; lo nuevo es solo aleatorización + shadow-log + CLI; (b) el poder estadístico es el agujero fatal: asignación por sesión = cluster-randomized; a 5% son 1-2 sesiones holdout/semana — `memo proof` honesto diría "n insuficiente" casi siempre.

**Plan — recortado a "modo campaña"**:

1. `MEMO_RECALL_ABLATION_RATE` (float, default 0.0) en `flags_recall.py`; asignación determinística por sesión (hash session_id de `identity.py` + fecha). Uso previsto: `memo proof --collect 2w --rate 0.2` — experimento acotado que se corre a propósito asumiendo la degradación, NO ambiente perpetuo a 5%.
2. Holdout en AMBOS paths del hook (`cli_recall_hook.py` subprocess Y `recall_server.py` daemon — trampa conocida de los 2 paths): rank_hits igual, suprime inyección, loguea `{would_inject_ids, scores, arm}` en recall_hook.log (con rotación); cuando escribe el daemon, el arm viaja en el request. `grounding.py` estampa arm.
3. Decidir explícitamente qué side-effects corre el holdout (session-dedup registry `get_recalled_ids`, next_turn, prompt_trail) para no corromper estado que otras features leen.
4. Lector por-arm: reusar `ablation_stats` como base, no escribir uno nuevo; re-ask vía `_reask_tokens` existente; persistencia en `token_ledger.py`. Exclusiones: turnos con guard/interject pendiente nunca se ablacionan.
5. `cli_proof.py` (`memo proof --window`): deltas con n por brazo; re-ask rate con Wilson CI (proporción, mejor potenciada que tokens/turno); se niega a concluir bajo `MEMO_PROOF_MIN_SAMPLES`. Kill-switch = rate 0 vía markdown config.
6. Bonus real: turnos holdout donde el usuario re-preguntó = labels positivos fuertes para `eval_recall.harvest` (alimenta el gate de HyPE).

**Esfuerzo**: 1-2 semanas de código + semanas de pared juntando muestras. **Riesgos**: degradación deliberada de sesiones reales (por eso campaña, no ambiente); re-ask es heurística; tests sin índice vivo (tmp_cfg + stub embedder, trampa falso-verde conocida); el resultado puede ser incómodo — feature, no bug.

**Cómo se mide.** Esta idea ES la medición. Gate de éxito propio: al cierre de la campaña, `memo proof` emite un delta con CI que no cruza cero (re-ask rate como outcome primario) o emite "n insuficiente" — y en ese caso el output igual sirve: corrige el número del README de estimado a "no concluyente, medido".

---

## Orden del roadmap y dependencias

```
Fase A (paralelo, ~2 semanas):
  A1. Chronicle (4-8 días)           — riesgo bajo, demo asset, cero dependencias
  A2. memo onboard (~1 sem)          — quick-win distribución, independiente

Fase B (~2-3 semanas):
  B1. HyPE                           — la apuesta de calidad de retrieval del trimestre
  B2. MCPB Node bootstrap (4-7 días) — en paralelo si hay banda; independiente

Fase C (condicional):
  C1. RCT modo campaña (1-2 sem código) — DESPUÉS de HyPE: los labels holdout
      alimentan el harvest que valida HyPE, y proof mide el sistema CON HyPE
  C2. Total Recall fase 1 FTS5-only (~1 sem) — cuando haya demanda real de
      verbatim; fase 2 embeddings solo si el uso léxico lo justifica

Diferida / re-evaluar:
  Gateway — esperar señal de que opencode/hermes se USAN de verdad (hoy
  0 consults ≈ 0 uso); resolver antes la frontera con synapse (doc 1 página)
```

**Dependencias y colisiones concretas:**

- **HyPE ↔ Total Recall fase 2 ↔ backfill de onboard**: tres cargas MLX nocturnas pesadas. Nunca concurrentes (exit-144) — todo entra secuencial en la cadena `memo-nightly` existente; el backfill de HyPE (2-3 noches) no se solapa con un backfill verbatim.
- **RCT → HyPE**: dependencia blanda bidireccional — labels positivos del holdout mejoran el harvest de `eval_recall`; correr proof post-HyPE mide el sistema real. Orden: HyPE primero.
- **Gateway → token_ledger**: el rediseño del split grounded/consults del ledger es prerequisito interno del gateway; si se hace, hacerlo ANTES de publicar números by-agent.
- **Chronicle** no depende de nada y no bloquea nada — por eso va primera junto con onboard.
- **Regla transversal (no negociable)**: todo flag nuevo nace OFF; todo flip pasa por su gate (`eval recall --gate` / bench per-bucket / métrica funnel / gate humano de prosa según corresponda); ningún número se publica sin medición que lo respalde.

## Hallazgos laterales del análisis (fuera de alcance de este doc, accionables aparte)

1. **Bug synapse→memo**: 407/408 consults de synapse en `recall.log` son `{"prompt": "", "hits": [], "latency_ms": 0, "via": "cli:search"}` — query vacía cada pocos segundos. El consumo real de synapse hoy es ~0 y las stats de usefulness/roi están infladas. Bug del lado synapse (shellea `memo search` con query vacía); ticket aparte en el repo synapse.
2. **Gate curado saturado**: prec@5=0.2 es el máximo teórico del regression set actual (prompts de respuesta única). Toda mejora de retrieval se mide con `memo eval bench` per-bucket + labels harvested; el gate curado queda como guardia anti-regresión, no como métrica de progreso.
3. **"1.08M tokens saved" es estimación por flag**, no medición causal; la data observacional (grounded gasta más, Δ−2287, confundida por dificultad) no la respalda. El RCT (idea 6) existe para corregir ese número.

## Referencias

Archivos citados por los auditores: `src/memo/cli_dream_passes.py`, `store/schema.py`, `store/queries.py`, `store/episode_store.py`, `recall_logic.py`, `integrations/wrap.py`, `dashboard_metrics.py`, `token_ledger.py`, `release_mcpb.py`, `eval/regression_labels.json`, `docs/eval/capability-baseline-and-levers.md`. Informe crudo del workflow: run `wf_38c809eb-0d1` (36 agentes, 3.05M tokens de subagentes, 397 tool calls).
