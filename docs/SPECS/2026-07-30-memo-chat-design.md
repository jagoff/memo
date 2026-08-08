Status: shipped in #149 (feat: native memo chat (synapse chat rescue), merged 2026-08-01). PR body explicitly states "Implements docs/SPECS/2026-07-30-memo-chat-design.md". `src/memo/chat/`, `src/memo/cli_chat.py`, `src/memo/server_chat.py`, `com.memo.chat` launchd all present. Known backlog: eval-chat gate 13/22 green, parked per the spec's own pragmatism decision.

# Chat nativo en memo — rescate lean del chat de synapse

**Fecha:** 2026-07-30
**Estado:** aprobado (brainstorming validado sección por sección; reescrito post-#136)
**Decisión:** memo gana una superficie de chat sobre memoria propia, reimplementando lean
las capacidades vivas del chat de synapse. Rewrite sobre primitivas de memo — no port
verbatim.

## Contexto

La deprecación big-bang de synapse
([spec](2026-07-30-synapse-deprecation-design.md), PR #136) adoptó los daemons de
infraestructura (recall, nightly, vault-ingest) y **mató el chat sin reemplazo**,
dejándolo explícitamente como spec futura. Este documento es esa spec.

Estado del mundo tras #136:

- Fleet 100% `com.memo.*`; nada escucha en :8765; `com.fer.whatsapp-bridge` también
  apagado.
- Código fuente del chat: `~/repos/_archived/synapse` (tag `deprecation-final`) —
  `web-chat/` (React UI), stack de retrieval/síntesis, `eval/regression_corpus.json`.
- Estado/señales: `~/.memo-daemon-backups/20260730T213401-synapse-final/dot-synapse/state/`
  (`feedback/`, `eval/`, `user_model.json`, `digests/`).
- memflow también deprecado (2026-07-30) — el chat es memo-only por construcción.

Lo que se rescata (era el valor vivo del dashboard synapse): chat sobre memoria con
síntesis MLX warm (~6-8 s), stack de calidad de retrieval, feedback 👍👎 con
generalización semántica, corpus de regresión, y las capacidades de aprendizaje
(insight, crystallize, digest).

## Decisiones (fijadas en brainstorming 2026-07-30)

1. **Chat completo en memo**: UI + síntesis warm + stack de calidad.
2. **Capa de aprendizaje migra toda**, reimplementada lean (ver mapeo).
3. **Chat memo-only**: cero lectura de memflow (además, ya no existe).
4. **Enfoque**: rewrite lean nativo. Las capacidades migran; el código no.
5. **Continuidad**: uso personal, pragmático; el gate es el corpus de regresión, no
   paridad A/B en vivo (synapse ya no corre).

## Arquitectura del chat

El fast path de synapse (retrieval memo por subprocess + fusión vault + síntesis MLX
warm propia) colapsa a in-process dentro de memo:

- **Superficie**: `memo-daemon` HTTP gana la superficie de chat completa. `/chat/stream`
  (SSE) ya existe en `server.py`; se agregan endpoints de feedback y gestión de fuentes
  que servía el dashboard synapse. El daemon escucha **:8765** (continuidad de
  bookmarks).
- **Pipeline** (`memo/chat/`, paquete chico): `search_hybrid` (memoria) + `repo_search`
  (vault) en paralelo → RRF fusion → normalización de scores por grupo → dedup de
  chunks/near-duplicates → boost por votos 👍 → relevance floor → síntesis con MLX warm
  in-process. El MoE 30B pasa a residir en el proceso de memo (un solo modelo warm,
  mismo budget de 36 GB).
- **Se portan como algoritmos** (funciones puras reescritas lean, con tests propios):
  RRF fusion, score_norm por grupo, sources_dedup (colapso de chunks `(§N/M)` + merge
  de snippets), relevance floor, fulldoc inline, query_rewrite de follow-ups,
  multi_query gateado por categoría de query, title_boost (vuelve a su origen:
  `memo/retrieval_boost.py`).
- **Se descartan**: HyDE, query_decompose, multi_hop, rerank cross-encoder (solo vivían
  en el path bloqueante legacy o estaban apagados en producción).
- **UI**: `web-chat` (React) se copia del repo archivado a `memo/web-chat/` apuntando al
  daemon de memo. El dashboard de observabilidad de synapse (`web/`) queda muerto.
- **Config**: los ~20 knobs `SYNAPSE_CHAT_*` se reducen a los que estaban activos en el
  plist de producción, renombrados `MEMO_CHAT_*`, con los valores de producción como
  defaults de código. El zoo de timeouts/transports/scales de subprocess murió con el
  subprocess.

## Capa de aprendizaje — mapeo a contrapartes memo

Regla: cada capacidad se mapea a la contraparte memo existente y se extiende; solo lo
sin contraparte se escribe nuevo.

| Synapse | Destino en memo |
|---|---|
| feedback 👍👎 + boost/filtro de fuentes (exacto + semántico) | Nuevo `memo/chat/feedback.py`; store append-only en state de memo; generalización semántica con el embedder in-process |
| eval-chat + corpus + gate de regresión | Extiende `cli_eval`; se migran `regression_corpus.json` + labels verbatim |
| insight (respuesta → propuesta de memoria) | Se pliega al pipeline capture/graduation existente (propuesta + threshold, nunca write directo) |
| crystallize (sesión → memoria) | Comando nuevo `memo crystallize` sobre capture/consolidate existentes |
| SSM user_model | Lean: el routing prior murió con la federación; sobrevive solo el threshold adaptativo de insight por dominio, plegado a stats de feedback |
| goal_model | Metas = memorias memo con tipo/tag `goal`, surfaced en briefing; sin read model aparte |
| morning_digest | Extiende `memo briefing` + agent launchd nuevo (`com.memo.briefing`; el daemon synapse murió en #136) |
| dream-synthesis | Ya cubierto por `memo dream` (`com.memo.dream`); solo verificar que los passes de síntesis nocturna que aportaba synapse tengan equivalente |
| contacts_alias + whatsapp_live | Port lean a `memo/chat/` (queries por nombre sobre conversaciones). **Dependencia**: requiere reinstalar `whatsapp-bridge` + `com.memo.whatsapp-ingest`; si no se reinstala, la feature entra apagada |
| ocr_enrich | Diferido — fuera de v1 |

## Ops

Se **extiende el grupo `memo ops` existente** (`cli_ops.py`, creado en #136) con
`install|uninstall|status <service>`, modelado en el `ops.py` de synapse pero
memo-owned. Servicios nuevos a instalar: `memo-daemon` (chat + síntesis),
`whatsapp-ingest`, `briefing`. Los ya instalados (recall-daemon, nightly, vault-ingest,
dream) se incorporan a `status`. Templates en `launchd/` con el patrón existente
`__HOME__`/`__MEMO_BIN__`. Labels `com.memo.*`.

## Migración de estado

Se migran las **señales** desde el backup
(`~/.memo-daemon-backups/20260730T213401-synapse-final/`), no los derivados:

- Migran: eventos de feedback (jsonl), decisiones de insight, corpus eval + labels
  (también en `eval/` del repo archivado).
- Ya migrado en #136: tombstones de `ingest_exclude` (verificar).
- Se recomputa en memo: user_model, boosts, thresholds.
- Queda en el backup (no migra): ledger, trust, packets, snapshots runtime.
- Herramienta: script one-off `migrate_synapse_state.py` en memo.

## Secuencia

1. **Pipeline + tests**: `memo/chat/` (algoritmos portados TDD) + endpoints en
   memo-daemon + feedback store.
2. **Gate de regresión**: migrar corpus + labels; `eval-chat` verde contra el chat de
   memo; bench de latencia ≤ ~6-8 s warm (referencia histórica synapse; in-process
   debería mejorar).
3. **Migrar estado** (script) + UI (`memo/web-chat/`).
4. **Instalar**: `memo ops install memo-daemon` → chat vivo en :8765. Smoke manual de
   web-chat (votos, chips de fuentes, fulldoc, follow-ups).
5. **Capa de aprendizaje**: feedback semántico, insight, crystallize, briefing/digest,
   whatsapp (si se reinstala el bridge).
6. **Replicar en la otra Mac** (`memo ops install` por máquina).

No hay bootout ni switchover: nada de synapse corre. Cada etapa es aditiva.

## Testing

- Algoritmos portados: TDD, tests unitarios propios en memo (RED → GREEN), sin copiar
  suites de synapse salvo fixtures de datos.
- Gate de regresión (corpus + labels) = criterio de aceptación del pipeline.
- Bench de latencia warm/cold como verificación de performance.
- Smoke manual de web-chat.

## Riesgos

1. **Residencia del 30B**: el MoE pasa a vivir en memo-daemon; verificar convivencia
   con recall-daemon (4B) dentro del budget de 36 GB antes de instalarlo como agent.
2. **Sin A/B en vivo**: synapse ya no corre; el corpus de regresión es la única
   referencia de paridad. Tuning perdido por knobs descartados: riesgo residual
   aceptado (un solo usuario).
3. **whatsapp-bridge apagado**: `whatsapp_live` nace apagada si el bridge no se
   reinstala; decidir en implementación.
4. **Multi-máquina**: instalación por máquina; la otra Mac requiere su propio
   `memo ops install`.

## Fuera de alcance

- OCR de adjuntos citados en chat (diferido).
- Resurrección de cualquier pieza del control-plane de federación.
- Dashboard de observabilidad (`web/` de synapse).
- memflow (deprecado 2026-07-30).
