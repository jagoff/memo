# Memo Chat Learning Layer (plan 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** completar la capa de aprendizaje del chat de memo (diferida del plan 1): insight 👁 (respuesta→propuesta de memoria), crystallize (sesión→memoria), digest en el briefing, metas por tag `goal`, y WhatsApp vivo (contacts_alias + whatsapp_live + ingest launchd).

**Architecture:** todo se apoya en primitivas memo existentes — insight guarda vía `mem.save` con tag `_uncertain` (quarantine existente, gradúa `dream_graduate`); crystallize consume `chat/sessions.SessionStore` y sintetiza con el `ChatBackend` de `Memory`; el digest es una `*_lines()` más en `briefing.py` (patrón `dream_digest_lines`); whatsapp_live lee el `messages.db` del bridge en modo ro y reemplaza el retrieval semántico cuando la query pide conversación reciente (fuente autoritativa, no fusión); el ingest ya existe (`memo import whatsapp`) — solo se agenda con WatchPaths vía `memo ops`.

**Spec fuente:** tabla "Capa de aprendizaje" de `docs/SPECS/2026-07-30-memo-chat-design.md`. Algoritmos fieles al synapse archivado (file:line citados por task).

## Global Constraints

- Branch `feat/memo-chat-learning` (worktree `~/repos/memo/.worktrees/chat-learning`); master protegido → PR al final. Commits `<type>: <desc>`, SIN atribución.
- Verificación por task: `PYTHONPATH=src ~/repos/memo/.venv/bin/python -m pytest <files> -v`; antes de commitear: scope chat -q + `PYTHONPATH=src ~/repos/memo/.venv/bin/python -m mypy src/memo/chat/ src/memo/ops_launchd.py` + `~/repos/memo/.venv/bin/ruff check src tests` + `uvx ruff@0.16.0 format --check .` (el CI formatea también markdown). TDD estricto (RED capturado → GREEN).
- El quality gate del CI usa `eval/quality_baseline.json` — si un task agrega complejidad/broad-excepts deliberados, correr `~/repos/memo/.venv/bin/python scripts/quality_gate.py --update` y commitear el baseline EN el mismo commit, mencionándolo.
- CodeQL corre en el PR: regex sobre input de usuario SIN ambigüedad cuantificador-adyacente (nada de `\s+(.+?)[x]*$`; boundaries determinísticos `\S`).
- Sin deps nuevas. Cero imports de synapse. Los knobs nuevos van a `ChatConfig` (env-only) y sus nombres al set `owned` de `flags.unknown_memo_vars` (patrón del plan 1).
- Sin LLM en el hot path de retrieval: insight corre DESPUÉS del `done` (no bloquea la respuesta); en prod synapse `SYNAPSE_INSIGHT_LLM=0` — el detector es heurístico-only (NO portar el judge LLM).
- Facts de primitivas memo (verificadas en 3cfe5b89): `extract_and_save_text(mem, cfg, text, *, merge_tags, title, type_, ...)` `capture_core.py:1303`; quarantine = tag `_uncertain` + `dream_graduate.run_graduation` (untag nocturno); `SessionStore` en `chat/sessions.py` (`get(session_id) -> list[{"role","text","ts"}]`); briefing se extiende editando `memo_native_briefing_lines`/`cli_briefing.py` (sin registry), patrón `dream_digest_lines` `briefing.py:252`; `mem.store.list_by_tag(tag, limit)` `store/queries.py:1574` (match exacto); NO existe include_tags en search; `Memory._ensure_chat() -> ChatBackend` con `.chat(model, messages, options) -> dict` y `chat_json` NO existe — parsear JSON del content a mano (patrón `expand._content_of`).
- WhatsApp bridge: DB en `~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db` (existe; `MEMO_WHATSAPP_DB` flag ya registrado). `memo import whatsapp --all-chats --json` ya existe en memo.

---

### Task 0: Branch
- [ ] Ya creado por el controller: worktree `~/repos/memo/.worktrees/chat-learning`, branch `feat/memo-chat-learning` desde `3cfe5b89`. Verificar `git status` limpio.

---

### Task 1: `memo/chat/contacts_alias.py`

**Files:** Create `src/memo/chat/contacts_alias.py`, Test `tests/test_chat_contacts_alias.py`

**Interfaces — Produces:** `build_index(contacts_dir: Path) -> dict[str, str]` (trigger folded → jid; pura, sin cache); `resolve_jid(query: str, index: dict[str, str]) -> str | None`. (El caller cachea; sin TTL interno — lean vs synapse `contacts_alias.py:64-146`.)

Algoritmo fiel (synapse `contacts_alias.py:95-194`): por cada `.md` del dir (excluir prefijos `.`/`_`): extraer `wa_jid` con regex `(?im)^-[ \t]*\*\*[ \t]*wa[_\s-]?jid[ \t]*\*\*[ \t]*:[ \t]*(.*)$` (fallback label `jid`); sin jid → skip. Triggers = {stem del archivo} ∪ {campo `Apodo`} ∪ {tokens `[a-záéíóúñ]{3,}` del campo `Apellido / nombre completo`/`Full name`} ∪ {triggers de parentesco del campo `Relación`/`Relation` vía mapa `_KINSHIP`: mama/mamá/madre→{mama,madre,mami}, papa/papá/padre→{papa,padre,papi}, hermano/a→{hermano,hermana}, hijo/a→{hijo,hija}, esposa/mujer→{esposa,mujer}, esposo/marido→{esposo,marido}}. Fold NFKD lowercase sin diacríticos, len≥2. **Trigger compartido por 2+ jids se descarta del índice.** `resolve_jid`: fold query, matchear cada trigger como `\b{re.escape(trig)}\b`; exactamente 1 jid distinto → devolverlo; 0 o ambiguo → None. (Sin fallback GBrain — muerto.)

- [ ] Test RED: nota fixture con wa_jid/Apodo/Relación mamá → index con triggers esperados; trigger ambiguo entre 2 notas → ausente; `resolve_jid("qué me dijo mi mamá?")` → jid; query sin contacto → None; nota sin wa_jid → ignorada.
- [ ] Implementar → GREEN → contrato → commit `feat(chat): contacts alias index from vault notes`

---

### Task 2: `memo/chat/whatsapp_live.py`

**Files:** Create `src/memo/chat/whatsapp_live.py`, Test `tests/test_chat_whatsapp_live.py`

**Interfaces — Produces:** `bridge_db_path() -> Path` (env `MEMO_WHATSAPP_DB` o default `~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db`); `resolve_chats(query, db, contacts_index) -> list[tuple[str, str]]` (jid, label); `last_messages(db, chat_jid, *, limit, today_only) -> list[dict]`; `format_transcript(label, msgs) -> str`; `recency_conversation_intent(q) -> bool`; `singular_last_intent(q) -> bool`; `today_only_intent(q) -> bool`. Todas fail-soft: cualquier excepción → resultado vacío/False.

Fiel a synapse `whatsapp_live.py`: conexión `sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)` (**ro, NUNCA immutable** — esconde filas del WAL). Query mensajes (`:209-252`):
```sql
SELECT datetime(timestamp, 'localtime') AS ts, is_from_me, content FROM messages
 WHERE chat_jid = ? AND content IS NOT NULL AND TRIM(content) != ''
 [AND date(timestamp,'localtime') = date('now','localtime')]
 ORDER BY timestamp DESC LIMIT ?
```
clamp limit [1, 200 today/100], revertir a cronológico. Resolución: primero `contacts_alias.resolve_jid`; si no, `SELECT jid,name FROM chats` y matchear tokens significativos del name como palabra completa (excluir stopwords genéricas); grupos `@g.us` solo si `SELECT COUNT(DISTINCT sender) FROM messages WHERE chat_jid=? AND is_from_me=0 AND sender IS NOT NULL AND TRIM(sender)!=''` ≤ 1. Intents (regex sin ambigüedad ReDoS): recency = `(últim[oa]s? (mensajes?|conversaci[oó]n)|qué (me )?dijo|conversaci[oó]n con|last messages?|what did .{1,40} say)` case-insensitive; singular = "último mensaje" sin plural; today = `\b(hoy|today)\b`.

- [ ] Test RED con sqlite tmp (crear tablas chats/messages con fixtures): resolve por contacts index (prioridad), resolve por nombre de chat, grupo multi-sender excluido, last_messages orden+today_only+limit clamp, intents, db inexistente → [] sin excepción.
- [ ] Implementar → GREEN → contrato → commit `feat(chat): live whatsapp read-through from bridge db`

---

### Task 3: Hook de pipeline — WA live exclusivo

**Files:** Modify `src/memo/chat/pipeline.py`, `src/memo/chat/config.py` (+knob), Test extiende `tests/test_chat_pipeline.py`

**Interfaces:** `ChatConfig` gana `whatsapp_live: bool` (env `MEMO_CHAT_WHATSAPP_LIVE`, default True) y `contacts_dir: Path | None` (env `MEMO_CHAT_CONTACTS_DIR`, default `None` → `<vault>/Obsidian/Contacts` si `memory.cfg` expone vault path, sino skip). Registrar ambos nombres en el set `owned` de `flags.py`.

En `chat_stream`, ANTES del retrieval semántico (tras el rewrite): si `cfg.whatsapp_live and recency_conversation_intent(memo_query)`: `resolve_chats(...)` → `last_messages(...)` → si hay mensajes, construir fuente sintética fiel a synapse `stream.py:1296-1320`: `{"id": f"wa-live:{label_folded}", "source": "memory", "type": "whatsapp_live", "title": f"WhatsApp · {label} — {fecha_último}", "snippet": format_transcript(...), "score": 0.99, "normalized_score": 0.99}` y **reemplazar todo el retrieval** (skip search/repo_search/multi-query/fusión: `sources = [wa_src]`, sin feedback ni floor) → context → síntesis normal sobre esa única fuente. Si no resuelve/no hay mensajes → flujo semántico normal sin ruido. Cualquier excepción del path WA → flujo normal (try/except envolvente).

- [ ] Test RED: fake memory + monkeypatch de `whatsapp_live.resolve_chats`/`last_messages` → query "qué me dijo Ana hoy?" produce context con única fuente `wa-live:*` y NUNCA llama `memory.search` (fake que falla si se llama); query normal → flujo previo intacto; excepción en resolve → flujo normal.
- [ ] Implementar → GREEN → contrato completo → commit `feat(chat): exclusive whatsapp live source for recency queries`

---

### Task 4: Insight — detector + threshold adaptativo

**Files:** Create `src/memo/chat/insight.py`, Test `tests/test_chat_insight.py`

**Interfaces — Produces:**
- `@dataclass(frozen=True) InsightCandidate`: `title, body, tags: list[str], confidence: float, score: int, suggested_type: str, chat_session_id: str, chat_turn_id: str, schema: str = "memo.chat.insight_candidate.v1"` + `.to_dict()`.
- `detect(question, answer, sources, *, threshold) -> InsightCandidate | None` — heurístico puro, sin LLM.
- `insight_threshold(query, feedback_events: list[ChatFeedback]) -> int` — adaptativo.
- `is_duplicate(memory, candidate) -> bool`.

Detector fiel a synapse `insight.py:240-383` (SIN el LLM judge — prod lo tenía off):
1. **Goal fast-path**: regex `_GOAL_RE = (decidimos|vamos a|quiero|planeo|el objetivo es|we will|i want to|the goal is)` (word-boundaries, case-insensitive) sobre `f"{question} {answer}"` → candidato `suggested_type="note"`, tags `["goal"]`, score 55, confidence 0.6 (memo no tiene tipo goal; la meta es una note taggeada `goal` — spec plan 1).
2. **Gates** (None si): `len(answer) < 200`; `len(sources) < 2`; frase negativa (`no encontré|no encuentro|no sé|sin resultados`); auto-referencia (`el chat|este sistema|esta respuesta`); ≥70% de líneas empiezan con `-`/`*`/`•`.
3. **Score 0-50**: `+20` si ≥2 citas `[n]` inline; `+10` si contiene hoy/ayer/mañana/"20" (año); `+10` si verbo de decisión (decidimos|decidí|optamos|elegimos|acordamos|resolvimos|definimos); `+min(15, entidades_capitalizadas_únicas*5)`; cap 50. **Final = heur*2** (0-100).
4. `final >= threshold` o None. Título = pregunta[:80] sin `?`; tags = ["decision" si verbo]+["temporal" si fecha]+hasta 3 entidades lowercase; tipo = decision/fact/note; body = primer párrafo del answer cortado en borde de oración ≤600; confidence = score/100.

Threshold fiel-lean a synapse `user_model.py:636-659` (sin accept-rate — no recolectamos rejects): dominio = primer match de `{"decision": r"\b(decision|decisión|decidimos|decid[ií]|elegimos|acordamos|optamos)\b", "personal": r"\b(familia|amigo|coach|cliente|coaching|scrum|agile|personal)\b", "technical": r"\b(python|code|bug|test|api|deploy|backend|memo)\b"}`; si dominio y `count(feedback rating=="up" cuyo query matchea el dominio) >= 5` → **75**, sino **90**. `is_duplicate`: `memory.search(candidate.title, limit=3)` → top-1 título substring bidireccional case-insensitive.

- [ ] Test RED: goal fast-path; cada gate; score compuesto exacto (fixture con 2 citas+verbo+2 entidades = (20+10+10)*2=80 → pasa threshold 75, no 90); threshold 90 default / 75 con 5 ups técnicos; dedup por título.
- [ ] Implementar → GREEN → contrato → commit `feat(chat): heuristic insight detector with adaptive threshold`

---

### Task 5: Insight en el pipeline + `/api/insight/capture`

**Files:** Modify `src/memo/chat/pipeline.py`, `src/memo/chat/http.py`, `src/memo/chat/config.py` (knob `insight: bool` = `MEMO_CHAT_INSIGHT`, default True, + owned set), Tests extienden pipeline+http

**Pipeline:** tras yield del `done` (solo path de síntesis, no fulldoc/error): computar `insight_threshold(question, FeedbackStore(cfg.feedback_dir).load())`, `detect(...)`; si candidato y `not is_duplicate(memory, c)` → `yield {"type": "insight_proposal", "candidate": c.to_dict()}` (shape que la UI ya consume, `web-chat/src/types.ts:157-178`). Todo el bloque en try/except (nunca rompe el stream ya emitido).

**HTTP:** `POST /api/insight/capture` reemplaza el 501 (plan 1): body `{"candidate": {...}}` → validar title/body no vacíos (400 si no); `tags = candidate.tags + ["chat-capture"]` (+`"_uncertain"` si `score < 90` — quarantine del pipeline capture existente, gradúa dream); `mem.save(text=body, title=title, type_=suggested_type_válido_o_note, tags=tags)` — verificar firma real de `mem.save` (`memory/write_ops.py:306`) y adaptar kwargs; append línea a `cfg.state_dir/"chat"/"insights"/"captures.jsonl"` `{memoria_id, title, score, chat_session_id, captured_at}`; respuesta `{"ok": True, "memoria_id": ...}` (shape `CaptureResult` de types.ts).

- [ ] Test RED pipeline: fake con answer larga+2 fuentes+verbo decisión → aparece `insight_proposal` tras `done`; answer corta → no aparece; excepción en detect → stream intacto.
- [ ] Test RED http: capture con candidato válido → 200 ok + memoria_id + jsonl línea; body vacío → 400; score<90 → save llamado con `_uncertain` (fake memory graba kwargs).
- [ ] Implementar → GREEN → contrato (mypy incluye http) → commit `feat(chat): insight proposal emission and capture endpoint`

---

### Task 6: Crystallize

**Files:** Create `src/memo/chat/crystallize.py`, Modify `src/memo/cli_chat.py` (subcomando), Test `tests/test_chat_crystallize.py`

**Interfaces:** `crystallize_session(memory, session_id: str | None = None, *, n_turns: int = 30, dry_run: bool = False) -> dict` — toma la sesión (o la más reciente vía `SessionStore.list_sessions(limit=1)`), formatea transcript (últimos 4000 chars, `ROLE: text[:500]` por línea) y sintetiza con `memory._ensure_chat().chat(model, messages, options={"temperature":0.1,"max_tokens":600})` usando el prompt VERBATIM de synapse `crystallize.py:31-48` (JSON con title/situation/decisions/learnings/goal_progress/body/tags); parseo tolerante (patrón `_content_of` + slice `{...}`); fallo LLM → fallback heurístico (título `Session <id12> — <fecha>`, body con turn count). Escribe `mem.save` `type_="decision"`, text = título + body + secciones `**Decisions:**`/`**Learnings:**`/`**Goal progress:**` (`- item`), cap 3000 chars, tags = hasta 8 del LLM + `["session-crystal"]`. Dedup window: `cfg.state_dir/"chat"/"crystallize_last.json"` — < 1800s desde el último run del MISMO session_id → `{"ok": False, "skipped": True}`. CLI: `memo chat crystallize [SESSION_ID] [--dry-run]` imprime resultado.

- [ ] Test RED: fake backend devolviendo JSON válido → save con secciones y tags; backend que lanza → fallback heurístico; dedup window (segundo call skipped, tmp state); sesión inexistente → error limpio; CLI --dry-run no escribe.
- [ ] Implementar → GREEN → contrato → commit `feat(chat): session crystallize command`

---

### Task 7: Digest de chat + metas en el briefing

**Files:** Modify `src/memo/briefing.py`, `src/memo/cli_briefing.py` (si aplica), flag nuevo, Test `tests/test_briefing_chat_digest.py`

**Interfaces:** `chat_digest_lines(mem, state_dir: Path) -> list[str]` en `briefing.py` (patrón exacto `dream_digest_lines` `briefing.py:252`): (a) captures de insight últimas 24h (`chat/insights/captures.jsonl` — count + último título); (b) sesiones de chat con actividad 24h (`SessionStore.list_sessions` sobre `state_dir/chat/sessions`); (c) **metas activas**: `mem.store.list_by_tag("goal", limit=50)` → hasta 3 títulos no archivados. Vacío→[]; todo con `contextlib.suppress`. Gate: flag `MEMO_BRIEFING_CHAT_DIGEST` default ON — registrarlo como FlagSpec en el `flags_<group>.py` que ya contenga los `MEMO_BRIEFING_*` (mirar dónde vive `MEMO_BRIEFING_DREAM_DIGEST` y copiar el patrón exacto). Insertar la llamada en `memo_native_briefing_lines` después del bloque proactive.

- [ ] Test RED: fixtures jsonl+sessions+fake store.list_by_tag → líneas esperadas; sin datos → []; flag off → no aparece (test del compositor con monkeypatch env).
- [ ] Implementar → GREEN → contrato → commit `feat(briefing): chat digest and active goals section`

---

### Task 8: Ops — servicio whatsapp-ingest

**Files:** Modify `src/memo/ops_launchd.py`, `src/memo/cli_ops.py`, Test extiende `tests/test_ops_launchd.py`

**Interfaces:** `render_whatsapp_ingest_plist(memo_bin, home) -> str` — label `com.memo.whatsapp-ingest`, `ProgramArguments = [memo_bin, "import", "whatsapp", "--all-chats", "--json"]`, `RunAtLoad true`, **`WatchPaths` = [<bridge_db>, <bridge_db>-wal]** (de `whatsapp_live.bridge_db_path()`), `ThrottleInterval 300`, logs `~/Library/Logs/memo/whatsapp-ingest.log`, env MEMO_* forwardeado (mismo helper del plan 1) — fiel al plist synapse del backup (WatchPaths+throttle, NO StartCalendarInterval). `install_whatsapp_ingest`/`uninstall_whatsapp_ingest` (mismo patrón bootout/bootstrap/RuntimeError). CLI: `service` choice pasa a `["chat", "whatsapp-ingest"]` en install/uninstall.

- [ ] Test RED (puros, plutil condicional como plan 1): render contiene label/WatchPaths/throttle/args; install escribe+bootout+bootstrap orden (mocked); choice inválido rechazado.
- [ ] Implementar → GREEN → contrato → commit `feat(ops): whatsapp-ingest launchd service`

---

### Task 9: Integración final

**Files:** Modify `README.md` (inventario si suma comandos), `CLAUDE.md` (sección chat: learning layer), quality baseline si hace falta.

- [ ] Suite chat completa -q verde; mypy; ruff whole-tree; `uvx ruff@0.16.0 format --check .`; `scripts/quality_gate.py` (update si rojo, commitear baseline).
- [ ] Smoke real: `PYTHONPATH=src ... -m memo.cli chat serve` foreground → una pregunta que dispare insight (answer larga) → verificar frame `insight_proposal` y capture endpoint end-to-end con curl; `memo chat crystallize --dry-run`; `memo briefing | grep -i chat`.
- [ ] Docs surgical; commit `docs: chat learning layer`.
- [ ] PR a master: `gh pr create` con resumen por task + gates; checks verdes → squash merge (mismo flow del plan 1: quality gate/CodeQL/format md son los rojos típicos).

## Self-review
- Cobertura de la tabla del spec: feedback (plan 1 ✔); eval-chat (plan 1 ✔); insight → Tasks 4-5 ✔; crystallize → Task 6 ✔; SSM threshold → Task 4 (lean: ups-only, sin accept-rate — decisión documentada) ✔; goal_model → tags `goal` + briefing Task 7 ✔; morning_digest → Task 7 (lean: solo señales vivas; federación muerta descartada) ✔; dream-synthesis → ya cubierto por memo dream (verificado: synthesize_cross_cluster/distill/chronicle) — sin task ✔; contacts_alias+whatsapp_live → Tasks 1-3 ✔; whatsapp-ingest → Task 8 (comando ya existe en memo) ✔; ocr_enrich → fuera (spec) ✔.
- Sin placeholders: algoritmos con fórmulas/regex/SQL verbatim citados; los pocos "verificar firma real" tienen file:line y fallback.
- Consistencia: knobs nuevos todos vía ChatConfig + owned set; stores bajo `state_dir/chat/`; patrones de test idénticos al plan 1.
