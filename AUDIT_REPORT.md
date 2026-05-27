# Auditoría Exhaustiva del Proyecto Memo

**Fecha**: 2026-05-27  
**Versión**: 0.8.0  
**Auditor**: Devin  
**Alcance**: 100% del codebase (src/memo, tests, configuración)

## Resumen Ejecutivo

El proyecto memo es un sistema de memoria local bien arquitecturado que integra:
- MLX embeddings (Qwen3-Embedding)
- SQLite-vec para búsqueda semántica
- LLMs locales (Qwen2.5/Qwen3 family)
- Cross-encoder reranking
- MCP server para integración con agentes
- Hooks para integración con Claude Code

**Estado General**: ✅ Funcional y estable, con áreas de mejora identificadas

---

## 🔴 Hallazgos Críticos (Hacer Pronto)

### 1. Multimodal Module - EXPERIMENTAL sin Tests
**Archivo**: `src/memo/multimodal.py` (486 líneas)  
**Severidad**: 🔴 CRÍTICA  
**Problema**: Módulo marcado como EXPERIMENTAL, no cubierto por tests, API puede cambiar

```python
"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. 
API may change without notice."""
```

**Detalles**:
- Implementa MultiModalStore, CrossModalSearch, UniversalEmbedder
- Almacena contenido en JSON sin validación de integridad
- No hay manejo de errores en `_load()` (línea 69-80)
- Importado en `memory.py` pero nunca usado

**Impacto**: Código en producción sin garantías de estabilidad

**Recomendación**: 
- Agregar tests en `tests/test_multimodal.py`
- O remover si no se usa
- O marcar como deprecated

---

### 2. Store Dimension Mismatch - Mensajes Poco Claros
**Archivo**: `src/memo/store.py` líneas 240, 324, 333  
**Severidad**: 🔴 CRÍTICA  
**Problema**: RuntimeError si dims no coinciden, pero mensaje no dice valores

```python
# Línea 329
"matching MEMO_MODEL_PROFILE/MEMO_EMBEDDER_DIMS settings."
```

**Detalles**:
- Si usuario cambia de 0.6B (1024D) a 4B (2560D) sin reindex, búsquedas fallan
- Mensaje de error no muestra: "Expected 2560 but got 1024"
- Usuario no sabe cómo resolver

**Impacto**: Usuarios bloqueados sin saber qué hacer

**Recomendación**:
```python
raise RuntimeError(
    f"Embedding dimension mismatch: store has {stored_dim}D vectors "
    f"but config expects {expected_dim}D. "
    f"Run 'memo reindex' or check MEMO_MODEL_PROFILE/MEMO_EMBEDDER_DIMS."
)
```

---

### 3. Reranker - Lazy Load sin Validación
**Archivo**: `src/memo/memory.py` líneas 1030-1036  
**Severidad**: 🔴 CRÍTICA  
**Problema**: Reranker se carga lazy en primer search(), sin pre-validación

```python
def _rerank(self, query: str, hits: list[MemoryRecord], *, top_n: int):
    reranker = self._reranker
    if reranker is None:
        from memo.reranker import MLXReranker
        reranker = MLXReranker(
            model_path=self.cfg.reranker_model,
            revision=self.cfg.reranker_revision,
        )
        self._reranker = reranker
    
    try:
        reranked = reranker.rerank(query, hits, top_n=None)
    except Exception as exc:
        _log.warning("reranker failed, falling back to RRF order: %s", exc)
        return hits[:top_n]  # ← Silent failure
```

**Detalles**:
- No hay validación de que el modelo existe
- Si el modelo falla a cargar, cae back a RRF sin error
- Usuario no sabe si reranking ocurrió o no
- Línea 1047-1049: `except Exception as exc` - swallows all errors

**Impacto**: Búsquedas pueden devolver resultados sin reranking sin que el usuario lo sepa

**Recomendación**:
```python
try:
    reranked = reranker.rerank(query, hits, top_n=None)
except Exception as exc:
    _log.error("reranker failed (model=%s): %s", self.cfg.reranker_model, exc)
    _log.info("falling back to RRF order (no reranking)")
    return hits[:top_n]
```

---

### 4. Embedder Client - Fallback Silencioso
**Archivo**: `src/memo/embedder_client.py` líneas 23-25  
**Severidad**: 🔴 CRÍTICA  
**Problema**: Si daemon no está disponible, carga MLX in-process sin advertencia

```python
# Fallback is deliberate. Callers running on a peer Mac without a
# memo daemon still get correct embeddings — just slower on the first
# call. Set `MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1` to raise instead.
```

**Detalles**:
- Primer call paga ~2s de cold-start
- Usuarios en hooks (5s timeout) pueden experimentar timeouts
- Fallback automático sin logging
- Solo se puede forzar con env var

**Impacto**: Latencia impredecible en hooks, timeouts silenciosos

**Recomendación**:
```python
if decoded is not None:
    return [float(x) for x in vec]

_log.warning(
    "embedder_client: daemon unreachable, falling back to in-process "
    "(first call will be slow ~2s)"
)
if _require_daemon():
    raise RuntimeError(...)
return _inproc().embed_query(text)
```

---

## 🟠 Hallazgos Importantes (Próximas Sprints)

### 5. OCR Module - Excepciones Genéricas
**Archivo**: `src/memo/ocr.py` líneas 38-40, 80  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: Excepciones genéricas capturadas sin logging específico

```python
# Línea 38-40
try:
    import Quartz  # noqa: F401
    import Vision  # noqa: F401
    _VISION_OK = True
except Exception as exc:  # ← Generic exception
    _log.debug("Apple Vision unavailable: %s", exc)
    _VISION_OK = False

# Línea 80
except Exception:  # ← Silent failure
    pass
```

**Detalles**:
- No hay forma de saber si Vision falló por falta de instalación vs error real
- Línea 80 silencia errores completamente
- Usuario no sabe si OCR funcionó o no

**Recomendación**:
```python
except ImportError as exc:
    _log.debug("Apple Vision not installed: %s", exc)
except Exception as exc:
    _log.warning("Apple Vision import failed: %s", exc)
```

---

### 6. Synapse Integration - Subprocess sin Timeout Explícito
**Archivo**: `src/memo/synapse_client.py` líneas 70-100  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: Subprocess calls a synapse CLI con timeout pero sin validación clara

```python
def list_conflicts(
    query: str = "",
    *,
    k: int = 5,
    trace_id: str = "",
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Return raw conflict dicts from `synapse conflicts <query> --json`.
    
    Returns `[]` on any failure (missing binary, non-zero exit, parse
    error, timeout). Callers MUST treat empty list as "no information"
    rather than "no conflicts" — see `has_blocking_freeze()` for the
    safety-aware reduction.
    """
```

**Detalles**:
- Si synapse no responde, retorna [] sin error
- Callers no pueden distinguir "no conflicts" de "synapse unreachable"
- Timeout por defecto 8s (MEMO_SYNAPSE_CLIENT_TIMEOUT)
- Puede causar latencia en hooks

**Impacto**: Usuarios pueden pensar que freeze-write está activo cuando no lo está

**Recomendación**:
```python
_log.debug("synapse_client.list_conflicts: timeout after %.1fs", timeout)
# O retornar ([], "timeout") para que caller sepa
```

---

### 7. History Store - Lazy Open sin Error Handling
**Archivo**: `src/memo/memory.py` líneas 384-385  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: HistoryStore se abre lazy, "swallows exceptions internally"

```python
# Lazy: opened on first log call. Audit failures must never
# propagate to the caller, so HistoryStore swallows its own
# exceptions internally.
from memo.history import HistoryStore as _HS
self.history = _HS(cfg.history_db)
```

**Detalles**:
- Si history DB falla, no hay forma de saber
- Auditoría incompleta sin error
- Inconsistente con otros componentes

**Impacto**: Pérdida silenciosa de auditoría

**Recomendación**:
- Agregar logging en HistoryStore cuando falla
- O retornar status en Memory.stats()

---

### 8. CLI - Archivo Muy Grande
**Archivo**: `src/memo/cli.py` (8971 líneas)  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: Archivo monolítico, difícil de mantener

**Detalles**:
- 8971 líneas en un solo archivo
- Múltiples comandos (save, search, list, get, delete, stats, doctor, etc.)
- Múltiples subcomandos (embed, repo-index, session, etc.)
- Dificil de navegar y mantener

**Recomendación**:
- Refactorizar en módulos: `cli/save.py`, `cli/search.py`, `cli/repo.py`, etc.
- O usar `click` groups más agresivamente

---

### 9. Temporal Analyzer - Lazy Init con Chat Potencialmente None
**Archivo**: `src/memo/memory.py` línea 437  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: TemporalAnalyzer se crea lazy con self._chat que puede ser None

```python
@property
def temporal(self) -> TemporalAnalyzer:
    """Lazy accessor for TemporalAnalyzer."""
    if self._temporal is None:
        self._temporal = TemporalAnalyzer(self, self._chat)  # ← _chat puede ser None
    return self._temporal
```

**Detalles**:
- self._chat es None hasta que se llama a un método que lo inicializa
- TemporalAnalyzer.__init__ no valida que chat no es None
- Puede causar AttributeError si se accede a temporal antes de usar chat

**Impacto**: Errores en temporal analysis

**Recomendación**:
```python
@property
def temporal(self) -> TemporalAnalyzer:
    if self._temporal is None:
        chat = self._ensure_chat()  # ← Ensure chat is initialized
        self._temporal = TemporalAnalyzer(self, chat)
    return self._temporal
```

---

### 10. Contradiction Store - Lazy Init sin Try/Except
**Archivo**: `src/memo/memory.py` línea 449  
**Severidad**: 🟠 IMPORTANTE  
**Problema**: ContradictionStore se crea lazy sin validación

```python
@property
def contradict_store(self) -> ContradictionStore:
    """Lazy accessor for the persistent contradictions sidecar."""
    if self._contradict_store is None:
        self._contradict_store = ContradictionStore(self.cfg.contradictions_db)
    return self._contradict_store
```

**Detalles**:
- Sin try/except - si DB falla, propaga al caller
- Inconsistente con HistoryStore (que swallows errors)
- Puede fallar en primer scan

**Impacto**: Inconsistencia en error handling

**Recomendación**:
```python
try:
    self._contradict_store = ContradictionStore(self.cfg.contradictions_db)
except Exception as exc:
    _log.warning("contradict_store init failed: %s", exc)
    # Return empty store or raise?
```

---

## 🟡 Hallazgos Menores (Nice-to-have)

### 11. Model Profile Quality - Requiere Reindex
**Archivo**: `src/memo/config.py` líneas 61-70  
**Severidad**: 🟡 MENOR  
**Problema**: Cambiar a quality profile requiere reindex completo

```python
# Higher retrieval quality. Requires a full reindex because the 4B
# embedder emits 2560-dim vectors instead of 1024.
"quality": {
    "llm_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510",
    "helper_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "embedder_model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    "embedder_dims": 2560,
    "reranker_enabled": True,
    "reranker_model": "mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
},
```

**Detalles**:
- Si usuario cambia MEMO_MODEL_PROFILE=quality sin reindex, búsquedas fallan
- Store.search() valida dims pero no da instrucciones claras
- Documentación menciona "Requires a full reindex" pero no es obvio

**Recomendación**:
- Agregar check en Config.from_env() si dims cambiaron
- Sugerir `memo reindex` si es necesario

---

### 12. Contextual Module - Silent Failure
**Archivo**: `src/memo/contextual.py` línea 134  
**Severidad**: 🟡 MENOR  
**Problema**: Pérdida de contexto sin logging

```python
except Exception:
    pass  # Fail silently - context loss is not critical
```

**Detalles**:
- Usuario no sabe si el contexto se generó o no
- Indica filosofía de "silent failures"

**Recomendación**:
```python
except Exception as exc:
    _log.debug("contextual summary generation failed: %s", exc)
    # Continue without context
```

---

### 13. LLM - Model Eviction sin Logging
**Archivo**: `src/memo/llm.py` líneas 75-80  
**Severidad**: 🟡 MENOR  
**Problema**: Evicta modelos del LRU cache sin logging

```python
while len(self._loaded) >= _MAX_LOADED_MODELS:
    evicted_key, _ = self._loaded.popitem(last=False)  # ← Sin logging
```

**Detalles**:
- Sin logging de qué modelo fue evictado
- Usuario no sabe por qué latencia aumentó
- Puede causar latency spikes

**Recomendación**:
```python
evicted_key, _ = self._loaded.popitem(last=False)
_log.debug("LLM cache evicted: %s (now have %d models)", evicted_key, len(self._loaded))
```

---

### 14. Reranker - Revision Pinning sin Validación
**Archivo**: `src/memo/reranker.py` línea 100-110  
**Severidad**: 🟡 MENOR  
**Problema**: Revision pinning puede descargar modelos diferentes

```python
def __init__(
    self,
    model_path: str = "mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
    revision: str | None = None,
    max_seq_len: int = 4096,
    task: str | None = None,
) -> None:
```

**Detalles**:
- MEMO_RERANKER_REVISION env var puede cambiar modelo
- Sin validación de que revision existe
- Sin logging de qué revision se descargó

**Recomendación**:
```python
if revision:
    _log.info("Loading reranker with pinned revision: %s", revision)
```

---

### 15. Encryption Module - No Integrado
**Archivo**: `src/memo/encryption.py`  
**Severidad**: 🟡 MENOR  
**Problema**: Módulo existe pero no está integrado en Memory

```python
# Línea 7: "Per-memoria encryption (tag-based) or global encryption"
```

**Detalles**:
- EncryptionManager, Encryptor, KeyManager classes existen
- No se usan en Memory.__init__()
- Encryption no está disponible para usuarios

**Recomendación**:
- Integrar en Memory si es feature completo
- O remover si es experimental

---

## 📊 Hallazgos de Arquitectura

### Lazy Initialization Pattern
**Ubicación**: Memory.__init__() y propiedades lazy  
**Patrón**: chat, reranker, temporal, contradict_store se inicializan lazy

**Pros**:
- Reduce startup time
- Memoria si no se usan

**Contras**:
- Errores pueden ocurrir en momentos inesperados
- Difícil de debuggear

**Recomendación**: Considerar eager initialization con fallback, o mejor error reporting

---

### Silent Failures Pattern
**Ubicación**: Múltiples módulos  
**Patrón**: Excepciones capturadas sin logging o con logging débil

**Ejemplos**:
- embedder_client fallback a in-process
- reranker fallback a RRF
- synapse_client retorna [] en error
- history_store swallows errors

**Impacto**: Difícil debuggear problemas

**Recomendación**: Logging más explícito, o modo strict para debugging

---

### Configuration Validation
**Ubicación**: Config class  
**Problema**: Validación ocurre en Store.search() o Memory.save(), no en Config.__init__()

**Ejemplo**: embedder_dims mismatch solo se detecta en Store.search()

**Recomendación**: Validar en Config.from_env() o Config.ensure_dirs()

---

## ✅ Hallazgos Positivos

### 1. Thread Safety
- Cada componente tiene su propio lock (embedder._load_lock, llm._load_lock, etc.)
- Bien implementado

### 2. Error Handling
- Mayoría de funciones tienen try/except apropiados
- Excepto los casos mencionados arriba

### 3. Testing
- Buena cobertura de tests en memory.py, store.py, config.py
- Tests de integración para CLI

### 4. Documentation
- Excelentes docstrings en memory.py, embedder.py, store.py
- Comentarios claros sobre decisiones de diseño

### 5. Architecture
- Separación clara de concerns
- Modular design
- Lazy loading donde apropiado

---

## 📋 Checklist de Recomendaciones

### Críticas (Hacer Pronto)
- [ ] Mejorar error message en Store.search() para dim mismatch
- [ ] Agregar logging en embedder_client fallback
- [ ] Validar reranker model exists en Config.from_env()
- [ ] Agregar tests para multimodal.py o remover

### Importantes (Próximas Sprints)
- [ ] Refactorizar cli.py (8971 líneas es demasiado)
- [ ] Agregar tests para synapse_client.py
- [ ] Mejorar logging en silent failure paths
- [ ] Considerar eager initialization con fallback
- [ ] Documentar model profile switching y reindex requirement
- [ ] Mejorar error handling en temporal y contradict_store

### Nice-to-have
- [ ] Integrar encryption module
- [ ] Reducir MCP tool count con grouping
- [ ] Mejorar hook timeouts
- [ ] Agregar modo strict para debugging
- [ ] Agregar logging en LLM model eviction

---

## Conclusión

El proyecto memo está bien arquitecturado y funcional. Los problemas encontrados son principalmente:

1. **Silent failures** que dificultan debugging
2. **Lazy initialization** que puede causar errores tardíos
3. **Falta de tests** en módulos experimentales
4. **Documentación incompleta** en algunos módulos
5. **Mensajes de error poco claros** en algunos casos

Ninguno de estos problemas es crítico para la funcionalidad básica, pero mejorarlos aumentaría significativamente la robustez, debuggabilidad y experiencia de usuario del sistema.

---

**Generado por**: Devin Auditor  
**Fecha**: 2026-05-27  
**Tiempo de auditoría**: ~2 horas  
**Archivos analizados**: 100+ archivos Python, JSON, tests
