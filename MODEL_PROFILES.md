# Model Profiles Guide

memo tiene 3 perfiles de modelos predefinidos. Cambiar entre ellos requiere cuidado.

## Perfiles Disponibles

### `light` (menor latencia)
```
Embedder: Qwen3-Embedding-0.6B (1024D)
LLM: Qwen2.5-7B-Instruct
Helper: Qwen2.5-3B-Instruct
Reranker: DISABLED
```

**Cuándo usar**: CI, máquinas lentas, latencia crítica (hooks)

**Ventajas**: Rápido (~500ms recall-hook), bajo VRAM

**Desventajas**: Calidad de búsqueda menor, sin reranking

---

### `balanced` (default)
```
Embedder: Qwen3-Embedding-0.6B (1024D)
LLM: Qwen2.5-7B-Instruct
Helper: Qwen2.5-3B-Instruct
Reranker: Qwen3-Reranker-0.6B (ENABLED)
```

**Cuándo usar**: Producción, máquinas normales (M1+)

**Ventajas**: Buen balance calidad/latencia, reranking mejora MRR 30-60%

**Desventajas**: Requiere ~6GB VRAM

---

### `quality` (mejor búsqueda)
```
Embedder: Qwen3-Embedding-4B (2560D)
LLM: Qwen3-30B-A3B-Instruct
Helper: Qwen2.5-3B-Instruct
Reranker: Qwen3-Reranker-0.6B (ENABLED)
```

**Cuándo usar**: Búsqueda crítica, máquinas potentes (M2+ con 36GB+)

**Ventajas**: Mejor calidad de búsqueda, embeddings más ricos

**Desventajas**: 
- Requiere ~12GB VRAM
- **REQUIERE REINDEX COMPLETO** (cambio de dimensionalidad)
- Más lento (~2s recall-hook)

---

## Cambiar Perfiles

### Opción 1: Env var (temporal)
```bash
MEMO_MODEL_PROFILE=quality memo search "query"
```

### Opción 2: Config file (~/.config/memo/config.toml)
```toml
[storage]
data_dir = "~/Documents/memo"

[models]
model_profile = "quality"
```

### Opción 3: Env vars específicas (override)
```bash
export MEMO_EMBEDDER_MODEL="mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
export MEMO_EMBEDDER_DIMS=2560
export MEMO_LLM_MODEL="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ"
```

---

## ⚠️ IMPORTANTE: Cambiar de 0.6B a 4B Embedder

Si cambias de `balanced` (1024D) a `quality` (2560D):

### 1. Backup
```bash
cp ~/.local/share/memo/memvec.db ~/.local/share/memo/memvec.db.backup
```

### 2. Reindex
```bash
rm ~/.local/share/memo/memvec.db
memo reindex
```

**Por qué**: El store valida que los vectores coincidan con la dimensionalidad esperada. Si no reindexas, `memo search` fallará con:
```
Embedding dimension mismatch: store has 1024D vectors but config expects 2560D.
Fix: rm ~/.local/share/memo/memvec.db && memo reindex
```

### 3. Esperar
Reindex puede tomar minutos dependiendo del tamaño del corpus.

---

## Validación Automática

`memo` valida el modelo reranker en `Config.from_env()`:
- Si el modelo no existe en HuggingFace, logs warning
- Reintenta en primer `search()` (lazy load)
- Si falla, fallback a RRF (sin reranking)

---

## Debugging

### Ver config actual
```bash
memo stats
```

### Ver logs de carga de modelos
```bash
MEMO_RECALL_DEBUG=1 memo search "query"
```

### Validar embedder
```bash
memo embed-daemon status
```

---

## Recomendaciones

| Máquina | Perfil | Razón |
|---------|--------|-------|
| M1/M2 (8GB) | `light` | Evita OOM |
| M1/M2 (16GB) | `balanced` | Default recomendado |
| M2 Pro/Max (32GB+) | `quality` | Mejor búsqueda |
| CI/Testing | `light` | Rápido, determinista |
