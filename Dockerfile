# memo — imagen CPU (Linux/cross-platform). Sin MLX: search/recall/save.
# Reranker + ask/synthesize/dream son MLX-only (Apple Silicon) y quedan OFF
# con un mensaje claro. NO es "memo completo" — es la puerta de entrada.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMO_EMBEDDER_BACKEND=st \
    MEMO_ST_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B \
    MEMO_EMBEDDER_DIMS=1024 \
    MEMO_DATA_DIR=/data \
    HF_HOME=/opt/hf-cache

# torch CPU-only primero (evita bajar el build CUDA gigante), después memo[cpu].
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install "mlx-memo[cpu]"

# Pre-hornear el modelo de embeddings en la imagen: el primer run NO baja nada.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

# El índice sqlite (memvec.db, graph/history) vive bajo state_dir, NO data_dir.
# Apuntarlo dentro del volumen para que UN solo `-v vol:/data` persista índice
# + markdown entre contenedores. (ENV acá, post-bake, para no invalidar la cache.)
ENV MEMO_STATE_DIR=/data/state

# Usuario no-root + volumen de datos persistente.
RUN useradd -m memo && mkdir -p /data /data/state /opt/hf-cache \
    && chown -R memo:memo /data /opt/hf-cache
USER memo
VOLUME /data

# `docker run IMG` -> memo doctor (self-check).
# Otros usos: `docker run IMG memo save '...'` / `docker run IMG memo search '...'`
# Servidor MCP stdio:  `docker run -i IMG memo-mcp`
CMD ["memo", "doctor"]
