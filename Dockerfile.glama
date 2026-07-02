# Glama sandbox image: start the MCP stdio server by default so Glama can run
# protocol introspection (tools/list, resources/list, prompts/list).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMO_NONINTERACTIVE=1 \
    MEMO_AUTO_UPDATE=0 \
    MEMO_MCP_PROFILE=core \
    MEMO_EMBEDDER_BACKEND=st \
    MEMO_ST_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B \
    MEMO_EMBEDDER_DIMS=1024 \
    MEMO_DATA_DIR=/data \
    MEMO_STATE_DIR=/data/state \
    HF_HOME=/opt/hf-cache

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install "mlx-memo[cpu]"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

RUN useradd -m memo && mkdir -p /data/state /opt/hf-cache \
    && chown -R memo:memo /data /opt/hf-cache

USER memo
VOLUME /data

CMD ["memo-mcp"]
