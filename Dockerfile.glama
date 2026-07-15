# Glama sandbox image: build memo from this checkout, then run the packaged
# MCP stdio server for protocol introspection.
FROM python:3.14-slim@sha256:d3400aa122fa42cf0af0dbe8ec3091b047eac5c8f7e3539f7135e86d855dc015 AS builder

WORKDIR /src
COPY . .
RUN python -m pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

FROM python:3.14-slim@sha256:d3400aa122fa42cf0af0dbe8ec3091b047eac5c8f7e3539f7135e86d855dc015 AS runtime

ARG EXPECTED_VERSION
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMO_NONINTERACTIVE=1 \
    MEMO_UPDATE_CHECK_ENABLED=0 \
    MEMO_AUTO_UPDATE=0 \
    MEMO_STATUSLINE_SELFHEAL=0 \
    MEMO_HOOK_SELFHEAL=0 \
    MEMO_MCP_PROFILE=agent \
    MEMO_EMBEDDER_BACKEND=st \
    MEMO_ST_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B \
    MEMO_EMBEDDER_DIMS=1024 \
    MEMO_DATA_DIR=/data \
    MEMO_STATE_DIR=/data/state \
    HF_HOME=/opt/hf-cache \
    HF_MODEL_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

COPY --from=builder /dist/ /tmp/dist/
RUN wheel=$(find /tmp/dist -name '*.whl' -print -quit) \
    && test -n "$wheel" \
    && test -n "$EXPECTED_VERSION" \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install "${wheel}[cpu]" \
    && EXPECTED_VERSION="$EXPECTED_VERSION" python -c "import os; import memo; expected = os.environ['EXPECTED_VERSION']; installed = memo.__version__; assert installed == expected, f'{installed} != {expected}'" \
    && rm -rf /tmp/dist

RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['MEMO_ST_EMBEDDER_MODEL'], revision=os.environ[\"HF_MODEL_REVISION\"])"

RUN useradd -m memo && mkdir -p /data/state /opt/hf-cache \
    && chown -R memo:memo /data /opt/hf-cache

USER memo
VOLUME /data

CMD ["memo-mcp"]
