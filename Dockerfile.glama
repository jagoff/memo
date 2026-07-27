# Glama sandbox image: build memo from this checkout, then run the packaged
# MCP stdio server for protocol introspection.
FROM python:3.14-slim@sha256:d3400aa122fa42cf0af0dbe8ec3091b047eac5c8f7e3539f7135e86d855dc015 AS builder

WORKDIR /src
# Explicit build allowlist: never send a developer's ignored vault, .env or
# runtime state into a layer/cache. Keep this aligned with hatch force-include.
COPY pyproject.toml uv.lock README.md LICENSE server.json ./
COPY src ./src
COPY .agents ./.agents
COPY .claude-plugin ./.claude-plugin
COPY commands ./commands
COPY hooks ./hooks
COPY plugins ./plugins
COPY skills ./skills
COPY statusline ./statusline
RUN mkdir -p /dist \
    && python -m pip install --no-cache-dir uv==0.11.21 \
    && uv export --frozen --extra cpu --no-dev --no-emit-project \
        --no-emit-package torch --no-emit-package cuda-bindings \
        --no-emit-package cuda-pathfinder --no-emit-package cuda-toolkit \
        --no-emit-package nvidia-cublas --no-emit-package nvidia-cuda-cupti \
        --no-emit-package nvidia-cuda-nvrtc --no-emit-package nvidia-cuda-runtime \
        --no-emit-package nvidia-cudnn-cu13 --no-emit-package nvidia-cufft \
        --no-emit-package nvidia-cufile --no-emit-package nvidia-curand \
        --no-emit-package nvidia-cusolver --no-emit-package nvidia-cusparse \
        --no-emit-package nvidia-cusparselt-cu13 --no-emit-package nvidia-nccl-cu13 \
        --no-emit-package nvidia-nvjitlink --no-emit-package nvidia-nvshmem-cu13 \
        --no-emit-package nvidia-nvtx --no-emit-package triton \
        --format requirements-txt --output-file /dist/runtime-requirements.txt \
    && uv build --wheel --out-dir /dist

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
    MEMO_ST_EMBEDDER_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
    MEMO_EMBEDDER_DIMS=1024 \
    MEMO_DATA_DIR=/data \
    MEMO_STATE_DIR=/data/state \
    HF_HOME=/opt/hf-cache

COPY --from=builder /dist/ /tmp/dist/
RUN wheel=$(find /tmp/dist -name '*.whl' -print -quit) \
    && test -n "$wheel" \
    && test -n "$EXPECTED_VERSION" \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install --require-hashes -r /tmp/dist/runtime-requirements.txt \
    && python -m pip install --no-deps "$wheel" \
    && EXPECTED_VERSION="$EXPECTED_VERSION" python -c "import os; import memo; expected = os.environ['EXPECTED_VERSION']; installed = memo.__version__; assert installed == expected, f'{installed} != {expected}'" \
    && rm -rf /tmp/dist

RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['MEMO_ST_EMBEDDER_MODEL'], revision=os.environ['MEMO_ST_EMBEDDER_REVISION'])"

# Network is allowed only while the exact model revision is populated above.
# Runtime resolves the same configured commit exclusively from that cache.
ENV HF_HUB_OFFLINE=1

RUN useradd -m memo && mkdir -p /data/state /opt/hf-cache \
    && chown -R memo:memo /data /opt/hf-cache

USER memo
VOLUME /data

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["memo", "config", "validate"]

CMD ["memo-mcp"]
