"""memo — local MCP memory backed by Markdown, sqlite-vec, and MCP.

100% local stack — zero Ollama, zero cloud APIs:

- LLM: `mlx-lm` running quantized Qwen models on Apple Silicon Metal
  (in-process, no daemon) for ask/synthesis/dream.
- Embedder: MLX Qwen3-Embedding on Apple Silicon, or CPU
  sentence-transformers (`STEmbedder`) on Linux/Intel macOS.
- Vector store: `sqlite-vec` (single file, no daemon, no Qdrant).
- Storage of record: markdown files under `MEMO_DATA_DIR`, or under
  `<vault>/<SYSTEM_DIR>/AI/memory/` when `MEMO_MEMORIES_IN_VAULT=1`.
- MCP server: `fastmcp`, profile-gated from the 14-tool `agent`
  surface through the 129-tool `full` surface.

Public API:

    from memo import Memory, Config

    cfg = Config.from_env()
    mem = Memory(cfg)
    mem.save(content="...", title="...", tags=["x", "y"])
    hits = mem.search("query", limit=10)
"""

import os

# Silence HuggingFace hub progress/download bars globally, before any model
# load. Otherwise embedder/llm/reranker loads, prewarm, and daemon startups
# leak repeated "Fetching N files / Download complete 0.00B" noise for
# already-cached models. setdefault honors an explicit override (e.g. the
# installer may set "0" to show real first-download progress).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from memo.mlx_gpu import suppress_swig_deprecation_warnings

suppress_swig_deprecation_warnings()

from memo.config import Config  # noqa: E402
from memo.memory import Memory, MemoryRecord  # noqa: E402

# Single source of truth lives in pyproject.toml `[project] version`.
# Resolve at import time from the installed distribution metadata so
# `memo.__version__` always matches `pip show mlx-memo`. Falls back to
# a sentinel when running from an uninstalled checkout.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("mlx-memo")
except PackageNotFoundError:  # pragma: no cover — editable install w/o metadata
    __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover — defensive
    __version__ = "0.0.0+unknown"

__all__ = ["Config", "Memory", "MemoryRecord", "__version__"]
