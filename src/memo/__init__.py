"""memo — local MCP memory backed by Markdown, sqlite-vec, and MCP.

100% local stack — zero Ollama, zero cloud APIs:

- LLM: `mlx-lm` running quantized Qwen models on Apple Silicon Metal
  (in-process, no daemon) for ask/synthesis/dream.
- Embedder: MLX Qwen3-Embedding on Apple Silicon, or CPU
  sentence-transformers (`STEmbedder`) on Linux.
- Vector store: `sqlite-vec` (single file, no daemon, no Qdrant).
- Storage of record: markdown files under `MEMO_DATA_DIR`, or under
  `<vault>/<SYSTEM_DIR>/AI/memory/` when `MEMO_MEMORIES_IN_VAULT=1`.
- MCP server: `fastmcp`, profile-gated from the 41-tool `agent`
  surface through the 164-tool `full` surface.

Public API:

    from memo import Memory, Config

    cfg = Config.from_env()
    mem = Memory(cfg)
    mem.save(content="...", title="...", tags=["x", "y"])
    hits = mem.search("query", limit=10)
"""

import os
from pathlib import Path

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
#
# Distribution metadata is only a *snapshot* of that. For an editable install
# the snapshot goes stale the instant the version is bumped — nothing rewrites
# `.dist-info` until someone reinstalls — so trusting it makes freshly bumped
# source report the previous release (this is how 4.9.3's code reported 4.9.2).
# A source checkout is therefore authoritative for its own version; an
# installed wheel has no pyproject above it and falls through to the metadata.


def _checkout_version(pkg_init: Path) -> str | None:
    """Version declared by the mlx-memo checkout that owns ``pkg_init``.

    ``None`` whenever ``pkg_init`` is not the ``<repo>/src/memo/__init__.py`` of
    an mlx-memo checkout — which is every installed distribution, and also any
    unrelated project that happens to vendor a ``src/memo/``.
    """
    if pkg_init.parent.parent.name != "src":
        return None
    try:
        text = (pkg_init.parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    if 'name = "mlx-memo"' not in text:
        return None
    import tomllib

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    declared = data.get("project", {}).get("version")
    return declared if isinstance(declared, str) else None


def _resolve_version() -> str:
    checkout = _checkout_version(Path(__file__).resolve())
    if checkout is not None:
        return checkout
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _version

        return _version("mlx-memo")
    except PackageNotFoundError:  # pragma: no cover — checkout w/o metadata
        return "0.0.0+unknown"
    except Exception:  # pragma: no cover — defensive
        return "0.0.0+unknown"


__version__ = _resolve_version()

__all__ = ["Config", "Memory", "MemoryRecord", "__version__"]
