"""Platform capability probes (leaf module — no intra-memo imports).

Used to pick the embedder backend and gate MLX-only features (reranker,
LLM chat) on non-Apple-Silicon hosts. Kept dependency-free and fully typed
so `config.py` (phase-1 strict mypy) can import it without coupling.

The MLX runtime ships wheels for Apple Silicon only. On Linux (incl. Ubuntu)
and Intel macs memo falls back to the CPU `sentence-transformers` backend
(see `embedder_select.make_embedder`).
"""

from __future__ import annotations

import importlib.util
import platform
import sys


def is_apple_silicon() -> bool:
    """True on macOS arm64 — the only platform where MLX wheels exist."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """True if the `mlx_lm` runtime can be imported on this host.

    Cheap: probes the import spec without importing (no MLX cold-start).
    Used by `embedder_select.resolve_backend` for the `auto` default.
    """
    if not is_apple_silicon():
        return False
    return importlib.util.find_spec("mlx_lm") is not None
