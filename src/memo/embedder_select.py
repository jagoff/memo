"""Single decision point: which embedder backend to construct.

`MLXEmbedder` (Apple Silicon) vs `STEmbedder` (CPU `sentence-transformers`,
for Linux/Ubuntu). Every in-process embedder construction —
the `Memory` facade, the `embedder_client` daemon fallback, and bulk
`memo ingest` — routes through `make_embedder` so the backend choice lives
in one place.

Selection (`cfg.embedder_backend`, env `MEMO_EMBEDDER_BACKEND`):
    "auto" (default) → MLX when the runtime is importable, else ST.
    "mlx"            → force MLX (errors later if MLX is absent).
    "st"             → force the CPU backend.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from memo.embed_base import EmbedderBase
from memo.errors import MemoError
from memo.model_pins import ModelPinError, model_identity
from memo.platform_detect import mlx_available

if TYPE_CHECKING:
    from memo.config import Config


def resolve_backend(cfg: Config) -> str:
    """Return the effective backend id: "mlx" or "st"."""
    raw = (cfg.embedder_backend or "auto").strip().lower()
    if raw in ("mlx", "st"):
        return raw
    if mlx_available():
        return "mlx"
    if sys.platform.startswith("linux"):
        return "st"
    raise MemoError(
        f"Automatic embedding is unavailable on unsupported platform {sys.platform!r}: "
        "MLX requires Apple Silicon, while automatic CPU/ST fallback is supported "
        "only on Linux. Set MEMO_EMBEDDER_BACKEND=st explicitly for a manual "
        "sentence-transformers setup."
    )


def active_embedder_identity(cfg: Config) -> str:
    """Return the exact model identity that owns vectors and cache entries."""
    if resolve_backend(cfg) == "st":
        model = cfg.st_embedder_model
        revision = cfg.st_embedder_revision
    else:
        model = cfg.embedder_model
        revision = cfg.embedder_revision
    try:
        return model_identity(model, revision)
    except ModelPinError:
        # Preserve legacy/test vector ownership metadata. Production Config.from_env
        # rejects mutable remote overrides before Memory reaches this point.
        return model


def make_embedder(cfg: Config, *, cache_size: int | None = None) -> EmbedderBase:
    """Construct the in-process embedder for this host/config.

    `cache_size` is forwarded to the MLX backend's query cache; the ST backend
    ignores it (CPU query embeds are cheap and uncached).
    """
    if resolve_backend(cfg) == "st":
        from memo.embedder_st import STEmbedder

        return STEmbedder(
            model_path=cfg.st_embedder_model,
            revision=cfg.st_embedder_revision,
            expected_dims=cfg.embedder_dims,
        )
    from memo.embedder import MLXEmbedder

    return MLXEmbedder(
        model_path=cfg.embedder_model,
        revision=cfg.embedder_revision,
        expected_dims=cfg.embedder_dims,
        cache_size=cache_size,
    )
