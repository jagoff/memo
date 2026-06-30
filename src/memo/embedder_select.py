"""Single decision point: which embedder backend to construct.

`MLXEmbedder` (Apple Silicon) vs `STEmbedder` (CPU `sentence-transformers`,
for Linux/Ubuntu and Intel macs). Every in-process embedder construction —
the `Memory` facade, the `embedder_client` daemon fallback, and bulk
`memo ingest` — routes through `make_embedder` so the backend choice lives
in one place.

Selection (`cfg.embedder_backend`, env `MEMO_EMBEDDER_BACKEND`):
    "auto" (default) → MLX when the runtime is importable, else ST.
    "mlx"            → force MLX (errors later if MLX is absent).
    "st"             → force the CPU backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memo.embed_base import EmbedderBase
from memo.platform_detect import mlx_available

if TYPE_CHECKING:
    from memo.config import Config


def resolve_backend(cfg: Config) -> str:
    """Return the effective backend id: "mlx" or "st"."""
    raw = (cfg.embedder_backend or "auto").strip().lower()
    if raw in ("mlx", "st"):
        return raw
    return "mlx" if mlx_available() else "st"


def make_embedder(cfg: Config, *, cache_size: int | None = None) -> EmbedderBase:
    """Construct the in-process embedder for this host/config.

    `cache_size` is forwarded to the MLX backend's query cache; the ST backend
    ignores it (CPU query embeds are cheap and uncached).
    """
    if resolve_backend(cfg) == "st":
        from memo.embedder_st import STEmbedder

        return STEmbedder(
            model_path=cfg.st_embedder_model,
            expected_dims=cfg.embedder_dims,
        )
    from memo.embedder import MLXEmbedder

    return MLXEmbedder(
        model_path=cfg.embedder_model,
        expected_dims=cfg.embedder_dims,
        cache_size=cache_size,
    )
