"""Shared embedder interface for the memo / memflow / rag ecosystem.

All concrete embedders (MLX in-process, socket-client daemon, sentence-transformers)
can implement this base so retrieval pipelines accept any backend without coupling
to MLX or the daemon transport.

Not an ABC: memo's MLXEmbedder defers mlx imports (CLAUDE.md invariant 4). Using
ABCMeta would make the abstract-method check fire before mlx is even importable on
non-Apple-Silicon machines. The contract is documented here and satisfied implicitly
(duck typing) or explicitly via inheritance.

Structural layout:
  - MLXEmbedder        → inherits EmbedderBase, overrides embed_query (asymmetric prefix)
  - SocketEmbedder     → inherits EmbedderBase, delegates to daemon
  - rag ST embedder    → can inherit or duck-type (symmetric, no prefix)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class EmbedderBase:
    """Template base shared by all memo-ecosystem embedders.

    Invariants every implementation must preserve:
    - embed() takes a *sequence of strings*, never a bare string.
    - Output shape: list[list[float]] where each inner list has length == dims.
    - embed_query() applies any asymmetric retrieval prefix (override when needed).
    """

    @property
    def dims(self) -> int:
        """Embedding dimensionality (must match the vector store schema)."""
        raise NotImplementedError

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document strings (NO query prefix).

        Args:
            inputs: Non-empty sequence of strings to embed.

        Returns:
            List of float vectors, one per input, each of length ``dims``.
        """
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (apply asymmetric prefix if needed).

        Default: calls embed([query]) and returns the first element.
        Override when the model uses a special query-side instruction prefix
        (e.g. Qwen3-Embedding asymmetric retrieval).
        """
        result = self.embed([query])
        if not result:
            raise RuntimeError("embed() returned empty list for single query")
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Alias for embed() with explicit list type."""
        return self.embed(texts)

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return ""

    @property
    def is_warm(self) -> bool:
        """True if pre-loaded (no cold-start latency expected)."""
        return False

    def health(self) -> dict[str, Any]:
        """Return a health/status dict for observability. Best-effort."""
        return {"dims": self.dims, "model": self.model_name, "warm": self.is_warm}
