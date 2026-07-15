"""CPU embedder backend (`sentence-transformers`) for non-Apple-Silicon hosts.

memo's primary embedder is MLX (`embedder.MLXEmbedder`), which only runs on
Apple Silicon. On Linux/Ubuntu and Intel macs this `STEmbedder` provides the
same `EmbedderBase` surface backed by a CPU `sentence-transformers` model, so
search / recall / save work without MLX. Selected by `embedder_select`.

Default model is `Qwen/Qwen3-Embedding-0.6B` — the SAME family and dimensionality
(1024) as memo's default MLX quant (`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`),
so the vec0 schema (`embedder_dims=1024`) is unchanged and MLX invariant #1
(asymmetric retrieval prefix on queries only) is preserved verbatim.

Cross-backend note: MLX-4bit and ST-fp vectors live in slightly different regions
of the space — an Ubuntu node is a STANDALONE corpus, not a vec-coherent peer of
a Mac in the trinity. See `docs/ubuntu.md`.

The `sentence_transformers` import is deferred to first use (mirrors MLX invariant
#4): constructing an `STEmbedder` never imports torch, so `embedder_select` can
return one cheaply and the selection path stays light.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Sequence
from typing import Any

from memo.embed_base import EmbedderBase

_DEFAULT_ST_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class STEmbedder(EmbedderBase):
    """In-process CPU embedder. Drop-in for `MLXEmbedder`.

    Args:
        model_path: HF id loadable by `SentenceTransformer` (fp weights, NOT an
            `mlx-community/*` quant — those need MLX).
        revision: Optional exact Hugging Face commit or revision. Pinning keeps
            preloaded/offline images and runtime model selection identical.
        expected_dims: Asserted against the loaded model's reported dimension.
            A mismatch (e.g. config pinned to a 2560-dim profile while this model
            yields 1024) raises early instead of corrupting the vec0 table.
        max_seq_len: Tokens kept per input. 512 mirrors memo's MLX path.
        device: torch device. Defaults to "cpu"; pass "cuda" if a GPU is present.
        batch_size: encode() batch size.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_ST_MODEL,
        revision: str | None = None,
        expected_dims: int = 1024,
        max_seq_len: int = 512,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_path = model_path
        self.revision = revision.strip() if revision else None
        self.expected_dims = expected_dims
        self.max_seq_len = max_seq_len
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._load_lock = threading.Lock()

    # -- internal -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._load_lock.acquire(timeout=120.0):
            raise RuntimeError("STEmbedder model load timed out after 120s")
        try:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - install-time guard
                raise RuntimeError(
                    "STEmbedder needs the optional `sentence-transformers` "
                    "dependency. Install with: pipx install 'mlx-memo[cpu]' "
                    "(or pip install 'mlx-memo[cpu]'). See docs/ubuntu.md."
                ) from exc
            model = SentenceTransformer(
                self.model_path,
                device=self.device,
                revision=self.revision,
            )
            # Not all backbones expose a settable max_seq_length; non-fatal.
            with contextlib.suppress(AttributeError, TypeError):
                model.max_seq_length = self.max_seq_len
            dim = model.get_sentence_embedding_dimension()
            if dim is not None and int(dim) != self.expected_dims:
                raise RuntimeError(
                    f"STEmbedder model {self.model_path!r} produced dim={dim} but "
                    f"config expects {self.expected_dims}. Set MEMO_ST_EMBEDDER_MODEL "
                    f"to a {self.expected_dims}-dim model, or set MEMO_EMBEDDER_DIMS "
                    f"to {dim} (then reindex with 'memo reindex --rebuild')."
                )
            self._model = model
        finally:
            self._load_lock.release()

    # -- public -------------------------------------------------------------

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        """L2-normalised embeddings, one per input (NO query prefix)."""
        if isinstance(inputs, str):
            raise TypeError("embed: pass Sequence[str], not bare str. Wrap as `[text]`.")
        items = list(inputs)
        if not items:
            return []
        self._ensure_loaded()
        vecs = self._model.encode(
            items,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        out = [[float(x) for x in row] for row in vecs]
        if out and len(out[0]) != self.expected_dims:
            raise RuntimeError(
                f"STEmbedder produced dim={len(out[0])} but expected "
                f"{self.expected_dims}. Check MEMO_ST_EMBEDDER_MODEL."
            )
        return out

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with memo's asymmetric retrieval prefix.

        Imports the canonical prefix lazily from `memo.embedder` so the
        query-side instruction stays a single source of truth (MLX invariant #1).
        """
        q = (query or "").strip()
        if not q:
            return [0.0] * self.expected_dims
        from memo.embedder import _QUERY_INSTRUCTION_PREFIX

        return self.embed([_QUERY_INSTRUCTION_PREFIX + q])[0]

    def unload(self) -> None:
        """Drop the model reference. Idempotent."""
        with self._load_lock:
            self._model = None

    @property
    def dims(self) -> int:
        return self.expected_dims

    @property
    def model_name(self) -> str:
        return f"{self.model_path}@{self.revision}" if self.revision else self.model_path

    @property
    def is_warm(self) -> bool:
        return self._model is not None
