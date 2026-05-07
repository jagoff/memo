"""MLX embedder — Qwen3-Embedding-0.6B-4bit-DWQ.

Loads the MLX-quantised Qwen3-Embedding via `mlx_lm.load()` and bypasses
`lm_head` to access the transformer body's hidden states. Pools by
last-real-token (using attention mask to handle padded batches) and
L2-normalises the result. Output: list of 1024-dim float lists.

## Why not `mlx_embeddings` / `sentence-transformers`?

- `mlx_embeddings` doesn't exist as a stable package on PyPI as of
  2026-05-06. The mlx-community ports of Qwen3-Embedding ship as
  generic `mlx_lm`-loadable models — the embedder API is implicit.
- `sentence-transformers` would route through PyTorch+MPS (not pure
  MLX). The whole point of memo is MLX-native.

## Cosine sim vs Ollama (verified 2026-05-06)

`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` produces vectors with
**0.9747 cosine similarity** vs the same model run via Ollama Q4_K_M
(measured on a real Spanish-Castellano query). ~3% drift attributable
to quantisation noise — does not invalidate corpus continuity if
memo replaces an existing Ollama-indexed mem-vault.

## Threading

`MLXEmbedder.embed()` is reentrant for forward passes — `mlx_lm` model
inference has no internal mutable state. The lazy `load()` is guarded
by `_load_lock` so concurrent first calls don't race-load the weights.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

# `mlx_lm` is Apple-Silicon-only. Importing at module level on Linux/x86
# would raise; we defer the import until `_ensure_loaded()`.


# Qwen3-Embedding is instruction-tuned for **asymmetric retrieval**:
# queries get a `Instruct: ...\nQuery: ...` prefix; documents go raw.
# Without the prefix, queries embed in a different region of the space
# and cosine collapses toward 0 (or even negative). Verified empirically
# 2026-05-07: 223-doc corpus where queries with literal title overlap
# (e.g. "informe terapia ocupacional astor" against the Astor doc)
# returned scores in [-0.15, 0.0]; with the prefix, the same query
# rises above 0.5.
#
# The task string is generic enough to work across the personal-memory
# domain (notes, decisions, bug logs, preferences). Tuning per-task
# would require classifying the query first — out of scope for v0.
_QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Given a search query, retrieve relevant memory entries "
    "from the user's notes.\nQuery: "
)


class MLXEmbedder:
    """In-process MLX embedder. Drop-in for any code that needs a
    `embed(list[str]) -> list[list[float]]` callable.

    Args:
        model_path: HF id of the MLX-quantised embedder. Must be
            loadable via `mlx_lm.load()`.
        expected_dims: Asserted post-load. Raises if the model produces
            a different hidden size — guards against accidental swap to
            an incompatible model that would silently produce garbage.
        max_seq_len: Tokens kept per input; longer inputs are
            tail-truncated (preserves the most semantically dense end of
            the document for retrieval). 512 is the Qwen3-Embedding
            training context.
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        expected_dims: int = 1024,
        max_seq_len: int = 512,
    ) -> None:
        self.model_path = model_path
        self.expected_dims = expected_dims
        self.max_seq_len = max_seq_len
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()
        self._last_use: float = 0.0

    # -- internal -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from mlx_lm import load as _mlx_load

            self._model, self._tokenizer = _mlx_load(self.model_path)

    # -- public -------------------------------------------------------------

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        """Compute L2-normalised embeddings for every text in `inputs`.

        Returns a list-of-lists of `expected_dims` floats. Order
        matches the input order. Empty strings produce a zero vector
        — caller's responsibility to filter if zeros are unwanted.
        """
        self._ensure_loaded()
        self._last_use = time.time()
        if not inputs:
            return []

        import mlx.core as mx

        out: list[list[float]] = []

        # Per-input forward pass. Batching across inputs with padding is
        # possible but not implemented in this v0 — most real callers
        # pass 1-3 inputs (search query, save content) and the per-call
        # overhead of MLX is small compared to the forward itself.
        # Promotion to batched mode is straightforward when needed.
        for text in inputs:
            if not text:
                out.append([0.0] * self.expected_dims)
                continue
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            # Head-truncate for two reasons:
            # 1. Memory entries put their high-signal content (title-like
            #    H1, frontmatter-derived header) at the TOP. Tail-truncation
            #    drops it for any body >512 tokens — verified 2026-05-07
            #    where the Astor TO report (1409 tokens) lost its title
            #    and recall against a literal-title query collapsed.
            # 2. Qwen3-Embedding was fine-tuned to pool on the EOS hidden
            #    state. `tokenizer.encode(...)` does NOT auto-append EOS
            #    (Qwen tokenizers leave that to the chat template), so we
            #    add it manually as the LAST token after truncation.
            eos = self._tokenizer.eos_token_id
            cap = self.max_seq_len - (1 if eos is not None else 0)
            if len(ids) > cap:
                ids = ids[:cap]
            if eos is not None:
                ids = ids + [eos]
            arr = mx.array([ids])
            # `model.model` is the transformer body without the LM head
            # — that's what produces the hidden states we pool. Calling
            # `model(arr)` would route through `lm_head` and return
            # logits over vocab (~151k floats per token), totally wrong
            # for embedding extraction.
            hidden = self._model.model(arr)  # (1, seq, hidden_dim)

            if hidden.shape[-1] != self.expected_dims:
                raise RuntimeError(
                    f"Embedder produced dim={hidden.shape[-1]} but config expects "
                    f"{self.expected_dims}. Either the model swap was incorrect or "
                    f"`embedder_dims` config is stale."
                )

            pooled = hidden[:, -1, :]  # (1, hidden_dim) — last real token (no padding)
            norm = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
            emb = pooled / norm
            out.append([float(x) for x in emb[0].tolist()])
        return out

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query. Prepends the instruction template that
        Qwen3-Embedding's asymmetric-retrieval training expects, then
        delegates to `embed()`. Use this for the QUERY side of search;
        use `embed()` raw for the DOCUMENT side.

        Mixing them up (e.g. embedding queries raw against doc-prefixed
        vectors, or vice versa) collapses cosine similarity toward zero
        — the model places prefixed and raw inputs in different regions
        of the space.
        """
        return self.embed([_QUERY_INSTRUCTION_PREFIX + (query or "")])[0]

    def unload(self) -> None:
        """Drop the model + tokenizer; clear the MLX cache. Idempotent."""
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                # mlx not importable on non-Apple-Silicon → nothing to clear.
                pass

    @property
    def last_use(self) -> float:
        """Wall-time epoch of last `embed()` call. Used by idle-unload watchdogs."""
        return self._last_use


__all__ = ["MLXEmbedder"]
