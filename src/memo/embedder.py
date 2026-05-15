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

        Batched: tokenizes all inputs, right-pads to the longest in the
        batch, runs ONE forward pass through the transformer, and pools
        each row at its individual EOS position. Causal attention means
        the EOS token only sees real content (everything to its right
        is padding it ignores), so padding doesn't pollute the pooled
        embedding. Empirically ~3-5x faster than the per-input loop on
        batches of 50+ entries (e.g. `memo reindex` against a corpus).

        Returns a list-of-lists of `expected_dims` floats, order-aligned
        with `inputs`. Empty strings produce a zero vector — caller's
        responsibility to filter if zeros are unwanted.
        """
        self._ensure_loaded()
        self._last_use = time.time()
        if not inputs:
            return []

        import mlx.core as mx

        eos = self._tokenizer.eos_token_id
        # Cap leaves room for the EOS slot we append.
        cap = self.max_seq_len - (1 if eos is not None else 0)

        # Tokenize + truncate + append EOS for each input. Track real
        # length (= EOS position + 1) for per-row pooling later.
        tok_ids: list[list[int]] = []
        eos_idx: list[int] = []  # position of the EOS in each row (the pool target)
        skip_mask: list[bool] = []  # True for empty inputs (zero-vector shortcut)

        for text in inputs:
            if not text:
                tok_ids.append([])
                eos_idx.append(-1)
                skip_mask.append(True)
                continue
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            # Head-truncate (preserves title-like header) + append EOS
            # (Qwen3-Embedding pools on EOS hidden state).
            if len(ids) > cap:
                ids = ids[:cap]
            if eos is not None:
                ids = [*ids, eos]
            tok_ids.append(ids)
            eos_idx.append(len(ids) - 1)
            skip_mask.append(False)

        # If everything is empty, short-circuit.
        if all(skip_mask):
            return [[0.0] * self.expected_dims for _ in inputs]

        # Right-pad to the max real length so all sequences fit a single
        # tensor. The pad value can be anything — under causal attention
        # padding positions never feed into the EOS pool. We use eos_id
        # (or 0 as a fallback) just for cleanliness.
        pad_id = eos if eos is not None else 0
        max_len = max(len(t) for t, sk in zip(tok_ids, skip_mask, strict=True) if not sk)
        padded: list[list[int]] = []
        # For empty rows we still emit a placeholder padded vector of
        # length 1; we'll mask those out post-forward.
        for ids, sk in zip(tok_ids, skip_mask, strict=True):
            if sk:
                padded.append([pad_id] * max_len)
            else:
                padded.append(ids + [pad_id] * (max_len - len(ids)))

        arr = mx.array(padded)
        # `model.model` is the transformer body without the LM head —
        # that's what produces the hidden states we pool. Calling
        # `model(arr)` would route through `lm_head` and return logits
        # over vocab (~151k floats per token), totally wrong here.
        hidden = self._model.model(arr)  # (B, T, H)

        if hidden.shape[-1] != self.expected_dims:
            raise RuntimeError(
                f"Embedder produced dim={hidden.shape[-1]} but config expects "
                f"{self.expected_dims}. Either the model swap was incorrect or "
                f"`embedder_dims` config is stale."
            )

        # Per-row pool: gather hidden[i, eos_idx[i], :]. Easiest in MLX
        # without fancy gather is a small Python loop — the heavy work
        # (the forward pass) is already done in one batched call.
        out: list[list[float]] = []
        for i, sk in enumerate(skip_mask):
            if sk:
                out.append([0.0] * self.expected_dims)
                continue
            row = hidden[i : i + 1, eos_idx[i], :]  # (1, H)
            norm = mx.sqrt(mx.sum(row * row, axis=-1, keepdims=True))
            emb = row / norm
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


def assert_valid_embedding(
    embedding: list[float], expected_dims: int, *, context: str = "",
) -> None:
    """Validate one vector's shape + norm. Raises with a context-aware
    message on violation. Use at every embed boundary that writes to
    the index — silent malformed vectors corrupt retrieval until the
    next reindex sweeps them out.

    Past silent-failure mode (v0.3.0): string-as-Sequence-of-chars
    cascade returned variable-dim outputs (135, 512, 2465...). The
    Metal kernel didn't error; the embedder yielded whatever shape
    the partial recovery produced. Catching this at the boundary
    means the next regression of the same shape surfaces immediately
    instead of poisoning N records before anyone notices.
    """
    if len(embedding) != expected_dims:
        raise ValueError(
            f"embedding dim mismatch: got {len(embedding)}, "
            f"want {expected_dims}{(' [' + context + ']') if context else ''}"
        )
    norm = sum(x * x for x in embedding) ** 0.5
    if not (0.5 < norm < 1.5):
        raise ValueError(
            f"embedding norm out of L2-normalised range: {norm:.4f} "
            f"(expected ≈ 1.0)"
            f"{(' [' + context + ']') if context else ''}"
        )


__all__ = ["MLXEmbedder", "assert_valid_embedding"]
