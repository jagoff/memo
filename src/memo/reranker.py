"""MLX cross-encoder reranker — Qwen3-Reranker-0.6B-MLX-8Bit (default).

Reranks candidates from `Memory.search()` by feeding each `(query, doc)`
pair through the Qwen3-Reranker model and scoring it as
`P(yes | query, doc)` over the model's "yes"/"no" output logits.

## Why a reranker on top of vec + bm25?

Bi-encoders (vec) and BM25 score `query` and `doc` independently and
fuse late. Cross-encoders read the pair jointly and capture
interactions (subtle phrasing, negation, partial matches) the
bi-encoder misses. Empirically lifts MRR by 30-60% on RAG retrieval
benchmarks at the cost of one extra forward pass per candidate —
acceptable in memo where K is small (≤50).

## Model choice

Qwen3-Reranker is the same family as memo's existing embedder
(Qwen3-Embedding) and chat (Qwen2.5-Instruct), so adding it imports
zero new tokenizer/architecture surface. License Apache 2.0 — safe
to ship as default in a public MIT package.

Sizes available (HF MLX community):

- `mku64/Qwen3-Reranker-0.6B-mlx-8Bit` — default. ~700 MB, ~20ms/pair on M3 warm.
- `kerncore/Qwen3-Reranker-0.6B-MLX-4bit` — ~400 MB, slightly less recall.
- `vserifsaglam/Qwen3-Reranker-4B-4bit-MLX` — ~2.5 GB, +10-15% MRR estimated.

Override via `MEMO_RERANKER_MODEL` env var.

## Inference contract

The model is a generative LLM trained to answer "yes" / "no" to the
question "does this document match the query?". We don't run
generation — we run a single forward pass and extract the next-token
distribution at the last position, then take
`softmax({logit_yes, logit_no})[1]` as the relevance score.

See the original Qwen3-Reranker model card for the prompt format.
We mirror the official prefix/suffix bracketing so quantised MLX
weights agree with the BF16 reference scoring within rounding noise.

## Threading

Same shape as `MLXEmbedder` / `MLXChat`: forward passes are reentrant;
the lazy `load()` is guarded by a single lock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from memo.mlx_gpu import gpu_guard, suppress_swig_deprecation_warnings

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from memo.memory import MemoryRecord


# Same chat-template wrapper Qwen3-Reranker is fine-tuned with. The
# `<think></think>` shell is empty because we pre-bake the assistant
# prefix and read the next-token logits at "yes"/"no" — we never let
# the model actually generate a thinking trace.
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Generic instruction that works across memo's domain (notes,
# decisions, bug logs, preferences). Kept short so it doesn't dominate
# the input context for short notes. Mirror of the embedder's
# `_QUERY_INSTRUCTION_PREFIX` design.
_DEFAULT_TASK = (
    "Given a search query, retrieve relevant memory entries from the user's personal notes."
)


class MLXReranker:
    """In-process MLX cross-encoder reranker.

    Args:
        model_path: HF id of the MLX-quantised reranker.
        max_seq_len: Tokens kept per `(query, doc)` pair. Tail-truncated
            on the doc side because the prompt structure is
            `prefix + query + doc + suffix`; truncating the head would
            chop the instruction. 4096 is well within Qwen3's 32k
            window and keeps even very long memories intact.
        task: Optional instruction string. Defaults to a generic
            personal-memory phrasing.
    """

    def __init__(
        self,
        model_path: str = "mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
        revision: str | None = None,
        max_seq_len: int = 4096,
        task: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.max_seq_len = max_seq_len
        self.task = task or _DEFAULT_TASK

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._load_lock = threading.Lock()
        self._loaded_at: float | None = None

    def _resolve_model_path(self) -> str:
        if not self.revision:
            return self.model_path
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.model_path, revision=self.revision)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Timeout after 30s to avoid indefinite hang if load stalls
        if not self._load_lock.acquire(timeout=30.0):
            raise RuntimeError("Reranker model load timed out after 30s")
        try:
            if self._model is not None:
                return
            suppress_swig_deprecation_warnings()
            from mlx_lm import load  # deferred — Apple-Silicon-only import

            t0 = time.time()
            load_path = self._resolve_model_path()
            if self.revision:
                _log.info(
                    "Loading reranker with pinned revision: %s (path=%s)",
                    self.revision,
                    load_path,
                )
            loaded = load(load_path)
            self._model = loaded[0]
            self._tokenizer = loaded[1]
            self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._no_id = self._tokenizer.convert_tokens_to_ids("no")
            if self._yes_id is None or self._no_id is None:
                raise RuntimeError(
                    f"Reranker tokenizer for {self.model_path!r} does not "
                    "expose 'yes'/'no' tokens — model probably isn't a "
                    "Qwen3-Reranker variant."
                )
            self._loaded_at = time.time() - t0
            _log.debug("Reranker loaded in %.2fs", self._loaded_at)
        finally:
            self._load_lock.release()

    def _format(self, query: str, doc: str) -> str:
        return f"{_PREFIX}<Instruct>: {self.task}\n<Query>: {query}\n<Document>: {doc}{_SUFFIX}"

    def score(self, query: str, doc: str) -> float:
        """Score a single `(query, doc)` pair → `P(yes)` in [0, 1]."""
        self._ensure_loaded()
        import mlx.core as mx

        model = self._model
        tokenizer = self._tokenizer
        yes_id = self._yes_id
        no_id = self._no_id
        if model is None or tokenizer is None or yes_id is None or no_id is None:
            raise RuntimeError("Reranker failed to load.")

        text = self._format(query, doc)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Tail-truncate if a long doc blows past the window. Preserves
        # the prefix + query + early doc which carries the most signal.
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        # Serialize the GPU forward+materialization against all other MLX
        # work in this process; the Metal default stream is global and
        # concurrent eval aborts the interpreter (see memo.mlx_gpu).
        with gpu_guard():
            arr = mx.array([ids])
            # Run the transformer BODY (no LM head), then project ONLY the
            # last token's hidden state through the head. Scoring reads a
            # single position, so the usual full-logits path wastes the head
            # matmul (hidden × 151k vocab) over every OTHER position — the
            # dominant cost once the body is done (~9% of the 4B forward,
            # ~20% of the 0.6B). Slicing before the head is mathematically
            # identical (a position-wise linear projection) and is bit-exact
            # on fp/8-bit weights; on a 4-bit model the quantized head matmul
            # is shape-sensitive, so P(yes) can shift at the ~1/64
            # quantization step — ranking-safe (the retrieved set is
            # unchanged; verified test_head_slice_matches_full_forward_ranking).
            hidden = model.model(arr)  # (1, T, H) — body only
            last = self._apply_lm_head(model, hidden[:, -1, :])  # (1, V)
            y = float(last[0, yes_id])
            n = float(last[0, no_id])
        # Softmax over the (no, yes) pair only — we don't care about
        # the rest of the vocab. Numerical-stable form.
        m = max(y, n)
        e_y = math.exp(y - m)
        e_n = math.exp(n - m)
        return e_y / (e_y + e_n)

    @staticmethod
    def _apply_lm_head(model: Any, hidden: Any) -> Any:
        """Project hidden states ``(…, H)`` → logits ``(…, vocab)`` via the
        model's LM head.

        Qwen3-Reranker ties its input embeddings to the output projection
        (`tie_word_embeddings=True`), so there is no separate ``lm_head``
        module — the head is ``model.embed_tokens.as_linear`` (exactly what
        the model's own ``__call__`` applies after the body). Untied variants
        expose ``lm_head`` directly. Raises (never silently mis-scores) if
        neither is present.
        """
        lm_head = getattr(model, "lm_head", None)
        if callable(lm_head):
            return lm_head(hidden)
        embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
        as_linear = getattr(embed_tokens, "as_linear", None)
        if callable(as_linear):
            return as_linear(hidden)
        raise RuntimeError(
            "reranker model exposes neither `lm_head` nor tied "
            "`embed_tokens.as_linear`; cannot project hidden states to logits"
        )

    def rerank(
        self,
        query: str,
        hits: list[MemoryRecord],
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[MemoryRecord]:
        """Score every hit against `query`, return them re-ordered by
        descending `P(yes)`. Each returned `MemoryRecord` has its
        `score` overwritten with the rerank probability so downstream
        callers (CLI display, MCP responses) see the new ranking.

        Args:
            query: User query.
            hits: Output of `Memory.search()` (or any list of records
                with `title` + `body` populated).
            top_n: If given, truncate to top-N after rerank.
            body_chars: Body truncation per candidate before scoring.
                The reranker is per-pair sequential, so longer docs
                blow up latency linearly. 1200 chars (~300 tokens)
                covers the title + lead paragraphs which carry the
                bulk of retrieval signal; the tail is rarely
                discriminative for ranking decisions.

        Returns: new list, never mutates input.
        """
        if not hits:
            return []
        self._ensure_loaded()

        scored: list[tuple[float, MemoryRecord]] = []
        for h in hits:
            # Compose like the embedder: title carries dense signal,
            # body carries detail. Reranker reads the whole thing.
            body = (h.body or "")[:body_chars]
            doc = f"{h.title}\n\n{body}" if body else h.title
            p = self.score(query, doc)
            scored.append((p, h))

        scored.sort(key=lambda t: t[0], reverse=True)
        out = [replace(h, score=p) for p, h in scored]
        if top_n is not None:
            out = out[:top_n]
        return out

    def unload(self) -> None:
        """Release the loaded model and clear GPU caches.

        Idempotent — safe to call even when not loaded. Mirrors the
        pattern from ``MLXEmbedder.unload()`` so the ``memo-mcp``
        daemon can cycle models without restarting.
        """
        from contextlib import suppress

        # Take _load_lock before zeroing any field so we don't race
        # _ensure_loaded() into a half-loaded state (spurious RuntimeError).
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._yes_id = None
            self._no_id = None
            self._loaded_at = None
            with suppress(ImportError, AttributeError):
                import mlx.core as mx

                # Serialize the cache flush against concurrent mx.eval /
                # float(scalar) in other threads; clear_cache() racing a live
                # Metal command buffer aborts the interpreter (memo.mlx_gpu).
                with gpu_guard():
                    mx.clear_cache()

    def warmup(self) -> None:
        """Force-load + run one tiny scoring pass so the first real
        query in a session doesn't pay the JIT/graph-construction cost.
        Call from `memo prewarm` (SessionStart hook).
        """
        self._ensure_loaded()
        # Tiny dummy pair — just to populate the MLX graph cache.
        self.score("test query", "test document")
