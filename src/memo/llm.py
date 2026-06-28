"""MLX chat LLM wrapper — Qwen2.5-Instruct family.

Two-tier setup mirroring obsidian-rag:

- **Helper tier** (`Qwen2.5-3B-Instruct-4bit`): deterministic
  (`temperature=0`, `seed=42`) for tasks that need reproducibility:
  title extraction, tag suggestion, dedup classification, content
  classification. Cheap (~1.9 GB).
- **Chat tier** (`Qwen2.5-7B-Instruct-4bit`): synthesis,
  consolidation, conflict resolution. Higher quality but non-trivial
  VRAM (~4.3 GB). Same options dict (`temperature=0` for memory
  consolidation determinism — we don't want the same input
  consolidating differently across runs).

## Memory management

Both models cached in `_loaded` as a small LRU. Single-tenant
eviction not implemented in v0 — Apple Silicon 36 GB box can fit both
+ embedder + Obsidian + Claude Code without pressure. If it grows to
30B-A3B-class models, port the LRU eviction + idle-unload watchdog
from `rag/llm_backend.py`.

## Threading

Same as the embedder: forward passes are reentrant; the lazy load
under `_load_lock` is the only critical section. Tests can
monkeypatch `MLXChat.chat` directly to skip the load.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

from memo.mlx_gpu import gpu_guard, suppress_swig_deprecation_warnings

_log = logging.getLogger(__name__)
_MAX_LOADED_MODELS = 2


def _prompt_cache_enabled() -> bool:
    """Prefix/KV cache is opt-in (default OFF).

    It only pays off in a long-lived process (the `memo-mcp` HTTP daemon) where
    the byte-identical system-prompt prefix can be reused across requests
    instead of re-prefilled. In one-shot CLI invocations it is inert, so we
    keep it off unless `MEMO_PROMPT_CACHE` is set — the daemon turns it on.
    Output is byte-identical either way (greedy/temp=0).
    """
    from memo.flags import flag_bool

    return flag_bool("MEMO_PROMPT_CACHE")


def _apply_chat_template(tok: Any, **kw: Any) -> Any:
    """Call `tok.apply_chat_template`, dropping `enable_thinking` gracefully
    on tokenizers that don't accept it (non-Qwen3, older mlx-lm)."""
    try:
        return tok.apply_chat_template(**kw)
    except TypeError:
        kw.pop("enable_thinking", None)
        return tok.apply_chat_template(**kw)


class MLXChat:
    """Thin chat wrapper around `mlx_lm.generate()`.

    API:

        chat = MLXChat()
        out = chat.chat(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            messages=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": 512},
        )
        # → {"message": {"content": "..."}}
    """

    def __init__(self) -> None:
        # OrderedDict used as an LRU cache: most-recently-used moves to end.
        self._loaded: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._load_lock = threading.Lock()
        self._last_use: dict[str, float] = {}
        # Prefix prompt-cache (opt-in via MEMO_PROMPT_CACHE). Per-model:
        # (cache_obj, token_ids_held, model_obj_id). model_obj_id auto-
        # invalidates when _ensure_model returns a freshly reloaded model.
        # Guarded by _gen_lock — generations that share the cache must be
        # serialized (the cache mutates in place during decode).
        self._prompt_cache: dict[str, tuple[Any, list[int], int]] = {}
        self._gen_lock = threading.Lock()

    # -- internal -----------------------------------------------------------

    def _prompt_cache_prepare(
        self, model: str, m: Any, prompt_tokens: list[int]
    ) -> tuple[list[int], Any]:
        """Return (feed_tokens, cache) reusing the longest common prefix.

        Call under `_gen_lock`. Reuses the KV state held for `model` up to the
        divergence point with `prompt_tokens`, trims the cache there, and
        returns only the new suffix to prefill. Fresh cache on model reload
        (id mismatch) or a non-trimmable cache. Does NOT persist the entry —
        `_prompt_cache_commit` does that on clean completion.
        """
        from mlx_lm.models.cache import (  # type: ignore[import-not-found]
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        model_key = id(m)
        entry = self._prompt_cache.get(model)
        cache = None
        held: list[int] = []
        if entry is not None:
            cache, held, held_key = entry
            if held_key != model_key or not can_trim_prompt_cache(cache):
                cache, held = None, []
        if cache is None:
            cache = make_prompt_cache(m)
            held = []

        common = 0
        for a, b in zip(held, prompt_tokens, strict=False):
            if a != b:
                break
            common += 1
        if common >= len(prompt_tokens):  # never feed an empty suffix
            common = len(prompt_tokens) - 1

        n_trim = len(held) - common
        if n_trim > 0:
            trimmed = trim_prompt_cache(cache, n_trim)
            if trimmed != n_trim:  # couldn't trim exactly → rebuild
                cache = make_prompt_cache(m)
                common = 0
        return prompt_tokens[common:], cache

    def _prompt_cache_commit(self, model: str, m: Any, full_tokens: list[int], cache: Any) -> None:
        """Persist cache + the token sequence it now holds (prompt + decoded).

        Call under `_gen_lock`, after a clean generation.
        """
        self._prompt_cache[model] = (cache, full_tokens, id(m))

    def _ensure_model(self, model: str) -> tuple[Any, Any]:
        suppress_swig_deprecation_warnings()
        if model in self._loaded:
            # Lock-free hot path. A concurrent eviction (popitem under
            # _load_lock) can drop `model` between the membership test and
            # the reorder, so guard move_to_end and fall through to the
            # locked slow path on KeyError rather than crashing the caller.
            try:
                self._loaded.move_to_end(model)
                return self._loaded[model]
            except KeyError:
                pass
        # Timeout after 30s to avoid indefinite hang if load stalls
        if not self._load_lock.acquire(timeout=30.0):
            raise RuntimeError(f"LLM model load timed out after 30s for {model}")
        try:
            if model in self._loaded:
                self._loaded.move_to_end(model)
                return self._loaded[model]
            from mlx_lm import load as _mlx_load

            # Evict the least-recently-used model before loading a new one
            # to stay within the _MAX_LOADED_MODELS budget. This prevents
            # OOM on long sessions that call helper + chat + reranker in
            # sequence without unloading.
            while len(self._loaded) >= _MAX_LOADED_MODELS:
                evicted_key, _ = self._loaded.popitem(last=False)
                self._last_use.pop(evicted_key, None)
                self._prompt_cache.pop(evicted_key, None)
                _log.debug(
                    "LLM cache evicted: %s (loading %s, now have %d models)",
                    evicted_key,
                    model,
                    len(self._loaded),
                )
                try:
                    import mlx.core as mx

                    # Serialize against concurrent mx.eval in other threads;
                    # clear_cache() racing a live Metal command buffer aborts
                    # the interpreter (memo.mlx_gpu).
                    with gpu_guard():
                        mx.clear_cache()
                except Exception as exc:
                    # Don't swallow: a failed cache flush means the evicted
                    # model's buffers may still be resident, which is exactly
                    # what pushes a 36GB box toward OOM when cycling
                    # helper+chat+reranker. Surface it so it's diagnosable.
                    _log.warning(
                        "LLM cache: mx.clear_cache() failed after evicting %s: %s",
                        evicted_key,
                        exc,
                    )

            loaded = _mlx_load(model)
            self._loaded[model] = (loaded[0], loaded[1])
        finally:
            self._load_lock.release()
        return self._loaded[model]

    # -- public -------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion.

        `options` keys honoured:
        - `temperature` (float, default 0.0)
        - `top_p` (float, default 1.0)
        - `seed` (int, default 42)
        - `num_predict` / `max_tokens` (int, default 512)

        Returns Ollama-shape dict (`{"message": {"content": "..."}}`)
        for trivial drop-in compatibility with existing call sites.
        """
        suppress_swig_deprecation_warnings()
        from mlx_lm import generate as _mlx_generate
        from mlx_lm.sample_utils import make_sampler

        opts = options or {}
        temperature = float(opts.get("temperature", 0.0))
        top_p = float(opts.get("top_p", 1.0))
        # `seed` is not consumed by `mlx_lm.generate` directly in current
        # versions — determinism comes from `temperature=0` (greedy).
        # We keep the kwarg for API symmetry with Ollama callers.
        max_tokens = int(opts.get("num_predict") or opts.get("max_tokens") or 512)
        # `thinking` — pass True to enable chain-of-thought on Qwen3 models.
        # Ignored gracefully on tokenizers that don't support enable_thinking.
        thinking: bool = bool(opts.get("thinking", False))

        m, tok = self._ensure_model(model)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        if _prompt_cache_enabled():
            # Prefix-cache path: drive stream_generate so we can capture the
            # decoded token ids (needed to extend the cache); accumulate text.
            # Byte-identical to the non-cached greedy result.
            from mlx_lm import stream_generate as _mlx_stream

            prompt_tokens = list(
                _apply_chat_template(
                    tok,
                    conversation=messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            )
            from memo.mlx_gpu import _gpu_tl as _gpu_tl_ref

            _gl_timeout = getattr(_gpu_tl_ref, "timeout", None)
            _lock_acquired = self._gen_lock.acquire(
                timeout=_gl_timeout if _gl_timeout is not None else -1
            )
            if not _lock_acquired:
                raise TimeoutError(
                    f"MLX _gen_lock not acquired within {_gl_timeout}s — "
                    "an abandoned thread may still hold it"
                )
            try:
                feed, cache = self._prompt_cache_prepare(model, m, prompt_tokens)
                parts: list[str] = []
                gen_tokens: list[int] = []
                committed = False
                try:
                    with gpu_guard():
                        for resp in _mlx_stream(
                            m,
                            tok,
                            feed,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            prompt_cache=cache,
                        ):
                            if getattr(resp, "finish_reason", None) is None:
                                tk = getattr(resp, "token", None)
                                if tk is not None:
                                    gen_tokens.append(int(tk))
                            parts.append(getattr(resp, "text", "") or "")
                    self._prompt_cache_commit(
                        model,
                        m,
                        prompt_tokens + gen_tokens,
                        cache,
                    )
                    committed = True
                finally:
                    if not committed:
                        self._prompt_cache.pop(model, None)
                    self._last_use[model] = time.time()
            finally:
                self._gen_lock.release()
            return {"message": {"content": ("".join(parts) or "").strip()}}

        prompt = _apply_chat_template(
            tok,
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        with gpu_guard():
            text = _mlx_generate(
                m,
                tok,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        # `mlx_lm.generate` returns either a bare str (newer versions)
        # or an object with `.text`. Normalise both.
        content = text if isinstance(text, str) else getattr(text, "text", str(text))
        self._last_use[model] = time.time()
        return {"message": {"content": (content or "").strip()}}

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion — yields token deltas (str) as they
        are decoded. Same options as `chat()`. Empty deltas are skipped.

        Consumer pattern:
            for delta in chat.chat_stream(model, messages, options):
                process(delta)

        Raises `RuntimeError` if the installed `mlx_lm` lacks
        `stream_generate` (mlx-lm < 0.18).
        """
        suppress_swig_deprecation_warnings()
        try:
            from mlx_lm import stream_generate as _mlx_stream
        except ImportError as exc:
            raise RuntimeError(
                "MLXChat.chat_stream requires mlx-lm with stream_generate "
                "(mlx-lm >= 0.18). Upgrade with: pip install -U mlx-lm"
            ) from exc
        from mlx_lm.sample_utils import make_sampler

        opts = options or {}
        temperature = float(opts.get("temperature", 0.0))
        top_p = float(opts.get("top_p", 1.0))
        max_tokens = int(opts.get("num_predict") or opts.get("max_tokens") or 512)
        # `thinking` — pass False to disable chain-of-thought on Qwen3 models;
        # otherwise <think>…</think> leaks into the streamed deltas uncleaned.
        # Default off to match `chat()` — the stream has no think-tag stripping.
        thinking: bool = bool(opts.get("thinking", False))

        m, tok = self._ensure_model(model)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        if _prompt_cache_enabled():
            # Prefix-cache path: feed only the suffix beyond the cached
            # system-prompt prefix. Output is byte-identical (greedy/temp=0).
            prompt_tokens = list(
                _apply_chat_template(
                    tok,
                    conversation=messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            )
            from memo.mlx_gpu import _gpu_tl as _gpu_tl_ref

            _gl_timeout = getattr(_gpu_tl_ref, "timeout", None)
            _lock_acquired = self._gen_lock.acquire(
                timeout=_gl_timeout if _gl_timeout is not None else -1
            )
            if not _lock_acquired:
                raise TimeoutError(
                    f"MLX _gen_lock not acquired within {_gl_timeout}s — "
                    "an abandoned thread may still hold it"
                )
            try:
                feed, cache = self._prompt_cache_prepare(model, m, prompt_tokens)
                gen_tokens: list[int] = []
                committed = False
                try:
                    with gpu_guard():
                        for resp in _mlx_stream(
                            m,
                            tok,
                            feed,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            prompt_cache=cache,
                        ):
                            if getattr(resp, "finish_reason", None) is None:
                                tk = getattr(resp, "token", None)
                                if tk is not None:
                                    gen_tokens.append(int(tk))
                            delta = getattr(resp, "text", "") or ""
                            if delta:
                                yield delta
                    self._prompt_cache_commit(
                        model,
                        m,
                        prompt_tokens + gen_tokens,
                        cache,
                    )
                    committed = True
                finally:
                    if not committed:
                        # Interrupted/errored: cache offset no longer maps to a
                        # known token sequence — drop it so the next call rebuilds.
                        self._prompt_cache.pop(model, None)
                    self._last_use[model] = time.time()
            finally:
                self._gen_lock.release()
            return

        prompt = _apply_chat_template(
            tok,
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        try:
            with gpu_guard():
                for resp in _mlx_stream(
                    m,
                    tok,
                    prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    delta = getattr(resp, "text", "") or ""
                    if delta:
                        yield delta
        finally:
            self._last_use[model] = time.time()

    def unload(self, model: str | None = None) -> bool:
        """Drop one (or all) loaded models from memory. Returns True if
        anything was actually unloaded.

        Useful from a memory-pressure watchdog or when consolidating a
        large batch and we want to free the small helper before loading
        the chat-tier.
        """
        with self._load_lock:
            if model is None:
                if not self._loaded:
                    return False
                self._loaded.clear()
                self._last_use.clear()
                self._prompt_cache.clear()
            else:
                if model not in self._loaded:
                    return False
                self._loaded.pop(model, None)
                self._last_use.pop(model, None)
                self._prompt_cache.pop(model, None)
            try:
                import mlx.core as mx

                # Serialize the cache flush against concurrent mx.eval in
                # other threads; clear_cache() racing a live Metal command
                # buffer aborts the interpreter (memo.mlx_gpu).
                with gpu_guard():
                    mx.clear_cache()
            except (ImportError, AttributeError):
                # mlx not importable on non-Apple-Silicon → nothing to clear.
                pass
        return True


__all__ = ["MLXChat"]
