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

import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

_MAX_LOADED_MODELS = 2


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

    # -- internal -----------------------------------------------------------

    def _ensure_model(self, model: str) -> tuple[Any, Any]:
        if model in self._loaded:
            return self._loaded[model]
        with self._load_lock:
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
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass

            loaded = _mlx_load(model)
            self._loaded[model] = (loaded[0], loaded[1])
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
        from mlx_lm import generate as _mlx_generate
        from mlx_lm.sample_utils import make_sampler

        opts = options or {}
        temperature = float(opts.get("temperature", 0.0))
        top_p = float(opts.get("top_p", 1.0))
        # `seed` is not consumed by `mlx_lm.generate` directly in current
        # versions — determinism comes from `temperature=0` (greedy).
        # We keep the kwarg for API symmetry with Ollama callers.
        max_tokens = int(opts.get("num_predict") or opts.get("max_tokens") or 512)

        m, tok = self._ensure_model(model)
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        sampler = make_sampler(temp=temperature, top_p=top_p)
        text = _mlx_generate(
            m, tok, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False,
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

        m, tok = self._ensure_model(model)
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        sampler = make_sampler(temp=temperature, top_p=top_p)
        try:
            for resp in _mlx_stream(
                m, tok, prompt, max_tokens=max_tokens, sampler=sampler,
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
            else:
                if model not in self._loaded:
                    return False
                self._loaded.pop(model, None)
                self._last_use.pop(model, None)
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
        return True


__all__ = ["MLXChat"]
