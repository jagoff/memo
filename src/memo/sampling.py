"""Per-request client-sampling state for synthesis tools.

Mirrors the ``_trace.py`` pattern: MCP tools that synthesize set a
``SamplingState`` in a contextvar; ``SamplingChat`` (returned by
``Memory._ensure_chat()`` when ``MEMO_SAMPLING_SYNTH_ENABLED`` is on) reads
the state at every ``.chat()`` call and routes synthesis to the client's
model, falling back to local MLX on any failure — sticky for the rest of
the request so one broken sampler never produces a retry storm.

The router deliberately binds NOTHING at construction time: the facade and
``AdvancedConsolidator`` cache chat instances across requests, so any
request-scoped data must live in the contextvar, not on the instance.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

# (messages, options) -> answer text; raises on any failure.
Sampler = Callable[[list[dict[str, str]], dict[str, Any]], str]


@dataclass
class SamplingState:
    """Mutable per-request sampling budget + attribution."""

    sampler: Sampler
    calls_left: int
    sticky_off: bool = False
    used_client: bool = False
    model_hint: str = "client"

    def usable(self) -> bool:
        return not self.sticky_off and self.calls_left > 0


_state: ContextVar[SamplingState | None] = ContextVar("memo_sampling_state", default=None)


def current_state() -> SamplingState | None:
    return _state.get()


@contextmanager
def sampling_scope(state: SamplingState) -> Iterator[SamplingState]:
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


class _NullChat:
    """Fail-open stand-in when MLX is unavailable: ``score_grounding`` catches
    the raise and returns None (never punish a claim because the judge is
    unavailable)."""

    def chat(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise RuntimeError("no local LLM available for grounding")


class SamplingChat:
    """Chat router: client sampler when a usable state is active, else MLX.

    Same call surface as ``MLXChat`` (``chat`` / ``chat_stream``, Ollama-shape
    return). Safe to cache anywhere — all request-scoped data lives in the
    contextvar.
    """

    def __init__(self, mlx_factory: Callable[[], Any]) -> None:
        self._mlx_factory = mlx_factory
        self._mlx: Any | None = None
        self._mlx_lock = threading.Lock()

    def mlx_fallback(self) -> Any:
        if self._mlx is None:
            with self._mlx_lock:
                if self._mlx is None:
                    self._mlx = self._mlx_factory()
        return self._mlx

    def grounding_backend(self) -> Any:
        try:
            return self.mlx_fallback()
        except Exception:
            return _NullChat()

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = current_state()
        if state is not None and state.usable():
            state.calls_left -= 1
            try:
                text = state.sampler(messages, options or {})
                state.used_client = True
                return {"message": {"content": text}}
            except Exception:
                _log.debug("client sampling failed; sticky fallback to MLX", exc_info=True)
                state.sticky_off = True
        return self.mlx_fallback().chat(model, messages, options)

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        # Streaming over MCP sampling is not supported; always local.
        return self.mlx_fallback().chat_stream(model, messages, options)  # type: ignore[no-any-return]


def grounding_chat(chat: Any) -> Any:
    """Unwrap a ``SamplingChat`` to its local backend for grounding judges.

    Grounding must never burn a client-sampling round trip (spec). Plain
    backends pass through untouched.
    """
    fn = getattr(chat, "grounding_backend", None)
    return fn() if callable(fn) else chat


__all__ = [
    "Sampler",
    "SamplingChat",
    "SamplingState",
    "current_state",
    "grounding_chat",
    "sampling_scope",
]
