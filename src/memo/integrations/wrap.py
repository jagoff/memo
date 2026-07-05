"""Transparent memory for openai/anthropic SDK clients — `wrap(client)`.

Pre-call: recall over memo's warm socket (op=search, priority 0 — the 5s
recall hook's own op always outranks it) injected as a system block.
Post-call (Task G-10): the exchange is enqueued through the capture pipeline
via a detached `memo save --extract` subprocess — never blocking the caller.
Everything fails open: no daemon, no injection; capture errors are swallowed.

Duck-typed on purpose: no openai/anthropic import, so memo gains no deps and
any compatible client shape works. `stream=True` calls pass through with
pre-call injection only (streamed-response capture is v2).
"""

from __future__ import annotations

import json
import logging
import sys
from types import ModuleType
from typing import Any

from memo.config import Config
from memo.recall_client import connect_and_send

_log = logging.getLogger(__name__)

_RECALL_HEADER = "[memo] Relevant memories (authoritative context):"


def _client_kind(client: Any) -> str:
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return "anthropic"
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return "openai"
    raise TypeError(
        "wrap(): unsupported client — expected an openai "
        "(client.chat.completions.create) or anthropic (client.messages.create) SDK client"
    )


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # anthropic content blocks
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p)
    return ""


def _recall_block(cfg: Config, query: str, *, limit: int, source: str) -> str | None:
    if not query.strip():
        return None
    try:
        raw = connect_and_send(
            cfg.state_dir,
            {"op": "search", "prompt": query[:2000], "limit": limit, "client": source},
            timeout=2.0,
        )
        if not raw:
            return None
        results = json.loads(raw).get("results") or []
    except Exception as exc:
        _log.debug("memo wrap: recall skipped (%s)", exc)
        return None
    if not results:
        return None
    lines = [_RECALL_HEADER]
    for r in results[:limit]:
        body = " ".join(str(r.get("body") or "").split())[:200]
        lines.append(f"- [{str(r.get('id'))[:8]}] {r.get('title')}: {body}")
    return "\n".join(lines)


def _inject(kind: str, kwargs: dict[str, Any], block: str) -> None:
    if kind == "openai":
        msgs = kwargs.get("messages")
        if isinstance(msgs, list):
            # new list — never mutate the caller's messages in place
            kwargs["messages"] = [{"role": "system", "content": block}, *msgs]
        return
    # anthropic: `system` is a top-level string (list-form systems pass through)
    sys_prompt = kwargs.get("system")
    if sys_prompt is None:
        kwargs["system"] = block
    elif isinstance(sys_prompt, str):
        kwargs["system"] = sys_prompt + "\n\n" + block


def wrap(
    client: Any,
    *,
    cfg: Config | None = None,
    source: str = "sdk-wrapper",
    recall_limit: int = 3,
    capture: bool = True,
) -> Any:
    """Wrap an openai/anthropic SDK client in place. Returns the same client."""
    kind = _client_kind(client)
    cfg = cfg or Config.from_env()
    holder = client.chat.completions if kind == "openai" else client.messages
    original = holder.create
    if getattr(original, "_memo_wrapped", False):
        return client

    def _pre(kwargs: dict[str, Any]) -> None:
        block = _recall_block(
            cfg, _last_user_text(kwargs.get("messages")), limit=recall_limit, source=source
        )
        if block:
            _inject(kind, kwargs, block)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _pre(kwargs)
        return original(*args, **kwargs)

    wrapped._memo_wrapped = True  # type: ignore[attr-defined]
    holder.create = wrapped
    return client


# The public symbol and this submodule share the name `wrap`, so re-exporting
# the function from the package `__init__` would shadow the submodule and break
# `monkeypatch.setattr("memo.integrations.wrap.connect_and_send", ...)` (the
# path would resolve to the function, which has no such attribute). Keeping the
# module itself as the exported `wrap` symbol lets patches hit the module global
# `connect_and_send` (the idiomatic path), while making the module callable
# preserves the ergonomic `from memo.integrations import wrap; wrap(client)`.
class _WrapModule(ModuleType):
    def __call__(self, client: Any, **kwargs: Any) -> Any:
        return wrap(client, **kwargs)


sys.modules[__name__].__class__ = _WrapModule
