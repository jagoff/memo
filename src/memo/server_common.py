from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from memo.memory import Memory

_log = logging.getLogger("memo.server")


async def run_synth[T](memory: Memory, ctx: Any, fn: Callable[[], T]) -> tuple[T, str]:
    """Run a sync synthesis closure with per-request client sampling.

    Returns ``(result, synthesizer_label)``. Inside MCP scope with
    MEMO_SAMPLING_SYNTH_ENABLED on and a sampling-capable client, synthesis
    routes to the client's model; otherwise MLX. Never raises because of
    sampling — ``fn``'s own errors pass through untouched.
    """
    import anyio

    from memo.sampling import sampling_scope, state_from_ctx

    state = state_from_ctx(ctx)

    def _run() -> T:
        if state is None:
            return fn()
        with sampling_scope(state):
            return fn()

    result = await anyio.to_thread.run_sync(_run)
    if state is not None and state.used_client:
        label = f"client:{state.model_hint}"
    else:
        # Attribution is cosmetic — never let a label lookup break the tool.
        try:
            label = f"mlx:{memory.cfg.llm_model}"
        except Exception:
            label = "mlx:unknown"
    return result, label


def now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _mcp_client_name() -> str | None:
    """Best-effort name of the connected MCP client, from the initialize
    handshake's ``clientInfo.name`` (e.g. ``devin`` / ``opencode`` / ``devin-desktop``).
    Lets every MCP consult self-attribute even when the caller passed no
    ``source=`` and set no ``MEMO_SOURCE`` — so agent-class consumers stop
    showing up as the anonymous ``mcp:unknown``. Fully guarded: returns None off
    a request or on any FastMCP/MCP API drift."""
    try:
        from fastmcp.server.dependencies import get_context

        ctx = get_context()
        name = ctx.session.client_params.clientInfo.name  # type: ignore[union-attr]
        return (name or "").strip().lower() or None
    except Exception:
        return None


def log_consult(
    memory: Memory,
    *,
    tool: str,
    query: str,
    hits: list[dict[str, Any]],
    t0_ms: int,
    source: str = "",
) -> None:
    """Record an MCP consult into the shared recall ring buffer.

    ``source`` identifies the calling layer. Attribution precedence:
    explicit ``source=`` → ``MEMO_SOURCE`` env (mirrors the CLI's
    ``log_cli_consult``) → the MCP client's declared ``clientInfo.name``. The
    last tier means agent-class clients (devin / opencode / devin-desktop …) are
    attributed automatically from the handshake instead of showing up as the
    anonymous ``mcp:unknown`` consumer — no per-call args or env needed.
    """
    try:
        from memo.dashboard import append_recall_log
        from memo.flags import flag_str

        src = (source or flag_str("MEMO_SOURCE") or "").strip().lower() or _mcp_client_name()
        append_recall_log(
            memory.cfg.state_dir,
            prompt=query or "",
            hits=hits or [],
            via=f"mcp:{tool}",
            source=src,
            latency_ms=now_ms() - t0_ms,
        )
    except Exception as exc:
        _log.warning("consult recall-log write failed for %s: %s", tool, exc)
