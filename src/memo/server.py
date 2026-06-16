"""MCP server — `memo-mcp` entry point.

Exposes the `Memory` API as MCP tools so any MCP-aware client (Claude
Code, Devin, Claude Desktop, etc) can save / search / list / get /
update / delete entries from natural language.

Tools are deliberately small and verb-shaped so the LLM picks them up
without prompt engineering. Each tool returns plain JSON-serialisable
dicts (no SimpleNamespaces, no dataclasses) — the MCP transport
serialises everything anyway, but a dict surfaces field names in the
tool result message which helps the LLM decide what to do next.

Run via the installed entry point:

    memo-mcp

Or programmatically:

    from memo.server import build_server
    server = build_server()
    server.run()
"""

from __future__ import annotations

import os
from typing import Any, cast

from fastmcp import FastMCP

from memo import server_analytics as _srv_analytics
from memo import server_asof as _srv_asof
from memo import server_backup as _srv_backup
from memo import server_cache as _srv_cache
from memo import server_collaborative as _srv_collaborative
from memo import server_consolidate as _srv_consolidate
from memo import server_contextual as _srv_contextual
from memo import server_contradict as _srv_contradict
from memo import server_core_history as _srv_core_history
from memo import server_core_records as _srv_core_records
from memo import server_core_search as _srv_core_search
from memo import server_encrypt as _srv_encrypt
from memo import server_entities as _srv_entities
from memo import server_feedback as _srv_feedback
from memo import server_graph as _srv_graph
from memo import server_health as _srv_health
from memo import server_import_export as _srv_import_export
from memo import server_links as _srv_links
from memo import server_multimodal as _srv_multimodal
from memo import server_query as _srv_query
from memo import server_reflect as _srv_reflect
from memo import server_repo as _srv_repo
from memo import server_resources as _srv_resources
from memo import server_share as _srv_share
from memo import server_sync as _srv_sync
from memo import server_synthesis as _srv_synthesis
from memo import server_temporal as _srv_temporal
from memo import server_version as _srv_version
from memo._trace import TRACE_HEADER, trace_scope
from memo.config import Config
from memo.memory import Memory


def _make_trace_middleware() -> Any:
    """Bridge the synapse trace header into the shared trace contextvar.

    Synapse forwards its ``synapse://trace/<id>`` two ways: the
    ``SYNAPSE_TRACE_ID`` env var (subprocess path) and the
    ``x-synapse-trace-id`` HTTP header (warm-daemon path). The env var is read
    in `write_ops`; this middleware covers the header path so a warm-daemon
    write stamps the same trace id and the ledger trails stitch into one span.

    Returns ``None`` when FastMCP middleware isn't available so `build_server`
    can skip wiring it without failing.
    """
    try:
        from fastmcp.server.dependencies import get_http_headers
        from fastmcp.server.middleware import Middleware
    except ImportError:
        return None

    class _TraceMiddleware(Middleware):  # type: ignore[misc]
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            trace_id = ""
            try:
                headers = get_http_headers() or {}
                trace_id = (headers.get(TRACE_HEADER) or "").strip()
            except Exception:
                trace_id = ""
            if not trace_id:
                return await call_next(context)
            with trace_scope(trace_id):
                return await call_next(context)

    return _TraceMiddleware()


# Server-level memory-first directive. FastMCP surfaces `instructions` to the
# MCP client as a server system prompt, so this travels with the connection and
# reaches EVERY client (Codex, Gemini, Cursor, Windsurf, opencode, Claude
# Desktop) without a per-client instruction file. Strong-but-bounded: consult
# first for anything prior work might cover, skip for pure coding/math/general
# knowledge so it doesn't add noise where memo can't help.
_SERVER_INSTRUCTIONS = (
    "memo is the user's durable semantic memory — decisions, facts, preferences, "
    "and past work, indexed from their Obsidian vault. Before answering anything "
    "that earlier work might already cover (\"what did we decide\", \"where did I "
    "leave X\", \"what do you know about Y\", status/context/preference questions), "
    "call `memory_unified_briefing` FIRST (or `memory_search` / `memory_ask`) and "
    "build on what it returns — treat surfaced memorias as established facts and "
    "contradict one only explicitly. Pass `source=\"<your client>\"` on the read "
    "tools so the consult is attributed in memo's log. Skip the lookup only for "
    "pure coding, math, or general knowledge clearly outside the user's stored "
    "memory. Persist durable outcomes with `memory_save` so the next session "
    "inherits them."
)


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the MCP server. Accepts an explicit `Memory` for tests.

    The default constructs from `Config.from_env()` — production runs
    pick up `MEMO_*` env vars set by the calling shell or by Claude
    Code's `claude mcp add` invocation.
    """
    if memory is None:
        memory = Memory(Config.from_env())

    server = FastMCP("memo", instructions=_SERVER_INSTRUCTIONS)

    # Stitch the synapse trace header into the shared trace contextvar so
    # warm-daemon writes carry the same trace id as the subprocess path.
    _trace_mw = _make_trace_middleware()
    if _trace_mw is not None:
        server.add_middleware(_trace_mw)

    # Stable and advanced domain tool modules register their @server.tool()
    # closures here. Presence on the MCP surface does not by itself mean a
    # feature is part of memo's stable core contract; see experimental_index.md.
    # Skip when MEMO_MCP_SLIM=1 — reduces ~116 tools to ~26 core inline tools
    # for local/constrained LLMs where tool-definition tokens are expensive.
    from memo.flags import flag_bool as _flag_bool
    if not _flag_bool("MEMO_MCP_SLIM"):
        _srv_repo.register(server, memory)
        _srv_entities.register(server, memory)
        _srv_temporal.register(server, memory)
        _srv_contradict.register(server, memory)
        _srv_consolidate.register(server, memory)
        _srv_synthesis.register(server, memory)
        _srv_reflect.register(server, memory)
        _srv_graph.register(server, memory)
        _srv_health.register(server, memory)
        _srv_contextual.register(server, memory)
        _srv_links.register(server, memory)
        _srv_version.register(server, memory)
        _srv_query.register(server, memory)
        _srv_backup.register(server, memory)
        _srv_sync.register(server, memory)
        _srv_cache.register(server, memory)
        _srv_encrypt.register(server, memory)
        _srv_share.register(server, memory)
        _srv_analytics.register(server, memory)
        _srv_import_export.register(server, memory)
        _srv_feedback.register(server, memory)
        _srv_multimodal.register(server, memory)
        _srv_collaborative.register(server, memory)
        _srv_asof.register(server, memory)
        # ToolSpec-registry tools (new pattern — see mcp_tools.py)
        from memo.mcp_tools import register_all as _register_mcp_tools

        _register_mcp_tools(server, memory)
    _srv_core_records.register(server, memory)
    _srv_core_search.register(server, memory)
    _srv_core_history.register(server, memory)
    _srv_resources.register(server, memory)

    @server.custom_route("/chat/stream", methods=["POST"])
    async def chat_stream_route(request):  # type: ignore[no-untyped-def]
        """Real token streaming for chat synthesis (SSE).

        MCP `tools/call` can only return a single result, so the warm-daemon
        chat path otherwise has to buffer the whole answer before the client
        sees anything. This route exposes `Memory.chat_ask_stream` directly:
        emits one SSE `data:` line per event (context → token* → done), so the
        first token reaches the caller right after prefill instead of after the
        full decode. Output is identical to the non-streaming tool; only the
        delivery is incremental. Consumed by Synapse's MemoBackend.
        """
        import json as _json

        from starlette.responses import StreamingResponse

        try:
            body = await request.json()
        except Exception:
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        question = str(body.get("question") or "")
        if not question.strip():
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "empty question"}, status_code=400)
        k = int(body.get("k") or 7)
        type_ = body.get("type") or None
        history = body.get("history") or None
        context = body.get("context") or None

        def _gen():
            try:
                for ev in memory.chat_ask_stream(
                    question,
                    k=k,
                    type_=type_,
                    history=history,
                    context=context,
                ):
                    yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
            except Exception as exc:  # never hang the client mid-stream
                err = {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
                yield f"data: {_json.dumps(err, ensure_ascii=False)}\n\n"

        # Starlette drives the sync generator in a worker thread; the MLX
        # forward is GIL/Metal-bound and already serialized, so one stream at
        # a time is fine for the single-user daemon.
        return StreamingResponse(_gen(), media_type="text/event-stream")

    return server


def main() -> None:
    """Entry point for `memo-mcp` console script.

    Default transport is stdio (one client per process — Claude Code, Codex,
    etc.). Set ``MEMO_MCP_TRANSPORT=http`` (or ``streamable-http``/``sse``) to
    run as a long-lived HTTP daemon instead, so an external service can call
    the warm tools (e.g. ``memory_chat_ask``) without paying the ~per-process
    MLX cold-load. ``MEMO_MCP_HOST``/``MEMO_MCP_PORT`` pick the bind (default
    127.0.0.1:18768). One ``Memory`` instance is built here and reused across
    every request, so the embedder / reranker / synthesis LLM stay resident.
    """
    from memo.flags import flag_int, flag_str

    transport = (flag_str("MEMO_MCP_TRANSPORT") or "stdio").strip().lower()
    if transport in ("http", "streamable-http", "sse"):
        # Long-lived daemon: enable the prompt cache + a larger query-embedding
        # cache BEFORE building Memory, so the embedder constructed inside
        # build_server() actually picks them up (it reads the cache size at
        # construction time).
        os.environ.setdefault("MEMO_PROMPT_CACHE", "1")
        os.environ.setdefault("MEMO_QUERY_CACHE_SIZE", "500")
        server = build_server()
        host = flag_str("MEMO_MCP_HOST") or "127.0.0.1"
        # flag_int falls back to the registered default (18768) on a malformed
        # value, so no try/except is needed here.
        port = flag_int("MEMO_MCP_PORT") or 18768
        # transport is validated against the allowed set just above.
        server.run(transport=cast(Any, transport), host=host, port=port)
    else:
        server = build_server()
        server.run()


if __name__ == "__main__":
    main()
