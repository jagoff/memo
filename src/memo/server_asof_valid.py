"""MCP tools — valid-time as-of (record-level world-validity) domain.

Distinct from ``server_asof.py``, which does TRANSACTION-time reconstruction via
``time_machine.reconstruct`` (what the corpus *looked like* at a past point given
the audit log). These tools instead filter by each record's VALID-time interval
(``COALESCE(valid_at, created) <= as_of AND (invalid_at IS NULL OR invalid_at >
as_of)``), so a since-superseded fact resurfaces exactly as it stood *in the
world* at ``as_of``. Routed straight through the live ``Memory.search`` /
``Memory.ask`` index with ``as_of=``; no snapshot reconstruction.

Registered by ``build_server()`` via ``register(server, memory)`` under the
advanced (full) profile gate, alongside the transaction-time as-of tools. Both
are read/consult verbs and take ``source=`` for attribution (see CLAUDE.md's
attribution contract) — no cognition/suggest surface is added.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.asof import validate_as_of
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_search_valid_as_of(
        query: str,
        as_of: str,
        limit: int = 10,
        type: str | None = None,
        mode: str = "hybrid",
        source: str = "",
    ) -> dict[str, Any]:
        """Search durable memories as they were VALID at a past instant (valid-time).

        Unlike ``memo_search_as_of`` (transaction-time snapshot replay), this
        filters each record by its world-validity interval: a record surfaces
        only when ``as_of`` falls in ``[valid_at (or created), invalid_at)``. Use
        it to recall what was TRUE at a date even after the fact was later
        superseded — a since-invalidated record resurfaces at its historical
        instant instead of being hidden by the default now-gate.

        Args:
            query: Free-text query.
            as_of: ``YYYY-MM-DD`` or full ISO-8601 timestamp — the world-time to
                view the corpus at.
            limit: Max results.
            type: Optional record-type filter.
            mode: ``hybrid`` (default), ``vec``, or ``bm25``.
            source: Attributes the consult log to the calling layer.

        Returns: ``{"as_of", "results": [...]}`` where each result has the same
        shape as ``memo_search`` items.
        """
        t0 = now_ms()
        validate_as_of(as_of)
        records = memory.search(query, limit=limit, type_=type, mode=mode, as_of=as_of)
        results = [r.to_dict() for r in records]
        log_consult(
            memory, tool="search_valid_as_of", query=query, hits=results, t0_ms=t0, source=source
        )
        return {"as_of": as_of, "results": results}

    @annotated_tool(server, **READ_ONLY)
    def memo_ask_valid_as_of(
        question: str,
        as_of: str,
        k: int = 5,
        type: str | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        """RAG answer grounded in memories VALID at a past instant (valid-time).

        Same contract as ``memo_ask`` but retrieval is filtered to records whose
        world-validity interval contains ``as_of`` (see ``memo_search_valid_as_of``).
        Distinct from ``memo_ask_as_of``, which rewinds the *whole corpus* via
        transaction-time snapshot replay.

        Args:
            question: Free-text question.
            as_of: ``YYYY-MM-DD`` or full ISO 8601.
            k: Top-k memories fed to the model as context.
            type: Optional record-type filter.
            source: Attributes the consult log to the calling layer.

        Returns: the ``memo_ask`` answer envelope plus an ``as_of`` echo.
        """
        t0 = now_ms()
        validate_as_of(as_of)
        res = memory.ask(question, k=k, type_=type, as_of=as_of)
        cites = res.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        log_consult(
            memory, tool="ask_valid_as_of", query=question, hits=hit_dicts, t0_ms=t0, source=source
        )
        return {**res, "as_of": as_of}
