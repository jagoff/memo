from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from memo.context_pack import build_context_pack, consult_hits_from_pack
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_context_pack(
        question: str,
        k: int = 5,
        type_: str | None = None,
        snippet_chars: int = 800,
        source: str = "",
    ) -> dict[str, Any]:
        """Build a composed context pack for a question without calling the LLM.

        Returns current facts, supporting context, stale/conflicting context, and
        a compact summary. Use this when search hits need interpretation before
        answering. `source` attributes consult logging.
        """
        from memo.flags import flag_bool

        if not flag_bool("MEMO_CONTEXT_PACK"):
            return {
                "status": "disabled",
                "reason": "MEMO_CONTEXT_PACK=0 disables explicit context-pack tools.",
                "question": question,
            }
        t0 = now_ms()
        hits = memory.search(
            question,
            limit=k,
            type_=type_,
            mode="hybrid",
            disable_reranker=True,
            read_through=True,
            quality_rerank=True,
        )
        pack = build_context_pack(question, hits, snippet_chars=snippet_chars)
        log_consult(
            memory,
            tool="context_pack",
            query=question,
            hits=consult_hits_from_pack(pack),
            t0_ms=t0,
            source=source,
        )
        return asdict(pack)
