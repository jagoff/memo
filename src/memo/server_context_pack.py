from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from memo.context_pack import build_context_pack
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_context_pack(
        question: str,
        k: int = 5,
        type_: str | None = None,
        snippet_chars: int = 800,
    ) -> dict[str, Any]:
        """Build a composed context pack for a question without calling the LLM.

        Returns current facts, supporting context, stale/conflicting context, and
        a compact summary. Use this when search hits need interpretation before
        answering.
        """
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
        return asdict(pack)
