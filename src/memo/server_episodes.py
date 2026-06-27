"""MCP tools — episodic memory (session search).

Exposes the episode index (Phase 1's `memo resume` semantic layer) so agents can
ask "what was the session about X" over MCP — the same derived index, a second
surface. Retrieval only (no cognition verb): it returns resumable sessions, it
does not orchestrate. Registered by `build_server()` via `register(server, memory)`.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_episodes_search(query: str, limit: int = 10) -> dict[str, Any]:
        """Find past work *sessions* by meaning — memo's episodic memory.

        Distinct from `memo_search` (durable facts/decisions): this searches the
        semantic index of your *sessions* — what you were working on, across
        agents — and returns each as a resumable session. Use it to answer
        "which session was about X" or "where did I work on Y".

        Args:
            query: Free-text description of the session you're looking for.
            limit: Max sessions to return.

        Returns: `{"query", "results": [{session_id, agent, score, summary,
        cwd, resume_command}]}`. Empty `results` ⇒ the index is empty (run
        `memo episodes index`) or episodic memory is disabled.
        """
        from memo.resume._index import semantic_search

        hits = semantic_search(memory.cfg, query, k=limit, allow_cold=True)
        return {
            "query": query,
            "results": [
                {
                    "session_id": h.session_id,
                    "agent": h.agent,
                    "score": h.metadata.get("score"),
                    "summary": h.summary or h.title,
                    "cwd": h.cwd,
                    "resume_command": list(h.resume_command),
                }
                for h in hits
            ],
        }
