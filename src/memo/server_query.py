"""MCP tools — saved-queries domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_query_save(
        name: str,
        query_text: str,
        type_filter: str | None = None,
        tags_filter: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Save a query for reuse.

        Stores a query with its parameters so it can be executed later
        by name. Useful for frequently-used complex searches.

        Args:
            name: Query name (unique).
            query_text: The search query text.
            type_filter: Optional type filter.
            tags_filter: Optional tag filter.
            date_from: Optional start date (ISO format).
            date_to: Optional end date (ISO format).
            search_mode: Search mode (vec, bm25, hybrid).
            limit: Result limit.
            description: Optional description.
        """
        memory.query_composer.query_store.save_query(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=tags_filter,
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        return {"status": "saved", "name": name}

    @annotated_tool(server, **READ_ONLY)
    def memo_query_list() -> list[dict[str, Any]]:
        """List all saved queries.

        Returns a list of all saved queries with their parameters.
        Useful for discovering available queries to execute.
        """
        queries = memory.query_composer.query_store.list_queries()
        return [q.__dict__ for q in queries]

    @annotated_tool(server, **READ_ONLY)
    def memo_query_run(
        name: str,
    ) -> dict[str, Any]:
        """Execute a saved query.

        Executes a previously saved query by name and returns the results.

        Args:
            name: The name of the saved query to execute.
        """
        query = memory.query_composer.query_store.get_query(name)
        if not query:
            return {"error": "Query not found", "name": name}

        result = memory.query_composer.execute_query(query)
        return {
            "query_name": result.query_name,
            "count": result.count,
            "executed_at": result.executed_at,
            "results": [r.to_dict() for r in result.results],
        }

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_query_delete(
        name: str,
    ) -> dict[str, Any]:
        """Delete a saved query.

        Removes a saved query by name.

        Args:
            name: The name of the query to delete.
        """
        success = memory.query_composer.query_store.delete_query(name)
        return {"success": success, "name": name}
