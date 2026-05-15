"""Query composition and saved queries — build, save, reuse complex searches.

Allows users to:
- Compose complex queries with filters (type, tags, date ranges)
- Save queries for reuse
- Execute saved queries by name
- Share queries between sessions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Query:
    """A saved search query."""
    name: str
    query_text: str
    type_filter: str | None
    tags_filter: list[str] | None
    date_from: str | None
    date_to: str | None
    search_mode: str  # vec, bm25, hybrid
    limit: int
    description: str | None
    created_at: str


@dataclass
class QueryResult:
    """Result of executing a saved query."""
    query_name: str
    results: list[Any]
    count: int
    executed_at: str


class QueryStore:
    """Stores saved queries.

    Args:
        state_dir: Directory to store query state.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.queries_file = state_dir / "saved_queries.json"
        self._queries: dict[str, Query] = {}
        self._load()

    def _load(self) -> None:
        """Load queries from disk."""
        if self.queries_file.is_file():
            try:
                data = json.loads(self.queries_file.read_text(encoding="utf-8"))
                for name, q in data.items():
                    self._queries[name] = Query(**q)
            except Exception:
                self._queries = {}

    def _save(self) -> None:
        """Save queries to disk."""
        try:
            data = {name: q.__dict__ for name, q in self._queries.items()}
            self.queries_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def save_query(
        self,
        name: str,
        query_text: str,
        type_filter: str | None = None,
        tags_filter: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        description: str | None = None,
    ) -> None:
        """Save a query.

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
        self._queries[name] = Query(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=tags_filter,
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._save()

    def get_query(self, name: str) -> Query | None:
        """Get a saved query by name."""
        return self._queries.get(name)

    def list_queries(self) -> list[Query]:
        """List all saved queries."""
        return list(self._queries.values())

    def delete_query(self, name: str) -> bool:
        """Delete a saved query."""
        if name in self._queries:
            del self._queries[name]
            self._save()
            return True
        return False


class QueryComposer:
    """Composes and executes complex queries.

    Args:
        memory: The Memory instance to search.
        query_store: The QueryStore for saved queries.
    """

    def __init__(self, memory: Any, query_store: QueryStore) -> None:
        self.memory = memory
        self.query_store = query_store

    def execute_query(self, query: Query) -> QueryResult:
        """Execute a saved query.

        Args:
            query: The Query to execute.

        Returns:
            QueryResult with hits and metadata.
        """
        # Execute the base search
        hits = self.memory.search(
            query.query_text,
            limit=query.limit,
            mode=query.search_mode,
        )

        # Apply filters
        filtered = []
        for hit in hits:
            # Type filter
            if query.type_filter and hit.type != query.type_filter:
                continue

            # Tag filter
            if query.tags_filter and not any(tag in hit.tags for tag in query.tags_filter):
                continue

            # Date filter
            if query.date_from or query.date_to:
                try:
                    hit_date = datetime.fromisoformat(hit.updated.replace("Z", "+00:00"))
                    if query.date_from:
                        from_date = datetime.fromisoformat(query.date_from)
                        if hit_date < from_date:
                            continue
                    if query.date_to:
                        to_date = datetime.fromisoformat(query.date_to)
                        if hit_date > to_date:
                            continue
                except Exception:
                    pass

            filtered.append(hit)

        return QueryResult(
            query_name=query.name,
            results=filtered,
            count=len(filtered),
            executed_at=datetime.now(UTC).isoformat(),
        )

    def compose_and_save(
        self,
        name: str,
        query_text: str,
        type_filter: str | None = None,
        tags_filter: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        description: str | None = None,
    ) -> QueryResult:
        """Compose, save, and execute a query.

        Args:
            name: Query name.
            query_text: Search text.
            type_filter: Optional type filter.
            tags_filter: Optional tag filter.
            date_from: Optional start date.
            date_to: Optional end date.
            search_mode: Search mode.
            limit: Result limit.
            description: Optional description.

        Returns:
            QueryResult with execution results.
        """
        self.query_store.save_query(
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

        query = self.query_store.get_query(name)
        if not query:
            raise ValueError(f"Failed to save query: {name}")

        return self.execute_query(query)


__all__ = [
    "Query",
    "QueryComposer",
    "QueryResult",
    "QueryStore",
]
