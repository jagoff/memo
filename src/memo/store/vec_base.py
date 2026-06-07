"""Shared vector store interface for the memo / rag ecosystem.

memo's VecStore and rag's SqliteVecCollection both wrap sqlite-vec but expose
different API surfaces. VecStoreBase documents the minimal shared contract so:
- rag's SqliteVecCollection can inherit and align with memo's optimised
  implementation (thread-local connections, WAL config).
- Search pipelines (synapse, memflow) can type-hint VecStoreBase without
  coupling to memo internals.
- New backends (cloud vec DB, DuckDB-vec) only need to implement this interface.

Deliberately NOT an ABC: memo's VecStore uses mixin composition and Python's
ABCMeta doesn't resolve abstract method satisfaction through mixin MRO chains
at class-creation time. The contract is documented here and enforced at runtime
via NotImplementedError.
"""

from __future__ import annotations

from typing import Any


class VecStoreBase:
    """Template base for sqlite-vec-backed stores.

    Implementors must provide a ``dims: int`` attribute and concrete
    implementations of all methods listed below.
    """

    # dims: int must be exposed (instance attribute or @property).

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k vector search by cosine similarity.

        Returns row dicts with at least: id, path, title, type, tags,
        created, updated, score (float, cosine similarity).
        """
        raise NotImplementedError

    def search_bm25(
        self,
        query: str,
        *,
        limit: int,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search (FTS5 or tantivy). Same row dict shape as search()."""
        raise NotImplementedError

    def search_fuzzy(
        self,
        query: str,
        *,
        limit: int,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuzzy/typo-tolerant search via tantivy. Falls back to BM25 when unavailable."""
        return self.search_bm25(query, limit=limit, type_=type_, exclude_types=exclude_types)

    def upsert(
        self,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        embedding: list[float],
        extra: dict[str, Any] | None = None,
        body_text: str = "",
    ) -> None:
        """Insert-or-replace a document with its embedding vector."""
        raise NotImplementedError

    def delete(self, id_: str) -> bool:
        """Remove a document by id. Returns True if a row was deleted."""
        raise NotImplementedError

    def count(self) -> int:
        """Total documents in the store."""
        raise NotImplementedError

    # -- optional (override when supported) --------------------------------

    def get(self, id_: str) -> dict[str, Any] | None:
        """Fetch a single document by id."""
        raise NotImplementedError

    def find_by_prefix(self, prefix: str, limit: int = 10) -> list[str]:
        """Return ids whose prefix matches."""
        raise NotImplementedError

    def list_recent(
        self,
        limit: int = 20,
        type_: str | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent documents by updated timestamp."""
        raise NotImplementedError

    def touch(self, ids: list[str], *, ts: str | None = None) -> None:
        """Update access timestamp for given ids."""

    def has_vector(self, id_: str) -> bool:
        raise NotImplementedError

    def rebuild_fts(self) -> None:
        """Rebuild the full-text search index. No-op if not supported."""

    def close(self) -> None:
        """Release any held resources."""
