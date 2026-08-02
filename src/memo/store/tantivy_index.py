"""Tantivy-backed FTS index for memo memories.

Optional: only active when `tantivy` is installed AND `MEMO_FTS_BACKEND != fts5`.
When absent, all code paths fall back transparently to the FTS5 implementation
in queries.py. This is the only file in memo that imports `tantivy`.
"""

from __future__ import annotations

import logging
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

_log = logging.getLogger("memo.store.tantivy")


@lru_cache(maxsize=1)
def _tantivy_available() -> bool:
    try:
        import tantivy  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=4096)
def _fold_diacritics(text: str) -> str:
    """Strip combining diacritical marks so 'decisión' matches 'decision'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    ).lower()


class TantivyFTSIndex:
    """Tantivy index wrapping memo's title+tags+body FTS schema.

    Thread safety: a single `threading.Lock` serialises local operations. Writer
    leases are short-lived and guarded by a cross-process flock, so a long-lived
    MCP reader never monopolises Tantivy's single-writer lock and blocks CLI or
    daemon updates.
    """

    def __init__(self, index_dir: Path, *, writer_heap_mb: int = 50) -> None:
        import tantivy

        sb = tantivy.SchemaBuilder()
        # id is stored so we can retrieve it from search results.
        sb.add_text_field("id", stored=True)
        # BM25-indexed fields — body is not stored (it lives on disk as .md).
        sb.add_text_field("title", stored=False)
        sb.add_text_field("tags", stored=False)
        sb.add_text_field("body", stored=False)
        self._schema = sb.build()

        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir = index_dir
        self._index = tantivy.Index(self._schema, path=str(index_dir))
        self._writer_heap_bytes = writer_heap_mb * 1024 * 1024
        self._pending: list[tuple[str, tuple[str, ...]]] = []
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open_or_create(cls, index_dir: Path) -> TantivyFTSIndex:
        return cls(index_dir)

    @staticmethod
    def exists(index_dir: Path) -> bool:
        return (index_dir / "meta.json").exists()

    # -- write -----------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Tantivy index is closed")

    @contextmanager
    def _writer_lease(self) -> Iterator[Any]:
        """Yield the exclusive Tantivy writer and release it after one commit."""
        import contextlib
        import fcntl

        lock_path = self._index_dir / ".memo-writer.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            writer = self._index.writer(heap_size=self._writer_heap_bytes)
            try:
                yield writer
            finally:
                # Joining merge threads consumes the writer and deterministically
                # releases Tantivy's own lock before the outer flock is dropped.
                with contextlib.suppress(Exception):
                    writer.wait_merging_threads()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        """Commit pending writes and close this local index handle."""
        import contextlib

        if self._closed:
            return
        with contextlib.suppress(Exception):
            self.commit()
        with self._lock:
            self._pending.clear()
            self._closed = True

    def add_document(self, id_: str, title: str, tags: str, body: str) -> None:
        with self._lock:
            self._ensure_open()
            self._pending.append(
                (
                    "add",
                    (
                        id_,
                        _fold_diacritics(title),
                        _fold_diacritics(tags),
                        _fold_diacritics(body),
                    ),
                )
            )

    def delete_document(self, id_: str) -> None:
        with self._lock:
            self._ensure_open()
            self._pending.append(("delete", (id_,)))

    def commit(self) -> None:
        import tantivy

        with self._lock:
            self._ensure_open()
            if self._pending:
                with self._writer_lease() as writer:
                    for operation, values in self._pending:
                        if operation == "delete":
                            writer.delete_documents("id", values[0])
                            continue
                        doc = tantivy.Document()
                        doc.add_text("id", values[0])
                        doc.add_text("title", values[1])
                        doc.add_text("tags", values[2])
                        doc.add_text("body", values[3])
                        writer.add_document(doc)
                    writer.commit()
                self._pending.clear()
            # reload() stays inside the lock so a concurrent search cannot read
            # a stale snapshot between the commit and refresh.
            self._index.reload()

    def rebuild(self, records: list[dict[str, Any]]) -> None:
        """Clear the index and bulk-index all records in one commit."""
        import tantivy

        with self._lock:
            self._ensure_open()
            self._pending.clear()
            with self._writer_lease() as writer:
                writer.delete_all_documents()
                for r in records:
                    doc = tantivy.Document()
                    doc.add_text("id", r.get("id") or "")
                    doc.add_text("title", _fold_diacritics(r.get("title") or ""))
                    doc.add_text("tags", _fold_diacritics(r.get("tags") or ""))
                    doc.add_text("body", _fold_diacritics(r.get("body") or ""))
                    writer.add_document(doc)
                writer.commit()
            self._index.reload()

    # -- search ----------------------------------------------------------------

    _FIELD_BOOSTS: ClassVar[dict[str, float]] = {"title": 5.0, "tags": 3.0, "body": 1.0}
    _FUZZY_FIELDS: ClassVar[dict[str, tuple[bool, int, bool]]] = {
        "title": (False, 1, True),  # (prefix, distance, transpose_cost_one)
        "tags": (False, 1, True),
        "body": (False, 1, True),
    }

    def search_bm25(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """BM25 search with field boosts. Returns [{id, score}] (score in [0,1))."""
        return self._run_query(
            _fold_diacritics(query),
            limit,
            fuzzy=False,
        )

    def search_fuzzy(
        self,
        query: str,
        limit: int = 10,
        edit_distance: int = 1,
    ) -> list[dict[str, Any]]:
        """Fuzzy BM25 (typo-tolerant). Returns [{id, score}]."""
        return self._run_query(
            _fold_diacritics(query),
            limit,
            fuzzy=True,
        )

    def _run_query(
        self, query_str: str, limit: int, *, fuzzy: bool = False
    ) -> list[dict[str, Any]]:
        if not query_str.strip():
            return []
        try:
            kwargs: dict[str, Any] = {
                "field_boosts": self._FIELD_BOOSTS,
            }
            if fuzzy:
                kwargs["fuzzy_fields"] = self._FUZZY_FIELDS
            q = self._index.parse_query(query_str, ["title", "tags", "body"], **kwargs)
        except Exception as exc:
            _log.debug("tantivy parse_query failed for %r: %s", query_str, exc)
            return []
        # Refresh on every query so a long-lived MCP/daemon reader observes
        # commits made by another memo process.
        with self._lock:
            self._ensure_open()
            self._index.reload()
            searcher = self._index.searcher()
        try:
            results = searcher.search(q, limit)
        except Exception as exc:
            _log.debug("tantivy search failed for %r: %s", query_str, exc)
            return []
        out: list[dict[str, Any]] = []
        # tantivy ≥0.24: search() returns SearchResult with a .hits attribute
        # (list of (score, DocAddress) tuples). Older versions returned a bare
        # list. Handle both by checking for the .hits attribute.
        hits = getattr(results, "hits", results)
        for score, addr in hits:  # type: ignore[union-attr]  # tantivy Hits is iterable at runtime
            try:
                doc = searcher.doc(addr)
                id_val = doc.get_first("id")
            except Exception:  # noqa: S112
                continue
            if id_val is None:
                continue
            # Normalize positive tantivy BM25 score to [0,1) via s/(s+1).
            out.append({"id": str(id_val), "score": score / (score + 1.0)})
        return out
