"""Tantivy-backed FTS index for memo memories.

Optional: only active when `tantivy` is installed AND `MEMO_FTS_BACKEND != fts5`.
When absent, all code paths fall back transparently to the FTS5 implementation
in queries.py. This is the only file in memo that imports `tantivy`.
"""

from __future__ import annotations

import logging
import threading
import unicodedata
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

    Thread safety: a single `threading.Lock` serialises all writer calls.
    The IndexWriter is held open for the lifetime of this object; one commit
    per mutating operation keeps the on-disk index current.
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
        self._index = tantivy.Index(self._schema, path=str(index_dir))
        self._writer = self._index.writer(heap_size=writer_heap_mb * 1024 * 1024)
        self._lock = threading.Lock()

    @classmethod
    def open_or_create(cls, index_dir: Path) -> TantivyFTSIndex:
        return cls(index_dir)

    @staticmethod
    def exists(index_dir: Path) -> bool:
        return (index_dir / "meta.json").exists()

    # -- write -----------------------------------------------------------------

    def close(self) -> None:
        """Release the tantivy writer and index resources."""
        import contextlib

        with contextlib.suppress(Exception):
            self._writer.commit()

    def add_document(self, id_: str, title: str, tags: str, body: str) -> None:
        import tantivy

        doc = tantivy.Document()
        doc.add_text("id", id_)
        doc.add_text("title", _fold_diacritics(title))
        doc.add_text("tags", _fold_diacritics(tags))
        doc.add_text("body", _fold_diacritics(body))
        with self._lock:
            self._writer.add_document(doc)

    def delete_document(self, id_: str) -> None:
        with self._lock:
            self._writer.delete_documents("id", id_)

    def commit(self) -> None:
        # reload() inside the lock so a concurrent search can't read a stale
        # snapshot in the window between commit() and reload().
        with self._lock:
            self._writer.commit()
            self._index.reload()

    def rebuild(self, records: list[dict[str, Any]]) -> None:
        """Clear the index and bulk-index all records in one commit."""
        import tantivy

        with self._lock:
            self._writer.delete_all_documents()
            for r in records:
                doc = tantivy.Document()
                doc.add_text("id", r.get("id") or "")
                doc.add_text("title", _fold_diacritics(r.get("title") or ""))
                doc.add_text("tags", _fold_diacritics(r.get("tags") or ""))
                doc.add_text("body", _fold_diacritics(r.get("body") or ""))
                self._writer.add_document(doc)
            self._writer.commit()
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
            except Exception:
                continue
            if id_val is None:
                continue
            # Normalize positive tantivy BM25 score to [0,1) via s/(s+1).
            out.append({"id": str(id_val), "score": score / (score + 1.0)})
        return out
