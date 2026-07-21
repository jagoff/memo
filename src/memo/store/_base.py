from __future__ import annotations

import sqlite3
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tantivy_index import TantivyFTSIndex


class _StoreBase:
    """Shared attribute/method contract so the mixins type-check against the
    composed VecStore. Real implementations live in the mixins; these are
    typed stubs (ellipsis bodies) mypy accepts without return-checks."""

    db_path: Path
    dims: int
    embedder_model: str
    vec_quant: str
    _quant_int8: bool
    _has_pattern_cols: bool
    _local: threading.local
    tantivy_index_dir: Path
    _tantivy_inst: TantivyFTSIndex | None
    _tantivy_init_lock: threading.Lock
    _tantivy_write_lock: threading.Lock

    @property
    def _conn(self) -> sqlite3.Connection: ...  # type: ignore[empty-body]
    def _tx(self) -> AbstractContextManager[sqlite3.Connection]: ...  # type: ignore[empty-body]
    def _checkpoint(self) -> None: ...
    def _delete_repo_file_rows(self, cx: sqlite3.Connection, file_ids: list[str]) -> None: ...
    def _get_tantivy(self) -> TantivyFTSIndex | None: ...  # type: ignore[empty-body]
    def _rebuild_tantivy_from_sqlite(self) -> None: ...
    def _mark_tantivy_unhealthy(self) -> None: ...
    def _vec_table_dims(self, table: str) -> int | None: ...
    def _vec_table_dtype(self, table: str) -> str: ...  # type: ignore[empty-body]
    def _vec_dtype_ddl(self) -> str: ...  # type: ignore[empty-body]
    def _vec_bind_new(self) -> str: ...  # type: ignore[empty-body]
    def _vec_bind_stored(self) -> str: ...  # type: ignore[empty-body]
    def _create_vec_tables(self, conn: sqlite3.Connection) -> None: ...
    def _run_migrations(self) -> None: ...
    def set_user_version(self, version: int) -> None: ...
