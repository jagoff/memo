from __future__ import annotations

import sqlite3
import threading
from contextlib import AbstractContextManager
from pathlib import Path


class _StoreBase:
    """Shared attribute/method contract so the mixins type-check against the
    composed VecStore. Real implementations live in the mixins; these are
    typed stubs (ellipsis bodies) mypy accepts without return-checks."""

    db_path: Path
    dims: int
    _local: threading.local

    @property
    def _conn(self) -> sqlite3.Connection: ...  # type: ignore[empty-body]
    def _tx(self) -> AbstractContextManager[sqlite3.Connection]: ...  # type: ignore[empty-body]
    def _checkpoint(self) -> None: ...
    def _delete_repo_file_rows(self, cx: sqlite3.Connection, file_ids: list[str]) -> None: ...
