from __future__ import annotations

from types import ModuleType

from memo.mlx_gpu import suppress_swig_deprecation_warnings


def import_sqlite_vec() -> ModuleType:
    """Import sqlite-vec without its Python 3.14 SWIG metadata noise.

    The filter stays installed because ``swigvarlink`` emits the same
    third-party deprecation while the interpreter is shutting down.
    """
    suppress_swig_deprecation_warnings()
    import sqlite_vec  # type: ignore[import-not-found]

    return sqlite_vec
