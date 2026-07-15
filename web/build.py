"""Source-tree compatibility wrapper for :mod:`memo.web_build`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from memo import web_build as _impl  # noqa: E402

build = _impl.build
collect_data = _impl.collect_data


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


if __name__ == "__main__":
    _impl.main()
