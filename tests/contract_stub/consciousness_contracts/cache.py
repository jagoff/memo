from __future__ import annotations

from typing import Any


class ContentHashCache:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._items.get(key)

    def put(self, key: str, value: Any) -> None:
        self._items[key] = value


_DEFAULT_CACHE = ContentHashCache()


def get_default_cache() -> ContentHashCache:
    return _DEFAULT_CACHE
