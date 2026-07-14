from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class MemoUri:
    resource_type: str
    resource_id: str | None
    subpath: str | None


def is_memo_uri(uri: str) -> bool:
    return uri.startswith("memo://")


def parse_uri(uri: str) -> MemoUri | None:
    parsed = urlparse(uri)
    if parsed.scheme != "memo" or not parsed.netloc:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    return MemoUri(
        resource_type=parsed.netloc,
        resource_id=parts[0] if parts else None,
        subpath="/".join(parts[1:]) if len(parts) > 1 else None,
    )


def device_id() -> str:
    return hashlib.sha256(platform.node().encode()).hexdigest()[:16]
