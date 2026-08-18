"""Splits a Messages-API payload into a cache-stable prefix and a live zone.

The economics of this whole package hinge on one rule: a cache read costs 0.1x
a fresh input token, so a transform that rewrites the cached prefix on every
turn can easily cost more than it saves. Transforms that touch the prefix must
be deterministic and session-stable; `prefix_fingerprint` is what proves it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

LIVE_TURNS_DEFAULT = 2


@dataclass
class Zones:
    system: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    frozen_messages: list[dict] = field(default_factory=list)
    live_messages: list[dict] = field(default_factory=list)

    def to_payload(self, original: dict) -> dict:
        """Reassemble, preserving every key the proxy does not own."""
        out = dict(original)
        if self.system or "system" in original:
            out["system"] = self.system
        if self.tools or "tools" in original:
            out["tools"] = self.tools
        out["messages"] = [*self.frozen_messages, *self.live_messages]
        return out


def _as_list(value: object) -> list[dict]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return []


def split(payload: dict, *, live_turns: int = LIVE_TURNS_DEFAULT) -> Zones:
    """Partition a request payload. Never raises on a malformed payload."""
    if not isinstance(payload, dict):
        return Zones()
    messages = payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    cut = max(0, len(messages) - live_turns)
    return Zones(
        system=_as_list(payload.get("system")),
        tools=_as_list(payload.get("tools")),
        frozen_messages=list(messages[:cut]),
        live_messages=list(messages[cut:]),
    )


def prefix_fingerprint(zones: Zones) -> str:
    """sha256 of everything the provider will cache. Live zone excluded."""
    blob = json.dumps(
        [zones.system, zones.tools, zones.frozen_messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
