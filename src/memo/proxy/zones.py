"""Splits a Messages-API payload into a cache-stable prefix and a live zone.

The economics of this whole package hinge on one rule: a cache read costs 0.1x
a fresh input token, so a transform that rewrites the cached prefix on every
turn can easily cost more than it saves. Transforms that touch the prefix must
be deterministic and session-stable; `prefix_fingerprint` is what proves it.

Two fingerprints, two different questions:

* `prefix_fingerprint` — sha256 of everything the provider will cache for
  THIS request (`system`, `tools`, `frozen_messages`). A strict, within-one-
  request check: "is this exactly what I expect the cached span to be right
  now." `frozen_messages` is included on purpose here.
* `stable_head_fingerprint` — sha256 of ONLY `system` and `tools`. Provider
  prompt caching matches the LONGEST CACHED PREFIX, so a growing
  conversation is cache-FRIENDLY: appending new turns leaves the earlier
  cached bytes valid, and only pays for the small new suffix. `frozen_messages`
  is therefore expected to change every turn as the live window advances —
  that is normal traffic, not instability. What actually destroys a cache
  hit is REWRITING content that was already cached: `system` or `tools`
  changing mid-session. `stable_head_fingerprint` is the property to compare
  ACROSS turns/requests in one session (see `memo.proxy.server`'s runtime
  drift check) — `prefix_fingerprint` compared across turns would false-
  positive on every single turn of a real conversation, since
  `frozen_messages` legitimately differs turn to turn.
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
    """sha256 of everything the provider will cache. Live zone excluded.

    Strict, single-request check — see the module docstring for why this is
    NOT the right thing to compare across turns/requests in a session
    (`frozen_messages` legitimately differs turn to turn as the live window
    advances; use `stable_head_fingerprint` for that comparison instead).
    """
    blob = json.dumps(
        [zones.system, zones.tools, zones.frozen_messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stable_head_fingerprint(zones: Zones) -> str:
    """sha256 of ONLY `system` and `tools` — the part of the cached prefix
    that must never be REWRITTEN mid-session (as opposed to `frozen_messages`,
    which is expected to GROW every turn and is excluded here). See the
    module docstring: this is the fingerprint to compare across turns of one
    session; `prefix_fingerprint` is not, because it would false-positive on
    ordinary conversation growth.
    """
    blob = json.dumps(
        [zones.system, zones.tools],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
