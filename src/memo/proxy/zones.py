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

`whole_history_scope`/`scan_scope` govern a related but distinct question:
which MESSAGES a content transform (structmap, delta, jsoncrush, toolresults,
pixel — not `ToolSchemas`, which always scans the whole prefix) is allowed to
scan. The cache rule above says a transform touching the prefix must be
deterministic; it does NOT say the prefix is off-limits. A transform that
maps the same block to the same bytes every time — independent of anything
outside that block, or (for structmap/delta) dependent only on the blocks
that precede it — produces a byte-identical prefix turn after turn even when
it runs over `frozen_messages` too, so `MEMO_PROXY_CONTENT_SCOPE=all`
(default) widens these five past the live zone. The one-time cost is a single
re-cache the first time a transform starts touching an already-cached block;
after that the wider prefix is exactly as stable as the narrower one was.
`MEMO_PROXY_CONTENT_SCOPE=tail` restricts back to `live_messages` only, the
original scope, one flag away.
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


_CONTENT_SCOPE_FLAG = "MEMO_PROXY_CONTENT_SCOPE"


def whole_history_scope() -> bool:
    """True unless `MEMO_PROXY_CONTENT_SCOPE` (`flags_proxy.py`) is explicitly
    `"tail"`. Never raises — an unreadable flag falls back to the aggressive
    default, matching every other proxy flag's fail-open convention.

    This governs the five DETERMINISTIC content transforms (structmap, delta,
    jsoncrush, toolresults, pixel — see `scan_scope` below and each
    transform's own module docstring for its determinism argument), never
    `ToolSchemas`, which already scans the whole prefix (`system`/`tools`)
    regardless of this flag.
    """
    try:
        from memo.flags import flag_str

        return flag_str(_CONTENT_SCOPE_FLAG) != "tail"
    except Exception:
        return True


def scan_scope(zones: Zones) -> list[dict]:
    """Messages a deterministic content transform may scan and rewrite.

    Whole-history (default, `MEMO_PROXY_CONTENT_SCOPE=all`): the full
    conversation, frozen zone included. This is safe ONLY because the five
    transforms that call this are deterministic, content-only functions of a
    block and (for structmap/delta) the blocks that precede it — the same
    input therefore produces byte-identical output turn after turn, so a
    block already inside the provider's cached prefix is never rewritten to
    something different once cached. See the module docstring's cache rule
    and each transform's own docstring for why it qualifies.

    Tail-only (`MEMO_PROXY_CONTENT_SCOPE=tail`): just `zones.live_messages`,
    the original conservative scope, kept as a one-flag-away rollback.
    """
    if whole_history_scope():
        return [*zones.frozen_messages, *zones.live_messages]
    return zones.live_messages


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
