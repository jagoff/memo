"""Composes transforms over a split payload.

`plan` knows nothing about what any transform does — only its zone, whether it
is enabled, and that it must never be allowed to fail a request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

ZONE_PREFIX = "prefix"
ZONE_LIVE = "live"


@dataclass
class Context:
    state_dir: Path
    session_key: str
    project: str | None = None


@runtime_checkable
class Transform(Protocol):
    name: str
    zone: str

    def enabled(self) -> bool: ...

    def apply(self, zones: Zones, ctx: Context) -> int: ...


@dataclass
class TransformPlan:
    applied: list[str] = field(default_factory=list)
    est_saved_tokens: int = 0
    # Per-transform breakdown of est_saved_tokens — lets a downstream reporter
    # (e.g. `memo tokens --by-transform`) attribute savings honestly instead of
    # crediting the whole scalar to every transform that merely ran.
    saved_by: dict[str, int] = field(default_factory=dict)


def apply_all(zones: Zones, ctx: Context, transforms: list[Transform]) -> TransformPlan:
    """Run every enabled transform. One that raises is skipped, never fatal."""
    plan = TransformPlan()
    for transform in transforms:
        try:
            if not transform.enabled():
                continue
            saved = transform.apply(zones, ctx)
            # Coerce to int defensively: non-numeric and None both → 0
            try:
                saved_int = int(saved or 0)
            except (ValueError, TypeError):
                saved_int = 0
            plan.applied.append(transform.name)
            plan.est_saved_tokens += saved_int
            if saved_int:
                plan.saved_by[transform.name] = plan.saved_by.get(transform.name, 0) + saved_int
        except Exception:
            # Guard access to transform.name in case it raises
            try:
                transform_name = transform.name
            except Exception:
                transform_name = "<unknown>"
            _log.warning("proxy: transform %s failed; skipped", transform_name)
            continue
    return plan
