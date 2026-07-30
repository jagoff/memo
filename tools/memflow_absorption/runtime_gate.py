"""Offline integration adapters that fence callbacks before retired work begins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tools.memflow_absorption.control_record import validate_synapse_request
from tools.memflow_absorption.schemas import VerifiedControlRecord


def _guarded[T](
    control: VerifiedControlRecord,
    epoch: int,
    kind: Literal["startup", "write", "fallback"],
    callback: Callable[[], T],
) -> T:
    validate_synapse_request(control, epoch, kind=kind)
    return callback()


def before_listener_start[T](
    control: VerifiedControlRecord,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(control, epoch, "startup", callback)


def before_worker_start[T](
    control: VerifiedControlRecord,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(control, epoch, "startup", callback)


def before_write[T](
    control: VerifiedControlRecord,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(control, epoch, "write", callback)


def before_fallback[T](
    control: VerifiedControlRecord,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(control, epoch, "fallback", callback)


__all__ = [
    "before_fallback",
    "before_listener_start",
    "before_worker_start",
    "before_write",
]
