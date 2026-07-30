"""Offline integration adapters that fence callbacks before retired work begins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from memo.operational_roster import VerificationRoster
from tools.memflow_absorption.control_record import (
    ControlRecordCAS,
    fetch_current_verified_control,
    validate_synapse_request,
)


def _guarded[T](
    cas: ControlRecordCAS,
    roster: VerificationRoster,
    epoch: int,
    kind: Literal["startup", "write", "fallback"],
    callback: Callable[[], T],
) -> T:
    control = fetch_current_verified_control(cas, roster=roster)
    validate_synapse_request(control, epoch, kind=kind)
    return callback()


def before_listener_start[T](
    cas: ControlRecordCAS,
    roster: VerificationRoster,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(cas, roster, epoch, "startup", callback)


def before_worker_start[T](
    cas: ControlRecordCAS,
    roster: VerificationRoster,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(cas, roster, epoch, "startup", callback)


def before_write[T](
    cas: ControlRecordCAS,
    roster: VerificationRoster,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(cas, roster, epoch, "write", callback)


def before_fallback[T](
    cas: ControlRecordCAS,
    roster: VerificationRoster,
    epoch: int,
    callback: Callable[[], T],
) -> T:
    return _guarded(cas, roster, epoch, "fallback", callback)


__all__ = [
    "before_fallback",
    "before_listener_start",
    "before_worker_start",
    "before_write",
]
