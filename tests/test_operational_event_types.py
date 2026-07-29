from __future__ import annotations

import pytest

from memo.error_contract import OperationalError, OperationalErrorCode
from memo.operational_event_types import (
    DELIVERY_ENQUEUED,
    EVENT_TYPES,
    FOCUS_SET,
    PRESENCE_UPDATED,
    validate_event_payload,
)


def test_registry_is_closed_and_fully_qualified() -> None:
    assert FOCUS_SET in EVENT_TYPES
    assert DELIVERY_ENQUEUED in EVENT_TYPES
    assert PRESENCE_UPDATED in EVENT_TYPES
    assert all(name.startswith("memo.operational.") and name.endswith(".v1") for name in EVENT_TYPES)


def test_registry_validates_mapping_payload_and_rejects_unknown_names() -> None:
    validate_event_payload(FOCUS_SET, {"project": "memo", "summary": "ship"})
    with pytest.raises(OperationalError) as exc:
        validate_event_payload("focus.set", {})
    assert exc.value.code is OperationalErrorCode.INVALID_EVENT
    with pytest.raises(OperationalError):
        validate_event_payload(FOCUS_SET, ["not", "a", "mapping"])  # type: ignore[arg-type]
