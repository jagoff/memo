from __future__ import annotations

from unittest.mock import MagicMock

from memo.outcome import self_heal_execution_failures


def test_self_heal_execution_failures() -> None:
    mem = MagicMock()
    mem.store.get_health_batch.return_value = {
        "mem_fail_1": {"confidence": 0.9}
    }

    res = self_heal_execution_failures(mem, ["mem_fail_1"])
    assert res["healed"] == 1
    assert res["penalized"] == 1

    # Verify set_confidence_batch was called with reduced confidence (0.9 - 0.35 = 0.55)
    mem.store.set_confidence_batch.assert_called_once_with([("mem_fail_1", 0.55)])
