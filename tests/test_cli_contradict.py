from __future__ import annotations

from types import SimpleNamespace

from memo.cli_contradict import _older_first


def test_older_first_orders_by_instant_not_timestamp_text() -> None:
    earlier = SimpleNamespace(id="earlier", updated="2026-01-01T03:00:00+03:00")
    later = SimpleNamespace(id="later", updated="2026-01-01T00:30:00+00:00")

    rec_a, rec_b = _older_first(later, earlier)

    assert rec_a is earlier
    assert rec_b is later
