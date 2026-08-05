"""Regression: an operational snapshot written before a section existed must
not break the writer that owns that section.

Found running memo as an end user against a real install: `memo operational
signal remember` raised a bare `KeyError: 'signals'` from the CLI, and the
same call through MCP surfaced as the opaque "coordinated MCP write failed
safely". The live snapshot predates the `signals` section, and
`_read_snapshot` returns the persisted dict verbatim whenever the schema
string and journal heads match — the schema string did not change when the
section was added, so nothing ever backfilled the key. Readers were already
defensive (`.get("signals", {})`); only the writer indexed it directly.
"""

from __future__ import annotations

import json

from memo.operational import OperationalStore


def _strip_section(store: OperationalStore, section: str) -> None:
    """Rewrite the on-disk snapshot as an older memo version would have."""
    data = json.loads(store.snapshot_path.read_text(encoding="utf-8"))
    data.pop(section, None)
    store.snapshot_path.write_text(json.dumps(data), encoding="utf-8")


def test_remember_signal_survives_a_snapshot_without_the_signals_section(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    store.set_focus(project="memo", summary="seed the snapshot")
    _strip_section(store, "signals")

    signal = store.remember_signal(marker="watcher:repo:memo", epoch=1, actor_id="qa")

    assert signal.marker == "watcher:repo:memo"
    assert [row.marker for row in store.list_signals()] == ["watcher:repo:memo"]


def test_read_snapshot_backfills_every_missing_section(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    store.set_focus(project="memo", summary="seed the snapshot")
    for section in ("signals", "outcomes", "conflicts", "attention", "handoffs"):
        _strip_section(store, section)

    state = store.state()

    for section in ("signals", "outcomes", "conflicts", "attention", "handoffs", "focus"):
        assert section in state, f"{section} was not backfilled"
    # The surviving section keeps its data — backfill must not reset the file.
    assert "memo" in state["focus"]
