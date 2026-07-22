"""Timezone + boundary correctness for the bi-temporal validity gate.

`_validity_filter` compares stored `valid_at`/`invalid_at`/`created` (stamped by
`record._now_iso()` in the machine's LOCAL UTC offset, ms precision) against a
bound value as TEXT, lexicographically. So the bound MUST share that same
local-offset shape or the boundary skews by the machine's UTC offset. These
tests pin that invariant machine-offset-agnostically.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from memo.memory import Memory
from memo.memory.record import _now_iso
from memo.store.bm25_queries import _validity_filter


def _now_shift(delta: timedelta) -> str:
    """A local-offset, ms-precision ISO string shifted from now — the exact
    shape memo stamps validity columns with."""
    return (datetime.fromisoformat(_now_iso()) + delta).isoformat(timespec="milliseconds")


# --- unit: the bound value's shape ------------------------------------------


def test_default_gate_binds_local_offset_matching_stamp() -> None:
    """The default now-gate binds `now` in memo's stored offset (record._now_iso),
    NOT a hardcoded +00:00 — that offset skew was the bug."""
    sql, params = _validity_filter("meta.", include_invalid=False, as_of=None)
    assert "meta.invalid_at > ?" in sql
    assert len(params) == 1
    assert params[0][-6:] == _now_iso()[-6:]  # same UTC offset as the stamp
    parsed = datetime.fromisoformat(params[0])
    assert parsed.tzinfo is not None  # aware, local-offset shape


def test_as_of_bare_date_expands_to_end_of_day() -> None:
    """A bare-date as_of means "as of the END of that day"."""
    _sql, params = _validity_filter("", include_invalid=False, as_of="2026-06-15")
    assert len(params) == 2 and params[0] == params[1]
    assert params[0].startswith("2026-06-15T23:59:59")
    assert datetime.fromisoformat(params[0]).tzinfo is not None  # local offset, not bare


def test_as_of_offset_aware_converted_to_local_offset() -> None:
    """An offset-aware as_of is converted instant-preservingly to the local
    offset so it compares correctly against the local-offset stored columns."""
    _sql, params = _validity_filter("", include_invalid=False, as_of="2026-06-15T14:00:00+00:00")
    assert params[0][-6:] == _now_iso()[-6:]  # re-expressed in local offset
    assert datetime.fromisoformat(params[0]) == datetime.fromisoformat("2026-06-15T14:00:00+00:00")


# --- e2e: offset correctness through both store seams -----------------------


def test_default_recall_offset_correct_regardless_of_machine_offset(
    mem_with_stub: Memory,
) -> None:
    """A record whose interval closes in the PAST is excluded; one closing in
    the near FUTURE stays included — and this must hold on any machine offset.
    On a negative-offset box (e.g. UTC-3) the old +00:00 bound skewed the
    future case to a false EXCLUDE."""
    a = mem_with_stub.save(content="prod db is postgres", title="A", type_="fact")
    b = mem_with_stub.save(content="prod db is mysql", title="B", type_="fact")

    mem_with_stub.store.update_validity(
        id_=a.id, valid_at=a.valid_at, invalid_at=_now_shift(timedelta(minutes=-1))
    )
    mem_with_stub.store.update_validity(
        id_=b.id, valid_at=b.valid_at, invalid_at=_now_shift(timedelta(hours=1))
    )

    emb = [1.0, 0.0, 0.0, 0.0]
    vec_ids = {r["id"] for r in mem_with_stub.store.search(emb, limit=10)}
    bm_ids = {r["id"] for r in mem_with_stub.store.search_bm25("prod db", limit=10)}

    assert a.id not in vec_ids and a.id not in bm_ids  # past-closed → hidden
    assert b.id in vec_ids and b.id in bm_ids  # future-closed → still valid


def test_as_of_bare_date_includes_same_day_later_fact(mem_with_stub: Memory) -> None:
    """`as_of="2026-06-15"` must include a fact that became valid at 14:00 that
    same day (end-of-day expansion); the prior day must exclude it."""
    a = mem_with_stub.save(content="prod db is postgres", title="A", type_="fact")
    mem_with_stub.store.update_validity(
        id_=a.id, valid_at="2026-06-15T14:00:00-03:00", invalid_at=None
    )

    inc_bm = {r["id"] for r in mem_with_stub.store.search_bm25("prod db", limit=10, as_of="2026-06-15")}
    inc_vec = {
        r["id"] for r in mem_with_stub.store.search([1.0, 0.0, 0.0, 0.0], limit=10, as_of="2026-06-15")
    }
    assert a.id in inc_bm and a.id in inc_vec  # same-day-later fact included

    exc = {r["id"] for r in mem_with_stub.store.search_bm25("prod db", limit=10, as_of="2026-06-14")}
    assert a.id not in exc  # day before → excluded
