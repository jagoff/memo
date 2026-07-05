"""Recency band: newest durables union into the pool at the min_sim floor."""
from __future__ import annotations

from types import SimpleNamespace


def _row(id_: str, title: str) -> dict:
    return {
        "id": id_ * 4, "path": f"2026/07/{id_}.md", "title": title, "type": "note",
        "tags": [], "created": "2026-07-02T10:00:00", "updated": "2026-07-02T10:00:00",
        "body_hash": "", "extra": {},
    }


def test_fetch_recency_band_scores_at_floor():
    from memo.recall_logic import fetch_recency_band

    store = SimpleNamespace(list_recent=lambda **kw: [_row("aaaa1111", "Fresh")])
    mem = SimpleNamespace(store=store, _read_body=lambda p: "cuerpo fresco " * 10)
    band = fetch_recency_band(mem, days=7, exclude_types={"reference"}, floor=0.5)
    assert len(band) == 1
    assert band[0].score == 0.5
    assert band[0].body.startswith("cuerpo fresco")


def test_apply_recency_band_dedups_and_appends():
    from memo.recall_logic import apply_recency_band

    hit = SimpleNamespace(id="aaaa1111" * 4, score=0.9)
    dup = SimpleNamespace(id="aaaa1111" * 4, score=0.5)
    new = SimpleNamespace(id="bbbb2222" * 4, score=0.5)
    out = apply_recency_band([hit], [dup, new])
    assert [h.id for h in out] == [hit.id, new.id]
    assert out[0].score == 0.9  # semantic hit keeps its rank position


def test_fetch_recency_band_never_raises():
    from memo.recall_logic import fetch_recency_band

    def _boom(**kw):
        raise RuntimeError("db gone")

    mem = SimpleNamespace(store=SimpleNamespace(list_recent=_boom), _read_body=lambda p: "")
    assert fetch_recency_band(mem, days=7, exclude_types=None, floor=0.5) == []
