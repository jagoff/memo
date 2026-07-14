"""Dream graduation: grounded or corroborated _uncertain captures get untagged."""

from __future__ import annotations


def test_grounded_candidate_is_promoted(mock_memory, tmp_cfg, monkeypatch):
    from memo import dream_graduate

    rec = mock_memory.save(
        content="insight auto-capturado sobre el pipeline de release " * 2,
        title="Quarantined",
        tags=["_uncertain"],
    )
    monkeypatch.setattr(dream_graduate, "_grounded_ids", lambda state_dir: {rec.id[:8]})
    out = dream_graduate.run_graduation(tmp_cfg, mock_memory)
    assert [p["why"] for p in out["promoted"]] == ["grounded"]
    assert "_uncertain" not in mock_memory.get(rec.id).tags


def test_corroborated_candidate_is_promoted(mock_memory, tmp_cfg, monkeypatch):
    from memo import dream_graduate

    mock_memory.save(
        content="insight repetido en varias sesiones sobre mypy cache " * 2,
        title="Corroborated",
        tags=["_uncertain"],
    )
    monkeypatch.setattr(dream_graduate, "_grounded_ids", lambda state_dir: set())
    monkeypatch.setattr(dream_graduate, "_support_count", lambda mem, id_: 3)
    out = dream_graduate.run_graduation(tmp_cfg, mock_memory, min_support=2)
    assert [p["why"] for p in out["promoted"]] == ["corroborated"]


def test_unproven_candidate_stays_quarantined_and_dry_run_writes_nothing(
    mock_memory, tmp_cfg, monkeypatch
):
    from memo import dream_graduate

    rec = mock_memory.save(
        content="insight sin evidencia de uso sobre nada en particular " * 2,
        title="Unproven",
        tags=["_uncertain"],
    )
    monkeypatch.setattr(dream_graduate, "_grounded_ids", lambda state_dir: set())
    out = dream_graduate.run_graduation(tmp_cfg, mock_memory)
    assert out["promoted"] == []

    monkeypatch.setattr(dream_graduate, "_grounded_ids", lambda state_dir: {rec.id[:8]})
    out2 = dream_graduate.run_graduation(tmp_cfg, mock_memory, dry_run=True)
    assert len(out2["promoted"]) == 1
    assert "_uncertain" in mock_memory.get(rec.id).tags  # dry-run untouched


def test_support_count_returns_zero_when_column_absent(mock_memory):
    from memo.dream_graduate import _support_count

    assert _support_count(mock_memory, "deadbeef" * 4) == 0
