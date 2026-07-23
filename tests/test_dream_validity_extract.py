"""Dream validity-extract pass (Task 10) — gated, LLM off the hot path.

Covers:
- the completeness gate (the flag declares a graduation gate in dream_flags.GATES);
- the pass with a STUBBED analyzer (no real MLX): a stated-window fact gets its
  interval set, a no-window fact is untouched, and a stubbed LLM error lands in
  ``receipt["errors"]`` (never silently swallowed);
- the ``TemporalAnalyzer.extract_validity_window`` anti-hallucination guard
  (a date whose year is absent from the note text is rejected).
"""

from __future__ import annotations

from types import SimpleNamespace

from memo.cli_dream_passes import _run_validity_extract

# --- completeness gate (RED first) -------------------------------------------


def test_validity_extract_flag_declares_a_gate():
    """The dark flag MUST declare a graduation gate — CI-enforced by
    test_dream_flags.test_every_dark_flag_has_a_gate."""
    from memo.dream_flags import GATES

    assert "MEMO_DREAM_VALIDITY_EXTRACT_ENABLED" in GATES


# --- the pass (stubbed analyzer; no MLX) -------------------------------------


def _fresh_receipt() -> dict:
    return {"errors": []}


def test_pass_sets_interval_only_for_the_stated_window_record(mock_memory, monkeypatch):
    a = mock_memory.save(
        content="Our AWS reserved-instance contract runs through 2026-09-30.", type_="fact"
    )
    b = mock_memory.save(content="We prefer tabs over spaces in Python.", type_="fact")

    def fake_extract(record):
        if "contract runs through" in (record.body or ""):
            return {"invalid_at": "2026-09-30T00:00:00"}
        return None

    monkeypatch.setattr(mock_memory.temporal, "extract_validity_window", fake_extract)

    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)

    ga = mock_memory.get(a.id)
    gb = mock_memory.get(b.id)
    # Stated-window record: invalid_at closed at the stated date, valid_at kept.
    assert ga.invalid_at == "2026-09-30T00:00:00"
    assert ga.valid_at == a.valid_at
    # No-window record: fully untouched.
    assert gb.invalid_at is None
    assert gb.valid_at == b.valid_at
    # Receipt records the update and no errors.
    assert not receipt["errors"]
    updated_ids = {u["id"] for u in receipt["validity_extract"]["updated"]}
    assert a.id in updated_ids and b.id not in updated_ids


def test_pass_mirrors_interval_to_frontmatter(mock_memory, monkeypatch):
    a = mock_memory.save(content="License valid until 2026-12-31 per the vendor.", type_="fact")

    monkeypatch.setattr(
        mock_memory.temporal,
        "extract_validity_window",
        lambda record: {"invalid_at": "2026-12-31T00:00:00"},
    )
    _run_validity_extract(mock_memory, _fresh_receipt())

    md = mock_memory._resolve_existing(mock_memory.get(a.id).path).read_text()
    assert "invalid_at:" in md and "2026-12-31" in md


def test_stubbed_llm_error_lands_in_receipt_errors(mock_memory, monkeypatch):
    a = mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")

    def boom(record):
        raise RuntimeError("mlx exploded")

    monkeypatch.setattr(mock_memory.temporal, "extract_validity_window", boom)

    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)

    assert any("validity_extract" in e and "mlx exploded" in e for e in receipt["errors"])
    # Not swallowed but also not fatal: the record is left untouched.
    assert mock_memory.get(a.id).invalid_at is None


def test_dry_run_writes_nothing(mock_memory, monkeypatch):
    a = mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")
    monkeypatch.setattr(
        mock_memory.temporal,
        "extract_validity_window",
        lambda record: {"invalid_at": "2026-09-30T00:00:00"},
    )
    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt, dry_run=True)
    assert mock_memory.get(a.id).invalid_at is None
    # Still reported as a would-update.
    assert receipt["validity_extract"]["updated"]


# --- extractor anti-hallucination guard (fake chat; no MLX) ------------------


class _FakeChatJSON:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(self, model, messages, options=None):
        return {"message": {"content": self._content}}


def _record(body: str) -> SimpleNamespace:
    return SimpleNamespace(id="deadbeef", title="t", type="fact", body=body)


def test_extract_rejects_date_whose_year_is_absent_from_text(mock_memory):
    # Note text never mentions 2030 → a returned 2030 date is a hallucination.
    mock_memory._chat = _FakeChatJSON('{"valid_at": null, "invalid_at": "2030-12-31"}')
    got = mock_memory.temporal.extract_validity_window(
        _record("The contract runs through the end of the quarter.")
    )
    assert got is None


def test_extract_keeps_date_whose_year_is_in_text(mock_memory):
    mock_memory._chat = _FakeChatJSON('{"valid_at": null, "invalid_at": "2026-09-30"}')
    got = mock_memory.temporal.extract_validity_window(_record("Contract runs through 2026-09-30."))
    assert got == {"invalid_at": "2026-09-30T00:00:00"}


def test_extract_returns_none_when_no_window_stated(mock_memory):
    mock_memory._chat = _FakeChatJSON('{"valid_at": null, "invalid_at": null}')
    got = mock_memory.temporal.extract_validity_window(_record("We use Python here."))
    assert got is None
