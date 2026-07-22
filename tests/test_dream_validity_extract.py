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

import memo.cli_dream_passes as passes
from memo.cli_dream_passes import _mirror_validity_to_markdown, _run_validity_extract

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


# --- extract_validity_window: remaining parse/guard branches -----------------


class _FakeChatNone:
    """Chat double whose call returns None → chat_with_timeout yields None."""

    def chat(self, model, messages, options=None):
        return None


def test_extract_empty_body_returns_none_without_calling_llm(mock_memory):
    # Whitespace-only body short-circuits before the LLM is ever consulted.
    mock_memory._chat = _FakeChatNone()
    assert mock_memory.temporal.extract_validity_window(_record("   \n\t")) is None


def test_extract_none_chat_output_returns_none(mock_memory):
    mock_memory._chat = _FakeChatNone()
    assert mock_memory.temporal.extract_validity_window(_record("Valid until 2026.")) is None


def test_extract_strips_markdown_fences(mock_memory):
    mock_memory._chat = _FakeChatJSON(
        '```json\n{"valid_at": null, "invalid_at": "2026-09-30"}\n```'
    )
    got = mock_memory.temporal.extract_validity_window(_record("Contract ends 2026-09-30."))
    assert got == {"invalid_at": "2026-09-30T00:00:00"}


def test_extract_malformed_json_returns_none(mock_memory):
    mock_memory._chat = _FakeChatJSON("not json at all {{{")
    assert mock_memory.temporal.extract_validity_window(_record("Ends 2026-09-30.")) is None


def test_extract_non_dict_json_returns_none(mock_memory):
    mock_memory._chat = _FakeChatJSON("[1, 2, 3]")
    assert mock_memory.temporal.extract_validity_window(_record("Ends 2026-09-30.")) is None


def test_extract_skips_unparseable_date_value(mock_memory):
    # A bare year never normalizes to an ISO datetime → the field is dropped and,
    # with no other boundary, the whole extraction returns None.
    mock_memory._chat = _FakeChatJSON('{"valid_at": "2026", "invalid_at": null}')
    assert mock_memory.temporal.extract_validity_window(_record("Since 2026 we ship.")) is None


# --- _normalize_extracted_date: direct unit coverage -------------------------


def test_normalize_extracted_date_rejects_bare_year():
    from memo.temporal import _normalize_extracted_date

    assert _normalize_extracted_date("2026") is None
    assert _normalize_extracted_date("not-a-date") is None


def test_normalize_extracted_date_folds_offset_to_naive_utc():
    from memo.temporal import _normalize_extracted_date

    # A full ISO datetime with a +02:00 offset is converted instant-preservingly
    # to naive UTC (12:00+02:00 → 10:00).
    assert _normalize_extracted_date("2026-09-30T12:00:00+02:00") == "2026-09-30T10:00:00"


# --- _mirror_validity_to_markdown: direct unit coverage ----------------------


def test_mirror_returns_when_source_not_a_file(mock_memory):
    # A record whose path doesn't resolve to a real file is a silent no-op.
    ghost = SimpleNamespace(path="no/such/bucket/ghost.md")
    _mirror_validity_to_markdown(
        mock_memory, ghost, valid_at="2026-01-01T00:00:00", invalid_at=None
    )  # must not raise


def test_mirror_writes_valid_at_to_frontmatter(mock_memory):
    rec = mock_memory.save(content="A dated fact.", type_="fact")
    _mirror_validity_to_markdown(mock_memory, rec, valid_at="2026-01-01T00:00:00", invalid_at=None)
    md = mock_memory._resolve_existing(rec.path).read_text(encoding="utf-8")
    assert "valid_at:" in md and "2026-01-01" in md


# --- _run_validity_extract: remaining loop/error branches --------------------


def test_pass_skips_when_record_vanished(mock_memory, monkeypatch):
    mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")
    # get() returns None for the selected row (concurrent delete) → skipped.
    monkeypatch.setattr(mock_memory, "get", lambda *a, **k: None)
    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)
    assert receipt["validity_extract"]["scanned"] == 0
    assert receipt["validity_extract"]["updated"] == []
    assert receipt["errors"] == []


def test_pass_noop_when_extraction_reproduces_stored_window(mock_memory, monkeypatch):
    a = mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")
    # Extraction returns exactly the already-stored valid_at → no-op, not an update.
    monkeypatch.setattr(
        mock_memory.temporal,
        "extract_validity_window",
        lambda record: {"valid_at": record.valid_at},
    )
    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)
    assert receipt["validity_extract"]["updated"] == []
    assert receipt["errors"] == []
    assert mock_memory.get(a.id).invalid_at is None


def test_pass_frontmatter_mirror_failure_lands_in_receipt(mock_memory, monkeypatch):
    a = mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")
    monkeypatch.setattr(
        mock_memory.temporal,
        "extract_validity_window",
        lambda record: {"invalid_at": "2027-01-01T00:00:00"},
    )

    def boom(*a, **k):
        raise RuntimeError("mirror kaput")

    monkeypatch.setattr(passes, "_mirror_validity_to_markdown", boom)

    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)

    # Index write still happened; the mirror failure is surfaced, not fatal.
    assert mock_memory.get(a.id).invalid_at == "2027-01-01T00:00:00"
    assert any("frontmatter mirror" in e for e in receipt["errors"])
    assert receipt["validity_extract"]["updated"]


def test_pass_whole_pass_failure_lands_in_receipt(mock_memory, monkeypatch):
    mock_memory.save(content="Contract runs through 2026-09-30.", type_="fact")

    def boom(*a, **k):
        raise RuntimeError("store exploded")

    # get() raises OUTSIDE the per-record try → the whole-pass guard catches it.
    monkeypatch.setattr(mock_memory, "get", boom)

    receipt = _fresh_receipt()
    _run_validity_extract(mock_memory, receipt)
    assert any("validity_extract" in e and "store exploded" in e for e in receipt["errors"])
