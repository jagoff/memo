"""GC-11 Memory Passport — a versioned, vendor-neutral export of the durable
corpus with semantic fidelity (type, tags, provenance + verification state via
the ``extra`` bag), validated on import.

Fidelity contract (honest v1): the passport carries the *canonical* record —
what markdown is the source of truth for. Derived indexes (embeddings, graph
relations) are rebuilt by the receiving store; ids/updated are regenerated on
import. Provenance/verification live in ``extra`` and DO round-trip.

Pure functions take plain dicts / attribute objects, so they unit-test with no
Memory. The Exporter/Importer wire the real store.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memo.errors import ValidationError
from memo.passport import (
    PASSPORT_SCHEMA,
    build_passport,
    entry_from_record,
    normalize_for_import,
    validate_passport,
)


def _rec() -> SimpleNamespace:
    return SimpleNamespace(
        id="abcd1234ef",
        type="decision",
        title="int8 is the default",
        body="We ship int8 vec quantization by default.",
        tags=["project:memo"],
        created="2026-07-01T00:00:00Z",
        updated="2026-07-20T00:00:00Z",
        extra={"verification_state": "verified", "synthesis_sources": ["x", "y"]},
    )


# --- entry_from_record (pure) ------------------------------------------------


def test_entry_captures_canonical_fields_and_extra() -> None:
    e = entry_from_record(_rec())
    assert e["id"] == "abcd1234ef"
    assert e["type"] == "decision"
    assert e["tags"] == ["project:memo"]
    assert e["extra"]["verification_state"] == "verified"  # provenance survives


def test_entry_tolerates_missing_extra() -> None:
    rec = _rec()
    rec.extra = {}
    assert entry_from_record(rec)["extra"] == {}


# --- build_passport (pure envelope) ------------------------------------------


def test_build_wraps_entries_in_versioned_envelope() -> None:
    obj = build_passport(
        [entry_from_record(_rec())], generator="memo/3.10.0", exported_at="2026-07-22T00:00:00Z"
    )
    assert obj["schema"] == PASSPORT_SCHEMA
    assert obj["generator"] == "memo/3.10.0"
    assert obj["exported_at"] == "2026-07-22T00:00:00Z"
    assert obj["count"] == 1
    assert isinstance(obj["memories"], list) and len(obj["memories"]) == 1


# --- validate_passport -------------------------------------------------------


def test_validate_accepts_a_well_formed_passport() -> None:
    obj = build_passport([entry_from_record(_rec())], generator="memo", exported_at="t")
    validate_passport(obj)  # no raise


def test_validate_rejects_wrong_schema() -> None:
    with pytest.raises(ValidationError):
        validate_passport({"schema": "notmemo.v9", "memories": []})


def test_validate_rejects_non_dict_or_missing_memories() -> None:
    with pytest.raises(ValidationError):
        validate_passport(["not", "a", "dict"])
    with pytest.raises(ValidationError):
        validate_passport({"schema": PASSPORT_SCHEMA})  # no memories key
    with pytest.raises(ValidationError):
        validate_passport({"schema": PASSPORT_SCHEMA, "memories": "notalist"})


def test_validate_rejects_non_dict_entry() -> None:
    with pytest.raises(ValidationError):
        validate_passport({"schema": PASSPORT_SCHEMA, "memories": [42]})


# --- normalize_for_import (→ save kwargs shape) ------------------------------


def test_normalize_maps_body_to_content_and_preserves_fidelity() -> None:
    e = entry_from_record(_rec())
    norm = normalize_for_import(e)
    assert norm["content"] == "We ship int8 vec quantization by default."
    assert norm["type"] == "decision"
    assert norm["tags"] == ["project:memo"]
    assert norm["created"] == "2026-07-01T00:00:00Z"
    assert norm["extra"]["verification_state"] == "verified"


def test_round_trip_build_validate_normalize_preserves_canonical() -> None:
    original = _rec()
    obj = build_passport([entry_from_record(original)], generator="memo", exported_at="t")
    validate_passport(obj)
    norm = normalize_for_import(obj["memories"][0])
    assert norm["content"] == original.body
    assert norm["type"] == original.type
    assert norm["tags"] == original.tags
    assert norm["extra"] == original.extra


# --- Exporter/Importer round-trip (fake store, no MLX) -----------------------


def test_exporter_importer_round_trip_via_fake_store(tmp_path) -> None:
    from memo.import_export import Exporter, Importer

    rec = _rec()
    export_mem = SimpleNamespace(list=lambda limit: [rec], store=None)
    path = tmp_path / "brain.passport"

    result = Exporter(export_mem).export_passport(path)
    assert result.exported_count == 1 and result.format == "passport"
    assert path.is_file()

    saved: list[dict] = []
    import_mem = SimpleNamespace(save=lambda **kw: saved.append(kw) or SimpleNamespace(id="new"))
    ires = Importer(import_mem).import_passport(path)

    assert ires.imported_count == 1
    assert saved[0]["content"] == rec.body
    assert saved[0]["type_"] == "decision"  # import_records saves via type_=
    assert saved[0]["extra"]["verification_state"] == "verified"  # provenance preserved


def test_import_from_rejects_malformed_passport(tmp_path) -> None:
    from memo.errors import ValidationError
    from memo.import_export import ImportExportManager

    bad = tmp_path / "bad.passport"
    bad.write_text('{"schema": "wrong", "memories": []}', encoding="utf-8")
    mgr = ImportExportManager(SimpleNamespace(save=lambda **kw: None))
    with pytest.raises(ValidationError):
        mgr.import_from(bad, "passport")
