"""Bounded, redacted import of useful Synapse feedback evidence.

Synapse state is retirement evidence, never executable input.  This module
reads only the explicitly named regular files through ``SecureDirectory`` and
reduces them to ranking signals and operator-only eval fixtures.  In
particular, it never carries a chat answer, trace body, cache, or arbitrary
metadata into Memo.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from memo.atomic_io import open_secure_directory
from memo.memory import Memory
from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.schemas import SynapseDataReceipt


class SynapseDataError(RuntimeError):
    """The narrowly allowed Synapse evidence cannot be safely imported."""


_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_LEDGER = "ledger.jsonl"
_CHAT_TRACES = "observability/chat-traces.jsonl"
_PIPELINE_TRACES = "observability/chat_pipeline_trace.jsonl"
_EVAL_CORPUS = "eval/corpus.json"
_STAGING_RELATIVE = "operator-staging/synapse-eval-fixtures.json"
_STAGING_SCHEMA = "memo.synapse_eval_staging.v1"
_RECEIPT_SCHEMA = "memo.synapse_data_receipt.v1"


@dataclass(frozen=True)
class FeedbackImport:
    feedback_id: str
    source_id: str
    query: str
    rating: str
    answer: str = ""


@dataclass(frozen=True)
class EvalFixture:
    fixture_id: str
    query: str
    source_ids: tuple[str, ...]
    content_sha256: str
    answer: str = ""


@dataclass(frozen=True)
class SynapseDataBundle:
    feedback: tuple[FeedbackImport, ...]
    eval_fixtures: tuple[EvalFixture, ...]
    input_sha256: str
    skipped_feedback_ids: tuple[str, ...] = ()


def _normal_id(value: object) -> str:
    candidate = str(value or "").strip()
    return candidate.casefold() if _ID_RE.fullmatch(candidate) else ""


def _normal_query(value: object) -> str:
    if not isinstance(value, str):
        return ""
    # Queries are an explicit ranking input; retain neither answers nor free
    # form trace/metadata fields.  A bounded whitespace-normalized value also
    # makes the feedback-store's idempotency key deterministic.
    return " ".join(value.split())[:2000]


def _read_optional(directory: Any, relative: str) -> bytes | None:
    try:
        return directory.read_bytes(relative)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SynapseDataError(f"unsafe Synapse input: {relative}") from exc


def _jsonl(encoded: bytes | None, relative: str) -> list[dict[str, Any]]:
    if encoded is None:
        return []
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SynapseDataError(f"{relative} is not UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SynapseDataError(f"malformed JSON in {relative}:{number}") from exc
        if not isinstance(row, dict):
            raise SynapseDataError(f"non-object JSON in {relative}:{number}")
        rows.append(row)
    return rows


def _json_value(encoded: bytes | None, relative: str) -> object:
    if encoded is None:
        return []
    try:
        return json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SynapseDataError(f"malformed JSON in {relative}") from exc


def _load_state(
    state_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], object, str]:
    """Read the four admitted inputs from one descriptor-secure state root."""
    try:
        with open_secure_directory(state_dir) as directory:
            ledger = _read_optional(directory, _LEDGER)
            chat_traces = _read_optional(directory, _CHAT_TRACES)
            pipeline_traces = _read_optional(directory, _PIPELINE_TRACES)
            corpus = _read_optional(directory, _EVAL_CORPUS)
    except (OSError, ValueError) as exc:
        raise SynapseDataError("Synapse state directory is unsafe or unreadable") from exc
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "ledger_sha256": hashlib.sha256(ledger or b"").hexdigest(),
                "chat_traces_sha256": hashlib.sha256(chat_traces or b"").hexdigest(),
                "pipeline_traces_sha256": hashlib.sha256(pipeline_traces or b"").hexdigest(),
                "eval_corpus_sha256": hashlib.sha256(corpus or b"").hexdigest(),
            }
        )
    ).hexdigest()
    return (
        _jsonl(ledger, _LEDGER),
        _jsonl(chat_traces, _CHAT_TRACES),
        _jsonl(pipeline_traces, _PIPELINE_TRACES),
        _json_value(corpus, _EVAL_CORPUS),
        digest,
    )


def _queries_by_trace(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in rows:
        trace_id = _normal_id(row.get("trace_id"))
        # ``query_preview`` is accepted for the pipeline trace only.  It is
        # still a request, never an answer/body, and is only used to attach an
        # explicit feedback event to its own query.
        query = _normal_query(row.get("query") or row.get("query_preview"))
        if trace_id and query:
            queries.setdefault(trace_id, query)
    return queries


def _feedback_from_rows(
    ledger: Iterable[Mapping[str, Any]],
    trace_queries: Mapping[str, str],
    seen_ids: set[str],
) -> tuple[tuple[FeedbackImport, ...], tuple[str, ...]]:
    out: list[FeedbackImport] = []
    skipped: list[str] = []
    # Callers may restore seen ids from a case-insensitive receipt.  Invalid
    # values are not admission controls and must not affect the result.
    observed = {normalized for item in seen_ids if (normalized := _normal_id(item))}
    for row in ledger:
        if row.get("action") != "chat_feedback":
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        base_id = _normal_id(metadata.get("feedback_id") or row.get("action_id"))
        rating = str(metadata.get("rating") or "").strip().casefold()
        trace_id = _normal_id(row.get("trace_id"))
        query = _normal_query(metadata.get("query")) or trace_queries.get(trace_id, "")
        sources = metadata.get("source_ids")
        if (
            not base_id
            or rating not in {"up", "down"}
            or not query
            or not isinstance(sources, list)
        ):
            continue
        normalized_sources = tuple(
            dict.fromkeys(source for source in map(_normal_id, sources) if source)
        )
        for source_id in normalized_sources:
            feedback_id = base_id if len(normalized_sources) == 1 else f"{base_id}:{source_id}"
            if base_id in observed or feedback_id in observed:
                skipped.append(feedback_id)
                continue
            observed.add(feedback_id)
            out.append(
                FeedbackImport(
                    feedback_id=feedback_id,
                    source_id=source_id,
                    query=query,
                    rating=rating,
                )
            )
    return (
        tuple(sorted(out, key=lambda item: item.feedback_id)),
        tuple(sorted(dict.fromkeys(skipped))),
    )


def _fixtures_from_value(value: object) -> tuple[EvalFixture, ...]:
    rows = value.get("fixtures", []) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise SynapseDataError("eval/corpus.json must be an array or fixtures object")
    fixtures: list[EvalFixture] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixture_id = _normal_id(row.get("id") or row.get("fixture_id"))
        query = _normal_query(row.get("question") or row.get("query"))
        source_values = row.get("expected_source_ids") or row.get("source_ids")
        if not isinstance(source_values, list):
            continue
        source_ids = tuple(
            dict.fromkeys(source for source in map(_normal_id, source_values) if source)
        )
        # A down-voted/needs-review row is not a high-signal eval fixture.
        if row.get("needs_review") is True or str(row.get("rating") or "").casefold() == "down":
            continue
        if not fixture_id or not query or not source_ids or fixture_id in seen:
            continue
        seen.add(fixture_id)
        content_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {"fixture_id": fixture_id, "query": query, "source_ids": list(source_ids)}
            )
        ).hexdigest()
        fixtures.append(
            EvalFixture(
                fixture_id=fixture_id,
                query=query,
                source_ids=source_ids,
                content_sha256=content_sha256,
            )
        )
    return tuple(sorted(fixtures, key=lambda item: item.fixture_id))


def build_synapse_data_bundle(
    state_dir: Path, seen_ids: set[str] | None = None
) -> SynapseDataBundle:
    """Extract exactly the bounded evidence surface and bind its input digest."""
    ledger, chat_traces, pipeline_traces, corpus, input_sha256 = _load_state(state_dir)
    trace_queries = _queries_by_trace((*chat_traces, *pipeline_traces))
    feedback, skipped_feedback_ids = _feedback_from_rows(ledger, trace_queries, seen_ids or set())
    return SynapseDataBundle(
        feedback=feedback,
        eval_fixtures=_fixtures_from_value(corpus),
        input_sha256=input_sha256,
        skipped_feedback_ids=skipped_feedback_ids,
    )


def extract_synapse_feedback(state_dir: Path, seen_ids: set[str]) -> tuple[FeedbackImport, ...]:
    """Return unseen explicit feedback only; chat answers are always redacted."""
    return build_synapse_data_bundle(state_dir, seen_ids).feedback


def extract_synapse_eval_fixtures(state_dir: Path) -> tuple[EvalFixture, ...]:
    """Return high-signal eval metadata only; fixture answers are always empty."""
    return build_synapse_data_bundle(state_dir).eval_fixtures


def _receipt_from_event(event: Any) -> SynapseDataReceipt | None:
    if getattr(event, "op", "") != "receipt.synapse-data":
        return None
    payload = getattr(event, "payload", {})
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or metadata.get("schema") != _RECEIPT_SCHEMA:
        return None
    try:
        skipped = _canonical_skipped_feedback_ids(metadata.get("skipped_feedback_ids", []))
        if int(metadata["feedback_skipped"]) != len(skipped):
            return None
        return SynapseDataReceipt(
            attempt_id=str(metadata["attempt_id"]),
            input_sha256=str(metadata["input_sha256"]),
            feedback_imported=int(metadata["feedback_imported"]),
            feedback_skipped=len(skipped),
            eval_fixture_count=int(metadata["eval_fixture_count"]),
            event_ids=(str(event.event_id),),
            status="applied",
            skipped_feedback_ids=skipped,
        )
    except (KeyError, TypeError, ValueError, SynapseDataError):
        return None


def _prior_receipt(memory: Memory, attempt_id: str) -> SynapseDataReceipt | None:
    try:
        events = memory.operational.ledger.validated_events()
    except AttributeError as exc:
        raise SynapseDataError("Memo operational ledger is unavailable") from exc
    for event in reversed(events):
        receipt = _receipt_from_event(event)
        if receipt is not None and receipt.attempt_id == attempt_id:
            return receipt
    return None


def _fixture_dict(fixture: EvalFixture) -> dict[str, object]:
    return {
        "fixture_id": fixture.fixture_id,
        "query": fixture.query,
        "source_ids": list(fixture.source_ids),
        "content_sha256": fixture.content_sha256,
    }


def _staging_payload(memory: Memory, data: SynapseDataBundle) -> tuple[bytes | None, bytes]:
    try:
        with open_secure_directory(memory.cfg.state_dir) as directory:
            previous = _read_optional(directory, _STAGING_RELATIVE)
    except (AttributeError, OSError, ValueError) as exc:
        raise SynapseDataError("Memo operator staging authority is unavailable") from exc
    existing: dict[str, dict[str, object]] = {}
    if previous is not None:
        try:
            payload = json.loads(previous)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SynapseDataError("existing operator staging corpus is malformed") from exc
        if not isinstance(payload, dict) or payload.get("schema") != _STAGING_SCHEMA:
            raise SynapseDataError("existing operator staging corpus has an unknown schema")
        rows = payload.get("fixtures")
        if not isinstance(rows, list):
            raise SynapseDataError("existing operator staging corpus has invalid fixtures")
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("fixture_id"), str):
                existing[row["fixture_id"]] = dict(row)
    for fixture in data.eval_fixtures:
        existing[fixture.fixture_id] = _fixture_dict(fixture)
    encoded = canonical_json_bytes(
        {
            "schema": _STAGING_SCHEMA,
            "input_sha256": data.input_sha256,
            "fixtures": [existing[key] for key in sorted(existing)],
        }
    )
    return previous, encoded


def _write_staging(memory: Memory, encoded: bytes) -> None:
    try:
        with open_secure_directory(memory.cfg.state_dir) as directory:
            directory.atomic_write_bytes(_STAGING_RELATIVE, encoded, mode=0o600)
    except (AttributeError, OSError, ValueError) as exc:
        raise SynapseDataError("could not publish operator-only eval staging corpus") from exc


def _restore_staging(memory: Memory, previous: bytes | None) -> None:
    """Best-effort compensation for a later feedback failure; never touch the vault."""
    try:
        with open_secure_directory(memory.cfg.state_dir) as directory:
            if previous is not None:
                directory.atomic_write_bytes(_STAGING_RELATIVE, previous, mode=0o600)
                return
            parent, name = directory._parent(_STAGING_RELATIVE, create=False)
            try:
                os.unlink(name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
    except (AttributeError, FileNotFoundError, OSError, ValueError):
        # The original exception is the important failure; this is only a
        # compensation attempt and must not turn into an unsafe broad cleanup.
        return


def _existing_operation_keys(memory: Memory, source_id: str) -> set[str]:
    keys: set[str] = set()
    try:
        rows = memory.feedback_list(source_id=source_id, limit=500)
    except ValueError:
        return keys
    for row in rows:
        try:
            extra = json.loads(str(row.get("extra_json") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(extra, dict) and isinstance(extra.get("synapse_operation_key"), str):
            keys.add(extra["synapse_operation_key"])
    return keys


def _feedback_operation_key_state(
    memory: Memory, feedback_id: str, operation_key: str
) -> Literal["owned", "not_owned", "unknown"]:
    """Look up one primary-keyed feedback row without a bounded global scan."""
    try:
        row = memory.store._conn.execute(
            "SELECT extra_json FROM source_feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
    except Exception:
        return "unknown"
    if row is None:
        return "unknown"
    try:
        extra = json.loads(str(row["extra_json"] or "{}"))
    except json.JSONDecodeError:
        return "unknown"
    if not isinstance(extra, dict):
        return "not_owned"
    return "owned" if extra.get("synapse_operation_key") == operation_key else "not_owned"


def _canonical_skipped_feedback_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise SynapseDataError("skipped_feedback_ids must be a list or tuple of IDs")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SynapseDataError("skipped_feedback_ids contains a non-string ID")
        feedback_id = _normal_id(value)
        if not feedback_id:
            raise SynapseDataError("skipped_feedback_ids contains an invalid ID")
        normalized.append(feedback_id)
    return tuple(sorted(dict.fromkeys(normalized)))


def _rollback_imported_feedback(memory: Memory, feedback: dict[str, str]) -> None:
    """Delete only rows proven to have been created by this failed attempt."""
    if not feedback:
        return
    try:
        with memory.store._tx() as cursor:
            removable: list[str] = []
            for feedback_id, operation_key in feedback.items():
                row = cursor.execute(
                    "SELECT extra_json FROM source_feedback WHERE id = ?", (feedback_id,)
                ).fetchone()
                if row is None:
                    continue
                try:
                    extra = json.loads(str(row["extra_json"] or "{}"))
                except json.JSONDecodeError:
                    continue
                if isinstance(extra, dict) and extra.get("synapse_operation_key") == operation_key:
                    removable.append(feedback_id)
            if not removable:
                return
            placeholders = ",".join("?" for _ in removable)
            cursor.execute(
                f"DELETE FROM source_feedback_vec WHERE feedback_id IN ({placeholders})",  # noqa: S608
                removable,
            )
            cursor.execute(
                f"DELETE FROM source_feedback WHERE id IN ({placeholders})",  # noqa: S608
                removable,
            )
    except Exception as exc:
        raise SynapseDataError("could not roll back failed Synapse feedback import") from exc


def apply_synapse_data(
    memory: Memory, data: SynapseDataBundle, *, attempt_id: str
) -> SynapseDataReceipt:
    """Apply bounded signals atomically enough to fail before feedback on bad fixtures.

    Eval fixtures are staged first, so a staging validation/write error leaves
    neither ranking feedback nor an operational receipt.  A replay is bound to
    both the attempt id and the exact source-input digest.
    """
    if not _ATTEMPT_RE.fullmatch(attempt_id):
        raise SynapseDataError("attempt id is unsafe")
    if not re.fullmatch(r"[0-9a-f]{64}", data.input_sha256):
        raise SynapseDataError("input_sha256 must be a lowercase SHA-256")
    canonical_bundle_skipped_ids = _canonical_skipped_feedback_ids(data.skipped_feedback_ids)
    prior = _prior_receipt(memory, attempt_id)
    if prior is not None:
        if prior.input_sha256 != data.input_sha256:
            raise SynapseDataError("attempt id was already used for a different input bundle")
        return replace(prior, status="reused")

    # Validate all fixture fields before *any* write.  The dataclass is public,
    # so callers are not assumed to have used the extractor.
    for fixture in data.eval_fixtures:
        if (
            not _normal_id(fixture.fixture_id)
            or not _normal_query(fixture.query)
            or not fixture.source_ids
            or fixture.answer
            or not re.fullmatch(r"[0-9a-f]{64}", fixture.content_sha256)
        ):
            raise SynapseDataError("invalid or non-redacted eval fixture")
    for feedback in data.feedback:
        if (
            not _normal_id(feedback.feedback_id)
            or not _normal_id(feedback.source_id)
            or not _normal_query(feedback.query)
            or feedback.rating not in {"up", "down"}
            or feedback.answer
        ):
            raise SynapseDataError("invalid or non-redacted feedback import")

    previous_staging, staged = _staging_payload(memory, data)
    _write_staging(memory, staged)
    imported = 0
    skipped_ids: list[str] = list(canonical_bundle_skipped_ids)
    imported_feedback: dict[str, str] = {}
    try:
        for feedback in data.feedback:
            operation_key = hashlib.sha256(
                f"{attempt_id}\\0{feedback.feedback_id}".encode()
            ).hexdigest()
            existing = _existing_operation_keys(memory, feedback.source_id)
            if operation_key in existing:
                skipped_ids.append(feedback.feedback_id)
                continue
            # Do not replace an existing human/other-import vote for the same
            # source+query.  ``only_if_absent`` is the final race-safe guard.
            try:
                rows = memory.feedback_list(source_id=feedback.source_id, limit=500)
            except ValueError:
                skipped_ids.append(feedback.feedback_id)
                continue
            if any(str(row.get("query_text") or "") == feedback.query for row in rows):
                skipped_ids.append(feedback.feedback_id)
                continue
            # Supply the stable key as the stored feedback id too.  It makes
            # compensation safe even for an implementation that persists then
            # raises before returning a result.
            imported_feedback[operation_key] = operation_key
            result = memory.feedback_record(
                feedback.source_id,
                query_text=feedback.query,
                rating=feedback.rating,
                feedback_id=operation_key,
                only_if_absent=True,
                extra={
                    "origin": "synapse_data_import",
                    "synapse_operation_key": operation_key,
                },
            )
            result_id = str(result.get("feedback_id") or "")
            ownership = _feedback_operation_key_state(memory, operation_key, operation_key)
            if ownership == "not_owned":
                imported_feedback.pop(operation_key, None)
                skipped_ids.append(feedback.feedback_id)
                continue
            if ownership == "unknown":
                # Keep the candidate in ``imported_feedback`` so the exception
                # path performs an exact compensating lookup/delete.  Writing
                # a receipt while ownership is uncertain could orphan a signal.
                raise SynapseDataError("cannot verify imported Synapse feedback ownership")
            if result_id != operation_key:
                # The row is ours despite an incoherent return value: retain it
                # and count it as imported, so the receipt remains its witness.
                pass
            imported += 1
        # Dynamic skips (duplicate operation, missing source, pre-existing
        # query) must pass through the same casefold/validation policy as
        # caller-provided skips before they reach the receipt.
        canonical_skipped_ids = _canonical_skipped_feedback_ids(skipped_ids)
        metadata = {
            "schema": _RECEIPT_SCHEMA,
            "attempt_id": attempt_id,
            "input_sha256": data.input_sha256,
            "feedback_imported": imported,
            "feedback_skipped": len(canonical_skipped_ids),
            "skipped_feedback_ids": list(canonical_skipped_ids),
            "eval_fixture_count": len(data.eval_fixtures),
        }
        event = memory.operational.receipt(
            "synapse-data",
            subject_uri=f"memo://synapse-data/{attempt_id}",
            actor_id="memo-synapse-data-import",
            metadata=metadata,
        )
    except Exception:
        try:
            _rollback_imported_feedback(memory, imported_feedback)
        finally:
            _restore_staging(memory, previous_staging)
        raise
    return SynapseDataReceipt(
        attempt_id=attempt_id,
        input_sha256=data.input_sha256,
        feedback_imported=imported,
        feedback_skipped=len(canonical_skipped_ids),
        eval_fixture_count=len(data.eval_fixtures),
        event_ids=(event.receipt_id,),
        status="applied",
        skipped_feedback_ids=canonical_skipped_ids,
    )


__all__ = [
    "EvalFixture",
    "FeedbackImport",
    "SynapseDataBundle",
    "SynapseDataError",
    "apply_synapse_data",
    "build_synapse_data_bundle",
    "extract_synapse_eval_fixtures",
    "extract_synapse_feedback",
]
