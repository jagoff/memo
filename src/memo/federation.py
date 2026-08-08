"""Signed, ACL-aware federation bundles for independent Memo installations."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text
from memo.contracts import (
    MEMO_FEDERATION_SCHEMA,
    ActorIdentity,
    TrustTier,
    Visibility,
)
from memo.errors import FederationError, MemoError
from memo.util import utc_now_iso

_MAX_BUNDLE_BYTES = 100 * 1024 * 1024
_MAX_RECORDS = 10_000
_EXTRA_KEYS = {
    "owner_principal",
    "principals",
    "provenance",
    "trust_tier",
    "visibility",
}
_REMOTE_CLAIM_KEYS = {"learning", "outcome_stats", "priority"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _derived_bundle_id(bundle: dict[str, Any]) -> str:
    identity = dict(bundle)
    identity.pop("bundle_id", None)
    identity.pop("signature", None)
    return "bundle-" + _digest(identity)[:24]


def _key_bytes(key: bytes | str) -> bytes:
    raw = key if isinstance(key, bytes) else key.encode("utf-8")
    if len(raw) < 16:
        raise FederationError("federation signing key must be at least 16 bytes")
    return raw


def _principals(extra: dict[str, Any]) -> set[str]:
    raw = extra.get("principals")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(value).strip() for value in raw if str(value).strip()}


def visible_to(
    extra: dict[str, Any] | None,
    *,
    principal: str,
    owner_principal: str,
) -> bool:
    """Apply Memo's deny-by-default federation visibility policy."""
    metadata = dict(extra or {})
    try:
        visibility = Visibility(str(metadata.get("visibility") or Visibility.OWNER.value))
    except ValueError:
        return False
    if visibility is Visibility.LOCAL_ONLY:
        return False
    record_owner = str(metadata.get("owner_principal") or owner_principal)
    if principal == record_owner:
        return True
    if visibility is not Visibility.SHARED:
        return False
    allowed = _principals(metadata)
    return principal in allowed or "*" in allowed


def _read_bundle(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_BUNDLE_BYTES:
            raise FederationError("federation bundle exceeds the 100 MiB safety limit")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FederationError:
        raise
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise FederationError(f"cannot read federation bundle: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != MEMO_FEDERATION_SCHEMA:
        raise FederationError("unsupported federation bundle schema")
    return raw


def _verify_signature(bundle: dict[str, Any], key: bytes) -> None:
    signature = bundle.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "hmac-sha256":
        raise FederationError("missing or unsupported federation signature")
    unsigned = dict(bundle)
    unsigned.pop("signature", None)
    expected = hmac.new(key, _canonical(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature.get("digest") or ""), expected):
        raise FederationError("federation signature mismatch")


def _verify_manifest(
    memories: list[Any],
    operations: list[Any],
    manifest: Any,
) -> None:
    if len(memories) > _MAX_RECORDS:
        raise FederationError(f"bundle exceeds {_MAX_RECORDS} records")
    if not isinstance(manifest, dict):
        raise FederationError("federation manifest is missing")
    try:
        counts_match = int(manifest.get("memory_count", -1)) == len(memories) and int(
            manifest.get("operation_count", -1)
        ) == len(operations)
    except (ValueError, TypeError) as exc:
        raise FederationError("federation manifest count is malformed") from exc
    digests_match = manifest.get("memories_sha256") == _digest(memories) and manifest.get(
        "operations_sha256"
    ) == _digest(operations)
    if not counts_match or not digests_match:
        raise FederationError("federation manifest mismatch")


def _verify_entries(memories: list[Any], operations: list[Any]) -> None:
    source_ids = [str(row.get("source_id") or "") for row in memories if isinstance(row, dict)]
    if len(source_ids) != len(memories) or any(not source_id for source_id in source_ids):
        raise FederationError("federation memory entry is malformed")
    if len(set(source_ids)) != len(source_ids):
        raise FederationError("federation bundle contains duplicate source ids")
    if any(not isinstance(row, dict) for row in operations):
        raise FederationError("federation operation entry is malformed")


def _imported_extra(
    incoming: Any,
    *,
    bundle: dict[str, Any],
    source_id: str,
    principal: str,
    trust_peer: bool,
) -> dict[str, Any]:
    incoming_extra = incoming if isinstance(incoming, dict) else {}
    extra = {key: incoming_extra[key] for key in _EXTRA_KEYS if key in incoming_extra}
    remote_claims = {
        key: incoming_extra[key] for key in _REMOTE_CLAIM_KEYS if key in incoming_extra
    }
    extra["trust_tier"] = (
        TrustTier.TOOL_OBSERVED.value if trust_peer else TrustTier.EXTERNAL_UNTRUSTED.value
    )
    provenance = dict(extra.get("provenance") or {})
    provenance.update(
        {
            "source_client": "memo-federation",
            "route_reason": "signed_bundle_import",
            "evidence_uris": [f"memo://federation/{bundle['source_device']}/{source_id}"],
        }
    )
    extra["provenance"] = provenance
    extra["federation"] = {
        "bundle_id": bundle["bundle_id"],
        "source_device": bundle["source_device"],
        "source_id": source_id,
        "principal": principal,
        "imported_at": utc_now_iso(),
        "signature_key_id": bundle["signature"]["key_id"],
    }
    if remote_claims:
        extra["federation"]["remote_claims"] = remote_claims
    return extra


class FederationManager:
    """Export and import authenticated subsets of a Markdown memory corpus."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def _records(self, *, with_bodies: bool = True) -> list[Any]:
        """Whole-corpus sweep. `with_bodies=False` for the metadata-only legs
        (preview, the already-applied receipt set) — reading and YAML-parsing
        every body just to drop it costs ~44s on a 10k-row corpus."""
        total = max(0, int(self.memory.store.count()))
        return self.memory.list(
            limit=max(1, min(total + 1, _MAX_RECORDS + 1)), with_bodies=with_bodies
        )

    def preview(
        self,
        *,
        principal: str,
        owner_principal: str | None = None,
    ) -> dict[str, Any]:
        owner = (owner_principal or self.memory.cfg.device_id).strip()
        target = principal.strip()
        if not target or not owner:
            raise FederationError("principal and owner_principal cannot be empty")
        visible = [
            {
                "id": record.id,
                "title": record.title,
                "type": record.type,
                "visibility": str((record.extra or {}).get("visibility") or Visibility.OWNER.value),
            }
            for record in self._records(with_bodies=False)
            if record.type != "secret"
            and visible_to(record.extra, principal=target, owner_principal=owner)
        ]
        return {
            "schema": MEMO_FEDERATION_SCHEMA,
            "principal": target,
            "owner_principal": owner,
            "count": len(visible),
            "memories": visible,
            "includes_journal": target == owner,
        }

    def export_bundle(
        self,
        output_path: Path,
        *,
        principal: str,
        signing_key: bytes | str,
        owner_principal: str | None = None,
    ) -> dict[str, Any]:
        key = _key_bytes(signing_key)
        owner = (owner_principal or self.memory.cfg.device_id).strip()
        target = principal.strip()
        if not target or not owner:
            raise FederationError("principal and owner_principal cannot be empty")

        memories: list[dict[str, Any]] = []
        for record in self._records():
            if record.type == "secret":
                continue
            extra = dict(record.extra or {})
            if not visible_to(extra, principal=target, owner_principal=owner):
                continue
            safe_extra = {key: extra[key] for key in _EXTRA_KEYS if key in extra}
            memories.append(
                {
                    "source_id": record.id,
                    "title": record.title,
                    "body": str(record.body or ""),
                    "type": record.type,
                    "tags": list(record.tags),
                    "created": record.created,
                    "updated": record.updated,
                    "extra": safe_extra,
                }
            )
        if len(memories) > _MAX_RECORDS:
            raise FederationError(f"bundle exceeds {_MAX_RECORDS} records")

        # Operational journals contain owner-only continuity. Shared recipients
        # get no journal rows; owners get complete device chains so hashes remain
        # independently verifiable and causality is not rewritten.
        operations = (
            [event.to_dict() for event in self.memory.operational.ledger.validated_events()]
            if target == owner
            else []
        )
        created_at = utc_now_iso()
        body: dict[str, Any] = {
            "schema": MEMO_FEDERATION_SCHEMA,
            "source_device": self.memory.cfg.device_id,
            "principal": target,
            "owner_principal": owner,
            "created_at": created_at,
            "memories": memories,
            "operations": operations,
            "manifest": {
                "memory_count": len(memories),
                "operation_count": len(operations),
                "memories_sha256": _digest(memories),
                "operations_sha256": _digest(operations),
            },
        }
        body["bundle_id"] = "bundle-" + _digest(body)[:24]
        signature = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
        bundle = {
            **body,
            "signature": {
                "algorithm": "hmac-sha256",
                "key_id": hashlib.sha256(key).hexdigest()[:12],
                "digest": signature,
            },
        }
        encoded = json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True)
        if len(encoded.encode("utf-8")) > _MAX_BUNDLE_BYTES:
            raise FederationError("federation bundle exceeds the 100 MiB safety limit")
        atomic_write_text(Path(output_path), encoded)
        return {
            "schema": MEMO_FEDERATION_SCHEMA,
            "bundle_id": body["bundle_id"],
            "output_path": str(output_path),
            "principal": target,
            "memories": len(memories),
            "operations": len(operations),
            "key_id": bundle["signature"]["key_id"],
        }

    @staticmethod
    def verify_bundle(
        input_path: Path,
        *,
        signing_key: bytes | str,
        principal: str | None = None,
    ) -> dict[str, Any]:
        key = _key_bytes(signing_key)
        bundle = _read_bundle(Path(input_path))
        _verify_signature(bundle, key)
        if principal is not None and bundle.get("principal") != principal:
            raise FederationError("federation bundle principal mismatch")
        memories = bundle.get("memories")
        operations = bundle.get("operations")
        if not isinstance(memories, list) or not isinstance(operations, list):
            raise FederationError("federation bundle payload must contain lists")
        _verify_manifest(memories, operations, bundle.get("manifest"))
        _verify_entries(memories, operations)
        if bundle.get("bundle_id") != _derived_bundle_id(bundle):
            raise FederationError("federation bundle id mismatch")
        return bundle

    def import_bundle(
        self,
        input_path: Path,
        *,
        signing_key: bytes | str,
        principal: str,
        trust_peer: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        bundle = self.verify_bundle(
            input_path,
            signing_key=signing_key,
            principal=principal,
        )
        memories = bundle["memories"]
        operations = bundle["operations"]
        self.memory.operational.ledger.validate_import_events(operations)
        receipt_exists = any(
            event.op == "receipt.federation.import"
            and (event.payload.get("metadata") or {}).get("bundle_id") == bundle["bundle_id"]
            for event in self.memory.operational.ledger.validated_events()
        )
        if dry_run:
            return {
                "schema": MEMO_FEDERATION_SCHEMA,
                "bundle_id": bundle["bundle_id"],
                "verified": True,
                "dry_run": True,
                "memories": len(memories),
                "operations": len(operations),
            }
        if receipt_exists:
            journal = self.memory.operational.ledger.import_events(operations)
            return {
                "schema": MEMO_FEDERATION_SCHEMA,
                "bundle_id": bundle["bundle_id"],
                "verified": True,
                "dry_run": False,
                "imported": 0,
                "revised": 0,
                "unchanged": len(memories),
                "failed": 0,
                "errors": [],
                "journal": journal,
                "idempotent_replay": True,
            }

        imported = 0
        revised = 0
        unchanged = 0
        errors: list[str] = []
        applied = {
            (
                str(federation.get("bundle_id") or ""),
                str(federation.get("source_id") or ""),
            )
            for record in self._records(with_bodies=False)
            if isinstance((record.extra or {}).get("federation"), dict)
            for federation in [(record.extra or {})["federation"]]
        }
        for item in memories:
            receipt_key = (str(bundle["bundle_id"]), str(item["source_id"]))
            if receipt_key in applied:
                unchanged += 1
                continue
            try:
                extra = _imported_extra(
                    item.get("extra"),
                    bundle=bundle,
                    source_id=str(item["source_id"]),
                    principal=principal,
                    trust_peer=trust_peer,
                )
                record = self.memory.save(
                    content=str(item.get("body") or ""),
                    title=str(item.get("title") or "untitled"),
                    type_=str(item.get("type") or "note"),
                    tags=[str(tag) for tag in item.get("tags") or ()],
                    created=str(item.get("created") or "") or None,
                    extra=extra,
                    topic_key=f"federation/{bundle['source_device']}/{item['source_id']}",
                    auto_project=False,
                    actor=ActorIdentity(
                        actor_id=f"federation:{bundle['source_device']}",
                        actor_kind="tool",
                        signature=str(bundle["signature"]["key_id"]),
                        source_client="memo-federation",
                    ),
                )
                if record.action == "revised":
                    revised += 1
                else:
                    imported += 1
            except (MemoError, OSError, TypeError, ValueError) as exc:
                errors.append(f"{str(item.get('source_id') or '')[:12]}: {exc}")
        journal = self.memory.operational.ledger.import_events(operations)
        if journal["imported"]:
            self.memory.operational.rebuild()
        if not errors:
            self.memory.operational.receipt(
                "federation.import",
                subject_uri=f"memo://federation/{bundle['bundle_id']}",
                actor_id="memo-federation",
                metadata={
                    "bundle_id": bundle["bundle_id"],
                    "source_device": bundle["source_device"],
                    "principal": principal,
                    "memories": len(memories),
                },
            )
        return {
            "schema": MEMO_FEDERATION_SCHEMA,
            "bundle_id": bundle["bundle_id"],
            "verified": True,
            "dry_run": False,
            "imported": imported,
            "revised": revised,
            "unchanged": unchanged,
            "failed": len(errors),
            "errors": errors,
            "journal": journal,
            "idempotent_replay": False,
        }


__all__ = ["FederationManager", "visible_to"]
