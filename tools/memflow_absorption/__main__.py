"""Command-line entry point for fail-closed Memflow absorption tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memo.config import Config
from memo.errors import SignatureError
from memo.memory import Memory
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalVerifier, SignatureEnvelope
from tools.memflow_absorption.control_record import (
    CONTROL_RECORD_DOMAIN,
    ControlRecordError,
    CutoverSafetyError,
    control_record_from_dict,
    verify_control_record,
)
from tools.memflow_absorption.inventory import (
    CONSUMER_INVENTORY_DOMAIN,
    InventoryError,
    consumer_inventory_from_dict,
    synapse_retirement_manifest_from_dict,
    verify_consumer_inventory,
)
from tools.memflow_absorption.manifest import (
    CAPABILITY_MANIFEST_DOMAIN,
    ManifestError,
    _usage_proof,
    audit_exclusions_from_dict,
    verify_audit_exclusions,
    verify_usage_proof,
)
from tools.memflow_absorption.safety import (
    assert_safe_attempt_root,
    independence_receipt_from_dict,
    independence_scan_from_dict,
    resolve_under_attempt,
    verify_independence_receipt,
)
from tools.memflow_absorption.schemas import SynapseRetirementState
from tools.memflow_absorption.snapshot import create_readonly_snapshot
from tools.memflow_absorption.synapse_catalog import discover_synapse_operations
from tools.memflow_absorption.synapse_data import (
    apply_synapse_data,
    build_synapse_data_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.memflow_absorption")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--attempt-root", type=Path, required=True)
    snapshot.add_argument("--attempt-id", required=True)
    snapshot.add_argument("--source", type=Path, required=True)
    snapshot.add_argument("--target-name", required=True)
    snapshot.add_argument("--manifest-sha256")
    snapshot.add_argument("--apply", action="store_true")
    for name in ("manifest", "inventory"):
        command = commands.add_parser(name)
        command.add_argument("--attempt-root", type=Path, required=True)
        command.add_argument("--attempt-id", required=True)
        command.add_argument("--manifest-sha256")
        command.add_argument("--apply", action="store_true")
    synapse_catalog = commands.add_parser("synapse-catalog")
    synapse_catalog.add_argument("--attempt-root", type=Path, required=True)
    synapse_catalog.add_argument("--attempt-id", required=True)
    synapse_catalog.add_argument("--snapshot", type=Path, required=True)
    synapse_catalog.add_argument("--apply", action="store_true")
    synapse_manifest = commands.add_parser("synapse-manifest")
    synapse_manifest.add_argument("--attempt-root", type=Path, required=True)
    synapse_manifest.add_argument("--attempt-id", required=True)
    synapse_manifest.add_argument("--snapshot", type=Path, required=True)
    synapse_manifest.add_argument("--usage-proof", type=Path, action="append", default=[])
    synapse_manifest.add_argument("--exclusion", type=Path, action="append", default=[])
    synapse_manifest.add_argument("--roster-root", type=Path, required=True)
    synapse_manifest.add_argument("--apply", action="store_true")
    synapse_data = commands.add_parser("synapse-data")
    synapse_data.add_argument("--attempt-id", required=True)
    synapse_data.add_argument(
        "--state-dir",
        "--synapse-state-dir",
        dest="state_dir",
        type=Path,
        required=True,
        help="Synapse state directory containing ledger.jsonl and eval/corpus.json",
    )
    synapse_data.add_argument("--seen-id", action="append", default=[])
    synapse_data.add_argument("--apply", action="store_true")
    synapse_preflight = commands.add_parser("synapse-preflight")
    synapse_preflight.add_argument("--control-record", type=Path, required=True)
    synapse_preflight.add_argument("--capability-manifest", type=Path, required=True)
    synapse_preflight.add_argument("--consumer-inventory", type=Path, required=True)
    synapse_preflight.add_argument("--consumer-plan", type=Path, required=True)
    synapse_preflight.add_argument("--roster-root", type=Path, required=True)
    synapse_preflight.add_argument("--apply", action="store_true")
    synapse_verify = commands.add_parser("synapse-verify")
    synapse_verify.add_argument("--control-record", type=Path, required=True)
    synapse_verify.add_argument("--inventory", type=Path, required=True)
    synapse_verify.add_argument("--retirement-manifest", type=Path, required=True)
    synapse_verify.add_argument("--post-stop-scan", type=Path, required=True)
    synapse_verify.add_argument("--post-reboot-scan", type=Path, required=True)
    synapse_verify.add_argument("--independence-receipt", type=Path, required=True)
    synapse_verify.add_argument("--roster-root", type=Path, required=True)
    synapse_verify.add_argument("--apply", action="store_true")
    return parser


def _snapshot(args: argparse.Namespace) -> dict[str, object]:
    target_name = Path(args.target_name)
    if target_name.name != args.target_name or args.target_name in {"", ".", ".."}:
        raise SystemExit("snapshot target name must be one safe filename")
    relative = f"snapshots/{args.target_name}"
    if not args.apply:
        root = assert_safe_attempt_root(args.attempt_root, args.attempt_id)
        return {
            "command": "snapshot",
            "dry_run": True,
            "source": str(args.source),
            "would_write": str(root / relative),
        }
    if args.manifest_sha256 is None:
        raise SystemExit(
            "snapshot --apply requires exact manifest SHA-256 via --manifest-sha256"
        )
    target = resolve_under_attempt(
        args.attempt_root,
        relative,
        args.attempt_id,
        args.manifest_sha256,
    )
    receipt = create_readonly_snapshot(args.source, target)
    return {"command": "snapshot", "dry_run": False, "receipt": receipt.to_dict()}


def _synapse_catalog(args: argparse.Namespace) -> dict[str, object]:
    root = assert_safe_attempt_root(args.attempt_root, args.attempt_id)
    rows = discover_synapse_operations(args.snapshot)
    if args.apply:
        raise SystemExit("synapse-catalog is inspection-only and never writes")
    encoded = json.dumps(
        [row.to_dict() for row in rows], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "command": "synapse-catalog",
        "dry_run": True,
        "attempt_root": str(root),
        "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
        "operation_count": len(rows),
        "excluded_count": sum(row.exclusion_reason is not None for row in rows),
        "operations": [row.source_operation for row in rows],
    }


def _read_canonical_object(path: Path, description: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{description} must be readable canonical JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != encoded:
        raise SystemExit(f"{description} must be canonical JSON object")
    return cast(dict[str, Any], value)


def _verified_receipt_ids(
    args: argparse.Namespace,
    roster: VerificationRoster,
    source_commit: str,
) -> tuple[list[str], list[str]]:
    try:
        proofs = [
            _usage_proof(_read_canonical_object(path, "usage proof"))
            for path in args.usage_proof
        ]
        exclusions = [
            audit_exclusions_from_dict(
                _read_canonical_object(path, "audit exclusion receipt")
            )
            for path in args.exclusion
        ]
        for proof in proofs:
            verify_usage_proof(proof, roster=roster)
            if proof.snapshot_commit_oid != source_commit:
                raise ManifestError("usage proof snapshot commit does not match Synapse source")
        for receipt in exclusions:
            verify_audit_exclusions(receipt, roster=roster)
    except (ManifestError, TypeError) as exc:
        raise SystemExit(f"synapse catalog preflight receipt validation failed: {exc}") from exc
    if len(proofs) != 2 or len({proof.device_id for proof in proofs}) != 2:
        raise SystemExit("synapse catalog preflight requires two distinct signed usage proofs")
    return (
        sorted(f"{proof.device_id}:{proof.key_id}" for proof in proofs),
        sorted(f"{receipt.signer_device_id}:{receipt.signer_key_id}" for receipt in exclusions),
    )


def _synapse_manifest(args: argparse.Namespace) -> dict[str, object]:
    root = assert_safe_attempt_root(args.attempt_root, args.attempt_id)
    rows = discover_synapse_operations(args.snapshot)
    try:
        roster = VerificationRoster.load(args.roster_root)
    except Exception as exc:  # roster failures are deliberately fail-closed at the CLI boundary.
        raise SystemExit("synapse catalog preflight cannot load verification roster") from exc
    source = _read_canonical_object(args.snapshot / "source.json", "Synapse source record")
    source_commit = source.get("source_commit")
    if not isinstance(source_commit, str):
        raise SystemExit("Synapse source record lacks a source commit")
    proof_ids, exclusion_ids = _verified_receipt_ids(args, roster, source_commit)
    if args.apply:
        raise SystemExit("synapse catalog preflight is inspection-only and never writes")
    encoded = json.dumps(
        [row.to_dict() for row in rows], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "command": "synapse-catalog-preflight",
        "dry_run": True,
        "attempt_root": str(root),
        "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
        "operation_count": len(rows),
        "excluded_operation_count": sum(row.exclusion_reason is not None for row in rows),
        "verified_usage_proof_ids": proof_ids,
        "verified_exclusion_receipt_ids": exclusion_ids,
        "readiness_claim": False,
    }


def _synapse_data(args: argparse.Namespace) -> dict[str, object]:
    bundle = build_synapse_data_bundle(args.state_dir, set(args.seen_id))
    result: dict[str, object] = {
        "command": "synapse-data",
        "dry_run": not args.apply,
        "input_sha256": bundle.input_sha256,
        "feedback_count": len(bundle.feedback),
        "feedback_skipped": len(bundle.skipped_feedback_ids),
        "eval_fixture_count": len(bundle.eval_fixtures),
    }
    if not args.apply:
        return result
    memory = Memory(Config.from_env())
    try:
        receipt = apply_synapse_data(memory, bundle, attempt_id=args.attempt_id)
    finally:
        memory.close()
    result["receipt"] = receipt.to_dict()
    result["dry_run"] = False
    return result


def _blank_signature(payload: dict[str, Any]) -> bytes:
    unsigned = dict(payload)
    unsigned["signature"] = ""
    return canonical_json_bytes(unsigned)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iso_timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str):
        raise SystemExit(f"{description} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{description} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise SystemExit(f"{description} timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _verification_roster(path: Path) -> VerificationRoster:
    try:
        return VerificationRoster.load(path)
    except Exception as exc:
        raise SystemExit("Synapse cutover cannot load verification roster") from exc


def _verify_raw_signature(
    payload: dict[str, Any],
    *,
    description: str,
    domain: str,
    roster: VerificationRoster,
    roster_version: int | None = None,
) -> None:
    key_id = payload.get("signer_key_id")
    signature = payload.get("signature")
    version = payload.get("roster_version") if roster_version is None else roster_version
    if (
        not isinstance(key_id, str)
        or not key_id
        or not isinstance(signature, str)
        or not signature
        or not isinstance(version, int)
        or version < 1
    ):
        raise SystemExit(f"{description} signature authority is malformed")
    try:
        key = roster.key(key_id)
        signer_device_id = payload.get("signer_device_id")
        if signer_device_id is not None and signer_device_id != key.device_id:
            raise SignatureError(f"{description} signer device does not own its key")
        OperationalVerifier().verify(
            domain=domain,
            payload=_blank_signature(payload),
            envelope=SignatureEnvelope(
                algorithm="ed25519",
                key_id=key_id,
                roster_version=version,
                signature=signature,
            ),
            roster=roster,
        )
    except (KeyError, SignatureError) as exc:
        raise SystemExit(f"{description} signature is invalid") from exc


def _synapse_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Inspect canonical artifacts only; never start, stop, or rewrite a service."""

    if args.apply:
        raise SystemExit("synapse-preflight is inspection-only and never applies changes")
    control = _read_canonical_object(args.control_record, "cutover control record")
    manifest = _read_canonical_object(args.capability_manifest, "capability manifest")
    inventory = _read_canonical_object(args.consumer_inventory, "consumer inventory")
    plan = _read_canonical_object(args.consumer_plan, "consumer replacement plan")
    roster = _verification_roster(args.roster_root)
    _verify_raw_signature(
        control,
        description="cutover control record",
        domain=CONTROL_RECORD_DOMAIN,
        roster=roster,
    )
    _verify_raw_signature(
        manifest,
        description="capability manifest",
        domain=CAPABILITY_MANIFEST_DOMAIN,
        roster=roster,
    )
    _verify_raw_signature(
        inventory,
        description="consumer inventory",
        domain=CONSUMER_INVENTORY_DOMAIN,
        roster=roster,
    )
    try:
        typed_inventory = consumer_inventory_from_dict(inventory)
        verify_consumer_inventory(typed_inventory, roster=roster)
    except InventoryError as exc:
        raise SystemExit("consumer inventory typed verification failed") from exc
    if control.get("synapse_state") in {"COMMITTED", "VERIFIED"}:
        raise SystemExit("synapse.cutover.retired")
    if (
        control.get("schema") != "memo.cutover_control_record.v1"
        or control.get("state") != "PREPARING"
        or control.get("synapse_state") != "PREPARING"
    ):
        raise SystemExit("Synapse preflight requires a signed PREPARING control record")
    if (
        manifest.get("schema") != "memo.cutover_capability_manifest.v1"
        or manifest.get("frozen") is not True
        or manifest.get("blockers") != []
        or not manifest.get("signature")
    ):
        raise SystemExit("Synapse capability manifest is not frozen and signed")
    machine_ids = manifest.get("machine_ids")
    if (
        not isinstance(machine_ids, list)
        or len(machine_ids) != 2
        or machine_ids != sorted(set(machine_ids))
    ):
        raise SystemExit("Synapse capability manifest lacks exact two-peer authority")
    operation_map = {
        "mappings": manifest.get("operation_mappings"),
        "registry_authority_sha256": manifest.get("registry_authority_sha256", ""),
        "fixture_authority_sha256": manifest.get("fixture_authority_sha256", ""),
    }
    if (
        manifest.get("operation_map_sha256")
        != _sha256_bytes(canonical_json_bytes(operation_map))
        or manifest.get("slo_baseline_sha256")
        != _sha256_bytes(canonical_json_bytes(manifest.get("slo_baselines")))
    ):
        raise SystemExit("Synapse capability manifest digest mismatch")
    rows = plan.get("rows")
    digest = plan.get("digest")
    surfaces = plan.get("covered_surfaces")
    if not isinstance(rows, list) or digest != _sha256_bytes(canonical_json_bytes(rows)):
        raise SystemExit("Synapse consumer-plan digest mismatch")
    required = {
        "process",
        "port",
        "launchagent",
        "mcp_gateway_route",
        "shell_config_path",
        "state_root",
    }
    if (
        not isinstance(surfaces, dict)
        or set(surfaces) != required
        or any(
            not isinstance(values, list)
            or not values
            or values != sorted(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in surfaces.values()
        )
    ):
        raise SystemExit("Synapse preflight surface coverage is incomplete")
    manifest_sha256 = _sha256_bytes(_blank_signature(manifest))
    inventory_sha256 = hashlib.sha256(typed_inventory.signed_bytes()).hexdigest()
    if (
        plan.get("capability_manifest_sha256") != manifest_sha256
        or plan.get("inventory_sha256") != inventory_sha256
        or surfaces != inventory.get("surface_observations")
    ):
        raise SystemExit(
            "Synapse consumer plan is not derived from verified inventory and manifest"
        )
    return {
        "command": "synapse-preflight",
        "dry_run": True,
        "preflight_passed": True,
        "capability_manifest_sha256": manifest_sha256,
        "consumer_plan_sha256": _sha256_bytes(canonical_json_bytes(plan)),
        "peer_count": 2,
        "covered_surfaces": sorted(required),
    }


def _synapse_verify(args: argparse.Namespace) -> dict[str, object]:
    """Inspect signed negative-scan inputs without touching a runtime."""

    if args.apply:
        raise SystemExit("synapse-verify is inspection-only and never applies changes")
    control = _read_canonical_object(args.control_record, "cutover control record")
    inventory = _read_canonical_object(args.inventory, "consumer inventory")
    manifest = _read_canonical_object(args.retirement_manifest, "retirement manifest")
    post_stop = _read_canonical_object(args.post_stop_scan, "post-stop scan")
    post_reboot = _read_canonical_object(args.post_reboot_scan, "post-reboot scan")
    receipt = _read_canonical_object(args.independence_receipt, "independence receipt")
    roster = _verification_roster(args.roster_root)
    try:
        typed_control = control_record_from_dict(control)
        verified_control = verify_control_record(
            expected_oid=typed_control.control_oid,
            roster=roster,
            record=typed_control,
            fetched_oid=typed_control.control_oid,
        )
        typed_inventory = consumer_inventory_from_dict(inventory)
        typed_manifest = synapse_retirement_manifest_from_dict(manifest)
        typed_stop = independence_scan_from_dict(post_stop)
        typed_reboot = independence_scan_from_dict(post_reboot)
        typed_receipt = independence_receipt_from_dict(receipt)
    except (ControlRecordError, InventoryError, CutoverSafetyError) as exc:
        raise SystemExit("Synapse verification artifact parsing failed") from exc
    if verified_control.synapse_state is not SynapseRetirementState.VERIFIED:
        raise SystemExit("Synapse independence receipt is not committed")
    receipt_sha256 = hashlib.sha256(typed_receipt.signed_bytes()).hexdigest()
    if (
        typed_receipt.control_oid != verified_control.previous_control_oid
        or receipt_sha256 != verified_control.independence_receipt_sha256
    ):
        raise SystemExit(
            "Synapse VERIFIED control does not commit the predecessor-bound receipt"
        )
    try:
        verify_independence_receipt(
            typed_receipt,
            verified_control,
            typed_inventory,
            typed_manifest,
            typed_stop,
            typed_reboot,
            roster=roster,
        )
    except CutoverSafetyError as exc:
        raise SystemExit("Synapse independence receipt verification failed") from exc
    manifest_sha256 = hashlib.sha256(typed_manifest.signed_bytes()).hexdigest()
    inventory_sha256 = hashlib.sha256(typed_inventory.signed_bytes()).hexdigest()
    return {
        "command": "synapse-verify",
        "dry_run": True,
        "independent": True,
        "retirement_epoch": verified_control.retirement_epoch,
        "synapse_manifest_sha256": manifest_sha256,
        "consumer_inventory_sha256": inventory_sha256,
        "independence_receipt_sha256": receipt_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "snapshot":
        result = _snapshot(args)
    elif args.command == "synapse-catalog":
        result = _synapse_catalog(args)
    elif args.command == "synapse-manifest":
        result = _synapse_manifest(args)
    elif args.command == "synapse-data":
        result = _synapse_data(args)
    elif args.command == "synapse-preflight":
        result = _synapse_preflight(args)
    elif args.command == "synapse-verify":
        result = _synapse_verify(args)
    else:
        root = assert_safe_attempt_root(args.attempt_root, args.attempt_id)
        if args.apply:
            if args.manifest_sha256 is None:
                raise SystemExit(
                    f"{args.command} --apply requires exact manifest SHA-256 "
                    "via --manifest-sha256"
                )
            assert_safe_attempt_root(
                root,
                args.attempt_id,
                require_sentinel=True,
                manifest_sha256=args.manifest_sha256,
            )
            raise SystemExit(
                f"{args.command} apply requires explicit signed inputs; use the Python API"
            )
        result = {
            "command": args.command,
            "dry_run": True,
            "attempt_root": str(root),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
