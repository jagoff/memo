"""Command-line entry point for fail-closed Memflow absorption tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from tools.memflow_absorption.manifest import (
    ManifestError,
    _usage_proof,
    audit_exclusions_from_dict,
    verify_audit_exclusions,
    verify_usage_proof,
)
from tools.memflow_absorption.safety import (
    assert_safe_attempt_root,
    resolve_under_attempt,
)
from tools.memflow_absorption.snapshot import create_readonly_snapshot
from tools.memflow_absorption.synapse_catalog import discover_synapse_operations


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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "snapshot":
        result = _snapshot(args)
    elif args.command == "synapse-catalog":
        result = _synapse_catalog(args)
    elif args.command == "synapse-manifest":
        result = _synapse_manifest(args)
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
