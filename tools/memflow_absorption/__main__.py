"""Command-line entry point for fail-closed Memflow absorption tooling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tools.memflow_absorption.safety import (
    assert_safe_attempt_root,
    resolve_under_attempt,
)
from tools.memflow_absorption.snapshot import create_readonly_snapshot


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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "snapshot":
        result = _snapshot(args)
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
