from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_BLOCKING_EXIT_CODES = {
    None: "not-checked",
    0: "survived",
    2: "interrupted",
    5: "no-tests",
    33: "no-tests",
    35: "suspicious",
}

# Mutmut 3.6 assigns every other exit code the ``suspicious`` status. Keep the
# known non-blocking statuses explicit so a new or corrupt code fails closed.
_NON_BLOCKING_EXIT_CODES = {
    -24,  # timeout (SIGXCPU)
    -11,  # segfault
    -9,  # segfault
    1,  # killed
    3,  # killed by pytest internal error
    24,  # timeout (SIGXCPU)
    34,  # skipped
    36,  # timeout
    37,  # caught by type checker
    152,  # timeout (SIGXCPU)
    255,  # timeout
}
_REQUIRED_METADATA_KEYS = {
    "durations_by_key",
    "estimated_durations_by_key",
    "exit_code_by_key",
}
_OPTIONAL_METADATA_KEYS = {"type_check_error_by_key"}
_ALLOWED_METADATA_KEYS = _REQUIRED_METADATA_KEYS | _OPTIONAL_METADATA_KEYS
_BASELINE_SCHEMA_VERSION = 1
_BASELINE_KEYS = {
    "schema_version",
    "blocking_count",
    "blocking_sha256",
    "blocking_by_reason",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _malformed(path: Path, detail: str) -> RuntimeError:
    return RuntimeError(f"malformed mutmut metadata {path}: {detail}")


def _mapping(path: Path, field: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _malformed(path, f"{field} must be an object")

    validated: dict[str, Any] = {}
    for mutant, field_value in value.items():
        if not isinstance(mutant, str) or not mutant:
            raise _malformed(path, f"invalid {field} key: mutant names must be non-empty strings")
        validated[mutant] = field_value
    return validated


def _read_results(path: Path) -> dict[str, int | None]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _malformed(path, str(error)) from error

    if not isinstance(payload, dict):
        raise _malformed(path, "top-level value must be an object")

    keys = set(payload)
    missing = sorted(_REQUIRED_METADATA_KEYS - keys)
    if missing:
        raise _malformed(path, f"missing required key(s): {', '.join(missing)}")
    unexpected = sorted(keys - _ALLOWED_METADATA_KEYS)
    if unexpected:
        raise _malformed(path, f"unexpected key(s): {', '.join(unexpected)}")

    results = _mapping(path, "exit_code_by_key", payload["exit_code_by_key"])

    validated: dict[str, int | None] = {}
    for mutant, exit_code in results.items():
        if exit_code is not None and type(exit_code) is not int:
            raise _malformed(path, f"exit code for {mutant!r} must be an integer or null")
        validated[mutant] = exit_code

    for field in ("durations_by_key", "estimated_durations_by_key"):
        durations = _mapping(path, field, payload[field])
        for mutant, duration in durations.items():
            if type(duration) not in (int, float):
                raise _malformed(
                    path,
                    f"invalid {field} value for {mutant!r}: expected a number",
                )

    type_errors = _mapping(
        path,
        "type_check_error_by_key",
        payload.get("type_check_error_by_key", {}),
    )
    for mutant, type_error in type_errors.items():
        if type_error is not None and not isinstance(type_error, str):
            raise _malformed(
                path,
                f"invalid type_check_error_by_key value for {mutant!r}: expected a string or null",
            )
    return validated


def blocking_mutants(root: Path) -> dict[str, str]:
    metadata = sorted(root.rglob("*.meta"))
    if not metadata:
        raise RuntimeError(f"no mutmut metadata found under {root}")

    blocked: dict[str, str] = {}
    seen: set[str] = set()
    for path in metadata:
        for mutant, exit_code in _read_results(path).items():
            if mutant in seen:
                raise _malformed(path, f"duplicate mutant name {mutant!r}")
            seen.add(mutant)

            reason = _BLOCKING_EXIT_CODES.get(exit_code)
            if reason is not None:
                blocked[mutant] = reason
            elif exit_code not in _NON_BLOCKING_EXIT_CODES:
                blocked[mutant] = "suspicious"

    if not seen:
        raise RuntimeError(f"no mutant results found in mutmut metadata under {root}")
    return dict(sorted(blocked.items()))


def blocking_summary(blocked: dict[str, str]) -> dict[str, Any]:
    """Return a deterministic, reviewable fingerprint of the blocking set."""

    canonical = "".join(f"{mutant}\t{reason}\n" for mutant, reason in sorted(blocked.items()))
    return {
        "schema_version": _BASELINE_SCHEMA_VERSION,
        "blocking_count": len(blocked),
        "blocking_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "blocking_by_reason": dict(sorted(Counter(blocked.values()).items())),
    }


def load_baseline(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid mutation baseline {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid mutation baseline {path}: top-level value must be an object")

    keys = set(payload)
    missing = sorted(_BASELINE_KEYS - keys)
    unexpected = sorted(keys - _BASELINE_KEYS)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected: {', '.join(unexpected)}")
        raise RuntimeError(f"invalid mutation baseline {path}: {'; '.join(detail)}")

    if payload["schema_version"] != _BASELINE_SCHEMA_VERSION:
        raise RuntimeError(
            f"invalid mutation baseline {path}: schema_version must be {_BASELINE_SCHEMA_VERSION}"
        )
    count = payload["blocking_count"]
    if type(count) is not int or count < 0:
        raise RuntimeError(
            f"invalid mutation baseline {path}: blocking_count must be a non-negative integer"
        )
    digest = payload["blocking_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError(
            f"invalid mutation baseline {path}: blocking_sha256 must be lowercase SHA-256"
        )
    reasons = payload["blocking_by_reason"]
    if not isinstance(reasons, dict) or any(
        not isinstance(reason, str)
        or not reason
        or type(reason_count) is not int
        or reason_count < 0
        for reason, reason_count in reasons.items()
    ):
        raise RuntimeError(
            f"invalid mutation baseline {path}: blocking_by_reason must map names to counts"
        )
    if sum(reasons.values()) != count:
        raise RuntimeError(
            f"invalid mutation baseline {path}: blocking_by_reason counts must sum to "
            "blocking_count"
        )
    return payload


def _print_blockers(blocked: dict[str, str]) -> None:
    for mutant, reason in blocked.items():
        print(f"{reason}: {mutant}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("mutants"))
    parser.add_argument(
        "--baseline",
        type=Path,
        help=(
            "Require the exact blocking-set fingerprint in this reviewed baseline. "
            "Without a baseline, every blocking mutant fails."
        ),
    )
    args = parser.parse_args(argv)
    blocked = blocking_mutants(args.root)
    if args.baseline is not None:
        expected = load_baseline(args.baseline)
        actual = blocking_summary(blocked)
        if actual == expected:
            print(
                "mutation gate: reviewed baseline matched "
                f"({actual['blocking_count']} blocking; "
                f"sha256={actual['blocking_sha256']})"
            )
            return 0
        print(f"mutation gate baseline mismatch: {args.baseline}")
        print(f"expected: {json.dumps(expected, sort_keys=True)}")
        print(f"actual:   {json.dumps(actual, sort_keys=True)}")
        print("review `mutmut results`; update tests or explicitly review a new baseline")
        return 1
    if not blocked:
        print("mutation gate: no surviving or incomplete mutants")
        return 0
    _print_blockers(blocked)
    print(f"mutation gate failed: {len(blocked)} blocking mutant(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
