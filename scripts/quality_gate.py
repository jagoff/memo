#!/usr/bin/env python3
"""Progressive quality budget for complexity and broad exceptions."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from memo.dev_audit import (
    BROAD_EXCEPTION_ALLOWED,
    BROAD_EXCEPTION_RATCHET_EXEMPTIONS,
    find_broad_exception_sites,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "eval" / "quality_baseline.json"
STRICT_MODULES: tuple[str, ...] = (
    "memo.cli_release",
    "memo.runtime.autoupdate",
    "memo.server",
    "memo.surface",
    "memo.capture_core",
)

Metrics = dict[str, dict[str, int]]
_COMPLEXITY_RE = re.compile(r"`(?P<function>[^`]+)` is too complex \((?P<value>\d+) > \d+\)")


def parse_ruff_complexity(payload: list[dict[str, Any]], root: Path) -> dict[str, int]:
    """Parse Ruff C901 JSON into stable path/function complexity budgets."""
    root = root.resolve()
    metrics: dict[str, int] = {}
    for diagnostic in payload:
        if diagnostic.get("code") != "C901":
            continue
        match = _COMPLEXITY_RE.fullmatch(str(diagnostic.get("message") or ""))
        if match is None:
            continue
        path = Path(str(diagnostic.get("filename") or "")).resolve()
        try:
            relpath = path.relative_to(root).as_posix()
        except ValueError:
            continue
        key = f"{relpath}::{match.group('function')}"
        value = int(match.group("value"))
        metrics[key] = max(value, metrics.get(key, 0))
    return dict(sorted(metrics.items()))


def _ruff_executable() -> str:
    sibling = Path(sys.executable).with_name("ruff")
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("ruff")
    if found is None:
        raise RuntimeError("ruff executable not found beside Python or on PATH")
    return found


def collect_complexity(root: Path) -> dict[str, int]:
    """Run Ruff C901 against memo sources and return current violations."""
    completed = subprocess.run(
        [
            _ruff_executable(),
            "check",
            str(root / "src" / "memo"),
            "--select",
            "C901",
            "--output-format",
            "json",
            "--exit-zero",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError("Ruff JSON output must be a list")
    return parse_ruff_complexity(payload, root)


def collect_broad_exceptions(
    root: Path,
    *,
    exemptions: set[tuple[str, str, int]] | None = None,
) -> dict[str, int]:
    """Count broad handlers, excluding every site the audit already covers.

    The count excludes both the ratchet exemptions and the sites lexically
    classified in ``BROAD_EXCEPTION_ALLOWED`` (tests/test_dev_audit.py), so
    the integer budget covers only files no stricter gate guards yet — a
    site classified in that gate is never billed here too.
    """
    metrics: dict[str, int] = {}
    source_root = root / "src" / "memo"
    allowed = (
        BROAD_EXCEPTION_RATCHET_EXEMPTIONS | BROAD_EXCEPTION_ALLOWED
        if exemptions is None
        else exemptions
    )
    for site in find_broad_exception_sites(source_root):
        if (site.relpath, site.scope, site.ordinal) in allowed:
            continue
        relpath = f"src/memo/{site.relpath}"
        metrics[relpath] = metrics.get(relpath, 0) + 1
    return metrics


def collect_metrics(root: Path) -> Metrics:
    """Collect every progressive metric from one checkout."""
    return {
        "complexity": collect_complexity(root),
        "broad_exceptions": collect_broad_exceptions(root),
    }


def compare(current: Metrics, baseline: Metrics) -> list[str]:
    """Report only new or increased debt; reductions and deletions pass."""
    issues: list[str] = []
    for metric in ("complexity", "broad_exceptions"):
        allowed_values = baseline.get(metric, {})
        for key, value in sorted(current.get(metric, {}).items()):
            allowed = allowed_values.get(key, 0)
            if value > allowed:
                issues.append(f"{metric} {key}: current {value} > baseline {allowed}")
    return issues


def _load_baseline(path: Path) -> Metrics:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"{path} must contain quality baseline version 1")
    return {
        "complexity": dict(raw.get("complexity") or {}),
        "broad_exceptions": dict(raw.get("broad_exceptions") or {}),
    }


def _write_baseline(path: Path, metrics: Metrics) -> None:
    document: dict[str, object] = {"version": 1, **metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Replace the committed baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)

    current = collect_metrics(ROOT)
    if args.update:
        if args.baseline.is_file():
            for issue in compare(current, _load_baseline(args.baseline)):
                print(f"increase: {issue}")
        _write_baseline(args.baseline, current)
        print(
            f"updated {args.baseline}: {len(current['complexity'])} complexity budgets, "
            f"{len(current['broad_exceptions'])} exception budgets"
        )
        return 0

    if not args.baseline.is_file():
        print(f"quality baseline missing: {args.baseline}; run with --update", file=sys.stderr)
        return 2
    issues = compare(current, _load_baseline(args.baseline))
    if issues:
        print("quality gate failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(
        f"quality gate passed: {len(current['complexity'])} complexity budgets, "
        f"{len(current['broad_exceptions'])} exception budgets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
