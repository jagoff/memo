"""Repository-history providers for co-change and cross-service evidence."""

from __future__ import annotations

import itertools
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

_SERVICE_ROOTS = frozenset(
    {
        "apps",
        "crates",
        "modules",
        "packages",
        "plugins",
        "services",
    }
)


def service_for_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) >= 2 and parts[0].lower() in _SERVICE_ROOTS:
        return f"{parts[0].lower()}/{parts[1]}"
    return None


def collect_git_change_signals(
    repo_root: Path,
    *,
    max_commits: int = 300,
    max_files_per_commit: int = 80,
    max_neighbors_per_path: int = 25,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Derive bounded, deterministic co-change signals from Git history."""
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "log",
            "-n",
            str(max(1, max_commits)),
            "--name-only",
            "--format=__MEMO_COMMIT__%H",
            "-z",
        ],
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "git log failed")

    commits = _parse_git_log(proc.stdout)
    file_commits: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    analyzed = 0
    truncated = 0
    for paths in commits:
        unique = sorted(set(paths))
        if not unique:
            continue
        analyzed += 1
        if len(unique) > max_files_per_commit:
            truncated += 1
            unique = unique[:max_files_per_commit]
        file_commits.update(unique)
        pair_counts.update(itertools.combinations(unique, 2))

    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_service_pairs = 0
    for (left, right), count in pair_counts.items():
        left_service = service_for_path(left)
        right_service = service_for_path(right)
        cross_service = bool(
            left_service and right_service and left_service != right_service
        )
        if cross_service:
            cross_service_pairs += 1
        denominator = max(1, min(file_commits[left], file_commits[right]))
        confidence = count / denominator
        common = {
            "count": count,
            "confidence": round(confidence, 6),
            "cross_service": cross_service,
            "services": [left_service, right_service],
        }
        adjacency[left].append({"path": right, **common})
        adjacency[right].append(
            {
                "path": left,
                **common,
                "services": [right_service, left_service],
            }
        )

    bounded: dict[str, list[dict[str, Any]]] = {}
    for path, neighbors in adjacency.items():
        bounded[path] = sorted(
            neighbors,
            key=lambda item: (
                -int(item["count"]),
                -float(item["confidence"]),
                str(item["path"]),
            ),
        )[:max_neighbors_per_path]

    return {
        "schema": "memo.repo_change_signals.v1",
        "provider": "git-history",
        "analyzed_commits": analyzed,
        "truncated_commits": truncated,
        "files_with_history": len(file_commits),
        "cochange_pairs": len(pair_counts),
        "cross_service_pairs": cross_service_pairs,
        "file_commit_counts": dict(sorted(file_commits.items())),
        "adjacency": dict(sorted(bounded.items())),
    }


def expand_cochange_paths(
    signals: dict[str, Any],
    seed_paths: list[str],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    adjacency = signals.get("adjacency")
    if not isinstance(adjacency, dict):
        return []
    best: dict[str, dict[str, Any]] = {}
    seed_set = set(seed_paths)
    for seed_rank, seed in enumerate(seed_paths):
        neighbors = adjacency.get(seed)
        if not isinstance(neighbors, list):
            continue
        for item in neighbors:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if not path or path in seed_set:
                continue
            score = float(item.get("confidence") or 0.0) + (
                float(item.get("count") or 0.0) / 1000.0
            )
            score /= seed_rank + 1
            current = best.get(path)
            if current is not None and float(current["score"]) >= score:
                continue
            best[path] = {
                "path": path,
                "score": score,
                "seed_path": seed,
                "count": int(item.get("count") or 0),
                "confidence": float(item.get("confidence") or 0.0),
                "cross_service": bool(item.get("cross_service")),
                "services": list(item.get("services") or []),
            }
    return sorted(
        best.values(),
        key=lambda item: (-float(item["score"]), str(item["path"])),
    )[:limit]


def _parse_git_log(raw: bytes) -> list[list[str]]:
    commits: list[list[str]] = []
    current: list[str] | None = None
    for piece in raw.decode("utf-8", errors="surrogateescape").split("\x00"):
        value = piece.strip("\n")
        if not value:
            continue
        if value.startswith("__MEMO_COMMIT__"):
            if current is not None:
                commits.append(current)
            current = []
            continue
        if current is not None:
            current.append(value)
    if current is not None:
        commits.append(current)
    return commits


__all__ = [
    "collect_git_change_signals",
    "expand_cochange_paths",
    "service_for_path",
]
