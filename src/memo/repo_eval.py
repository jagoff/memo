"""Symmetric evaluation for lexical-first versus graph-enriched repo search."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol


class RepoSearcher(Protocol):
    def repo_search(
        self,
        query: str,
        *,
        limit: int,
        repo: str | None,
        mode: str,
        scope: str,
    ) -> Iterable[Any]: ...


@dataclass(frozen=True)
class RepoEvalLabel:
    id: str
    query: str
    expected_paths: tuple[str, ...]
    repo: str | None = None
    scope: str = "all"


@dataclass(frozen=True)
class JudgeResult:
    relevant_ranks: tuple[int, ...]
    judged_results: int

    @property
    def first_relevant_rank(self) -> int | None:
        return self.relevant_ranks[0] if self.relevant_ranks else None


Judge = Callable[[RepoEvalLabel, list[dict[str, Any]]], JudgeResult]


@dataclass(frozen=True)
class RepoEvalRun:
    label_id: str
    query: str
    strategy: str
    mode: str
    scope: str
    repo: str | None
    elapsed_ms: float
    result_count: int
    zero_results: bool
    failure: str | None
    paths: tuple[str, ...]
    relevant_ranks: tuple[int, ...]


@dataclass(frozen=True)
class RepoEvalStrategy:
    strategy: str
    mode: str
    queries: int
    failures: int
    zero_results: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    search_elapsed_ms: float
    returned_results: int


@dataclass(frozen=True)
class RepoEvalReport:
    schema: str
    k: int
    labels: int
    total_session_elapsed_ms: float
    total_search_calls: int
    total_returned_results: int
    failures: int
    zero_results: int
    strategies: tuple[RepoEvalStrategy, ...]
    runs: tuple[RepoEvalRun, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRATEGIES: tuple[tuple[str, str], ...] = (
    ("grep-first", "lexical"),
    ("graph-first", "unified"),
)


def load_repo_eval_labels(path: Path) -> list[RepoEvalLabel]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "memo.eval.repo_search.labels.v1":
        raise ValueError("repo-search labels must use schema memo.eval.repo_search.labels.v1")
    cases = raw.get("queries")
    if not isinstance(cases, list) or not cases:
        raise ValueError("repo-search labels require a non-empty queries list")
    labels: list[RepoEvalLabel] = []
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"repo-search label #{index} must be an object")
        label_id = str(case.get("id") or f"q{index}").strip()
        query = str(case.get("query") or "").strip()
        expected = case.get("expected_paths")
        if not label_id or label_id in seen:
            raise ValueError(f"duplicate or empty repo-search label id: {label_id!r}")
        if not query:
            raise ValueError(f"repo-search label {label_id!r} has an empty query")
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"repo-search label {label_id!r} requires non-empty expected_paths"
            )
        scope = str(case.get("scope") or "all").strip().lower()
        if scope not in {"all", "production", "tests", "vendor"}:
            raise ValueError(f"repo-search label {label_id!r} has invalid scope {scope!r}")
        labels.append(
            RepoEvalLabel(
                id=label_id,
                query=query,
                expected_paths=tuple(str(item) for item in expected if str(item)),
                repo=str(case["repo"]) if case.get("repo") else None,
                scope=scope,
            )
        )
        seen.add(label_id)
    return labels


def deterministic_path_judge(
    label: RepoEvalLabel,
    hits: list[dict[str, Any]],
) -> JudgeResult:
    """Independent judge: only path labels matter, never ranker scores."""
    relevant: list[int] = []
    for rank, hit in enumerate(hits, start=1):
        path = str(hit.get("path") or "")
        if any(_path_matches(path, expected) for expected in label.expected_paths):
            relevant.append(rank)
    return JudgeResult(tuple(relevant), len(hits))


def evaluate_repo_search(
    searcher: RepoSearcher,
    labels: list[RepoEvalLabel],
    *,
    k: int = 10,
    repo: str | None = None,
    judge: Judge = deterministic_path_judge,
    strategies: tuple[tuple[str, str], ...] = STRATEGIES,
) -> RepoEvalReport:
    if k < 1:
        raise ValueError("k must be positive")
    if not labels:
        raise ValueError("at least one repo-search label is required")
    if not strategies:
        raise ValueError("at least one repo-search strategy is required")

    session_started = time.perf_counter()
    runs: list[RepoEvalRun] = []
    # Label outermost guarantees both strategies run adjacently against the
    # same repository state; strategy order is fixed and reported.
    for label in labels:
        for strategy, mode in strategies:
            started = time.perf_counter()
            failure: str | None = None
            hit_dicts: list[dict[str, Any]] = []
            try:
                hits = list(
                    searcher.repo_search(
                        label.query,
                        limit=k,
                        repo=repo or label.repo,
                        mode=mode,
                        scope=label.scope,
                    )
                )
                hit_dicts = [_hit_to_dict(hit) for hit in hits]
                judged = judge(label, hit_dicts)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                judged = JudgeResult((), 0)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            runs.append(
                RepoEvalRun(
                    label_id=label.id,
                    query=label.query,
                    strategy=strategy,
                    mode=mode,
                    scope=label.scope,
                    repo=repo or label.repo,
                    elapsed_ms=elapsed_ms,
                    result_count=len(hit_dicts),
                    zero_results=failure is None and not hit_dicts,
                    failure=failure,
                    paths=tuple(str(hit.get("path") or "") for hit in hit_dicts),
                    relevant_ranks=judged.relevant_ranks,
                )
            )

    metrics = tuple(_strategy_metrics(name, mode, runs, k=k) for name, mode in strategies)
    total_elapsed_ms = (time.perf_counter() - session_started) * 1000.0
    return RepoEvalReport(
        schema="memo.eval.repo_search.v1",
        k=k,
        labels=len(labels),
        total_session_elapsed_ms=total_elapsed_ms,
        total_search_calls=len(runs),
        total_returned_results=sum(run.result_count for run in runs),
        failures=sum(run.failure is not None for run in runs),
        zero_results=sum(run.zero_results for run in runs),
        strategies=metrics,
        runs=tuple(runs),
    )


def _strategy_metrics(
    strategy: str,
    mode: str,
    runs: list[RepoEvalRun],
    *,
    k: int,
) -> RepoEvalStrategy:
    selected = [run for run in runs if run.strategy == strategy]
    successful = [run for run in selected if run.failure is None]
    query_hits = sum(bool(run.relevant_ranks) for run in successful)
    relevant_results = sum(len(run.relevant_ranks) for run in successful)
    returned = sum(run.result_count for run in successful)
    reciprocal = sum(
        1.0 / run.relevant_ranks[0] for run in successful if run.relevant_ranks
    )
    denominator = max(1, len(selected))
    return RepoEvalStrategy(
        strategy=strategy,
        mode=mode,
        queries=len(selected),
        failures=sum(run.failure is not None for run in selected),
        zero_results=sum(run.zero_results for run in selected),
        recall_at_k=query_hits / denominator,
        precision_at_k=relevant_results / max(1, min(returned, len(selected) * k)),
        mrr=reciprocal / denominator,
        search_elapsed_ms=sum(run.elapsed_ms for run in selected),
        returned_results=returned,
    )


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        return dict(hit)
    to_dict = getattr(hit, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError(f"repo search returned unsupported hit type: {type(hit).__name__}")


def _path_matches(path: str, expected: str) -> bool:
    if any(char in expected for char in "*?["):
        return fnmatch(path, expected)
    return path == expected or path.endswith(f"/{expected}")


__all__ = [
    "STRATEGIES",
    "JudgeResult",
    "RepoEvalLabel",
    "RepoEvalReport",
    "RepoEvalRun",
    "RepoEvalStrategy",
    "deterministic_path_judge",
    "evaluate_repo_search",
    "load_repo_eval_labels",
]
