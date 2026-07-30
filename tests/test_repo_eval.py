from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.repo_eval import evaluate_repo_search, load_repo_eval_labels


class _Searcher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def repo_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **kwargs})
        if query == "boom" and kwargs["mode"] == "unified":
            raise RuntimeError("provider failed")
        if query == "zero":
            return []
        if kwargs["mode"] == "lexical":
            return [{"path": "src/noise.py"}]
        return [{"path": "src/target.py"}, {"path": "src/noise.py"}]


def test_repo_eval_is_symmetric_and_records_failures_and_zeros(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema": "memo.eval.repo_search.labels.v1",
                "queries": [
                    {
                        "id": "hit",
                        "query": "target",
                        "expected_paths": ["src/target.py"],
                        "scope": "production",
                    },
                    {"id": "zero", "query": "zero", "expected_paths": ["missing.py"]},
                    {"id": "boom", "query": "boom", "expected_paths": ["src/target.py"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    searcher = _Searcher()

    report = evaluate_repo_search(searcher, load_repo_eval_labels(labels_path), k=5)

    assert report.total_search_calls == 6
    assert report.failures == 1
    assert report.zero_results == 2
    assert report.total_session_elapsed_ms >= 0
    assert [call["mode"] for call in searcher.calls] == [
        "lexical",
        "unified",
        "lexical",
        "unified",
        "lexical",
        "unified",
    ]
    assert all(call["limit"] == 5 for call in searcher.calls)
    by_strategy = {row.strategy: row for row in report.strategies}
    assert by_strategy["graph-first"].recall_at_k > by_strategy["grep-first"].recall_at_k
    assert by_strategy["graph-first"].failures == 1
