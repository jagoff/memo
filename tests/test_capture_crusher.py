from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memo.capture_core import _score_rows_by_relevance, maybe_crush_json_capture
from memo.store.crush_cache import CrushCache


def _enable_crusher(monkeypatch: pytest.MonkeyPatch, *, ratio: str = "0.2") -> None:
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    monkeypatch.setenv("MEMO_CRUSHER_ROWS_KEEP_RATIO", ratio)


def _large_rows(count: int = 30) -> list[dict[str, object]]:
    return [
        {"id": index, "kind": "result", "payload": "shared payload " * 20}
        for index in range(count)
    ]


def test_ratio_one_returns_original_without_cache(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_crusher(monkeypatch, ratio="1.0")
    original = json.dumps(_large_rows())

    assert maybe_crush_json_capture(original, "", tmp_cfg) == (original, None)
    assert not (tmp_cfg.state_dir / "crush_cache").exists()


def test_exactly_ten_rows_are_not_replaced_by_ten_rows_and_marker(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_crusher(monkeypatch)
    original = json.dumps(_large_rows(10))

    assert maybe_crush_json_capture(original, "", tmp_cfg) == (original, None)


def test_marker_expansion_returns_original(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_crusher(monkeypatch)
    original = json.dumps([{"id": index} for index in range(11)])

    assert maybe_crush_json_capture(original, "", tmp_cfg) == (original, None)


def test_cache_write_failure_returns_original(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_crusher(monkeypatch)
    original = json.dumps(_large_rows())

    def fail_cache(self, hash_val: str, content: str) -> None:
        del self, hash_val, content
        raise OSError("disk full")

    monkeypatch.setattr(CrushCache, "cache", fail_cache)

    assert maybe_crush_json_capture(original, "", tmp_cfg) == (original, None)


def test_cache_verification_failure_returns_original(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_crusher(monkeypatch)
    original = json.dumps(_large_rows())
    monkeypatch.setattr(CrushCache, "retrieve", lambda self, hash_val, ttl_days=30: None)

    assert maybe_crush_json_capture(original, "", tmp_cfg) == (original, None)


def test_valid_compression_is_smaller_and_recoverable(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_crusher(monkeypatch)
    original = json.dumps(_large_rows(50))

    crushed, hash_val = maybe_crush_json_capture(original, "result", tmp_cfg)

    assert hash_val is not None
    assert len(crushed.encode("utf-8")) <= len(original.encode("utf-8")) * 0.95
    assert CrushCache(tmp_cfg.state_dir).retrieve(hash_val) == original


def test_mean_idf_prevents_long_boilerplate_from_winning() -> None:
    boilerplate = " ".join(f"common{index}" for index in range(30))
    rows: list[object] = [
        {"kind": "noise", "text": boilerplate, "position": index} for index in range(10)
    ]
    rows.append({"kind": "signal", "text": "needle", "position": "answer"})

    scores = _score_rows_by_relevance(rows, "")

    assert scores[-1] > scores[0]


def test_context_bonus_is_positive_but_bounded() -> None:
    rows: list[object] = [
        {"text": "alpha beta gamma delta epsilon"},
        {"text": "ordinary repeated row"},
    ]
    without_context = _score_rows_by_relevance(rows, "")[0]
    with_context = _score_rows_by_relevance(rows, "alpha beta gamma delta epsilon")[0]

    assert with_context > without_context
    assert with_context <= without_context * 1.25


def test_score_ties_are_deterministic() -> None:
    rows: list[object] = [{"same": "value"} for _ in range(12)]

    first = _score_rows_by_relevance(rows, "")
    second = _score_rows_by_relevance(rows, "")

    assert first == second
    assert len(set(first)) == 1


def test_shared_extract_path_crushes_for_extraction_but_grounds_on_original(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memo import capture_core

    user_text = "find the deployment answer"
    assistant_text = json.dumps(_large_rows())
    seen: dict[str, object] = {}

    def fake_crush(content: str, context: str, config) -> tuple[str, str | None]:
        seen["crusher"] = (content, context, config)
        return "CRUSHED FOR EXTRACTION", "abc123"

    def fake_extract(helper, model, user, assistant, *, state_dir):
        del helper, model, user, state_dir
        seen["extract_assistant"] = assistant
        return [
            {
                "title": "Deployment answer",
                "type": "note",
                "body": "The deployment answer remains grounded in the complete original transcript.",
                "tags": [],
            }
        ]

    def fake_grounding(helper, model, *, source: str, claim: str) -> float:
        del helper, model, claim
        seen["ground_source"] = source
        return 1.0

    monkeypatch.setenv("MEMO_GROUNDING_JUDGE", "1")
    monkeypatch.setattr(capture_core, "maybe_crush_json_capture", fake_crush)
    monkeypatch.setattr(capture_core, "extract_insights", fake_extract)
    monkeypatch.setattr(capture_core, "score_grounding", fake_grounding)
    monkeypatch.setattr(capture_core, "find_near_duplicate", lambda *args, **kwargs: None)
    monkeypatch.setattr(capture_core, "_passes_quality", lambda content: True)

    mem = MagicMock()
    mem._ensure_chat.return_value = object()
    mem.save.return_value = SimpleNamespace(
        id="saved-id",
        title="Deployment answer",
        type="note",
    )
    mem.fact_edges.query.return_value = []

    result = capture_core._extract_and_save(mem, tmp_cfg, user_text, assistant_text)

    assert result["saved"] == ["saved-id"]
    assert seen["crusher"] == (assistant_text, user_text, tmp_cfg)
    assert seen["extract_assistant"] == "CRUSHED FOR EXTRACTION"
    assert seen["ground_source"] == f"{user_text}\n{assistant_text}"
