"""Schema-validation test for eval/regression_labels.json.

This test validates the label file format WITHOUT running actual retrieval —
no MLX, no vault access, no live index required. It can run in any CI
environment, including GitHub Actions.

To run the live retrieval eval (requires the developer's vault):
    memo eval recall --labels eval/regression_labels.json --k 5 --force
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LABELS_PATH = Path(__file__).parent.parent / "eval" / "regression_labels.json"

_MIN_PROMPT_COUNT = 25
_MIN_SPANISH_PROMPTS = 2

# Characters / markers that identify a prompt as Spanish-language
_SPANISH_MARKERS = {"¿", "¡", "á", "é", "í", "ó", "ú", "ü", "ñ", "qué", "cómo", "cuál", "cuántas"}


def _is_spanish(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _SPANISH_MARKERS)


@pytest.fixture(scope="module")
def labels_data() -> dict:
    assert LABELS_PATH.exists(), f"Label file not found: {LABELS_PATH}"
    raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "Top-level JSON must be an object"
    return raw


@pytest.fixture(scope="module")
def prompts(labels_data: dict) -> list[dict]:
    return labels_data.get("prompts", [])


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


def test_schema_field_present(labels_data: dict) -> None:
    assert labels_data.get("schema") == "memo.eval_recall.labels.v1", (
        "Missing or wrong 'schema' field — must be 'memo.eval_recall.labels.v1'"
    )


def test_prompts_key_is_list(labels_data: dict) -> None:
    assert isinstance(labels_data.get("prompts"), list), "'prompts' must be a list"


def test_prompt_count(prompts: list[dict]) -> None:
    count = len(prompts)
    assert count >= _MIN_PROMPT_COUNT, (
        f"Expected at least {_MIN_PROMPT_COUNT} prompts, got {count}. "
        "Add more coverage scenarios to eval/regression_labels.json."
    )


def test_each_prompt_has_text(prompts: list[dict]) -> None:
    for i, p in enumerate(prompts):
        if isinstance(p, str):
            assert p.strip(), f"Prompt[{i}] bare string must be non-empty"
        elif isinstance(p, dict):
            assert isinstance(p.get("text"), str) and p["text"].strip(), (
                f"Prompt[{i}] missing or empty 'text' field: {p!r}"
            )
        else:
            pytest.fail(f"Prompt[{i}] must be a str or dict, got {type(p)}: {p!r}")


def test_each_prompt_has_relevant_field(prompts: list[dict]) -> None:
    """Each dict prompt must have a 'relevant' bool."""
    for i, p in enumerate(prompts):
        if isinstance(p, dict):
            assert "relevant" in p, f"Prompt[{i}] missing 'relevant' field: {p!r}"
            assert isinstance(p["relevant"], bool), (
                f"Prompt[{i}] 'relevant' must be a bool, got {type(p['relevant'])}: {p!r}"
            )


def test_each_prompt_has_at_least_one_signal(prompts: list[dict]) -> None:
    """Each dict prompt must have expect_ids, noise_tags, or relevant=True/False (any bool qualifies)."""
    for i, p in enumerate(prompts):
        if isinstance(p, dict):
            has_expect_ids = bool(p.get("expect_ids"))
            has_noise_tags = bool(p.get("noise_tags"))
            has_relevant = "relevant" in p
            assert has_expect_ids or has_noise_tags or has_relevant, (
                f"Prompt[{i}] must have at least one of: expect_ids, noise_tags, or relevant. Got: {p!r}"
            )


def test_expect_ids_are_strings_of_min_8_chars(prompts: list[dict]) -> None:
    for i, p in enumerate(prompts):
        if isinstance(p, dict):
            for j, eid in enumerate(p.get("expect_ids") or []):
                assert isinstance(eid, str), (
                    f"Prompt[{i}].expect_ids[{j}] must be a string, got {type(eid)}"
                )
                assert len(eid) >= 8, (
                    f"Prompt[{i}].expect_ids[{j}]={eid!r} is shorter than 8 chars "
                    "(required for reliable prefix matching)"
                )


# ---------------------------------------------------------------------------
# Coverage checks
# ---------------------------------------------------------------------------


def test_at_least_two_spanish_prompts(prompts: list[dict]) -> None:
    texts = [
        (p if isinstance(p, str) else p.get("text", ""))
        for p in prompts
    ]
    spanish = [t for t in texts if _is_spanish(t)]
    assert len(spanish) >= _MIN_SPANISH_PROMPTS, (
        f"Expected at least {_MIN_SPANISH_PROMPTS} Spanish-language prompts, "
        f"found {len(spanish)}. Add prompts containing '¿', accented chars, or "
        "Spanish keywords (qué, cómo, cuál, etc.)."
    )


def test_at_least_one_noise_probe(prompts: list[dict]) -> None:
    """At least one prompt should be relevant=False (a noise probe)."""
    noise_probes = [
        p for p in prompts
        if isinstance(p, dict) and p.get("relevant") is False
    ]
    assert noise_probes, (
        "No noise probes found (relevant=false). Add at least one prompt "
        "that is NOT expected to match real content, to verify noise@K."
    )


def test_at_least_one_relevant_prompt(prompts: list[dict]) -> None:
    relevant = [
        p for p in prompts
        if isinstance(p, dict) and p.get("relevant") is True
    ]
    assert relevant, "No relevant=true prompts found — add ground-truth prompts."


def test_relevant_terms_is_a_list(labels_data: dict) -> None:
    terms = labels_data.get("relevant_terms")
    if terms is not None:
        assert isinstance(terms, list), "'relevant_terms' must be a list of strings"
        for t in terms:
            assert isinstance(t, str), f"relevant_terms entry must be str, got {type(t)}: {t!r}"


def test_noise_tags_is_a_list(labels_data: dict) -> None:
    tags = labels_data.get("noise_tags")
    if tags is not None:
        assert isinstance(tags, list), "'noise_tags' must be a list of strings"
        for t in tags:
            assert isinstance(t, str), f"noise_tags entry must be str, got {type(t)}: {t!r}"


def test_noise_path_fragments_is_a_list(labels_data: dict) -> None:
    frags = labels_data.get("noise_path_fragments")
    if frags is not None:
        assert isinstance(frags, list), "'noise_path_fragments' must be a list of strings"
        for f in frags:
            assert isinstance(f, str), f"noise_path_fragments entry must be str, got {type(f)}: {f!r}"
