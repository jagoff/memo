from __future__ import annotations

from memo.graph_canonical import canonical_key, fold_key


def test_canonical_key_folds_separators_and_case() -> None:
    assert canonical_key("FastAPI") == "fastapi"
    assert canonical_key("fast api") == "fastapi"
    assert canonical_key("fast-api") == "fastapi"
    assert canonical_key("Fast_API") == "fastapi"


def test_canonical_key_strips_accents_and_punctuation() -> None:
    assert canonical_key("decisión!") == "decision"
    assert canonical_key("  Synapse.  ") == "synapse"


def test_fold_key_applies_alias_map() -> None:
    # postgres / postgresql collapse; normalization alone cannot do this
    assert fold_key("Postgres") == fold_key("PostgreSQL")
    assert fold_key("k8s") == fold_key("Kubernetes")


def test_fold_key_is_identity_for_unknown() -> None:
    assert fold_key("Memo") == "memo"
