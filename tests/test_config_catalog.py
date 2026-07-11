"""Typed catalog coverage for the configuration TUI."""

from __future__ import annotations

from memo.config import Config
from memo.flags import REGISTRY
from memo.tui.config.catalog import (
    PersistencePolicy,
    Visibility,
    build_catalog,
    catalog_by_key,
    domain_file_for_key,
    path_to_env,
    path_to_field,
)


def test_catalog_covers_every_config_field_and_flag_once() -> None:
    catalog = build_catalog()

    assert len({spec.key for spec in catalog}) == len(catalog)
    assert {spec.config_field for spec in catalog if spec.config_field} == set(Config.model_fields)
    assert {spec.env_name for spec in catalog if spec.env_name} >= set(REGISTRY)


def test_every_setting_has_explicit_policy_and_visibility() -> None:
    for spec in build_catalog():
        assert isinstance(spec.policy, PersistencePolicy)
        assert isinstance(spec.visibility, Visibility)
        assert spec.label
        assert spec.description


def test_runtime_controls_are_not_persistent() -> None:
    by_key = catalog_by_key()

    assert by_key["misc.noninteractive"].policy is PersistencePolicy.RUNTIME_ONLY
    assert by_key["session.agent_tty"].policy is PersistencePolicy.RUNTIME_ONLY


def test_catalog_owns_markdown_mappings() -> None:
    assert path_to_field()["storage.data_dir"] == "data_dir"
    assert path_to_env()["recall.top_k"] == "MEMO_RECALL_TOP_K"
    assert domain_file_for_key("recall.top_k") == "recall-config.md"
    assert domain_file_for_key("update.auto_update") == "advanced-config.md"


def test_model_profile_exposes_safe_choices() -> None:
    profile = catalog_by_key()["models.model_profile"]

    assert [choice.value for choice in profile.choices] == ["light", "balanced", "quality"]
