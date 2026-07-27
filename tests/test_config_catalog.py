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


def test_graph_flags_are_routed_to_graph_domain() -> None:
    assert REGISTRY["MEMO_GRAPH_PROJECTION_ENABLED"].group == "graph"
    assert path_to_env()["graph.signal_enabled"] == "MEMO_GRAPH_SIGNAL_ENABLED"
    assert path_to_env()["graph.signal_alpha"] == "MEMO_GRAPH_SIGNAL_ALPHA"
    assert domain_file_for_key("graph.signal_enabled") == "graph-config.md"


def test_model_profile_exposes_safe_choices() -> None:
    profile = catalog_by_key()["models.model_profile"]

    assert [choice.value for choice in profile.choices] == ["light", "balanced", "quality"]


def test_config_owned_env_vars_have_no_bogus_flag_alias() -> None:
    """MEMO_MODEL_PROFILE / MEMO_MEMORIES_IN_VAULT / MEMO_SINGLE_DB are bound to
    their canonical Config path via FIELD_BINDINGS. They are ALSO FlagSpecs (so
    config.py can read them via flag_bool), but ``path_to_env`` must not emit a
    second ``misc.*`` alias — that alias resolves + validates for `config set`
    yet writes a Markdown key the running Config never reads (silent divergence).
    """
    paths = path_to_env()
    assert "misc.model_profile" not in paths
    assert "misc.memories_in_vault" not in paths
    assert "misc.single_db" not in paths
    assert paths["models.model_profile"] == "MEMO_MODEL_PROFILE"
    assert paths["storage.memories_in_vault"] == "MEMO_MEMORIES_IN_VAULT"
    assert paths["storage.single_db"] == "MEMO_SINGLE_DB"
