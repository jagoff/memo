"""Tests for multi-vault federation module."""

import pytest

from memo.federation import (
    FederatedResult,
    FederationConfig,
    FederationSearcher,
    VaultConfig,
)


@pytest.fixture
def federation_config(tmp_cfg):
    """Fixture providing FederationConfig instance."""
    return FederationConfig(tmp_cfg.state_dir / "federation.json")


@pytest.fixture
def federation_searcher(federation_config):
    """Fixture providing FederationSearcher instance."""
    return FederationSearcher(federation_config)


def test_federation_config_init(federation_config):
    """Test FederationConfig initialization."""
    assert federation_config.config_path.parent.is_dir()


def test_federation_config_add_vault(federation_config):
    """Test adding a vault to federation."""
    federation_config.add_vault(
        name="test-vault",
        path="/path/to/memo",
        weight=1.5,
    )

    vault = federation_config.get_vault("test-vault")
    assert vault is not None
    assert vault.name == "test-vault"
    assert vault.path == "/path/to/memo"
    assert vault.weight == 1.5


def test_federation_config_get_vault(federation_config):
    """Test getting a vault."""
    federation_config.add_vault("test", "/path/to/memo")

    vault = federation_config.get_vault("test")
    assert vault is not None
    assert vault.name == "test"


def test_federation_config_get_vault_not_found(federation_config):
    """Test getting a non-existent vault."""
    vault = federation_config.get_vault("nonexistent")
    assert vault is None


def test_federation_config_list_vaults(federation_config):
    """Test listing all vaults."""
    federation_config.add_vault("v1", "/path1")
    federation_config.add_vault("v2", "/path2")

    vaults = federation_config.list_vaults()
    assert len(vaults) == 2
    assert all(isinstance(v, VaultConfig) for v in vaults)


def test_federation_config_remove_vault(federation_config):
    """Test removing a vault."""
    federation_config.add_vault("test", "/path/to/memo")

    assert federation_config.get_vault("test") is not None

    success = federation_config.remove_vault("test")
    assert success is True

    assert federation_config.get_vault("test") is None


def test_federation_config_get_enabled_vaults(federation_config):
    """Test getting only enabled vaults."""
    federation_config.add_vault("enabled1", "/path1")
    federation_config.add_vault("enabled2", "/path2")

    enabled = federation_config.get_enabled_vaults()
    assert len(enabled) == 2
    assert all(v.enabled for v in enabled)


def test_federation_config_persistence(tmp_cfg):
    """Test that vault config persists across instances."""
    config_path = tmp_cfg.state_dir / "federation.json"

    # Create first instance and add vault
    config1 = FederationConfig(config_path)
    config1.add_vault("test", "/path/to/memo", weight=1.5)

    # Create second instance and verify persistence
    config2 = FederationConfig(config_path)
    vault = config2.get_vault("test")

    assert vault is not None
    assert vault.path == "/path/to/memo"
    assert vault.weight == 1.5


def test_federation_searcher_init(federation_searcher):
    """Test FederationSearcher initialization."""
    assert federation_searcher.config is not None


def test_federation_searcher_search_no_vaults(federation_searcher):
    """Test search with no configured vaults."""
    results = federation_searcher.search("test query", limit=10)

    assert results == []


def test_federation_searcher_deduplicate_results(federation_searcher):
    """Test result deduplication."""
    results = [
        FederatedResult(
            memoria_id="abc123",
            vault_name="v1",
            title="Test",
            body="Content",
            score=0.8,
            type="note",
            tags=["test"],
        ),
        FederatedResult(
            memoria_id="abc123",
            vault_name="v2",
            title="Test",
            body="Content",
            score=0.9,
            type="note",
            tags=["test"],
        ),
        FederatedResult(
            memoria_id="def456",
            vault_name="v1",
            title="Test2",
            body="Content2",
            score=0.7,
            type="note",
            tags=["test"],
        ),
    ]

    deduped = federation_searcher._deduplicate_results(results)

    # Should keep only the higher-score abc123
    assert len(deduped) == 2
    assert any(r.memoria_id == "abc123" and r.score == 0.9 for r in deduped)
    assert any(r.memoria_id == "def456" for r in deduped)


def test_federation_searcher_rank_results(federation_searcher):
    """Test result ranking."""
    results = [
        FederatedResult(
            memoria_id="a",
            vault_name="v1",
            title="A",
            body="Content",
            score=0.5,
            type="note",
            tags=["test"],
        ),
        FederatedResult(
            memoria_id="b",
            vault_name="v1",
            title="B",
            body="Content",
            score=0.9,
            type="note",
            tags=["test"],
        ),
        FederatedResult(
            memoria_id="c",
            vault_name="v1",
            title="C",
            body="Content",
            score=0.7,
            type="note",
            tags=["test"],
        ),
    ]

    ranked = federation_searcher._rank_results(results)

    assert ranked[0].memoria_id == "b"  # Highest score
    assert ranked[-1].memoria_id == "a"  # Lowest score


def test_vault_config_dataclass():
    """Test VaultConfig dataclass structure."""
    config = VaultConfig(
        name="test",
        path="/path/to/memo",
        weight=1.5,
        enabled=True,
    )
    assert config.name == "test"
    assert config.path == "/path/to/memo"
    assert config.weight == 1.5
    assert config.enabled is True


def test_federated_result_dataclass():
    """Test FederatedResult dataclass structure."""
    result = FederatedResult(
        memoria_id="abc123",
        vault_name="test-vault",
        title="Test Title",
        body="Test content",
        score=0.85,
        type="decision",
        tags=["test", "decision"],
    )
    assert result.memoria_id == "abc123"
    assert result.vault_name == "test-vault"
    assert result.score == 0.85
    assert len(result.tags) == 2
