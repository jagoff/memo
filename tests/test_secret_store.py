"""Tests for secret storage (tier, flags, encryption, detection)."""

import pytest
from memo.tiers import DURABLE_TYPES, SECRET_KINDS


def test_secret_in_durable_types():
    """Secret tier should be in durable types."""
    assert "secret" in DURABLE_TYPES


def test_secret_kinds_defined():
    """All secret kinds should be defined."""
    expected_kinds = {"api_token", "password", "ssh_key", "db_credential", "certificate", "generic"}
    assert SECRET_KINDS == expected_kinds
