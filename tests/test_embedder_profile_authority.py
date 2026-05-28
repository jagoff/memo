"""M4: memo is the authoritative source of the embedder profile.

These tests pin the wire shape returned by:
1. The new MCP tool ``memory_get_embedder_profile`` — used by synapse and
   memflow at startup to verify compatibility.
2. ``_profile_status_report`` — extended to include a ``typed_profile``
   field that contracts-aware consumers can pluck.

If contracts is installed, the typed profile must roundtrip through
``EmbedderProfile.from_dict``.
"""

from __future__ import annotations

import pytest

from memo.cli import _profile_status_report, _typed_embedder_profile
from memo.config import Config

cc = pytest.importorskip("consciousness_contracts")


def test_typed_embedder_profile_returns_contract_shape(tmp_cfg: Config) -> None:
    typed = _typed_embedder_profile(tmp_cfg)
    assert typed is not None
    assert typed["schema"] == "consciousness.embedder_profile.v1"
    assert typed["model_id"] == tmp_cfg.embedder_model
    assert typed["dims"] == int(tmp_cfg.embedder_dims)
    assert typed["normalization"] == "l2"
    assert typed["provider"] == "memo"

    restored = cc.EmbedderProfile.from_dict(typed)
    assert restored.model_id == tmp_cfg.embedder_model
    assert restored.dims == int(tmp_cfg.embedder_dims)


def test_profile_status_report_includes_typed_profile(tmp_cfg: Config) -> None:
    report = _profile_status_report(tmp_cfg, include_db=False)
    assert "typed_profile" in report
    assert report["typed_profile"]["dims"] == int(tmp_cfg.embedder_dims)


def test_compatibility_check_detects_dim_drift(tmp_cfg: Config) -> None:
    """The fingerprint method makes silent dim mismatch impossible."""
    typed = _typed_embedder_profile(tmp_cfg)
    assert typed is not None
    p_memo = cc.EmbedderProfile.from_dict(typed)
    p_other = cc.EmbedderProfile(model_id=p_memo.model_id, dims=p_memo.dims + 1)
    assert not p_memo.is_compatible_with(p_other)
    assert p_memo.fingerprint() != p_other.fingerprint()


def test_mcp_tool_exposes_typed_profile(tmp_cfg: Config) -> None:
    """memory_get_embedder_profile must be registered and return the contract shape."""
    import asyncio

    from memo.memory import Memory
    from memo.server import build_server

    mem = Memory(tmp_cfg)
    server = build_server(memory=mem)
    tool = asyncio.run(server.get_tool("memory_get_embedder_profile"))
    assert tool is not None, "memory_get_embedder_profile not registered"

    payload = tool.fn()
    assert payload["schema"] == "consciousness.embedder_profile.v1"
    assert payload["model_id"] == tmp_cfg.embedder_model
    assert payload["dims"] == int(tmp_cfg.embedder_dims)
    assert payload["provider"] == "memo"
