"""Memo is the authoritative source of its native embedder profile."""

from __future__ import annotations

import hashlib
import json

from memo.cli_diag import _profile_status_report, _typed_embedder_profile
from memo.config import Config
from memo.embedder_select import active_embedder_identity


def _fingerprint(profile: dict) -> str:
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_typed_embedder_profile_returns_native_shape(tmp_cfg: Config) -> None:
    typed = _typed_embedder_profile(tmp_cfg)
    assert typed["schema"] == "memo.embedder_profile.v1"
    assert typed["model_id"] == active_embedder_identity(tmp_cfg)
    assert typed["dims"] == int(tmp_cfg.embedder_dims)
    assert typed["normalization"] == "l2"
    assert typed["provider"] == "memo"


def test_profile_status_report_includes_typed_profile(tmp_cfg: Config) -> None:
    report = _profile_status_report(tmp_cfg, include_db=False)
    assert report["typed_profile"]["schema"] == "memo.embedder_profile.v1"
    assert report["typed_profile"]["dims"] == int(tmp_cfg.embedder_dims)


def test_profile_fingerprint_detects_dim_drift(tmp_cfg: Config) -> None:
    profile = _typed_embedder_profile(tmp_cfg)
    drifted = {**profile, "dims": profile["dims"] + 1}
    assert _fingerprint(profile) != _fingerprint(drifted)


def test_mcp_tool_exposes_native_profile(tmp_cfg: Config) -> None:
    import asyncio

    from memo.memory import Memory
    from memo.server import build_server

    mem = Memory(tmp_cfg)
    try:
        server = build_server(memory=mem)
        tool = asyncio.run(server.get_tool("memo_get_embedder_profile"))
        assert tool is not None
        payload = tool.fn()
        assert payload["schema"] == "memo.embedder_profile.v1"
        assert payload["model_id"] == active_embedder_identity(tmp_cfg)
        assert payload["dims"] == int(tmp_cfg.embedder_dims)
        assert payload["provider"] == "memo"
    finally:
        mem.close()
