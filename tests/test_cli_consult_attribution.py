"""CLI consult attribution — trinity layers (synapse / memflow) shell out to the
`memo` CLI, so without source-tagging they show up as "silent" in
`memo usefulness` even though they read memo. `log_cli_consult` records the
consult ONLY when a source is given (so the developer's own interactive
`memo search` stays out of the stats)."""

from __future__ import annotations

import json

from memo.cli_common import log_cli_consult
from memo.config import Config
from memo.dashboard import read_recall_log


def _hits() -> list[dict]:
    return [{"id": "abc12345", "score": 1.2, "title": "a decision"}]


def test_logs_when_source_explicit(tmp_cfg: Config):
    log_cli_consult(tmp_cfg, verb="search", query="q", hits=_hits(), t0_ms=0, source="synapse")
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert len(rows) == 1
    assert rows[0]["source"] == "synapse"
    assert rows[0]["via"] == "cli:search"
    assert len(rows[0]["hits"]) == 1


def test_logs_from_env(tmp_cfg: Config, monkeypatch):
    monkeypatch.setenv("MEMO_SOURCE", "memflow")
    log_cli_consult(tmp_cfg, verb="recall", query="q", hits=_hits(), t0_ms=0, source=None)
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert len(rows) == 1
    assert rows[0]["source"] == "memflow"
    assert rows[0]["via"] == "cli:recall"


def test_silent_without_source(tmp_cfg: Config, monkeypatch):
    monkeypatch.delenv("MEMO_SOURCE", raising=False)
    log_cli_consult(tmp_cfg, verb="search", query="q", hits=_hits(), t0_ms=0, source=None)
    assert read_recall_log(tmp_cfg.state_dir, limit=10) == []


def test_explicit_source_overrides_env(tmp_cfg: Config, monkeypatch):
    monkeypatch.setenv("MEMO_SOURCE", "env-layer")
    log_cli_consult(tmp_cfg, verb="ask", query="q", hits=_hits(), t0_ms=0, source="synapse")
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert rows[0]["source"] == "synapse"


def test_mcp_consult_attributes_to_client_name(tmp_cfg: Config, monkeypatch):
    """An MCP consult with no explicit source / MEMO_SOURCE falls back to the
    client's handshake name — so devin/opencode/devin-desktop aren't 'mcp:unknown'."""
    import types as _t

    from memo import server_common

    monkeypatch.delenv("MEMO_SOURCE", raising=False)
    monkeypatch.setattr(server_common, "_mcp_client_name", lambda: "devin")
    mem = _t.SimpleNamespace(cfg=tmp_cfg)
    server_common.log_consult(mem, tool="search", query="q", hits=_hits(), t0_ms=0)
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert rows[0]["source"] == "devin"
    assert rows[0]["via"] == "mcp:search"


def test_mcp_explicit_source_beats_client_name(tmp_cfg: Config, monkeypatch):
    import types as _t

    from memo import server_common

    monkeypatch.delenv("MEMO_SOURCE", raising=False)
    monkeypatch.setattr(server_common, "_mcp_client_name", lambda: "devin")
    mem = _t.SimpleNamespace(cfg=tmp_cfg)
    server_common.log_consult(mem, tool="ask", query="q", hits=_hits(), t0_ms=0, source="synapse")
    assert read_recall_log(tmp_cfg.state_dir, limit=10)[0]["source"] == "synapse"


def test_attributed_consult_lands_in_usefulness(tmp_cfg: Config):
    """A source-tagged CLI consult must surface as a reader, not 'unknown'."""
    log_cli_consult(tmp_cfg, verb="recall", query="q", hits=_hits(), t0_ms=0, source="memflow")
    from memo.dashboard import consult_breakdown

    cb = consult_breakdown(tmp_cfg.state_dir, limit=50)
    names = [c["consumer"] for c in cb["consumers"]]
    assert "memflow" in names
    assert "memflow" not in cb["silent"]
    # serializable end-to-end (dashboard payload)
    json.dumps(cb, default=str)
