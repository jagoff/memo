"""Tests for the synapse-aware unified briefing composer.

Covers:

- `synapse_briefing_lines()` returns `[]` when synapse is not on PATH.
- When `synapse_client.get_packet` returns a real packet, the composer
  surfaces present_state + reality_conflicts + a health footer.
- Only `detected`/`acknowledged` conflicts surface; `resolved` and
  `archived` are filtered out.
- Item count is capped (top-3 each).
- Snippets are clipped to keep the briefing tight.
- `memory_unified_briefing` MCP tool returns the structured payload.
- The CLI briefing command swallows synapse errors and falls back to
  the local-only sections (no regression when synapse is absent).
"""

from __future__ import annotations

import json
from typing import Any

from click.testing import CliRunner

import memo.briefing as briefing_mod
import memo.cli_briefing as cli_briefing_mod
import memo.synapse_client as synapse_client


def _packet_fixture() -> dict[str, Any]:
    return {
        "schema": "synapse.consciousness_packet.v2",
        "trace_id": "synapse://trace/abc123def456",
        "status": "ready",
        "present_state": [
            {
                "source": "memflow",
                "title": "Current focus: Memo GC5 briefing unification",
                "snippet": "Wiring synapse packet into memo briefing without breaking single-Mac path.",
            },
            {
                "source": "memflow",
                "title": "Pending handoff from MacBook Air",
                "snippet": "Astor terapia: revisar borrador del informe TO.",
            },
            {
                "source": "memflow",
                "title": "Item 3",
                "snippet": "three",
            },
            {
                "source": "memflow",
                "title": "Item 4 should be dropped",
                "snippet": "four",
            },
        ],
        "reality_conflicts": [
            {
                "conflict_id": "C-open-1",
                "lifecycle_state": "detected",
                "severity": "high",
                "freeze_write": True,
                "summary": "Memo dice X pero Memflow dice ¬X sobre la decisión de auth.",
            },
            {
                "conflict_id": "C-ack-2",
                "lifecycle_state": "acknowledged",
                "severity": "medium",
                "freeze_write": False,
                "summary": "Estado intermedio para revisión humana.",
            },
            {
                "conflict_id": "C-resolved-3",
                "lifecycle_state": "resolved",
                "severity": "low",
                "freeze_write": False,
                "summary": "Ya cerrado — no debería aparecer.",
            },
            {
                "conflict_id": "C-archived-4",
                "lifecycle_state": "archived",
                "freeze_write": True,
                "summary": "Archivado — no debería aparecer.",
            },
        ],
    }


# -- happy path -----------------------------------------------------------


def test_returns_empty_when_synapse_unavailable(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: False)
    assert briefing_mod.synapse_briefing_lines("/tmp/wherever") == []


def test_returns_empty_when_packet_is_none(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: None)
    assert briefing_mod.synapse_briefing_lines("/tmp/wherever") == []


def test_returns_empty_when_packet_has_no_sections(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(
        synapse_client, "get_packet",
        lambda *a, **kw: {"status": "ready", "present_state": [], "reality_conflicts": []},
    )
    assert briefing_mod.synapse_briefing_lines("/tmp") == []


def test_renders_present_state_section(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: _packet_fixture())
    out = briefing_mod.synapse_briefing_lines("/tmp")
    md = "\n".join(out)
    assert "### Estado actual (Synapse)" in md
    assert "Current focus: Memo GC5 briefing unification" in md
    assert "Pending handoff from MacBook Air" in md
    # Item 4 must be dropped (top-3 cap).
    assert "Item 4" not in md


def test_renders_conflicts_filtering_resolved_and_archived(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: _packet_fixture())
    out = briefing_mod.synapse_briefing_lines("/tmp")
    md = "\n".join(out)
    assert "### Conflictos abiertos" in md
    assert "C-open-1" in md
    assert "C-ack-2" in md
    assert "C-resolved-3" not in md
    assert "C-archived-4" not in md
    # Freeze flag visible for the freeze_write conflict.
    assert "❄️" in md


def test_health_footer_carries_status_and_trace_short(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: _packet_fixture())
    out = briefing_mod.synapse_briefing_lines("/tmp")
    last_real = next(line for line in reversed(out) if line.strip())
    assert last_real.startswith("_Synapse:")
    assert "ready" in last_real
    # Trace short form drops the prefix and clips to 12 chars.
    assert "abc123def456" in last_real


def test_snippet_is_clipped(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    long_snippet = "x" * 500
    packet = {
        "status": "ready",
        "trace_id": "t/short",
        "present_state": [
            {"source": "memflow", "title": "Long item", "snippet": long_snippet},
        ],
        "reality_conflicts": [],
    }
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: packet)
    out = briefing_mod.synapse_briefing_lines("/tmp")
    snippet_line = next(line for line in out if line.lstrip().startswith("> "))
    body = snippet_line.lstrip()[2:]
    assert len(body) <= 161  # cap + ellipsis
    assert body.endswith("…")


def test_non_dict_rows_are_ignored(monkeypatch):
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(
        synapse_client, "get_packet",
        lambda *a, **kw: {
            "status": "partial",
            "trace_id": "t/x",
            "present_state": ["not-a-dict", 42, {"title": "ok"}],
            "reality_conflicts": [{"lifecycle_state": "detected", "summary": "hi"}, None],
        },
    )
    out = briefing_mod.synapse_briefing_lines("/tmp")
    md = "\n".join(out)
    assert "ok" in md
    # The non-dict junk must not have produced any extra markdown lines.
    assert "not-a-dict" not in md


def test_legacy_conflict_without_state_treated_as_detected(monkeypatch):
    """Memflow-projected conflicts may omit `lifecycle_state` — show them."""
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(
        synapse_client, "get_packet",
        lambda *a, **kw: {
            "status": "ready",
            "trace_id": "t/y",
            "present_state": [],
            "reality_conflicts": [{"conflict_id": "C-legacy", "summary": "no state set"}],
        },
    )
    out = briefing_mod.synapse_briefing_lines("/tmp")
    md = "\n".join(out)
    assert "C-legacy" in md


# -- MCP tool -------------------------------------------------------------


def test_mcp_unified_briefing_returns_payload(tmp_cfg, monkeypatch):
    import asyncio

    from memo.memory import Memory
    from memo.server import build_server

    mem = Memory(tmp_cfg)
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(synapse_client, "get_packet", lambda *a, **kw: _packet_fixture())

    server = build_server(memory=mem)
    fn = asyncio.run(server.get_tool("memory_unified_briefing")).fn
    out = fn(cwd="/tmp/sample")
    assert out["available"] is True
    assert "Conflictos abiertos" in out["markdown"]
    assert isinstance(out["lines"], list)
    assert len(out["lines"]) > 0
    assert len(out["markdown"]) <= 480


def test_mcp_unified_briefing_empty_when_synapse_missing(tmp_cfg, monkeypatch):
    import asyncio

    from memo.memory import Memory
    from memo.server import build_server

    mem = Memory(tmp_cfg)
    monkeypatch.setattr(synapse_client, "is_available", lambda: False)

    server = build_server(memory=mem)
    fn = asyncio.run(server.get_tool("memory_unified_briefing")).fn
    out = fn()
    assert out == {"available": False, "markdown": "", "lines": []}


def test_cli_briefing_emits_active_memory_block(tmp_cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_briefing_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setattr(
        "memo.session.list_sessions",
        lambda *a, **kw: [
            {
                "cwd": str(tmp_path),
                "session_id": "sid-1234",
                "updated": "2026-06-18T10:00:00+00:00",
                "summary": "ordenando la memoria activa",
                "running_summary": "Se está consolidando el bloque de memoria activa.",
                "project": "memo",
                "branch": "master",
                "turn_count": 3,
                "modified_files": ["src/memo/session.py"],
                "last_assistant_tail": "Quedó integrada en briefing y continuidad.",
                "prompt_trail": ["primer loop", "segundo loop"],
            }
        ],
    )

    class _FakeStore:
        def list_recent(self, *a, **kw):
            return []

    class _FakeMemory:
        def __init__(self, cfg):
            self.store = _FakeStore()

    monkeypatch.setattr("memo.memory.Memory", _FakeMemory)
    monkeypatch.setattr(briefing_mod, "synapse_briefing_lines", lambda cwd: [])

    result = CliRunner().invoke(cli_briefing_mod.briefing)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    md = payload["hookSpecificOutput"]["additionalContext"]
    assert "Memoria activa" in md
    assert "Última sesión en este proyecto" in md
    assert "Quedó integrada" in md
    assert "Loops abiertos (sesión)" in md


def test_cli_compact_briefing_caps_context_and_skips_synapse(tmp_cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_briefing_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setattr(
        "memo.session.list_sessions",
        lambda *a, **kw: [
            {
                "cwd": str(tmp_path),
                "session_id": "sid-compact",
                "updated": "2026-06-18T10:00:00+00:00",
                "summary": "resumen " + ("muy largo " * 80),
                "running_summary": "estado " + ("detallado " * 80),
                "project": "memo",
                "branch": "master",
                "turn_count": 12,
            }
        ],
    )

    class _FakeStore:
        def list_recent(self, *a, **kw):
            raise AssertionError("compact briefing must not scan open loops")

    class _FakeMemory:
        def __init__(self, cfg):
            self.store = _FakeStore()

    monkeypatch.setattr("memo.memory.Memory", _FakeMemory)
    monkeypatch.setattr(
        briefing_mod,
        "synapse_briefing_lines",
        lambda cwd: (_ for _ in ()).throw(AssertionError("compact briefing must skip Synapse")),
    )

    result = CliRunner().invoke(cli_briefing_mod.briefing, ["--compact"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    md = payload["hookSpecificOutput"]["additionalContext"]
    assert len(md) <= 480
    assert "sid-compact" in md
    assert "Memoria del día" not in md


def test_compact_text_preserves_limit_and_ellipsis() -> None:
    compact = briefing_mod.compact_text("alpha\n\n" + ("beta " * 200), max_chars=80)
    assert len(compact) <= 80
    assert compact.endswith("…")
    assert "\n\n" not in compact


# -- get_packet wrapper ---------------------------------------------------


def test_get_packet_returns_none_when_subprocess_fails(monkeypatch):
    monkeypatch.setattr(synapse_client, "_executable", lambda: "/does/not/exist")

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def _raise(*_a: Any, **_kw: Any):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(synapse_client.subprocess, "run", _raise)
    assert synapse_client.get_packet("q") is None


def test_get_packet_returns_none_on_non_zero_exit(monkeypatch):
    monkeypatch.setattr(synapse_client, "_executable", lambda: "/fake/synapse")

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "fail"

    monkeypatch.setattr(synapse_client.subprocess, "run", lambda *a, **kw: _Proc())
    assert synapse_client.get_packet("q") is None


def test_get_packet_returns_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(synapse_client, "_executable", lambda: "/fake/synapse")

    class _Proc:
        returncode = 0
        stdout = "{not-json"
        stderr = ""

    monkeypatch.setattr(synapse_client.subprocess, "run", lambda *a, **kw: _Proc())
    assert synapse_client.get_packet("q") is None


def test_get_packet_round_trips_dict(monkeypatch):
    monkeypatch.setattr(synapse_client, "_executable", lambda: "/fake/synapse")

    payload = {"schema": "v2", "status": "ready", "present_state": []}

    class _Proc:
        returncode = 0
        stdout = '{"schema": "v2", "status": "ready", "present_state": []}'
        stderr = ""

    monkeypatch.setattr(synapse_client.subprocess, "run", lambda *a, **kw: _Proc())
    out = synapse_client.get_packet("q")
    assert out == payload
