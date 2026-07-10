"""build_system_message — the human-visible 🧠 presence line (F1a)."""

from types import SimpleNamespace

from memo.recall_logic import build_system_message


def _hit(id_: str, title: str) -> SimpleNamespace:
    return SimpleNamespace(id=id_, title=title, body="", score=0.9)


def test_empty_hits_returns_empty_string() -> None:
    assert build_system_message([]) == ""


def test_counts_and_titles() -> None:
    msg = build_system_message(
        [_hit("a1b2c3d4e5", "sync tier decision"), _hit("f6e5d4c3b2", "delete rollback bug")]
    )
    assert msg.startswith("🧠 memo · 2: ")
    assert "sync tier decision" in msg
    assert "delete rollback bug" in msg


def test_truncates_to_max_chars_with_ellipsis() -> None:
    hits = [_hit(f"{i:08x}", "a very long memory title that keeps going " * 3) for i in range(5)]
    msg = build_system_message(hits, max_chars=80)
    assert len(msg) <= 80
    assert msg.endswith("…")


def test_untitled_hit_falls_back_to_short_id() -> None:
    msg = build_system_message([_hit("a1b2c3d4e5f6", "")])
    assert "a1b2c3d4" in msg


def test_newline_in_title_renders_single_line() -> None:
    """A hit whose title contains \\n must not produce a multi-line systemMessage."""
    msg = build_system_message([_hit("a1b2c3d4e5", "line one\nline two")])
    assert "\n" not in msg
    assert "line one" in msg
    assert "line two" in msg


def test_flag_default_is_on(monkeypatch) -> None:
    from memo.flags import flag_bool

    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)
    assert flag_bool("MEMO_RECALL_SYSTEM_MESSAGE") is True
    monkeypatch.setenv("MEMO_RECALL_SYSTEM_MESSAGE", "0")
    assert flag_bool("MEMO_RECALL_SYSTEM_MESSAGE") is False


def test_cite_instruction_flag_default_is_on(monkeypatch) -> None:
    from memo.flags import flag_bool

    monkeypatch.delenv("MEMO_RECALL_CITE_INSTRUCTION", raising=False)
    assert flag_bool("MEMO_RECALL_CITE_INSTRUCTION") is True
    monkeypatch.setenv("MEMO_RECALL_CITE_INSTRUCTION", "0")
    assert flag_bool("MEMO_RECALL_CITE_INSTRUCTION") is False


def test_cite_instruction_flag_gating(tmp_cfg, monkeypatch) -> None:
    """Full-hook: CITE_INSTRUCTION in additionalContext when flag on; absent when =0.

    Memory.search is stubbed to supply a deterministic hit so the hook always
    reaches the additionalContext rendering code. The flag-gating logic in
    cli_recall_hook.py (the ``if flag_bool("MEMO_RECALL_CITE_INSTRUCTION"):``
    block) is exercised through the real hook invocation via CliRunner.

    Route chosen: full-hook with Memory.search stub (not a bare unit test of
    the append logic). BM25 + saved-memory was attempted first but failed
    because the hook's in-process Memory instance uses Config.from_env() which
    can resolve a different db_path than the fixture's Memory; stubbing search
    directly is more robust and still exercises the real flag-gating code path.
    """
    import json

    from click.testing import CliRunner

    from memo.cli import cli
    from memo.memory import MemoryRecord
    from memo.recall_logic import CITE_INSTRUCTION

    # Stub embedder so no MLX model is loaded (nudge-building may call embed).
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0]] * len(inputs),
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: [1.0, 0.0, 0.0, 0.0],
    )

    # Isolated storage for the hook's Config.from_env()
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")

    # Fake hit with score > default min_sim (0.5) and body > default min_body_chars (40)
    _fake = MemoryRecord(
        id="a1b2c3d4e5f6a7b8",
        path="notes/deployment-decision.md",
        title="deployment decision",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="deployment orchestration decision about container systems " * 5,
        extra={},
        score=0.9,
    )
    monkeypatch.setattr("memo.memory.Memory.search", lambda self, q, **kw: [_fake])

    runner = CliRunner()
    payload = json.dumps({"prompt": "deployment orchestration container decision systems"})

    # Flag ON (default): CITE_INSTRUCTION must appear in additionalContext
    monkeypatch.delenv("MEMO_RECALL_CITE_INSTRUCTION", raising=False)
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())
    ctx = data.get("hookSpecificOutput", {}).get("additionalContext", "")
    assert ctx, "hook returned no additionalContext with flag on — check hook pipeline"
    assert CITE_INSTRUCTION in ctx

    # Flag OFF: CITE_INSTRUCTION must NOT appear
    monkeypatch.setenv("MEMO_RECALL_CITE_INSTRUCTION", "0")
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())
    ctx = data.get("hookSpecificOutput", {}).get("additionalContext", "")
    assert CITE_INSTRUCTION not in ctx


def test_system_message_flag_gating(tmp_cfg, monkeypatch) -> None:
    """Full-hook: top-level systemMessage present when flag on; absent when =0.

    Uses the same CliRunner + Memory.search-stub approach as
    test_cite_instruction_flag_gating.  monkeypatch.setenv is used for all
    env vars (not env= on runner.invoke) because Config.from_env() is called
    in-process inside the Click command and must see the vars at call time;
    the two sequential invocations also share storage-path state set before
    the first call.
    """
    import json

    from click.testing import CliRunner

    from memo.cli import cli
    from memo.memory import MemoryRecord

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0]] * len(inputs),
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: [1.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")

    _fake = MemoryRecord(
        id="a1b2c3d4e5f6a7b8",
        path="notes/deployment-decision.md",
        title="deployment decision",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="deployment orchestration decision about container systems " * 5,
        extra={},
        score=0.9,
    )
    monkeypatch.setattr("memo.memory.Memory.search", lambda self, q, **kw: [_fake])

    runner = CliRunner()
    payload = json.dumps({"prompt": "deployment orchestration container decision systems"})

    # Flag ON (default): systemMessage must be present and start with 🧠 memo ·
    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())
    assert "systemMessage" in data, "hook returned no systemMessage with flag on"
    assert data["systemMessage"].startswith("🧠 memo ·"), (
        f"unexpected systemMessage: {data['systemMessage']!r}"
    )

    # Flag OFF: systemMessage must NOT be present
    monkeypatch.setenv("MEMO_RECALL_SYSTEM_MESSAGE", "0")
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())
    assert "systemMessage" not in data, (
        f"systemMessage unexpectedly present: {data.get('systemMessage')!r}"
    )


# --- Subprocess-path recap composed into systemMessage (CC-native "※ recap:") --


def test_recap_line_folded_into_system_message_when_due(tmp_cfg, monkeypatch) -> None:
    """Full-hook: when MEMO_RECAP is on and cadence is due, the '※ recap:'
    line rides the SAME systemMessage field the 🧠 presence line uses, so
    Claude Code renders it as a proactive, user-visible dim line — composed,
    not clobbering the 🧠 line.
    """
    import json

    from click.testing import CliRunner

    from memo.cli import cli
    from memo.memory import MemoryRecord
    from memo.session import checkpoint, update_summary

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0]] * len(inputs),
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: [1.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)  # default on
    monkeypatch.delenv("MEMO_RECAP", raising=False)  # default on
    monkeypatch.setenv("MEMO_RECAP_EVERY_N", "6")

    _fake = MemoryRecord(
        id="a1b2c3d4e5f6a7b8",
        path="notes/deployment-decision.md",
        title="deployment decision",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="deployment orchestration decision about container systems " * 5,
        extra={},
        score=0.9,
    )
    monkeypatch.setattr("memo.memory.Memory.search", lambda self, q, **kw: [_fake])

    sid = "recap-sysmsg-sess-0001"
    for _ in range(6):
        checkpoint(tmp_cfg.state_dir, session_id=sid, cwd=str(tmp_cfg.data_dir))
    update_summary(tmp_cfg.state_dir, sid, "working on the recap systemMessage feature")

    runner = CliRunner()
    payload = json.dumps(
        {
            "prompt": "deployment orchestration container decision systems",
            "session_id": sid,
        }
    )
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())

    assert "systemMessage" in data, "hook returned no systemMessage with recap due"
    msg = data["systemMessage"]
    assert msg.startswith("🧠 memo ·"), f"presence line missing/clobbered: {msg!r}"
    assert "※ recap:" in msg, f"recap line missing from systemMessage: {msg!r}"
    assert "working on the recap systemMessage feature" in msg


def test_recap_line_absent_from_system_message_when_flag_off(tmp_cfg, monkeypatch) -> None:
    """MEMO_RECAP=0: systemMessage keeps the 🧠 presence line but never gains
    a '※ recap:' line, even when cadence would otherwise be due."""
    import json

    from click.testing import CliRunner

    from memo.cli import cli
    from memo.memory import MemoryRecord
    from memo.session import checkpoint, update_summary

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0]] * len(inputs),
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: [1.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)  # default on
    monkeypatch.setenv("MEMO_RECAP", "0")

    _fake = MemoryRecord(
        id="a1b2c3d4e5f6a7b8",
        path="notes/deployment-decision.md",
        title="deployment decision",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="deployment orchestration decision about container systems " * 5,
        extra={},
        score=0.9,
    )
    monkeypatch.setattr("memo.memory.Memory.search", lambda self, q, **kw: [_fake])

    sid = "recap-sysmsg-sess-0002"
    for _ in range(6):
        checkpoint(tmp_cfg.state_dir, session_id=sid, cwd=str(tmp_cfg.data_dir))
    update_summary(tmp_cfg.state_dir, sid, "working on the recap systemMessage feature")

    runner = CliRunner()
    payload = json.dumps(
        {
            "prompt": "deployment orchestration container decision systems",
            "session_id": sid,
        }
    )
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    data = json.loads(result.output.strip())

    assert "systemMessage" in data
    msg = data["systemMessage"]
    assert msg.startswith("🧠 memo ·")
    assert "※ recap:" not in msg


def test_daemon_path_folds_recap_into_system_message_when_due(monkeypatch, tmp_path) -> None:
    """Daemon (production) path: _recall_logic must fold the '※ recap:' line
    into systemMessage too, mirroring the subprocess path — otherwise a warm
    recall daemon silently never shows the CC-native recap line."""
    import json
    from types import SimpleNamespace

    from memo.recall_logic import _recall_logic
    from memo.session import checkpoint, update_summary

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)  # default on
    monkeypatch.delenv("MEMO_RECAP", raising=False)  # default on
    monkeypatch.setenv("MEMO_RECAP_EVERY_N", "6")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-daemon-sess-0001"
    for _ in range(6):
        checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))
    update_summary(state_dir, sid, "working on the daemon recap path")

    mem, _ = _daemon_stub_memory("sync tier decision")
    result, _log = _recall_logic(
        "que decidimos sobre el sync tier",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=state_dir),
        debug=False,
        session_id=sid,
    )
    payload = json.loads(result)
    msg = payload["systemMessage"]
    assert msg.startswith("🧠 memo · 1: ")
    assert "※ recap:" in msg
    assert "working on the daemon recap path" in msg


def test_cite_instruction_constant() -> None:
    from memo.recall_logic import CITE_INSTRUCTION

    # Task 3's parser depends on the [hex8] cite format this line teaches.
    assert "[a1b2c3d4]" in CITE_INSTRUCTION
    assert CITE_INSTRUCTION.startswith("_") and CITE_INSTRUCTION.endswith("_")


# --- Daemon-path (warm recall daemon) systemMessage parity ------------------
# The warm recall daemon short-circuits the hook: it returns a pre-built JSON
# string from _recall_logic and the hook prints it and exits, never reaching
# the subprocess-path systemMessage injection. So the daemon's own response
# builder (_recall_logic) must emit systemMessage itself — otherwise the
# NORMAL production config silently drops it. These exercise the REAL builder.


def _daemon_stub_memory(title: str):
    """Minimal Memory double for _recall_logic (vec mode never calls embed)."""
    from types import SimpleNamespace

    from memo.memory import MemoryRecord

    hit = MemoryRecord(
        id="a1b2c3d4e5f6a7b8",
        path="notes/x.md",
        title=title,
        type="decision",
        tags=[],
        created="2026-05-21T00:00:00+00:00",
        updated="2026-05-21T00:00:00+00:00",
        body="body " * 20,
        extra={},
        score=0.9,
    )

    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None, exclude_tags=None):
            return [hit]

    return StubMemory(), SimpleNamespace(state_dir=None)


def test_daemon_path_emits_system_message_when_flag_on(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)  # default on

    mem, _ = _daemon_stub_memory("sync tier decision")
    result, _log = _recall_logic(
        "que decidimos sobre el sync tier",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    payload = json.loads(result)
    assert payload["systemMessage"].startswith("🧠 memo · 1: ")
    assert "sync tier decision" in payload["systemMessage"]
    # additionalContext still present and independent of the presence line.
    assert payload["hookSpecificOutput"]["additionalContext"]


def test_daemon_path_omits_system_message_when_flag_off(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_SYSTEM_MESSAGE", "0")

    mem, _ = _daemon_stub_memory("sync tier decision")
    result, _log = _recall_logic(
        "que decidimos sobre el sync tier",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    payload = json.loads(result)
    assert "systemMessage" not in payload
    assert payload["hookSpecificOutput"]["additionalContext"]


# --- Daemon-path CITE_INSTRUCTION parity -----------------------------------


def test_daemon_path_emits_cite_instruction_when_flag_on(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from memo.recall_logic import CITE_INSTRUCTION, _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.delenv("MEMO_RECALL_CITE_INSTRUCTION", raising=False)  # default on

    mem, _ = _daemon_stub_memory("deployment decision")
    result, _log = _recall_logic(
        "deployment orchestration",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    payload = json.loads(result)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert CITE_INSTRUCTION in ctx


def test_daemon_path_omits_cite_instruction_when_flag_off(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from memo.recall_logic import CITE_INSTRUCTION, _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CITE_INSTRUCTION", "0")

    mem, _ = _daemon_stub_memory("deployment decision")
    result, _log = _recall_logic(
        "deployment orchestration",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    payload = json.loads(result)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert CITE_INSTRUCTION not in ctx
    # Context still present (memories rendered)
    assert ctx


# --- Daemon-path presence-recalls bump ------------------------------------


def test_daemon_path_bumps_presence_recalls(monkeypatch, tmp_path) -> None:
    """_recall_logic must bump presence recalls when it returns hits."""
    import json
    from types import SimpleNamespace

    from memo import presence
    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    before = presence.read_today(tmp_path)["recalls"]
    mem, _ = _daemon_stub_memory("sync tier decision")
    result, _log = _recall_logic(
        "sync tier",
        cwd=None,
        mem=mem,
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    payload = json.loads(result)
    assert payload["hookSpecificOutput"]["additionalContext"]  # sanity: got hits
    after = presence.read_today(tmp_path)["recalls"]
    assert after == before + 1
