"""GC-10 Dynamic Constraints — durable decisions/preferences projected into
each client's project-local instruction file as concrete, self-syncing rules.

The pure/writer functions take an explicit ``rules`` list (no Memory), so they
are unit-testable with plain tuples — mirroring dream_profile's testable core.
``gather_rules`` is a thin delegate to the shared motor (dream_profile), tested
by monkeypatching that seam.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import memo.constitution as ct
from memo.constitution import (
    RULES_END,
    RULES_START,
    render_rules_block,
    resync_rules_in_repo,
    upsert_rules_block,
    write_rules_for_clients,
    write_rules_to_file,
)

R1 = ("abcd1234ef", "Never git add -A in the shared worktree")
R2 = ("beef5678aa", "int8 vec quantization is the default")


# --- render_rules_block (pure, deterministic) --------------------------------


def test_render_empty_rules_is_empty_string() -> None:
    assert render_rules_block([]) == ""


def test_render_wraps_rules_between_markers_with_id_provenance() -> None:
    block = render_rules_block([R1, R2])
    assert block.startswith(RULES_START)
    assert block.rstrip().endswith(RULES_END)
    assert "- Never git add -A in the shared worktree `[abcd1234]`" in block
    assert "`[beef5678]`" in block  # id truncated to 8 chars


def test_render_is_deterministic_for_idempotency() -> None:
    assert render_rules_block([R1, R2]) == render_rules_block([R1, R2])


# --- upsert_rules_block (pure) -----------------------------------------------


def test_upsert_appends_when_absent_and_preserves_existing() -> None:
    existing = "# Project rules\n\nkeep me\n"
    out = upsert_rules_block(existing, render_rules_block([R1]))
    assert "keep me" in out
    assert out.count(RULES_START) == 1


def test_upsert_replaces_region_in_place_dropping_retired_rule() -> None:
    first = upsert_rules_block("# top\n", render_rules_block([R1, R2]))
    # R2 retired (superseded) → only R1 remains
    second = upsert_rules_block(first, render_rules_block([R1]))
    assert second.count(RULES_START) == 1  # not duplicated
    assert "`[abcd1234]`" in second
    assert "`[beef5678]`" not in second  # retired rule gone
    assert "# top" in second  # surrounding content preserved


def test_upsert_empty_block_removes_region() -> None:
    with_block = upsert_rules_block("# top\n\ntail\n", render_rules_block([R1]))
    cleared = upsert_rules_block(with_block, render_rules_block([]))
    assert RULES_START not in cleared
    assert "# top" in cleared and "tail" in cleared


def test_upsert_empty_block_when_absent_is_noop() -> None:
    existing = "# top\n"
    assert upsert_rules_block(existing, "") == existing


def test_upsert_is_idempotent() -> None:
    once = upsert_rules_block("# top\n", render_rules_block([R1, R2]))
    twice = upsert_rules_block(once, render_rules_block([R1, R2]))
    assert twice == once


# --- write_rules_to_file (I/O, tmp) ------------------------------------------


def test_write_creates_file_then_skips_when_current(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert write_rules_to_file(target, [R1, R2]) == "written"
    assert RULES_START in target.read_text(encoding="utf-8")
    # re-run identical rules → deterministic no-op
    assert write_rules_to_file(target, [R1, R2]) == "already current (skip)"


def test_write_updates_when_rules_change(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    write_rules_to_file(target, [R1, R2])
    assert write_rules_to_file(target, [R1]) == "written"
    body = target.read_text(encoding="utf-8")
    assert "`[beef5678]`" not in body


def test_write_removes_block_when_no_rules_remain(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    write_rules_to_file(target, [R1])
    assert write_rules_to_file(target, []) == "removed"
    assert RULES_START not in target.read_text(encoding="utf-8")


def test_write_no_rules_no_existing_is_skip(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert write_rules_to_file(target, []) == "no rules (skip)"
    assert not target.exists()


def test_write_dry_run_does_not_touch_disk(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert write_rules_to_file(target, [R1], dry_run=True) == "would write"
    assert not target.exists()


def test_rules_block_coexists_with_fixed_mandate(tmp_path: Path) -> None:
    from memo.cli_mandate import _MARKER, MANDATE_TEXT

    target = tmp_path / "AGENTS.md"
    target.write_text(MANDATE_TEXT, encoding="utf-8")
    write_rules_to_file(target, [R1])
    body = target.read_text(encoding="utf-8")
    assert _MARKER in body  # fixed mandate preserved
    assert RULES_START in body  # dynamic rules added alongside


# --- write_rules_for_clients (dedup shared files) ----------------------------


def test_write_for_clients_dedups_shared_agents_md(tmp_path: Path) -> None:
    results = write_rules_for_clients(["devin", "opencode"], [R1], cwd=tmp_path)
    paths = [rel for rel, _ in results]
    assert paths.count("AGENTS.md") == 1


def test_write_for_clients_creates_nested_path(tmp_path: Path) -> None:
    results = dict(write_rules_for_clients(["cursor"], [R1], cwd=tmp_path))
    assert ".cursor/rules/memo.md" in results
    assert (tmp_path / ".cursor" / "rules" / "memo.md").is_file()


# --- resync_rules_in_repo (only opted-in files) ------------------------------


def test_resync_only_touches_files_with_existing_block(tmp_path: Path) -> None:
    optedin = tmp_path / "AGENTS.md"
    write_rules_to_file(optedin, [R1, R2])
    # a file that never received rules must be left alone
    (tmp_path / ".cursor" / "rules").mkdir(parents=True)
    (tmp_path / ".cursor" / "rules" / "memo.md").write_text("# other\n", encoding="utf-8")

    results = dict(resync_rules_in_repo([R1], cwd=tmp_path))
    assert "AGENTS.md" in results
    assert ".cursor/rules/memo.md" not in results  # not opted in → untouched
    assert "`[beef5678]`" not in optedin.read_text(encoding="utf-8")  # R2 retired


def test_resync_no_opted_in_files_returns_empty(tmp_path: Path) -> None:
    assert resync_rules_in_repo([R1], cwd=tmp_path) == []


# --- gather_rules: graduated motor + directive-shape filter ------------------


def test_looks_like_rule_detects_directives_not_task_logs() -> None:
    assert ct._looks_like_rule("Never use `git add -A`")
    assert ct._looks_like_rule("preferí httpx sobre requests")
    assert not ct._looks_like_rule("bump version 2.10.0")
    assert not ct._looks_like_rule("pushed to master confirmed")


def test_is_rule_memory_by_type_and_shape() -> None:
    assert ct._is_rule_memory("feedback", "commit without asking")  # directive by type
    assert ct._is_rule_memory("preference", "delta for diffs")  # directive by type
    assert ct._is_rule_memory("decision", "Never `git add -A`")  # shaped decision
    assert not ct._is_rule_memory("decision", "bump version 2.10.0")  # log-shaped decision
    assert not ct._is_rule_memory("note", "checked pr status")  # episodic type


def test_gather_rules_filters_task_logs_and_passes_knobs(monkeypatch) -> None:
    called: dict[str, object] = {}
    R_PREF = ("f00d0001aa", "delta para diffs")
    R_LOG = ("10900002bb", "pusheado a master confirmado")

    def fake_gather(mem, cfg, *, k, min_used):
        called["k"], called["min_used"] = k, min_used
        return [R1, R2, R_PREF, R_LOG]  # R1 directive decision, R2 declarative decision

    types = {
        "abcd1234ef": "decision",  # R1 "Never git add -A" → kept (directive)
        "beef5678aa": "decision",  # R2 "int8 … default" → dropped (no marker)
        "f00d0001aa": "preference",  # kept by type
        "10900002bb": "note",  # task-log type → dropped
    }
    fake_mem = SimpleNamespace(get=lambda rid: SimpleNamespace(id=rid, type=types.get(rid)))
    monkeypatch.setattr("memo.dream_profile._gather_rules", fake_gather)

    ids = [rid for rid, _ in ct.gather_rules(fake_mem, object(), k=4, min_used=0.6)]
    assert ids == ["abcd1234ef", "f00d0001aa"]
    assert called == {"k": 4, "min_used": 0.6}


# --- CLI wiring (monkeypatch the store seam to avoid a live Memory) ----------


def test_cli_mandate_dynamic_installs_rules(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from memo.cli_mandate import mandate

    monkeypatch.setattr("memo.cli_mandate._gather_dynamic_rules", lambda: [R1])
    runner = CliRunner()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_STATE_DIR": str(tmp_path / "state"),  # isolate register_repo
        "MEMO_DATA_DIR": str(tmp_path / "data"),
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(mandate, ["--client", "codex", "--dynamic"], env=env)
        assert res.exit_code == 0, res.output
        body = Path("AGENTS.md").read_text(encoding="utf-8")
        assert RULES_START in body  # dynamic rules installed
        assert "<!-- memo-mandate -->" in body  # fixed mandate too (--dynamic implies write)


def test_cli_mandate_sync_regenerates_existing_block(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from memo.cli_mandate import mandate

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_rules_to_file(Path("AGENTS.md"), [R1, R2])
        # a later night: R2 superseded → gather now returns only R1
        monkeypatch.setattr("memo.cli_mandate._gather_dynamic_rules", lambda: [R1])
        res = runner.invoke(mandate, ["--sync"], env={"MEMO_NONINTERACTIVE": "1"})
        assert res.exit_code == 0, res.output
        body = Path("AGENTS.md").read_text(encoding="utf-8")
        assert "`[abcd1234]`" in body
        assert "`[beef5678]`" not in body  # retired rule swept on sync


# --- opted-in repo registry + nightly auto-sync pass -------------------------


def test_registry_empty_then_register_and_dedup(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert ct.registered_repos(state) == []
    repo = tmp_path / "repoA"
    repo.mkdir()
    ct.register_repo(state, repo)
    ct.register_repo(state, repo)  # idempotent
    assert ct.registered_repos(state) == [str(repo.resolve())]


def test_registry_tolerates_corrupt_file(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "mandate_repos.json").write_text("{not json", encoding="utf-8")
    assert ct.registered_repos(state) == []  # never raises


def test_mandate_sync_pass_refreshes_registered_repos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("memo.dream_profile._gather_rules", lambda mem, cfg, *, k, min_used: [R1])
    state = tmp_path / "state"
    repo = tmp_path / "repoA"
    repo.mkdir()
    # repo opted in: install a block with TWO rules, register it
    write_rules_to_file(repo / "AGENTS.md", [R1, R2])
    ct.register_repo(state, repo)
    # a registered-but-deleted repo must be skipped, not crash
    ct.register_repo(state, tmp_path / "gone")

    cfg = SimpleNamespace(state_dir=state)
    # gather_rules filters by memory type → mem.get must return rule-typed records
    fake_mem = SimpleNamespace(get=lambda rid: SimpleNamespace(id=rid, type="decision"))
    res = ct.run_mandate_sync_pass(cfg, fake_mem)

    assert res["status"] == "done"
    assert len(res["synced"]) == 1  # only the live repo
    body = (repo / "AGENTS.md").read_text(encoding="utf-8")
    assert "`[abcd1234]`" in body
    assert "`[beef5678]`" not in body  # R2 retired by the nightly sync


def test_mandate_sync_pass_noop_when_no_repos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("memo.dream_profile._gather_rules", lambda mem, cfg, *, k, min_used: [R1])
    fake_mem = SimpleNamespace(get=lambda rid: SimpleNamespace(id=rid, type="decision"))
    res = ct.run_mandate_sync_pass(SimpleNamespace(state_dir=tmp_path / "state"), fake_mem)
    assert res["status"] == "noop" and res["synced"] == []
