"""Unit tests for the journey-check harness.

Two layers are covered here, both MLX-free (they run on CI):

- ORCHESTRATION — aggregation, exit-code mapping, ``--only`` selection, the
  raising-check contract, and ``--json`` shape with STUBBED checks.
- CHECK BODIES — every real check function, ``JourneyContext``, and the private
  helpers, exercised against a **seeded isolated store built on a deterministic
  stubbed embedder** (``MLXEmbedder.embed`` monkeypatched + ``_mlx_available``
  forced True, mirroring ``tests/conftest.py``'s ``mem_with_stub``). This runs
  the pass/fail logic of each check without a real MLX forward pass, plus the
  skip and exception branches. A real MLX embed only ever happens on Apple
  Silicon under ``memo journey-check`` itself — those forward passes are the one
  thing the stub replaces.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo import journey_check as jc
from memo.cli_journey import journey_check
from memo.journey_check import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    compute_exit_code,
    run_all,
)
from memo.runtime import daemon as daemon_mod


def _stub(name: str, status: str) -> jc.Check:
    def _check(_ctx: object) -> CheckResult:
        return CheckResult(name, status, f"{name} detail")

    _check.__name__ = f"stub_{name}"
    return _check


def _install_stub_registry(monkeypatch, specs: list[tuple[str, str]]) -> None:
    """Replace the check registry with stubs and neuter store setup so run_all
    never touches MLX or the filesystem-backed store."""
    checks = [(name, _stub(name, status)) for name, status in specs]
    monkeypatch.setattr(jc, "_CHECKS", checks)
    monkeypatch.setattr(jc, "_STORE_CHECKS", frozenset())
    # JourneyContext with need_store=False still makes tmp dirs but skips seeding;
    # keep it cheap and MLX-free.
    monkeypatch.setattr(jc.JourneyContext, "_setup_store", lambda self: None)


# ── compute_exit_code ────────────────────────────────────────────────────────
def test_exit_code_zero_when_all_pass():
    results = [CheckResult("a", PASS), CheckResult("b", PASS)]
    assert compute_exit_code(results) == 0


def test_exit_code_nonzero_on_any_fail():
    results = [CheckResult("a", PASS), CheckResult("b", FAIL), CheckResult("c", WARN)]
    assert compute_exit_code(results) == 1


def test_warn_and_skip_do_not_fail_the_gate():
    results = [CheckResult("a", WARN), CheckResult("b", SKIP), CheckResult("c", PASS)]
    assert compute_exit_code(results) == 0


# ── run_all aggregation ──────────────────────────────────────────────────────
def test_run_all_runs_every_check(monkeypatch):
    _install_stub_registry(monkeypatch, [("auto-save", PASS), ("auto-recall", WARN)])
    results, code = run_all()
    assert [r.name for r in results] == ["auto-save", "auto-recall"]
    assert code == 0


def test_run_all_exit_code_reflects_a_failure(monkeypatch):
    _install_stub_registry(monkeypatch, [("auto-save", PASS), ("auto-recall", FAIL)])
    results, code = run_all()
    assert code == 1
    assert {r.name: r.status for r in results} == {"auto-save": PASS, "auto-recall": FAIL}


def test_run_all_only_selects_subset(monkeypatch):
    _install_stub_registry(
        monkeypatch, [("auto-save", FAIL), ("auto-recall", PASS), ("uses-memory", PASS)]
    )
    results, code = run_all(only=["auto-recall"])
    assert [r.name for r in results] == ["auto-recall"]
    # The failing check was not selected, so the gate is green.
    assert code == 0


def test_raising_check_becomes_fail_not_crash(monkeypatch):
    def _boom(_ctx: object) -> CheckResult:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(jc, "_CHECKS", [("auto-save", _boom)])
    monkeypatch.setattr(jc, "_STORE_CHECKS", frozenset())
    monkeypatch.setattr(jc.JourneyContext, "_setup_store", lambda self: None)
    results, code = run_all()
    assert code == 1
    assert results[0].status == FAIL
    assert "kaboom" in results[0].detail


def test_invalid_status_is_coerced_to_fail(monkeypatch):
    def _bogus(_ctx: object) -> CheckResult:
        return CheckResult("auto-save", "maybe")

    monkeypatch.setattr(jc, "_CHECKS", [("auto-save", _bogus)])
    monkeypatch.setattr(jc, "_STORE_CHECKS", frozenset())
    monkeypatch.setattr(jc.JourneyContext, "_setup_store", lambda self: None)
    results, code = run_all()
    assert results[0].status == FAIL
    assert code == 1


def test_run_all_skips_store_setup_when_no_store_check_selected(monkeypatch):
    """A check outside the store set must not trigger MLX seeding."""
    calls: list[int] = []

    def _spy(self: object) -> None:
        calls.append(1)

    monkeypatch.setattr(jc, "_CHECKS", [("live-wiring", _stub("live-wiring", PASS))])
    monkeypatch.setattr(jc, "_STORE_CHECKS", frozenset({"auto-save"}))
    monkeypatch.setattr(jc.JourneyContext, "_setup_store", _spy)
    run_all()
    assert calls == []


# ── CLI: --json shape + text output + exit code ──────────────────────────────
def test_cli_json_shape_and_exit_code(monkeypatch):
    _install_stub_registry(
        monkeypatch, [("auto-save", PASS), ("auto-recall", FAIL), ("ux-messages", WARN)]
    )
    result = CliRunner().invoke(journey_check, ["--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert [row["name"] for row in payload] == ["auto-save", "auto-recall", "ux-messages"]
    assert {row["name"]: row["status"] for row in payload} == {
        "auto-save": PASS,
        "auto-recall": FAIL,
        "ux-messages": WARN,
    }
    # Every row carries the CheckResult contract keys.
    for row in payload:
        assert set(row) == {"name", "status", "detail", "evidence"}


def test_cli_text_output_and_green_exit(monkeypatch):
    _install_stub_registry(monkeypatch, [("auto-save", PASS), ("ux-messages", WARN)])
    result = CliRunner().invoke(journey_check, [])
    assert result.exit_code == 0
    assert "journey-check" in result.output
    assert "auto-save" in result.output


def test_cli_only_flag_runs_subset(monkeypatch):
    _install_stub_registry(monkeypatch, [("auto-save", FAIL), ("auto-recall", PASS)])
    result = CliRunner().invoke(journey_check, ["--only", "auto-recall", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [row["name"] for row in payload] == ["auto-recall"]


# ── Stubbed embedder: run the real check bodies without a real MLX forward pass ─
def _stub_embed(self, inputs):
    """Deterministic bag-of-words embedder (mirrors conftest ``mem_with_stub``).

    Returns unit vectors sized to the embedder's configured dims, so a real
    seeded ``VecStore`` accepts them and shared tokens (deploy/token/zephyr)
    give the seeded fact a higher cosine than the unrelated prompt — enough for
    the check bodies to run their pass/fail logic end-to-end. No MLX load.
    """
    dims = self.expected_dims
    out = []
    for s in inputs:
        vec = [0.0] * dims
        for tok in (s or "").lower().split():
            vec[sum(ord(c) for c in tok) % dims] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        out.append([x / norm for x in vec])
    return out


@pytest.fixture
def seeded_ctx(monkeypatch):
    """A real, seeded ``JourneyContext`` backed by the deterministic stub embedder."""
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr(jc, "_mlx_available", lambda: True)
    ctx = jc.JourneyContext(need_store=True)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def no_mlx_ctx(monkeypatch):
    """A store-less ``JourneyContext`` with MLX forced absent (skip-branch driver)."""
    monkeypatch.setattr(jc, "_mlx_available", lambda: False)
    ctx = jc.JourneyContext(need_store=False)
    try:
        yield ctx
    finally:
        ctx.close()


# ── _mlx_available: both branches ────────────────────────────────────────────
def test_mlx_available_true_when_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    assert jc._mlx_available() is True


def test_mlx_available_false_when_import_fails(monkeypatch):
    # `None` in sys.modules makes `import mlx_lm` raise ImportError.
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    assert jc._mlx_available() is False


# ── JourneyContext seeding ───────────────────────────────────────────────────
def test_journey_context_seeds_store(seeded_ctx):
    assert seeded_ctx.mlx is True
    assert seeded_ctx.mem is not None
    assert seeded_ctx.seeded["fact_nonce"] == jc._FACT_NONCE
    assert seeded_ctx.seeded["decoy_count"] == len(jc._DECOYS)
    # The seeded fact id is a real record in the isolated store.
    assert len(str(seeded_ctx.seeded["fact_id"])) >= 8


# ── Store checks: happy path (bodies run past the MLX gate) ───────────────────
def _assert_ran(res, name):
    assert res.name == name
    assert res.status in jc._STATUSES
    assert res.status != SKIP  # the MLX gate did not short-circuit


def test_check_auto_save_runs_body(seeded_ctx):
    res = jc.check_auto_save(seeded_ctx)
    _assert_ran(res, "auto-save")
    # The capture path ran and reported per-candidate outcomes.
    assert "saved" in res.evidence
    assert "saved_types" in res.evidence


def test_check_auto_recall_runs_body(seeded_ctx):
    res = jc.check_auto_recall(seeded_ctx)
    _assert_ran(res, "auto-recall")
    assert {"surfaced", "latency_s", "within_budget", "negative_clean"} <= set(res.evidence)


def test_check_uses_memory_pass_and_abstain(seeded_ctx):
    nonce = str(seeded_ctx.seeded["fact_nonce"])
    fid = str(seeded_ctx.seeded["fact_id"])

    def _ask(prompt, k=5):
        if prompt == jc._MATCH_PROMPT:
            return {
                "answer": f"The deploy token is {nonce} (see {fid[:8]}).",
                "sources": [{"id": fid}],
            }
        return {"answer": "I couldn't find that in memory.", "sources": []}

    seeded_ctx.mem.ask = _ask  # type: ignore[method-assign]
    res = jc.check_uses_memory(seeded_ctx)
    assert res.name == "uses-memory"
    assert res.status == PASS
    assert res.evidence["contains_value"] and res.evidence["cites_id"] and res.evidence["abstained"]


def test_check_uses_memory_fail_when_value_absent(seeded_ctx):
    seeded_ctx.mem.ask = lambda prompt, k=5: {"answer": "no idea", "sources": [{"id": "x"}]}  # type: ignore[method-assign]
    res = jc.check_uses_memory(seeded_ctx)
    assert res.status == FAIL
    assert res.evidence["contains_value"] is False


def test_check_token_savings_runs_body(seeded_ctx):
    """Round-2 regression: the round-1 fix repointed this check at
    `historic["grounded"]`, but journey-check's `_recall()` writes
    `recall.log` with `client="journey-check"` (consults), never
    `grounding.log` — `grounded` can never move, so the check was a
    permanent, silent FAIL. `_assert_ran` alone doesn't catch this since it
    accepts FAIL; assert PASS + a real positive delta explicitly."""
    res = jc.check_token_savings(seeded_ctx)
    _assert_ran(res, "token-savings")
    assert {"before", "after", "delta", "recalls_logged"} <= set(res.evidence)
    assert res.status == PASS
    assert res.evidence["delta"] > 0


def test_check_ux_messages_warn_when_all_channels_deliver(seeded_ctx, monkeypatch):
    parsed = {
        "hookSpecificOutput": {"additionalContext": "<memo-recall>seeded ctx</memo-recall>"},
        "systemMessage": "memo surfaced 1 memory",
    }
    monkeypatch.setattr(jc, "_recall", lambda ctx, prompt, session_id=None: (parsed, 0.01))
    monkeypatch.setattr(jc, "_briefing_renders", lambda ctx: (True, ""))
    res = jc.check_ux_messages(seeded_ctx)
    assert res.name == "ux-messages"
    assert res.status == WARN
    assert res.evidence["notification_file"] is True


def test_check_ux_messages_fail_when_recall_silent(seeded_ctx, monkeypatch):
    monkeypatch.setattr(jc, "_recall", lambda ctx, prompt, session_id=None: ({}, 0.01))
    # A briefing error string is threaded into evidence as `briefing_error`.
    monkeypatch.setattr(jc, "_briefing_renders", lambda ctx: (False, "briefing blew up"))
    res = jc.check_ux_messages(seeded_ctx)
    assert res.status == FAIL
    assert res.evidence["additional_context"] is False
    assert res.evidence["briefing_error"] == "briefing blew up"


# ── Store checks: skip branch (MLX absent) ───────────────────────────────────
@pytest.mark.parametrize(
    "check, name",
    [
        (jc.check_auto_save, "auto-save"),
        (jc.check_auto_recall, "auto-recall"),
        (jc.check_uses_memory, "uses-memory"),
        (jc.check_token_savings, "token-savings"),
        (jc.check_ux_messages, "ux-messages"),
    ],
)
def test_store_check_skips_without_mlx(no_mlx_ctx, check, name):
    res = check(no_mlx_ctx)
    assert res.name == name
    assert res.status == SKIP


# ── _recall + _additional_context ────────────────────────────────────────────
def test_recall_returns_parsed_and_latency(seeded_ctx):
    parsed, latency = jc._recall(seeded_ctx, jc._MATCH_PROMPT, session_id="jc-t")
    assert isinstance(parsed, dict)
    assert latency >= 0.0


def test_additional_context_extracts_and_defaults():
    assert jc._additional_context({"hookSpecificOutput": {"additionalContext": "hi"}}) == "hi"
    assert jc._additional_context({"hookSpecificOutput": {}}) == ""
    assert jc._additional_context({}) == ""


# ── _briefing_renders ────────────────────────────────────────────────────────
def test_briefing_renders_against_seeded_store(seeded_ctx):
    ok, err = jc._briefing_renders(seeded_ctx)
    assert isinstance(ok, bool)
    assert isinstance(err, str)


def test_briefing_renders_reports_exception(seeded_ctx, monkeypatch):
    # A None entry makes `from memo.cli_briefing import briefing` raise ImportError,
    # exercising the defensive except path.
    monkeypatch.setitem(sys.modules, "memo.cli_briefing", None)
    ok, err = jc._briefing_renders(seeded_ctx)
    assert ok is False
    assert "ImportError" in err or err


def test_briefing_renders_reports_nonzero_exit(seeded_ctx, monkeypatch):
    import click

    @click.command()
    def _failing_briefing() -> None:
        raise SystemExit(3)

    monkeypatch.setattr("memo.cli_briefing.briefing", _failing_briefing)
    ok, err = jc._briefing_renders(seeded_ctx)
    assert ok is False
    assert err == "exit 3"


# ── check_install ────────────────────────────────────────────────────────────
def _fake_completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _install_run_memo(*, surface_nonce=True, doctor_ok=True):
    """Build a fake ``_run_memo`` dispatching on subcommand."""
    doctor_json = json.dumps(
        {
            "imports": [{"name": "mlx", "ok": True}],
            "storage": {"data_dir": {"ok": doctor_ok}},
        }
    )
    recall_ctx = (
        f"<memo-recall>{jc._INSTALL_NONCE} deploy setting</memo-recall>"
        if surface_nonce
        else "none"
    )
    recall_json = json.dumps(
        {
            "hookSpecificOutput": {
                "additionalContext": recall_ctx,
                "hookEventName": "UserPromptSubmit",
            }
        }
    )

    def _run(binary, args, env, *, stdin=None, timeout=180):
        sub = args[0]
        if sub == "onboard":
            return _fake_completed(args, 0, "{}")
        if sub == "doctor":
            return _fake_completed(args, 0, doctor_json)
        if sub == "save":
            return _fake_completed(args, 0, "saved")
        if sub == "recall-hook":
            return _fake_completed(args, 0, recall_json)
        return _fake_completed(args, 0, "")

    return _run


def test_check_install_skips_without_binary(monkeypatch):
    monkeypatch.setattr(jc.shutil, "which", lambda name: None)
    res = jc.check_install(types.SimpleNamespace())
    assert res.name == "install"
    assert res.status == SKIP


def test_check_install_pass(monkeypatch):
    monkeypatch.setattr(jc.shutil, "which", lambda name: "/fake/bin/memo")
    monkeypatch.setattr(jc, "_run_memo", _install_run_memo(surface_nonce=True))
    res = jc.check_install(types.SimpleNamespace())
    assert res.status == PASS
    assert res.evidence["recall_surfaced"] is True
    assert res.evidence["doctor_ok"] is True


def test_check_install_fail_when_recall_misses(monkeypatch):
    monkeypatch.setattr(jc.shutil, "which", lambda name: "/fake/bin/memo")
    monkeypatch.setattr(jc, "_run_memo", _install_run_memo(surface_nonce=False))
    res = jc.check_install(types.SimpleNamespace())
    assert res.status == FAIL
    assert res.evidence["recall_surfaced"] is False
    # The failure branch captures diagnostic stderr/stdout snippets.
    assert "recall_stdout" in res.evidence


# ── check_live_wiring: every branch ──────────────────────────────────────────
def _wire(monkeypatch, *, hook, daemon_running, mixed):
    monkeypatch.setattr("memo.cli_hooks.recall_hook_wired", lambda *a, **k: hook)
    monkeypatch.setattr(
        "memo.cli_diag._recall_daemon_health", lambda cfg: {"running": daemon_running}
    )
    warnings = ["memo and memo-mcp resolve to different environments"] if mixed else []
    monkeypatch.setattr(
        "memo.runtime.detect._runtime_install_report", lambda cwd=None: {"warnings": warnings}
    )


def test_live_wiring_pass(monkeypatch):
    _wire(monkeypatch, hook=True, daemon_running=True, mixed=False)
    res = jc.check_live_wiring(types.SimpleNamespace())
    assert res.status == PASS
    assert res.evidence["hook_wired"] and res.evidence["daemon_running"]


def test_live_wiring_warn_when_daemon_cold(monkeypatch):
    _wire(monkeypatch, hook=True, daemon_running=False, mixed=False)
    res = jc.check_live_wiring(types.SimpleNamespace())
    assert res.status == WARN


def test_live_wiring_fail_when_hook_unwired(monkeypatch):
    _wire(monkeypatch, hook=False, daemon_running=True, mixed=False)
    res = jc.check_live_wiring(types.SimpleNamespace())
    assert res.status == FAIL
    assert "hook" in res.detail.lower()


def test_live_wiring_fail_on_mixed_runtime(monkeypatch):
    _wire(monkeypatch, hook=True, daemon_running=True, mixed=True)
    res = jc.check_live_wiring(types.SimpleNamespace())
    assert res.status == FAIL
    assert res.evidence["mixed_runtime"] is True


# ── Install-check subprocess helpers ─────────────────────────────────────────
def test_fresh_home_env_is_day0(tmp_path):
    monkey_leak = "MEMO_SHOULD_NOT_LEAK"
    import os

    os.environ[monkey_leak] = "1"
    try:
        env = jc._fresh_home_env(tmp_path)
    finally:
        os.environ.pop(monkey_leak, None)
    assert env["HOME"] == str(tmp_path)
    assert env["MEMO_NONINTERACTIVE"] == "1"
    assert env["MEMO_AUTO_UPDATE"] == "0"
    assert env["MEMO_EMBEDDER_VIA_DAEMON"] == "0"
    # Inherited MEMO_* vars are stripped; only the Day-0 set remains.
    assert monkey_leak not in env
    assert Path(env["MEMO_DATA_DIR"]).is_dir()


def test_run_memo_executes_subprocess():
    res = jc._run_memo("/bin/echo", ["hello-journeycheck"], {"PATH": "/usr/bin:/bin"})
    assert res.returncode == 0
    assert "hello-journeycheck" in res.stdout


def test_doctor_no_critical_variants():
    ok, note = jc._doctor_no_critical(
        json.dumps({"imports": [{"name": "a", "ok": True}], "storage": {"data_dir": {"ok": True}}})
    )
    assert ok is True and "ok" in note

    bad_import, note2 = jc._doctor_no_critical(
        json.dumps({"imports": [{"name": "mlx", "ok": False}], "storage": {}})
    )
    assert bad_import is False and "failed imports" in note2

    no_storage, note3 = jc._doctor_no_critical(
        json.dumps({"imports": [], "storage": {"data_dir": {"ok": False}}})
    )
    assert no_storage is False and "data_dir missing" in note3

    not_json, note4 = jc._doctor_no_critical("not json at all")
    assert not_json is False and "JSON" in note4


def test_recall_hook_context_parses_last_json_line():
    stdout = (
        "some log noise\n"
        + json.dumps({"hookSpecificOutput": {"additionalContext": "found it"}})
        + "\n"
    )
    assert jc._recall_hook_context(stdout) == "found it"


def test_recall_hook_context_empty_when_no_json():
    assert jc._recall_hook_context("just plain text\nno json here") == ""
    assert jc._recall_hook_context("") == ""


# ── CLI: skipped-count line (cli_journey.py) ─────────────────────────────────
def test_cli_text_output_renders_skipped_count(monkeypatch):
    _install_stub_registry(monkeypatch, [("auto-save", PASS), ("auto-recall", SKIP)])
    result = CliRunner().invoke(journey_check, [])
    assert result.exit_code == 0
    assert "skipped" in result.output


# ── daemon._warm_embedder (PR #100's cold-first-recall fix) ───────────────────
class _FakeEmbedder:
    def __init__(self):
        self.embedded: list[list[str]] = []

    def embed(self, inputs):
        self.embedded.append(list(inputs))
        return [[0.0]]


def _warm_cfg(tmp_path, *, reranker=False):
    from memo.config import Config

    return Config(
        data_dir=tmp_path / "d",
        vault_path=None,
        state_dir=tmp_path / "s",
        reranker_enabled=reranker,
    )


def test_warm_embedder_warms_and_stamps(tmp_path, monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda cfg: fake)
    cfg = _warm_cfg(tmp_path)
    daemon_mod._warm_embedder(cfg)
    assert fake.embedded == [["warmup"]]
    assert (cfg.state_dir / ".prewarm_ts").is_file()


def test_warm_embedder_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_DISABLE", "1")
    called = []
    monkeypatch.setattr(
        "memo.embedder_select.make_embedder", lambda cfg: called.append(1) or _FakeEmbedder()
    )
    cfg = _warm_cfg(tmp_path)
    daemon_mod._warm_embedder(cfg)
    assert called == []
    assert not (cfg.state_dir / ".prewarm_ts").exists()


def test_warm_embedder_warms_reranker_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda cfg: _FakeEmbedder())
    warmed = []

    class _FakeReranker:
        def __init__(self, model_path=None, revision=None):
            pass

        def warmup(self):
            warmed.append(1)

    monkeypatch.setattr("memo.reranker.MLXReranker", _FakeReranker)
    cfg = _warm_cfg(tmp_path, reranker=True)
    daemon_mod._warm_embedder(cfg, warm_reranker=True)
    assert warmed == [1]
    assert (cfg.state_dir / ".prewarm_ts").is_file()


def test_warm_embedder_skips_reranker_when_flag_false(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda cfg: _FakeEmbedder())
    warmed = []

    class _FakeReranker:
        def __init__(self, **_):
            pass

        def warmup(self):
            warmed.append(1)

    monkeypatch.setattr("memo.reranker.MLXReranker", _FakeReranker)
    cfg = _warm_cfg(tmp_path, reranker=True)
    daemon_mod._warm_embedder(cfg, warm_reranker=False)
    assert warmed == []  # reranker skipped even though enabled
    assert (cfg.state_dir / ".prewarm_ts").is_file()


def test_warm_embedder_download_all_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda cfg: _FakeEmbedder())
    seen = []
    monkeypatch.setattr(
        "memo.model_pins.resolve_model_snapshot",
        lambda repo, revision: seen.append(repo) or "/snap",
    )
    cfg = _warm_cfg(tmp_path)
    daemon_mod._warm_embedder(cfg, download_all=True)
    assert len(seen) == 2  # llm + helper

    def _boom(repo, revision):
        raise RuntimeError("network down")

    monkeypatch.setattr("memo.model_pins.resolve_model_snapshot", _boom)
    # The download failure is caught and logged, never raised.
    daemon_mod._warm_embedder(_warm_cfg(tmp_path / "b", reranker=False), download_all=True)


def test_warm_embedder_swallows_embed_failure(tmp_path, monkeypatch):
    def _boom(cfg):
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr("memo.embedder_select.make_embedder", _boom)
    monkeypatch.setenv("MEMO_RECALL_DEBUG", "1")
    # Best-effort: a warm failure must never raise out of the hook path.
    daemon_mod._warm_embedder(_warm_cfg(tmp_path))


def test_prewarm_command_invokes_warm_embedder(monkeypatch):
    seen = []
    monkeypatch.setattr(daemon_mod, "_warm_embedder", lambda cfg, **kw: seen.append(kw))
    result = CliRunner().invoke(daemon_mod.prewarm, [])
    assert result.exit_code == 0
    assert seen and seen[0].get("download_all") is False
