"""Unit tests for the journey-check ORCHESTRATION layer.

These test aggregation, exit-code mapping, ``--only`` selection, the raising-check
contract, and ``--json`` shape with STUBBED checks — deterministic, no MLX, no
store. The real checks are the machine-local integration layer (MLX-gated) and
are exercised by running ``memo journey-check`` on Apple Silicon.
"""

from __future__ import annotations

import json

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
