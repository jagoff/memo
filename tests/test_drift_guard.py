"""GC-8 Drift Guard — catch code that violates your own durable constraints.

High-precision v1 by design (the #1 risk is false positives): a rule only
becomes an enforceable prohibition when it (a) carries a negative-polarity
marker (never / don't / avoid / nunca / evitá / …) AND (b) names the banned
pattern inside backticks or quotes. Semantic drift (LLM judgement) is a later,
gated addition. The pure functions take plain strings so they unit-test without
a Memory or a git repo.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from memo.drift_guard import (
    added_lines_from_diff,
    code_spans,
    parse_prohibitions,
    scan,
)

# rule = (memory_id, text)
R_BAN = ("abcd1234", "Never use `git add -A` in the shared worktree")
R_BAN_Q = ("beef5678", 'Avoid the "requests" library; prefer httpx')
R_POSITIVE = ("cafe9999", "Always run `pytest` before committing")  # not a prohibition
R_NOSPAN = ("dead0000", "Never commit secrets")  # prohibition but no delimited pattern


# --- code_spans (pure) -------------------------------------------------------


def test_code_spans_extracts_backticks_and_quotes() -> None:
    assert code_spans("use `git add -A` here") == ["git add -A"]
    assert code_spans('the "requests" lib') == ["requests"]
    assert code_spans("plain text") == []


# --- parse_prohibitions (pure) -----------------------------------------------


def test_parse_keeps_negative_rules_with_a_delimited_pattern() -> None:
    prohibitions = parse_prohibitions([R_BAN, R_BAN_Q])
    ids = {p.rule_id for p in prohibitions}
    assert ids == {"abcd1234", "beef5678"}
    ban = next(p for p in prohibitions if p.rule_id == "abcd1234")
    assert "git add -A" in ban.patterns


def test_parse_drops_positive_rules() -> None:
    assert parse_prohibitions([R_POSITIVE]) == []


def test_parse_drops_prohibitions_without_a_delimited_pattern() -> None:
    # honest v1 limitation: no backticks/quotes → nothing precise to match
    assert parse_prohibitions([R_NOSPAN]) == []


# --- added_lines_from_diff (pure) --------------------------------------------

_DIFF = """diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 import os
-os.system("git status")
+os.system("git add -A")
+x = 1
"""


def test_added_lines_tracks_path_and_ignores_context_and_removed() -> None:
    added = added_lines_from_diff(_DIFF)
    texts = [line for _path, line in added]
    assert 'os.system("git add -A")' in texts
    assert "x = 1" in texts
    assert "import os" not in texts  # context line ignored
    assert all(path == "app.py" for path, _ in added)
    assert not any("git status" in t for t in texts)  # removed line ignored


def test_added_lines_ignores_plusplusplus_header() -> None:
    added = added_lines_from_diff(_DIFF)
    assert not any("b/app.py" in line for _p, line in added)


# --- scan (pure, the payoff) -------------------------------------------------


def test_scan_flags_added_line_violating_a_prohibition() -> None:
    prohibitions = parse_prohibitions([R_BAN])
    added = added_lines_from_diff(_DIFF)
    violations = scan(prohibitions, added)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule_id == "abcd1234"
    assert v.pattern == "git add -A"
    assert v.path == "app.py"


def test_scan_clean_diff_has_no_violations() -> None:
    clean = """--- a/x.py
+++ b/x.py
@@ -0,0 +1 @@
+print("hello")
"""
    violations = scan(parse_prohibitions([R_BAN, R_BAN_Q]), added_lines_from_diff(clean))
    assert violations == []


def test_scan_matches_quoted_pattern_case_insensitively() -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+import Requests\n"
    violations = scan(parse_prohibitions([R_BAN_Q]), added_lines_from_diff(diff))
    assert len(violations) == 1
    assert violations[0].pattern == "requests"


# --- CLI wiring (monkeypatch the git + store seams) --------------------------


def test_cli_drift_reports_violation_and_strict_exits_nonzero(monkeypatch) -> None:
    from click.testing import CliRunner

    from memo.cli_drift import drift

    monkeypatch.setattr("memo.cli_drift._git_diff", lambda *, staged, ref: _DIFF)
    monkeypatch.setattr("memo.cli_drift._gather_rules", lambda: [R_BAN])
    runner = CliRunner()

    res = runner.invoke(drift, [], env={"MEMO_NONINTERACTIVE": "1"})
    assert res.exit_code == 0  # warn-only by default
    assert "git add -A" in res.output
    assert "1 drift violation" in res.output

    strict = runner.invoke(drift, ["--strict"], env={"MEMO_NONINTERACTIVE": "1"})
    assert strict.exit_code == 1  # gate mode


def test_cli_drift_clean_diff_is_silent_pass(monkeypatch) -> None:
    from click.testing import CliRunner

    from memo.cli_drift import drift

    clean = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+print('ok')\n"
    monkeypatch.setattr("memo.cli_drift._git_diff", lambda *, staged, ref: clean)
    monkeypatch.setattr("memo.cli_drift._gather_rules", lambda: [R_BAN])
    res = CliRunner().invoke(drift, [], env={"MEMO_NONINTERACTIVE": "1"})
    assert res.exit_code == 0
    assert "clean" in res.output


def test_git_diff_forwards_staged_ref_and_returns_stdout(monkeypatch) -> None:
    from memo.cli_drift import _git_diff

    completed = subprocess.CompletedProcess([], 0, stdout="diff body", stderr="")
    run = MagicMock(return_value=completed)
    monkeypatch.setattr("memo.cli_drift.subprocess.run", run)

    assert _git_diff(staged=True, ref="origin/master") == "diff body"
    run.assert_called_once_with(
        ["git", "diff", "--unified=0", "--no-color", "--cached", "origin/master"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "failure",
    [OSError("git unavailable"), subprocess.TimeoutExpired(["git", "diff"], timeout=30)],
)
def test_git_diff_degrades_to_empty_on_process_failure(monkeypatch, failure) -> None:
    from memo.cli_drift import _git_diff

    monkeypatch.setattr("memo.cli_drift.subprocess.run", MagicMock(side_effect=failure))

    assert _git_diff(staged=False, ref=None) == ""


def test_gather_rules_uses_environment_config_and_memory_facade(monkeypatch) -> None:
    from memo.cli_drift import _gather_rules

    cfg = object()
    memory = object()
    expected = [("rule-1", "Never use `unsafe_call`")]
    from_env = MagicMock(return_value=cfg)
    get_memory = MagicMock(return_value=memory)
    gather_rules = MagicMock(return_value=expected)
    monkeypatch.setattr("memo.config.Config.from_env", from_env)
    monkeypatch.setattr("memo.cli_common.get_memory", get_memory)
    monkeypatch.setattr("memo.constitution.gather_rules", gather_rules)

    assert _gather_rules() == expected
    from_env.assert_called_once_with()
    get_memory.assert_called_once_with(cfg)
    gather_rules.assert_called_once_with(memory, cfg)


def test_cli_drift_reports_when_there_are_no_changes(monkeypatch) -> None:
    from click.testing import CliRunner

    from memo.cli_drift import drift

    gather_rules = MagicMock()
    monkeypatch.setattr("memo.cli_drift._git_diff", lambda *, staged, ref: "\n")
    monkeypatch.setattr("memo.cli_drift._gather_rules", gather_rules)

    result = CliRunner().invoke(drift)

    assert result.exit_code == 0
    assert result.output == "no changes to check\n"
    gather_rules.assert_not_called()


def test_cli_drift_explains_when_rules_have_no_enforceable_prohibition(monkeypatch) -> None:
    from click.testing import CliRunner

    from memo.cli_drift import drift

    monkeypatch.setattr("memo.cli_drift._git_diff", lambda *, staged, ref: _DIFF)
    monkeypatch.setattr("memo.cli_drift._gather_rules", lambda: [R_POSITIVE])

    result = CliRunner().invoke(drift)

    assert result.exit_code == 0
    assert "no enforceable prohibitions" in result.output
