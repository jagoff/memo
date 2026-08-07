from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from memo.dev_audit import (
    BROAD_EXCEPTION_ALLOWED,
    BROAD_EXCEPTION_RATCHET_EXEMPTIONS,
    BROAD_EXCEPTION_TARGET_FILES,
    RAW_MEMO_ENV_ALLOWED,
    find_broad_exception_sites,
    find_raw_memo_env_reads,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "memo"


def _quality_gate() -> ModuleType:
    script = ROOT / "scripts" / "quality_gate.py"
    assert script.is_file(), "scripts/quality_gate.py must exist"
    spec = importlib.util.spec_from_file_location("memo_quality_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_memo_env_reads_are_classified() -> None:
    found = find_raw_memo_env_reads(SRC)
    unclassified = [
        f"{site.path}:{site.line}:{site.name}"
        for site in found
        if (site.relpath, site.name) not in RAW_MEMO_ENV_ALLOWED
    ]
    assert unclassified == []


def test_broad_exception_policy_targets_are_classified() -> None:
    found = find_broad_exception_sites(SRC)
    unclassified = [
        f"{site.path}:{site.line}:{site.scope}:{site.ordinal}"
        for site in found
        if site.relpath in BROAD_EXCEPTION_TARGET_FILES
        and (site.relpath, site.scope, site.ordinal) not in BROAD_EXCEPTION_ALLOWED
    ]
    assert unclassified == []

    # Reverse direction (allowlist ⊆ found): every allow-listed key must still
    # resolve to a real `except Exception` site, so a refactored-away entry
    # cannot linger as a stale broad-exception exemption.
    found_keys = {(site.relpath, site.scope, site.ordinal) for site in found}
    stale = sorted(BROAD_EXCEPTION_ALLOWED - found_keys)
    assert stale == []


def test_files_under_lexical_classification_carry_no_numeric_budget() -> None:
    """The two gates must not both bill the same site.

    For a file whose every broad catch is classified in
    BROAD_EXCEPTION_ALLOWED, the numeric ratchet must count zero — otherwise
    classifying a new fail-open site correctly (gate 1) still fails CI on the
    per-file budget (gate 2), which is what happened on commit a2706251.
    """
    gate = _quality_gate()
    counts = gate.collect_broad_exceptions(ROOT)

    for relpath in BROAD_EXCEPTION_TARGET_FILES:
        assert counts.get(f"src/memo/{relpath}", 0) == 0, (
            f"{relpath} is lexically classified but still carries a numeric budget"
        )


def test_broad_exception_ratchet_exemptions_are_exact_and_present() -> None:
    expected = {
        ("briefing.py", "proactive_compact_line", 1),
        ("cli_recall_hook.py", "_proactive_urgent_line", 1),
        ("constitution.py", "run_mandate_sync_pass", 1),
        ("repo_eval.py", "evaluate_repo_search", 1),
    }
    found = {(site.relpath, site.scope, site.ordinal) for site in find_broad_exception_sites(SRC)}

    assert expected == BROAD_EXCEPTION_RATCHET_EXEMPTIONS
    assert found >= BROAD_EXCEPTION_RATCHET_EXEMPTIONS


def test_exception_policy_doc_exists() -> None:
    policy = ROOT / "docs" / "engineering" / "exception-policy.md"
    text = policy.read_text(encoding="utf-8")
    assert "hook hot path" in text
    assert "user-visible CLI" in text
    assert "destructive write paths" in text
