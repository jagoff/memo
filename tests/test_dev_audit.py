from __future__ import annotations

from pathlib import Path

from memo.dev_audit import (
    BROAD_EXCEPTION_ALLOWED,
    BROAD_EXCEPTION_RATCHET_EXEMPTIONS,
    RAW_MEMO_ENV_ALLOWED,
    find_broad_exception_sites,
    find_raw_memo_env_reads,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "memo"


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
    target_files = {
        "recall_logic.py",
        "memory/write_ops.py",
        "cli_recall_hook.py",
        "store/queries.py",
    }
    unclassified = [
        f"{site.path}:{site.line}:{site.scope}:{site.ordinal}"
        for site in found
        if site.relpath in target_files
        and (site.relpath, site.scope, site.ordinal) not in BROAD_EXCEPTION_ALLOWED
    ]
    assert unclassified == []

    # Reverse direction (allowlist ⊆ found): every allow-listed key must still
    # resolve to a real `except Exception` site, so a refactored-away entry
    # cannot linger as a stale broad-exception exemption.
    found_keys = {(site.relpath, site.scope, site.ordinal) for site in found}
    stale = sorted(BROAD_EXCEPTION_ALLOWED - found_keys)
    assert stale == []


def test_broad_exception_ratchet_exemptions_are_exact_and_present() -> None:
    expected = {
        ("briefing.py", "proactive_compact_line", 1),
        ("cli_recall_hook.py", "_proactive_urgent_line", 1),
        ("constitution.py", "run_mandate_sync_pass", 1),
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
