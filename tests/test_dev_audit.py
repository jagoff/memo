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


def test_every_classified_file_is_under_lexical_enforcement() -> None:
    """An allowlist entry outside the target set would escape BOTH gates.

    scripts/quality_gate.py excludes classified sites from the integer budget,
    but tests/test_dev_audit.py only checks files in
    BROAD_EXCEPTION_TARGET_FILES. An entry naming a file outside that set would
    therefore be billed by neither gate.
    """
    classified_files = {relpath for relpath, _scope, _ordinal in BROAD_EXCEPTION_ALLOWED}
    assert classified_files <= BROAD_EXCEPTION_TARGET_FILES


def test_ratchet_exemptions_outside_target_files_stay_ratchet_protected() -> None:
    """BROAD_EXCEPTION_RATCHET_EXEMPTIONS does not reopen the ALLOWED seam.

    Three of its four entries (briefing.py, constitution.py, repo_eval.py) name
    files outside BROAD_EXCEPTION_TARGET_FILES, unlike BROAD_EXCEPTION_ALLOWED
    (see test_every_classified_file_is_under_lexical_enforcement above) — that
    is by design: ratchet exemptions cover isolated fail-open sites anywhere,
    not just the four lexically-audited targets.

    This is safe for two independent reasons this test pins:

    1. scripts/quality_gate.py's per-file budget is a general, unrestricted
       scan — it is not limited to target files the way the lexical test is.
       Recomputing counts with only BROAD_EXCEPTION_ALLOWED excluded (i.e. as
       if the ratchet exemption did not exist) shows each site would resolve
       to nonzero debt, so the exemption is a precise, singular carve-out, not
       a silent whole-file exclusion — removing it (without also touching the
       source) makes the site billed again immediately.
    2. Unlike BROAD_EXCEPTION_ALLOWED, which has no size/membership pin,
       BROAD_EXCEPTION_RATCHET_EXEMPTIONS is pinned to an exact literal by
       test_broad_exception_ratchet_exemptions_are_exact_and_present, so a new
       entry cannot be added silently — it requires a matching, reviewed
       change to that literal.
    """
    gate = _quality_gate()
    non_target_exemptions = {
        (relpath, scope, ordinal)
        for relpath, scope, ordinal in BROAD_EXCEPTION_RATCHET_EXEMPTIONS
        if relpath not in BROAD_EXCEPTION_TARGET_FILES
    }
    assert non_target_exemptions, "expected at least one non-target ratchet exemption"

    counts_without_ratchet_carveout = gate.collect_broad_exceptions(
        ROOT, exemptions=BROAD_EXCEPTION_ALLOWED
    )
    for relpath, _scope, _ordinal in non_target_exemptions:
        key = f"src/memo/{relpath}"
        assert counts_without_ratchet_carveout.get(key, 0) > 0, (
            f"{relpath}'s ratchet-exempted site carries zero numeric budget even "
            "without its own exemption entry — it is not a target file, so it "
            "would escape both gates"
        )


def test_exception_policy_doc_exists() -> None:
    policy = ROOT / "docs" / "engineering" / "exception-policy.md"
    text = policy.read_text(encoding="utf-8")
    assert "hook hot path" in text
    assert "user-visible CLI" in text
    assert "destructive write paths" in text
