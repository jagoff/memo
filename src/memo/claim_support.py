"""Pure evidence-ref detector for outcome claims (claim-support).

An outcome claim ("works / fixed / shipped / faster / secure / tested") should
carry a verifiable evidence ref (commit:/pr:/bench:/ci:/tests green). A claim
with NO ref, or a `commit:<sha>` that does not exist locally, is 'unsupported'
— the caller DOWNGRADES confidence (never drops the memory). Hedged or
first-person-intent statements are exempt. No LLM, no MLX.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# claim keyword -> short kind label
_CLAIM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(fixed|resolved|works|working|passes|passing)\b", re.I), "fixed"),
    (re.compile(r"\b(shipped|merged|released|deployed|landed)\b", re.I), "shipped"),
    (re.compile(r"\b(faster|speedup|sped up|optimiz(?:ed|es|ation))\b", re.I), "faster"),
    (re.compile(r"\b(secure|hardened|sanitiz(?:ed|es))\b", re.I), "secure"),
    (re.compile(r"\b(tested|covered by tests)\b", re.I), "tested"),
)

# exemptions: hedges + first-person intent/opinion (not an assertion of outcome)
_HEDGE_RE = re.compile(
    r"\b(i think|i believe|maybe|probably|might|should|seems|likely|not sure|"
    r"i guess|i'?m going to|i will|let'?s|we could|planning to|todo)\b",
    re.I,
)

# evidence refs
_COMMIT_RE = re.compile(r"\bcommit:([0-9a-f]{7,40})\b", re.I)
_OTHER_REF_RE = re.compile(
    r"\b(pr:#?\d+|bench:\S+|ci:\S+|tests?\s+(?:green|pass(?:ing|ed)?))\b", re.I
)


@dataclass(frozen=True)
class ClaimSupportResult:
    unsupported: bool
    claim_kind: str  # "" when the text makes no outcome claim
    reason: str


def _commit_exists(sha: str, repo_root: Path | None) -> bool:
    """True if `sha` resolves to a commit in `repo_root` (or cwd). Fail-open:
    if git is unavailable or errors, return True (do not penalize on tooling gaps)."""
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return proc.returncode == 0


def check_claim_support(text: str, *, repo_root: Path | None = None) -> ClaimSupportResult:
    """Classify an outcome claim's evidence. `unsupported=True` ⇒ caller downgrades."""
    if not text or not text.strip():
        return ClaimSupportResult(False, "", "empty")
    if _HEDGE_RE.search(text):
        return ClaimSupportResult(False, "", "hedged/first-person")

    kind = ""
    for pat, label in _CLAIM_PATTERNS:
        if pat.search(text):
            kind = label
            break
    if not kind:
        return ClaimSupportResult(False, "", "no outcome claim")

    commit_m = _COMMIT_RE.search(text)
    if commit_m:
        if _commit_exists(commit_m.group(1), repo_root):
            return ClaimSupportResult(False, kind, "backed by existing commit")
        return ClaimSupportResult(True, kind, f"commit:{commit_m.group(1)} does not exist locally")
    if _OTHER_REF_RE.search(text):
        return ClaimSupportResult(False, kind, "backed by evidence ref")
    return ClaimSupportResult(True, kind, f"{kind} claim with no evidence ref")
