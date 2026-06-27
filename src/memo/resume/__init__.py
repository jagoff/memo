"""memo resume — cross-agent session discovery + native resume.

Ported from synapse's resume package (Forma B): memo gains parity with
``synapse resume`` (discover recent/active sessions across codex, claude,
devin, gemini, opencode and resume them natively) while keeping its
individuality — memo's own recall-grounded session snapshots
(``state_dir/sessions/*.json``) are a first-class provider here, and there is
**zero** memflow/synapse coupling (the cross-machine/checkpoint federation
stays in synapse).
"""
from __future__ import annotations

# Re-exported at package level so monkeypatching ``memo.resume.os.execvp`` /
# ``memo.resume.shutil.which`` in tests propagates to submodules (module objects
# are singletons, so patching the attribute reaches every importer).
import os  # noqa: F401
import shutil  # noqa: F401

from ._formatting import (
    execute_resume_candidate,
    format_context_resume,
    format_resume_candidates,
    resolve_resume_candidate,
)
from ._orchestration import discover_resume_candidates
from ._parsers import (
    _claude_candidate,
    _codex_candidate,
    _devin_candidate,
    _gemini_candidate,
    _is_claude_subagent_transcript,
)
from ._providers import (
    ClaudeNativeProvider,
    CodexNativeProvider,
    DevinNativeProvider,
    GeminiNativeProvider,
    MemoSnapshotProvider,
    OpencodeNativeProvider,
    default_resume_providers,
)
from ._tui import pick_resume_candidate_interactive
from ._types import (
    MEMO_RESUME_REPORT_SCHEMA,
    RESUME_AGENT_CHOICES,
    ResumeAgent,
    ResumeCandidate,
    ResumeDiscoveryReport,
    ResumeMode,
    ResumeProvider,
    ResumeProviderError,
)

__all__ = [
    "MEMO_RESUME_REPORT_SCHEMA",
    "RESUME_AGENT_CHOICES",
    "ClaudeNativeProvider",
    "CodexNativeProvider",
    "DevinNativeProvider",
    "GeminiNativeProvider",
    "MemoSnapshotProvider",
    "OpencodeNativeProvider",
    "ResumeAgent",
    "ResumeCandidate",
    "ResumeDiscoveryReport",
    "ResumeMode",
    "ResumeProvider",
    "ResumeProviderError",
    "_claude_candidate",
    "_codex_candidate",
    "_devin_candidate",
    "_gemini_candidate",
    "_is_claude_subagent_transcript",
    "default_resume_providers",
    "discover_resume_candidates",
    "execute_resume_candidate",
    "format_context_resume",
    "format_resume_candidates",
    "pick_resume_candidate_interactive",
    "resolve_resume_candidate",
]
