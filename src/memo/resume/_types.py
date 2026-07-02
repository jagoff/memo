from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol


def utc_now_iso() -> str:
    """UTC now as an ISO-8601 string with a ``Z`` suffix.

    Inlined here (was ``synapse.models.utc_now_iso``) so the resume package
    carries no cross-repo dependency — memo owns its own resume surface.
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


ResumeAgent = Literal[
    "all", "codex", "claude", "devin", "devin-desktop", "gemini", "cursor", "opencode", "generic"
]
ResumeMode = Literal["native_resume", "context_resume", "inspect_only"]
_ResumeKey = Literal["up", "down", "enter", "quit", ""]

RESUME_AGENT_CHOICES = (
    "all",
    "codex",
    "claude",
    "devin",
    "devin-desktop",
    "gemini",
    "cursor",
    "opencode",
    "generic",
)
MEMO_RESUME_REPORT_SCHEMA = "memo.resume_candidates.v1"
_RESUME_ESCAPE_TIMEOUT_SECONDS = 0.25
_ANSI_RESET = "\x1b[0m"
_RESUME_CYAN = "\x1b[36m"
_RESUME_AGENT_COLORS = {
    "codex": "\x1b[36m",  # cyan
    "claude": "\x1b[32m",  # green
    "devin": "\x1b[35m",  # magenta/purple
    "devin-desktop": "\x1b[35m",
    "gemini": "\x1b[94m",  # bright blue
    "cursor": "\x1b[37m",
    "generic": "\x1b[90m",
}

_KNOWN_AGENTS = set(RESUME_AGENT_CHOICES) - {"all"}
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\a]*(?:\a|\x1b\\)")


@dataclass(frozen=True)
class ResumeCandidate:
    agent: str
    provider: str
    uri: str
    session_id: str
    title: str
    updated_at: str
    cwd: str = ""
    summary: str = ""
    status: str = ""  # "active" | "recent" | "stale" | "" (idle/ended)
    resume_mode: ResumeMode = "context_resume"
    resume_command: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResumeProviderError:
    provider: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "detail": self.detail}


@dataclass(frozen=True)
class ResumeDiscoveryReport:
    schema: str
    generated_at: str
    agent: str
    cwd: str
    limit: int
    include_all_cwd: bool
    candidates: list[ResumeCandidate] = field(default_factory=list)
    provider_errors: list[ResumeProviderError] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "agent": self.agent,
            "cwd": self.cwd,
            "limit": self.limit,
            "include_all_cwd": self.include_all_cwd,
            "candidates": [item.to_dict() for item in self.candidates],
            "provider_errors": [item.to_dict() for item in self.provider_errors],
        }


class ResumeProvider(Protocol):
    name: str

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]: ...
