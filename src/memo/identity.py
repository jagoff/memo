"""Stable process identity and canonical identity policy for memories.

The historical ``Identity``/``current`` API identifies the machine and agent
session. Canonical memory functions are colocated here so write, reindex,
migration, and diagnostics cannot drift into subtly different identity rules.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memo.errors import IdentityConflictError
from memo.project import LIFECYCLE_ARCHIVE_DIRS, slugify_project

GLOBAL_NAMESPACE = "_global"
UNSCOPED_NAMESPACE = "_unscoped"
PROJECT_NAMESPACE_PREFIX = "project:"

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Identity:
    """Who this memo process is. Immutable snapshot resolved at use time."""

    machine_id: str
    hostname: str
    session_id: str | None
    terminal: str | None

    @property
    def label(self) -> str:
        value = self.hostname
        if self.session_id:
            value = f"{value}·{self.session_id[:8]}"
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "session_id": self.session_id,
            "terminal": self.terminal,
            "label": self.label,
        }


def _hostname() -> str:
    try:
        return socket.gethostname().strip() or "unknown-host"
    except OSError:
        return "unknown-host"


def _session_id() -> str | None:
    for key in ("MEMO_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _terminal() -> str | None:
    for fd in (0, 1, 2):
        try:
            return os.ttyname(fd)
        except OSError:
            continue
    return None


def current(cfg: Any) -> Identity:
    """Resolve stable machine identity plus optional session provenance."""
    return Identity(
        machine_id=str(getattr(cfg, "device_id", "") or "unknown"),
        hostname=_hostname(),
        session_id=_session_id(),
        terminal=_terminal(),
    )


@dataclass(frozen=True)
class IdentityKeys:
    namespace: str
    topic_key: str | None
    normalized_title: str
    normalized_content_hash: str


def _canonical_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return _WS_RE.sub(" ", normalized.strip()).casefold()


def canonical_topic_key(value: str | None) -> str | None:
    """Canonical topic key, or ``None`` for absent/whitespace-only input."""
    if value is None:
        return None
    canonical = _canonical_identity_text(value)
    return canonical or None


def normalized_title(value: str) -> str:
    """Normalize title spelling without erasing word boundaries."""
    return _canonical_identity_text(value)


def normalized_content(value: str) -> str:
    """Canonical content used only for exact identity hashing.

    Internal whitespace remains significant. Only Unicode/newline spelling,
    trailing line whitespace, and outer whitespace are normalized.
    """
    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalized_content_hash(value: str) -> str:
    return hashlib.sha256(normalized_content(value).encode("utf-8")).hexdigest()


def _project_slugs(tags: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    slugs: list[str] = []
    invalid = False
    for tag in tags:
        folded = str(tag).strip().casefold()
        if not folded.startswith(PROJECT_NAMESPACE_PREFIX):
            continue
        slug = slugify_project(folded[len(PROJECT_NAMESPACE_PREFIX) :])
        if not slug:
            invalid = True
            continue
        if slug not in slugs:
            slugs.append(slug)
    return tuple(slugs), invalid


def namespace_for_write(tags: Sequence[str], *, auto_project: bool) -> str:
    """Derive the namespace for a new write and reject ambiguous project tags."""
    slugs, invalid = _project_slugs(tags)
    if invalid or len(slugs) > 1:
        raise IdentityConflictError(
            kind="ambiguous_namespace",
            incoming={"project_tags": slugs, "invalid_project_tag": invalid},
        )
    if slugs:
        return f"{PROJECT_NAMESPACE_PREFIX}{slugs[0]}"
    return UNSCOPED_NAMESPACE if auto_project else GLOBAL_NAMESPACE


def namespace_for_index(tags: Sequence[str], *, path: str) -> str | None:
    """Derive namespace for an existing Markdown/index row.

    Historical rows with incompatible/invalid project tags remain ambiguous
    (``None``). Untagged legacy paths are unscoped unless their first path
    component explicitly identifies the global/unscoped bucket.
    """
    slugs, invalid = _project_slugs(tags)
    if invalid or len(slugs) > 1:
        return None
    if slugs:
        return f"{PROJECT_NAMESPACE_PREFIX}{slugs[0]}"
    first = path.replace("\\", "/").strip("/").split("/", 1)[0]
    if first == GLOBAL_NAMESPACE:
        return GLOBAL_NAMESPACE
    if first == UNSCOPED_NAMESPACE:
        return UNSCOPED_NAMESPACE
    return UNSCOPED_NAMESPACE


def bucket_for_namespace(namespace: str) -> str:
    """Safe physical folder for a canonical namespace."""
    if namespace in {GLOBAL_NAMESPACE, UNSCOPED_NAMESPACE}:
        return namespace
    if not namespace.startswith(PROJECT_NAMESPACE_PREFIX):
        raise ValueError(f"unknown memory namespace: {namespace!r}")
    slug = slugify_project(namespace[len(PROJECT_NAMESPACE_PREFIX) :])
    if not slug:
        raise ValueError("project namespace must contain a non-empty slug")
    return f"_{slug}" if slug in LIFECYCLE_ARCHIVE_DIRS else slug


def identity_keys(
    *,
    title: str,
    content: str,
    tags: Sequence[str],
    topic_key: str | None,
    auto_project: bool,
) -> IdentityKeys:
    return IdentityKeys(
        namespace=namespace_for_write(tags, auto_project=auto_project),
        topic_key=canonical_topic_key(topic_key),
        normalized_title=normalized_title(title),
        normalized_content_hash=normalized_content_hash(content),
    )
