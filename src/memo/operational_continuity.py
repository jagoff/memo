"""Deterministic continuity composition from durable and operational Memo state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_MAX_CHARS = 12_000
_TERMINAL_DELIVERY_STATES = frozenset({"acknowledged", "expired"})


@dataclass(frozen=True)
class ContinuitySource:
    kind: str
    id: str
    title: str


@dataclass(frozen=True)
class ContinuityPacket:
    text: str
    sources: tuple[ContinuitySource, ...]
    omissions: tuple[str, ...]
    durable_available: bool
    operational_available: bool
    fallbacks: tuple[str, ...]


@dataclass
class _Section:
    title: str
    records: list[tuple[str, ContinuitySource | None]]


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _stable_source(kind: str, identifier: str, title: str) -> ContinuitySource:
    normalized = identifier.strip()
    if not normalized:
        normalized = hashlib.sha256(f"{kind}\0{title}".encode()).hexdigest()[:24]
    return ContinuitySource(kind=kind, id=normalized, title=title)


def _single_line(value: object, *, limit: int = 320) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _tuple_field(value: object, name: str) -> tuple[object, ...]:
    candidate = _field(value, name, ())
    if isinstance(candidate, (list, tuple, set, frozenset)):
        return tuple(candidate)
    return ()


class ContinuityComposer:
    """Compose a bounded resume packet without consulting retired runtimes."""

    def __init__(
        self,
        *,
        durable_briefing: Callable[..., str],
        coordination: Any,
        delivery: Any,
        presence: Any,
        sessions: Any,
        health: Callable[[], object],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.durable_briefing = durable_briefing
        self.coordination = coordination
        self.delivery = delivery
        self.presence = presence
        self.sessions = sessions
        self.health = health
        self.clock = clock or (lambda: datetime.now(UTC))

    def compose(
        self,
        *,
        query: str = "",
        cwd: str | None = None,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> ContinuityPacket:
        if isinstance(max_chars, bool) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        budget = min(max_chars, _DEFAULT_MAX_CHARS)
        omissions: list[str] = []
        fallbacks: list[str] = []
        sections = [
            _Section("Durable briefing", []),
            _Section("Handoffs", []),
            _Section("Pending deliveries and conflicts", []),
            _Section("Checkpoint", []),
            _Section("Runtime health", []),
        ]

        durable_available = self._collect_durable(
            sections[0], query=query, cwd=cwd, omissions=omissions, fallbacks=fallbacks
        )
        operational_reads = 0
        operational_reads += self._collect_handoffs(
            sections[1], omissions=omissions, fallbacks=fallbacks
        )
        operational_reads += self._collect_delivery(
            sections[2], omissions=omissions, fallbacks=fallbacks
        )
        project = Path(cwd).resolve().name if cwd else ""
        operational_reads += self._collect_conflicts(
            sections[2],
            project=project,
            omissions=omissions,
            fallbacks=fallbacks,
        )
        operational_reads += self._collect_checkpoint(
            sections[3],
            project=project,
            workspace=cwd,
            omissions=omissions,
            fallbacks=fallbacks,
        )
        operational_reads += self._collect_health(
            sections[4], omissions=omissions, fallbacks=fallbacks
        )
        operational_available = operational_reads > 0
        if not operational_available:
            sections[2].records.append(("Operational state unavailable.", None))

        text, included_sources = self._bounded_render(
            sections,
            omissions=omissions,
            budget=budget,
        )
        return ContinuityPacket(
            text=text,
            sources=tuple(included_sources),
            omissions=tuple(omissions),
            durable_available=durable_available,
            operational_available=operational_available,
            fallbacks=tuple(fallbacks),
        )

    def _collect_durable(
        self,
        section: _Section,
        *,
        query: str,
        cwd: str | None,
        omissions: list[str],
        fallbacks: list[str],
    ) -> bool:
        try:
            value = str(self.durable_briefing(query=query, cwd=cwd) or "").strip()
        except Exception:
            self._unavailable(
                "durable briefing unavailable", omissions=omissions, fallbacks=fallbacks
            )
            section.records.append(("Durable briefing unavailable.", None))
            return False
        if not value:
            self._unavailable(
                "durable briefing unavailable", omissions=omissions, fallbacks=fallbacks
            )
            section.records.append(("Durable briefing unavailable.", None))
            return False
        source = _stable_source("durable", hashlib.sha256(value.encode()).hexdigest(), "briefing")
        for line in value.splitlines():
            normalized = _single_line(line)
            if normalized:
                section.records.append((normalized, source))
        return True

    def _collect_handoffs(
        self,
        section: _Section,
        *,
        omissions: list[str],
        fallbacks: list[str],
    ) -> int:
        try:
            values = self.coordination.handoffs()
            rows = values.values() if isinstance(values, Mapping) else values
            active = sorted(
                (item for item in rows if _field(item, "status") == "open"),
                key=lambda item: (str(_field(item, "created_at")), str(_field(item, "id"))),
            )
        except Exception:
            self._unavailable("handoffs unavailable", omissions=omissions, fallbacks=fallbacks)
            section.records.append(("Handoffs unavailable.", None))
            return 0
        for item in active:
            identifier = str(_field(item, "id"))
            summary = _single_line(_field(item, "summary")) or "No summary"
            target = _single_line(_field(item, "to_actor")) or "unassigned"
            source = _stable_source("handoff", identifier, summary)
            section.records.append((f"- {identifier} → {target}: {summary}", source))
        return 1

    def _collect_delivery(
        self,
        section: _Section,
        *,
        omissions: list[str],
        fallbacks: list[str],
    ) -> int:
        try:
            rows = sorted(
                (
                    item
                    for item in self.delivery.deliveries()
                    if str(_field(item, "state")) not in _TERMINAL_DELIVERY_STATES
                ),
                key=lambda item: str(_field(item, "id")),
            )
        except Exception:
            self._unavailable("deliveries unavailable", omissions=omissions, fallbacks=fallbacks)
            section.records.append(("Deliveries unavailable.", None))
            return 0
        for item in rows:
            identifier = str(_field(item, "id"))
            state = str(_field(item, "state"))
            target = str(_field(item, "target_id"))
            attempt = int(str(_field(item, "attempt_count", 0)))
            source = _stable_source("delivery", identifier, state)
            section.records.append(
                (f"- Delivery {identifier} → {target}: {state} (attempt {attempt})", source)
            )
        return 1

    def _collect_conflicts(
        self,
        section: _Section,
        *,
        project: str,
        omissions: list[str],
        fallbacks: list[str],
    ) -> int:
        try:
            now = self.clock().astimezone(UTC)
            leases = self.presence.active(project=project, now=now)
            files = tuple(
                sorted({str(file) for lease in leases for file in _tuple_field(lease, "files")})
            )
            conflicts = sorted(
                self.presence.conflicts(project=project, files=files, now=now),
                key=lambda item: str(_field(item, "file")),
            )
        except Exception:
            self._unavailable(
                "presence conflicts unavailable", omissions=omissions, fallbacks=fallbacks
            )
            section.records.append(("Presence conflicts unavailable.", None))
            return 0
        for item in conflicts:
            file = str(_field(item, "file"))
            lease_ids = tuple(str(value) for value in _tuple_field(item, "lease_ids"))
            source = _stable_source("workspace_conflict", file, file)
            section.records.append((f"- Conflict {file}: {', '.join(lease_ids)}", source))
        return 1

    def _collect_checkpoint(
        self,
        section: _Section,
        *,
        project: str,
        workspace: str | None,
        omissions: list[str],
        fallbacks: list[str],
    ) -> int:
        try:
            checkpoint = self.sessions.latest_recoverable(
                project=project or None,
                workspace=workspace,
            )
        except Exception:
            self._unavailable("checkpoint unavailable", omissions=omissions, fallbacks=fallbacks)
            section.records.append(("Checkpoint unavailable.", None))
            return 0
        if checkpoint is not None:
            identifier = str(_field(checkpoint, "session_id"))
            summary = _single_line(_field(checkpoint, "summary")) or "No summary"
            branch = _single_line(_field(checkpoint, "branch"))
            head = _single_line(_field(checkpoint, "head"))
            reason = _single_line(_field(checkpoint, "recoverable_reason"))
            source = _stable_source("checkpoint", identifier, summary)
            section.records.append(
                (
                    f"- {identifier}: {summary} · {branch}@{head}"
                    + (f" · {reason}" if reason else ""),
                    source,
                )
            )
        return 1

    def _collect_health(
        self,
        section: _Section,
        *,
        omissions: list[str],
        fallbacks: list[str],
    ) -> int:
        try:
            health = self.health()
            if isinstance(health, Mapping):
                wire = json.dumps(
                    health,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                verdict = str(health.get("verdict", "unknown"))
            else:
                values = vars(health)
                wire = json.dumps(
                    values,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                verdict = str(getattr(health, "verdict", "unknown"))
        except Exception:
            self._unavailable(
                "runtime health unavailable", omissions=omissions, fallbacks=fallbacks
            )
            section.records.append(("Runtime health unavailable.", None))
            return 0
        source = _stable_source(
            "runtime_health", hashlib.sha256(wire.encode()).hexdigest(), verdict
        )
        section.records.append((f"- {verdict}: {_single_line(wire, limit=480)}", source))
        return 1

    @staticmethod
    def _unavailable(
        value: str,
        *,
        omissions: list[str],
        fallbacks: list[str],
    ) -> None:
        if value not in omissions:
            omissions.append(value)
        if value not in fallbacks:
            fallbacks.append(value)

    @staticmethod
    def _render(sections: list[_Section], omissions: list[str]) -> str:
        lines: list[str] = []
        for section in sections:
            lines.extend((f"## {section.title}", ""))
            lines.extend(record for record, _source in section.records)
            if not section.records:
                lines.append("None.")
            lines.append("")
        lines.extend(("## Omissions", ""))
        if omissions:
            lines.extend(f"- {item}" for item in omissions)
        else:
            lines.append("None.")
        return "\n".join(lines).rstrip() + "\n"

    def _bounded_render(
        self,
        sections: list[_Section],
        *,
        omissions: list[str],
        budget: int,
    ) -> tuple[str, list[ContinuitySource]]:
        text = self._render(sections, omissions)
        while len(text) > budget:
            trimmed = False
            for section in reversed(sections):
                sourced = next(
                    (
                        index
                        for index in range(len(section.records) - 1, -1, -1)
                        if section.records[index][1] is not None
                    ),
                    None,
                )
                if sourced is None:
                    continue
                section.records.pop(sourced)
                marker = f"{section.title.lower()} truncated by character budget"
                if marker not in omissions:
                    omissions.append(marker)
                trimmed = True
                break
            if not trimmed:
                break
            text = self._render(sections, omissions)
        if len(text) > budget:
            if budget == 1:
                text = "…"
            else:
                boundary = text.rfind("\n", 0, budget - 1)
                cut = boundary if boundary > 0 else budget - 1
                text = text[:cut].rstrip() + "…"
        sources: list[ContinuitySource] = []
        seen: set[tuple[str, str]] = set()
        for section in sections:
            for _record, source in section.records:
                if source is None or (source.kind, source.id) in seen:
                    continue
                sources.append(source)
                seen.add((source.kind, source.id))
        return text, sources


__all__ = ["ContinuityComposer", "ContinuityPacket", "ContinuitySource"]
