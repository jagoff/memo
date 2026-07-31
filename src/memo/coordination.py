"""Cross-agent coordination — live collision scan + LLM directives.

Implements docs/SPECS/2026-07-31-cross-agent-coordination-design.md: when two
or more *active* agent sessions touch the same resource (file, branch, daemon
label, or topic), a pure-code candidate pass finds the overlap, the local 4B
helper model judges it (``chat_with_timeout``, fail-open on timeout/garbage),
and the confirmed collision — one directive per side — lands in a sidecar
sqlite DB (``coordination.db``, same WAL/busy-timeout model as
``proactive/store.py``). Delivery rides the recall hook: a single indexed
sqlite read per turn (zero LLM, zero network) renders a ``<memo-coordination>``
block and stamps the side as delivered. The periodic trigger is a daemon
thread started from ``memo watch``, gated by ``MEMO_COORD_ENABLED``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

from memo.flags import flag_bool, flag_int
from memo.llm import ChatBackend
from memo.memory.record import chat_with_timeout, strip_llm_output
from memo.prompt_overrides import resolve_prompt

_log = logging.getLogger(__name__)

DEFAULT_SCAN_INTERVAL_S = 300
DEFAULT_ACTIVE_WINDOW_S = 21600
TOPIC_JACCARD = 0.35
RECENT_MEMORIES_PER_SESSION = 20
_RECENT_ROWS_SCANNED = 400
_MAX_SESSIONS = 50
_MAX_RESOURCES_PER_PAIR = 5
_MAX_JUDGED_PER_SCAN = 12
_JUDGE_TIMEOUT_S = 30.0
_SEVERITIES = frozenset({"info", "warn", "block"})
_ACTIVE_STATUSES = ("open", "delivered")

# Fail-open catch list for the scan/judge paths. Deliberately NOT a bare
# ``except Exception`` — the broad-exception ratchet budgets those per file.
_SCAN_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, ImportError)

_BRANCH_RE = re.compile(
    r"\b(?:feat|fix|chore|docs|refactor|test|perf|ci|build|release|hotfix)/[\w.\-/]+"
)
_BARE_BRANCHES = frozenset({"master", "main"})
_DAEMON_RE = re.compile(r"\bcom\.[a-z0-9][a-z0-9.\-]*[a-z0-9]\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9][\w\-]{2,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "over",
        "under",
        "after",
        "before",
        "while",
        "then",
        "been",
        "will",
        "your",
        "when",
        "not",
        "now",
    }
)

_JUDGE_SYSTEM_PROMPT = """You coordinate multiple AI coding agents working in parallel on one machine.
Given two agents' recent activity and one shared resource, decide whether they
are about to collide (duplicate work, breaking each other's runtime, or racing
on the same branch/PR). Respond with STRICT JSON only, no prose:
{"collision": true|false, "severity": "info"|"warn"|"block", "rationale": "...",
 "directive_a": "...", "directive_b": "..."}
directive_a is an instruction injected into agent A's next turn; directive_b
into agent B's. Keep each directive under 200 characters, concrete and
actionable. If there is no real conflict, return {"collision": false}."""


# ── records ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Collision:
    id: str
    session_a: str
    session_b: str
    resource: str
    kind: str  # file | branch | daemon | pr | topic
    severity: str  # info | warn | block
    rationale: str
    directive_a: str
    directive_b: str
    status: str  # open | delivered | resolved | stale
    created_at: str
    delivered_a: str | None
    delivered_b: str | None


@dataclass(frozen=True)
class Directive:
    collision_id: str
    session_id: str
    other_session: str
    resource: str
    kind: str
    severity: str
    rationale: str
    directive: str


@dataclass(frozen=True)
class SessionActivity:
    session_id: str
    project: str
    titles: tuple[str, ...]
    tokens: frozenset[str]
    files: frozenset[str]
    branches: frozenset[str]
    daemons: frozenset[str]
    focus: str


@dataclass(frozen=True)
class CollisionCandidate:
    session_a: str
    session_b: str
    resource: str
    kind: str


@dataclass(frozen=True)
class ScanResult:
    sessions: int
    candidates: int
    judged: int
    collisions: int
    skipped_active: int


# ── config knobs ─────────────────────────────────────────────────────────────


def coord_enabled() -> bool:
    return flag_bool("MEMO_COORD_ENABLED")


def scan_interval_s() -> float:
    value = flag_int("MEMO_COORD_SCAN_INTERVAL")
    return float(DEFAULT_SCAN_INTERVAL_S if value is None else value)


def active_window_s() -> float:
    value = flag_int("MEMO_COORD_ACTIVE_WINDOW")
    return float(DEFAULT_ACTIVE_WINDOW_S if value is None else value)


def coordination_db_path(cfg: Any) -> Path:
    return Path(cfg.state_dir) / "coordination.db"


def collision_id(session_a: str, session_b: str, resource: str) -> str:
    lo, hi = sorted((session_a, session_b))
    return hashlib.sha256(f"{lo}|{hi}|{resource}".encode()).hexdigest()[:16]


# ── sidecar store (pattern: proactive/store.py) ──────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS collisions (
    id            TEXT PRIMARY KEY,
    session_a     TEXT NOT NULL,
    session_b     TEXT NOT NULL,
    resource      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    severity      TEXT NOT NULL,
    rationale     TEXT,
    directive_a   TEXT NOT NULL,
    directive_b   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    TEXT NOT NULL,
    delivered_a   TEXT,
    delivered_b   TEXT);
CREATE INDEX IF NOT EXISTS idx_collisions_session_a ON collisions(session_a);
CREATE INDEX IF NOT EXISTS idx_collisions_session_b ON collisions(session_b);
"""

_COLUMNS = (
    "id, session_a, session_b, resource, kind, severity, rationale, "
    "directive_a, directive_b, status, created_at, delivered_a, delivered_b"
)


def _row_to_collision(row: sqlite3.Row) -> Collision:
    return Collision(**{key: row[key] for key in row.keys()})  # noqa: SIM118


class CoordinationStore:
    """WAL sidecar holding confirmed collisions and their delivery state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)

    def upsert(self, collision: Collision) -> bool:
        """Insert or refresh a collision. Open/delivered rows win (no re-judge):
        a recurring collision only refreshes a previously resolved/stale row."""
        row = self._conn.execute(
            "SELECT status FROM collisions WHERE id = ?", (collision.id,)
        ).fetchone()
        if row is not None and row["status"] in _ACTIVE_STATUSES:
            return False
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO collisions ({_COLUMNS}) "  # noqa: S608
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    collision.id,
                    collision.session_a,
                    collision.session_b,
                    collision.resource,
                    collision.kind,
                    collision.severity,
                    collision.rationale,
                    collision.directive_a,
                    collision.directive_b,
                    collision.status,
                    collision.created_at,
                    collision.delivered_a,
                    collision.delivered_b,
                ),
            )
        return True

    def active_ids(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT id FROM collisions WHERE status IN (?, ?)", _ACTIVE_STATUSES
        ).fetchall()
        return {r["id"] for r in rows}

    def list_collisions(
        self, *, statuses: tuple[str, ...] | None = None, limit: int = 50
    ) -> list[Collision]:
        sql = f"SELECT {_COLUMNS} FROM collisions "  # noqa: S608
        params: tuple[Any, ...] = ()
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f"WHERE status IN ({placeholders}) "
            params = statuses
        sql += "ORDER BY created_at DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_collision(r) for r in rows]

    def pending_directives(self, session_id: str) -> list[Directive]:
        """Open directives not yet delivered to `session_id`. One indexed read."""
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM collisions WHERE status = 'open' AND "  # noqa: S608
            "((session_a = ? AND delivered_a IS NULL) OR "
            " (session_b = ? AND delivered_b IS NULL)) ORDER BY created_at",
            (session_id, session_id),
        ).fetchall()
        out: list[Directive] = []
        for r in rows:
            side_a = r["session_a"] == session_id and r["delivered_a"] is None
            out.append(
                Directive(
                    collision_id=r["id"],
                    session_id=session_id,
                    other_session=r["session_b"] if side_a else r["session_a"],
                    resource=r["resource"],
                    kind=r["kind"],
                    severity=r["severity"],
                    rationale=r["rationale"] or "",
                    directive=r["directive_a"] if side_a else r["directive_b"],
                )
            )
        return out

    def mark_delivered(self, collision_id: str, session_id: str, now: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE collisions SET delivered_a = ? "
                "WHERE id = ? AND session_a = ? AND delivered_a IS NULL",
                (now, collision_id, session_id),
            )
            self._conn.execute(
                "UPDATE collisions SET delivered_b = ? "
                "WHERE id = ? AND session_b = ? AND delivered_b IS NULL",
                (now, collision_id, session_id),
            )
            self._conn.execute(
                "UPDATE collisions SET status = 'delivered' WHERE id = ? AND "
                "status = 'open' AND delivered_a IS NOT NULL AND delivered_b IS NOT NULL",
                (collision_id,),
            )

    def resolve(self, collision_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE collisions SET status = 'resolved' WHERE id = ?", (collision_id,)
            )
        return cur.rowcount > 0

    def expire_stale(self, cutoff_iso: str) -> int:
        """Mark open/delivered rows created before `cutoff_iso` as stale."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE collisions SET status = 'stale' WHERE status IN (?, ?) AND created_at < ?",
                (*_ACTIVE_STATUSES, cutoff_iso),
            )
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CoordinationStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ── gather: active sessions + capture memories + focus (all fail-open) ───────


def _parse_instant(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _active_session_snaps(state_dir: Path, *, now: datetime) -> list[dict[str, Any]]:
    from memo.session import list_sessions

    try:
        snaps = list_sessions(state_dir, limit=_MAX_SESSIONS)
    except (OSError, ValueError):
        return []
    cutoff = now - timedelta(seconds=active_window_s())
    out: list[dict[str, Any]] = []
    for snap in snaps:
        ts = _parse_instant(snap.get("updated"))
        if snap.get("session_id") and ts is not None and ts >= cutoff:
            out.append(snap)
    return out


def _row_session_id(row: dict[str, Any]) -> str:
    extra = row.get("extra") or {}
    provenance = extra.get("provenance")
    sid = provenance.get("session_id") if isinstance(provenance, dict) else None
    return str(sid or extra.get("session_id") or "")


def _capture_rows_by_session(mem: Any) -> dict[str, list[dict[str, Any]]]:
    try:
        rows = mem.store.list_recent(limit=_RECENT_ROWS_SCANNED)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error):
        _log.debug("coordination: capture rows unavailable", exc_info=True)
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sid = _row_session_id(row)
        if sid:
            grouped.setdefault(sid, []).append(row)
    return grouped


def _focus_by_project(cfg: Any) -> dict[str, str]:
    try:
        from memo.operational import OperationalStore

        state = OperationalStore(cfg.state_dir, device_id=cfg.device_id).state()
        focus = state.get("focus") or {}
        return {
            str(project): str(item.get("summary") or "")
            for project, item in focus.items()
            if isinstance(item, dict)
        }
    except (OSError, ValueError, TypeError, KeyError):
        _log.debug("coordination: operational focus unavailable", exc_info=True)
        return {}


def _row_files(row: dict[str, Any]) -> set[str]:
    extra = row.get("extra") or {}
    files: set[str] = set()
    for key in ("files_read", "files_modified"):
        values = extra.get(key)
        if isinstance(values, list):
            files.update(str(v).strip() for v in values if str(v).strip())
    return files


def build_activity(
    session_id: str,
    project: str,
    rows: list[dict[str, Any]],
    focus: str,
) -> SessionActivity:
    """Project a session's recent capture memories into overlap-detection sets."""
    recent = rows[:RECENT_MEMORIES_PER_SESSION]
    titles = tuple(str(r.get("title") or "") for r in recent)
    tags = (str(t) for r in recent for t in (r.get("tags") or []))
    text = " ".join((*titles, *tags, focus))
    tokens = frozenset(_TOKEN_RE.findall(text.lower())) - _STOPWORDS
    files = frozenset(f for r in recent for f in _row_files(r))
    branches = frozenset(_BRANCH_RE.findall(text)) | (tokens & _BARE_BRANCHES)
    daemons = frozenset(m.lower() for m in _DAEMON_RE.findall(text))
    return SessionActivity(
        session_id=session_id,
        project=project,
        titles=titles,
        tokens=tokens,
        files=files,
        branches=branches,
        daemons=daemons,
        focus=focus,
    )


def gather_activities(mem: Any, cfg: Any, *, now: datetime) -> tuple[SessionActivity, ...]:
    """Active sessions joined with capture memories + operational focus."""
    snaps = _active_session_snaps(Path(cfg.state_dir), now=now)
    if not snaps:
        return ()
    rows_by_session = _capture_rows_by_session(mem)
    focus_by_project = _focus_by_project(cfg)
    out: list[SessionActivity] = []
    for snap in snaps:
        sid = str(snap.get("session_id"))
        project = str(snap.get("project") or "")
        out.append(
            build_activity(
                sid, project, rows_by_session.get(sid, []), focus_by_project.get(project, "")
            )
        )
    return tuple(out)


# ── candidate detection (pure code, no LLM) ──────────────────────────────────


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pair_candidates(
    a: SessionActivity, b: SessionActivity, *, jaccard_threshold: float
) -> list[CollisionCandidate]:
    overlaps = (
        ("file", a.files & b.files),
        ("branch", a.branches & b.branches),
        ("daemon", a.daemons & b.daemons),
    )
    out = [
        CollisionCandidate(a.session_id, b.session_id, resource, kind)
        for kind, resources in overlaps
        for resource in sorted(resources)[:_MAX_RESOURCES_PER_PAIR]
    ]
    if not out and _jaccard(a.tokens, b.tokens) >= jaccard_threshold:
        shared = " ".join(sorted(a.tokens & b.tokens)[:6])
        out.append(CollisionCandidate(a.session_id, b.session_id, shared, "topic"))
    return out


def detect_candidates(
    activities: tuple[SessionActivity, ...], *, jaccard_threshold: float = TOPIC_JACCARD
) -> tuple[CollisionCandidate, ...]:
    return tuple(
        candidate
        for a, b in itertools.combinations(activities, 2)
        for candidate in _pair_candidates(a, b, jaccard_threshold=jaccard_threshold)
    )


# ── judge: 4B helper via chat_with_timeout (fail-open) ───────────────────────


def _activity_summary(activity: SessionActivity) -> str:
    lines = [f"session {activity.session_id} (project: {activity.project or 'unknown'})"]
    if activity.focus:
        lines.append(f"focus: {activity.focus[:200]}")
    if activity.titles:
        lines.append("recent activity: " + "; ".join(t for t in activity.titles[:8] if t)[:600])
    if activity.files:
        lines.append("files touched: " + ", ".join(sorted(activity.files)[:10])[:400])
    return "\n".join(lines)


def _judge_prompt(candidate: CollisionCandidate, a: SessionActivity, b: SessionActivity) -> str:
    return (
        f"Shared resource: {candidate.resource} (kind: {candidate.kind})\n\n"
        f"Agent A —\n{_activity_summary(a)}\n\n"
        f"Agent B —\n{_activity_summary(b)}\n\n"
        "Are these two live agents colliding on the shared resource?"
    )


def _parse_judgement(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(strip_llm_output(raw))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _collision_from_judgement(
    candidate: CollisionCandidate, data: dict[str, Any], *, now_iso: str
) -> Collision | None:
    if not data.get("collision"):
        return None
    directive_a = str(data.get("directive_a") or "").strip()
    directive_b = str(data.get("directive_b") or "").strip()
    if not directive_a or not directive_b:
        return None
    severity = str(data.get("severity") or "").strip().lower()
    return Collision(
        id=collision_id(candidate.session_a, candidate.session_b, candidate.resource),
        session_a=candidate.session_a,
        session_b=candidate.session_b,
        resource=candidate.resource,
        kind=candidate.kind,
        severity=severity if severity in _SEVERITIES else "warn",
        rationale=str(data.get("rationale") or "")[:400],
        directive_a=directive_a[:400],
        directive_b=directive_b[:400],
        status="open",
        created_at=now_iso,
        delivered_a=None,
        delivered_b=None,
    )


def judge_candidate(
    chat: ChatBackend,
    cfg: Any,
    candidate: CollisionCandidate,
    activities: dict[str, SessionActivity],
    *,
    now_iso: str,
) -> Collision | None:
    """Ask the 4B helper whether the candidate is a real collision. Fail-open."""
    a = activities.get(candidate.session_a)
    b = activities.get(candidate.session_b)
    if a is None or b is None:
        return None
    system = resolve_prompt("coordination", _JUDGE_SYSTEM_PROMPT, Path(cfg.state_dir))
    try:
        out = chat_with_timeout(
            chat,
            timeout=_JUDGE_TIMEOUT_S,
            model=cfg.helper_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _judge_prompt(candidate, a, b)},
            ],
            options={"temperature": 0.0, "max_tokens": 300, "thinking": False},
        )
    except _SCAN_ERRORS:
        _log.debug("coordination: judge call failed", exc_info=True)
        return None
    if out is None:  # timeout — a missed collision is the status quo ante
        return None
    raw = (out.get("message") or {}).get("content") or ""
    data = _parse_judgement(raw)
    if data is None:
        return None
    return _collision_from_judgement(candidate, data, now_iso=now_iso)


# ── scan orchestration ───────────────────────────────────────────────────────


def scan_collisions(
    mem: Any,
    cfg: Any,
    *,
    now: datetime | None = None,
    chat: ChatBackend | None = None,
) -> ScanResult:
    """One coordination pass: gather → candidates → judge → persist."""
    now_dt = now if now is not None else datetime.now(UTC)
    activities = gather_activities(mem, cfg, now=now_dt)
    candidates = detect_candidates(activities) if len(activities) >= 2 else ()
    if not candidates:
        return ScanResult(len(activities), 0, 0, 0, 0)
    now_iso = now_dt.isoformat()
    by_id = {activity.session_id: activity for activity in activities}
    judged = collisions = skipped = 0
    with CoordinationStore(coordination_db_path(cfg)) as store:
        stale_cutoff = (now_dt - timedelta(seconds=active_window_s())).isoformat()
        store.expire_stale(stale_cutoff)
        active = store.active_ids()
        for candidate in candidates[:_MAX_JUDGED_PER_SCAN]:
            cid = collision_id(candidate.session_a, candidate.session_b, candidate.resource)
            if cid in active:
                skipped += 1
                continue
            if chat is None:
                from memo.llm import MLXChat

                chat = MLXChat()
            judged += 1
            collision = judge_candidate(chat, cfg, candidate, by_id, now_iso=now_iso)
            if collision is not None and store.upsert(collision):
                collisions += 1
    return ScanResult(len(activities), len(candidates), judged, collisions, skipped)


# ── delivery (recall hook): pure sqlite read, zero LLM, zero network ─────────


def render_directives_block(directives: list[Directive]) -> str:
    lines = [
        "<memo-coordination>",
        "Live cross-agent coordination: another active agent overlaps with your work.",
    ]
    lines.extend(
        f"- [{d.severity}][{d.kind}] {d.resource} — other agent "
        f"{d.other_session[:8]}: {d.directive}"
        for d in directives
    )
    lines.append(
        "Follow each directive this turn. When a conflict no longer applies, run "
        f"`memo coordinate resolve {directives[0].collision_id}`."
    )
    lines.append("</memo-coordination>")
    return "\n".join(lines)


def deliver_pending_block(cfg: Any, session_id: str | None, *, now: datetime | None = None) -> str:
    """Render + stamp this session's pending directives. Never raises."""
    if not session_id or not coord_enabled():
        return ""
    try:
        with CoordinationStore(coordination_db_path(cfg)) as store:
            pending = store.pending_directives(session_id)
            if not pending:
                return ""
            ts = (now if now is not None else datetime.now(UTC)).isoformat()
            for directive in pending:
                store.mark_delivered(directive.collision_id, session_id, ts)
            return render_directives_block(pending)
    except (OSError, ValueError, sqlite3.Error):
        _log.debug("coordination: delivery failed", exc_info=True)
        return ""


# ── trigger: interval daemon thread for `memo watch` ─────────────────────────


def _run_one_scan() -> None:
    """Fresh Memory per scan (the watcher's own instance stays untouched)."""
    try:
        from memo.config import Config
        from memo.memory import Memory

        cfg = Config.from_env()
        mem = Memory(cfg)
        try:
            result = scan_collisions(mem, cfg)
        finally:
            mem.close()
        _log.debug("coordination scan: %s", result)
    except (*_SCAN_ERRORS, sqlite3.Error):
        _log.debug("coordination scan failed", exc_info=True)


def _scan_loop(stop: threading.Event, *, interval_s: float | None = None) -> None:
    interval = float(scan_interval_s() if interval_s is None else interval_s)
    while not stop.wait(interval):
        _run_one_scan()


def maybe_start_scan_thread(
    stop: threading.Event, *, interval_s: float | None = None
) -> threading.Thread | None:
    """Start the periodic collision scan (daemon thread) when enabled.

    Runs beside the watcher's debounce without ever blocking it; returns None
    when ``MEMO_COORD_ENABLED`` is off.
    """
    if not coord_enabled():
        return None
    thread = threading.Thread(
        target=_scan_loop,
        args=(stop,),
        kwargs={"interval_s": interval_s},
        name="memo-coordination-scan",
        daemon=True,
    )
    thread.start()
    return thread
