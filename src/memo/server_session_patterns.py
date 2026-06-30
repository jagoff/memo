"""Session patterns: session management, conflict detection, topic keys.

Session-aware memory primitives layered on memo's core storage:
- Session-aware storage with project+directory
- Topic key upserts (no duplicate flood)
- Exact deduplication with normalized hash
- Memory relations for conflict surfacing
- Soft delete with audit trail

These patterns enhance memo's core storage with session lifecycle,
conflict detection during save, and upsert semantics via topic_key.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import pathlib
from typing import Any

from memo.identity import _session_id

_log = logging.getLogger(__name__)

_VALID_RELATIONS = {
    "related",
    "compatible",
    "scoped",
    "conflicts_with",
    "supersedes",
    "not_conflict",
}


def _normalize_hash(title: str, type_: str, scope: str = "project") -> str:
    """Compute normalized hash for exact deduplication (session pattern)."""
    data = f"{title.lower().strip()}:{type_.lower().strip()}:{scope.lower().strip()}".encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _session_directory() -> str:
    """Get current working directory for session."""
    return os.getcwd()


def _project_from_cwd() -> str:
    """Detect project from cwd (5-case algorithm)."""
    cwd = os.getcwd()
    # Case 1: .memo/config.json exists (nearest to git root)
    config_path = _find_memo_config(cwd)
    if config_path:
        return config_path.parent.parent.name

    # Case 2: git root with origin remote (or bare git root)
    git_root = _find_git_root(cwd)
    if git_root:
        remote = _get_git_remote(git_root)
        if remote:
            return remote
        return git_root.name

    # Case 3 (formerly): inside git repo — already handled by Case 2 above.

    # Case 4: single git child
    child = _find_git_child(cwd)
    if child:
        return child.name

    # Case 5: directory basename fallback
    return cwd.split("/")[-1] or "default"


def _find_memo_config(cwd: str) -> Any:
    """Find nearest .memo/config.json within git root."""
    # Simplified: check cwd and parents

    path = pathlib.Path(cwd)
    for parent in [path, *path.parents]:
        config = parent / ".memo" / "config.json"
        if config.exists():
            return config
        # Stop at git root
        if (parent / ".git").exists():
            break
    return None


def _find_git_root(cwd: str) -> Any:
    """Find git root directory."""

    path = pathlib.Path(cwd)
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _get_git_remote(git_root: Any) -> str | None:
    """Get project name from git remote."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL (handles both .git and bare URLs)
            if url.endswith(".git"):
                url = url[:-4]
            return url.split("/")[-1]
    except Exception:
        _log.debug("git remote lookup failed")
    return None


def _find_git_child(cwd: str) -> Any | None:
    """Find single git repo child (depth=1, max 20 entries, 200ms timeout)."""
    import time

    path = pathlib.Path(cwd)
    if not path.is_dir():
        return None

    start = time.time()
    children = []
    skip = {
        ".git",
        "node_modules",
        "vendor",
        ".venv",
        "__pycache__",
        "target",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }

    try:
        for entry in path.iterdir():
            if time.time() - start > 0.2:
                break
            if entry.name.startswith(".") or entry.name in skip:
                continue
            if entry.is_dir() and (entry / ".git").exists():
                children.append(entry)
                if len(children) > 20:
                    return None  # Too many = ambiguous
    except Exception:
        return None

    return children[0] if len(children) == 1 else None


# -- Session Management (session pattern) ----------------------------------------

# Cache of memory instance IDs whose session tables have already been created.
# Avoids re-issuing 5 DDL statements on every tool call. Keyed on id(memory)
# so each fresh Memory object (e.g. in tests) still runs CREATE TABLE IF NOT EXISTS.
_session_schema_ensured_for: set[int] = set()


def _ensure_session_table(memory: Any) -> None:
    """Ensure sessions table exists (migration for existing DBs)."""
    mem_id = id(memory)
    if mem_id in _session_schema_ensured_for:
        return
    cx = memory.store._conn
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            directory TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            summary TEXT,
            status TEXT DEFAULT 'active'
        )
        """
    )
    cx.execute("CREATE INDEX IF NOT EXISTS idx_meta_session ON meta(session_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_meta_topic_key ON meta(topic_key)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_meta_hash ON meta(normalized_hash)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_meta_deleted ON meta(deleted_at)")
    _session_schema_ensured_for.add(mem_id)


# -- MCP Server Registration -------------------------------------------


def register(server: Any, memory: Any) -> None:
    """Register session-pattern MCP tools."""

    @server.tool()
    def mem_session_start(
        directory: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Start a new session (session pattern).

        Registers session start for the current project. The session ID is used
        to track work across the session lifecycle, enabling:
        - Context injection from previous sessions
        - Session summaries for warm restarts
        - Topic key upserts across sessions

        Args:
            directory: Explicit directory override (resolved first).
            cwd: Alias for directory.

        Returns session info with project detection results.
        """
        dir_override = directory or cwd
        project = _project_from_cwd()
        session_id = _session_id()
        now = datetime.datetime.now(datetime.UTC).isoformat()

        _ensure_session_table(memory)

        with memory.store._tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO sessions (id, project, directory, started_at, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (session_id, project, dir_override or _session_directory(), now),
            )

        return {
            "session_id": session_id,
            "project": project,
            "directory": dir_override or _session_directory(),
            "started_at": now,
            "status": "active",
        }

    @server.tool()
    def mem_session_end(
        summary: str | None = None,
    ) -> dict[str, Any]:
        """End the current session (session pattern).

        Saves a summary of what was accomplished, discovered, etc.
        The summary is used for context injection in future sessions.

        Args:
            summary: Session summary (Goal/Discoveries/Accomplished/Next Steps).

        Returns session ended confirmation.
        """
        session_id = _session_id()
        now = datetime.datetime.now(datetime.UTC).isoformat()

        _ensure_session_table(memory)

        with memory.store._tx() as cx:
            cx.execute(
                "UPDATE sessions SET ended_at = ?, summary = ?, status = 'completed' WHERE id = ?",
                (now, summary, session_id),
            )

        return {
            "session_id": session_id,
            "ended_at": now,
            "summary": summary,
            "status": "completed",
        }

    @server.tool()
    def mem_context(
        project: str | None = None,
        scope: str = "project",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Get context from previous sessions (session pattern).

        Returns formatted context from recent sessions for the project.
        This enables "warm restarts" where the agent picks up
        where it left off.

        Args:
            project: Project name (auto-detected if omitted).
            scope: project|personal|global
            limit: Number of recent sessions to include.

        Returns formatted context string with session summaries.
        """
        proj = project or _project_from_cwd()

        _ensure_session_table(memory)

        rows = memory.store._conn.execute(
            """
            SELECT id, directory, ended_at, summary
            FROM sessions
            WHERE project = ? AND status = 'completed' AND summary IS NOT NULL
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            (proj, limit),
        ).fetchall()

        if not rows:
            return {"project": proj, "context": "", "sessions": []}

        parts = [f"## Previous Sessions ({proj})"]
        for row in rows:
            parts.append(f"\n### Session {row['id'][:8]}")
            if row["directory"]:
                parts.append(f"Directory: {row['directory']}")
            if row["ended_at"]:
                parts.append(f"Ended: {row['ended_at']}")
            if row["summary"]:
                parts.append(f"\n{row['summary']}")

        return {
            "project": proj,
            "context": "\n\n".join(parts),
            "sessions": [dict(row) for row in rows],
        }

    @server.tool()
    def mem_timeline(
        observation_id: str,
        before: int = 5,
        after: int = 5,
    ) -> dict[str, Any]:
        """Get chronological context around an observation (session pattern).

        Returns observations from the same session before and after
        the specified observation, for temporal context.

        Args:
            observation_id: The ID to get timeline around.
            before: Number of observations before.
            after: Number of observations after.

        Returns timeline with surrounding observations.
        """
        # Get the target observation's session and time
        row = memory.store._conn.execute(
            "SELECT session_id, created FROM meta WHERE id = ?",
            (observation_id,),
        ).fetchone()

        if not row:
            return {"error": "observation not found", "id": observation_id}

        session_id = row["session_id"]
        created = row["created"]

        # Get before observations
        before_rows = memory.store._conn.execute(
            """
            SELECT id, title, type, created
            FROM meta
            WHERE session_id = ? AND created < ? AND id != ?
            ORDER BY created DESC
            LIMIT ?
            """,
            (session_id, created, observation_id, before),
        ).fetchall()

        # Get after observations
        after_rows = memory.store._conn.execute(
            """
            SELECT id, title, type, created
            FROM meta
            WHERE session_id = ? AND created > ? AND id != ?
            ORDER BY created ASC
            LIMIT ?
            """,
            (session_id, created, observation_id, after),
        ).fetchall()

        return {
            "observation_id": observation_id,
            "session_id": session_id,
            "before": [dict(r) for r in before_rows],
            "after": [dict(r) for r in after_rows],
        }

    @server.tool()
    def mem_judge(
        relation_id: int,
        relation: str,
        reason: str | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Resolve a memory conflict (session pattern).

        Records a verdict for a pending memory relation that was
        surfaced during save or conflict scan.

        Args:
            relation_id: The memory_relations ID.
            relation: related|compatible|scoped|conflicts_with|supersedes|not_conflict
            reason: Optional explanation.
            confidence: 0.0-1.0 confidence in the judgment.

        Returns the judged relation.
        """
        now = datetime.datetime.now(datetime.UTC).isoformat()

        if relation not in _VALID_RELATIONS:
            return {"error": f"invalid relation, must be one of {_VALID_RELATIONS}"}

        _ensure_session_table(memory)

        with memory.store._tx() as cx:
            cx.execute(
                """
                UPDATE memory_relations
                SET judgment_status = 'judged', relation = ?, reason = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (relation, reason, confidence, now, relation_id),
            )

        return {"relation_id": relation_id, "relation": relation, "status": "judged"}

    @server.tool()
    def mem_compare(
        memory_id_a: str,
        memory_id_b: str,
        relation: str,
        reasoning: str | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Persist semantic relation verdict between two memories (session pattern).

        This allows the agent to explicitly record how two memories
        relate (rather than conflict surfacing).

        Args:
            memory_id_a: First memory ID.
            memory_id_b: Second memory ID.
            relation: related|compatible|scoped|conflicts_with|supersedes|not_conflict
            reasoning: Brief explanation.
            confidence: 0.0-1.0 confidence.

        Returns the created relation sync_id.
        """
        import uuid

        if relation not in _VALID_RELATIONS:
            return {"error": f"invalid relation, must be one of {_VALID_RELATIONS}"}

        if relation == "not_conflict":
            return {"sync_id": "", "status": "no-op"}

        sync_id = f"rel-{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.UTC).isoformat()
        session_id = _session_id()

        _ensure_session_table(memory)

        with memory.store._tx() as cx:
            cx.execute(
                """
                INSERT INTO memory_relations
                (sync_id, source_id, target_id, relation, judgment_status, reason, confidence, session_id, created_at)
                VALUES (?, ?, ?, ?, 'judged', ?, ?, ?, ?)
                """,
                (
                    sync_id,
                    memory_id_a,
                    memory_id_b,
                    relation,
                    reasoning,
                    confidence,
                    session_id,
                    now,
                ),
            )

        return {"sync_id": sync_id, "status": "created"}

    @server.tool()
    def mem_suggest_topic_key(
        type: str = "note",
        title: str = "",
    ) -> dict[str, Any]:
        """Suggest a topic key for upsert semantics (session pattern).

        Topic keys turn mem_save into upserts: same project+scope+topic_key
        updates the existing memory instead of creating new ones.

        Args:
            type: Memory type (architecture, bugfix, decision, etc.)
            title: Memory title.

        Returns suggested topic_key in family/description format.
        """
        families = {
            "architecture": "architecture",
            "bugfix": "bug",
            "decision": "decision",
            "pattern": "pattern",
            "config": "config",
            "discovery": "discovery",
            "learning": "learning",
            "note": "note",
        }

        family = families.get(type, "note")

        # Generate description from title
        if title:
            # kebab-case the description
            desc = title.lower().strip().replace(" ", "-").replace("_", "-").replace("/", "-")
            # Strip common prefixes
            for prefix in ["fixed-", "added-", "updated-", "implemented-"]:
                if desc.startswith(prefix):
                    desc = desc[len(prefix) :]
            key = f"{family}/{desc}"
        else:
            key = f"{family}/untitled"

        return {"topic_key": key, "type": type, "title": title}

    @server.tool()
    def mem_current_project() -> dict[str, Any]:
        """Detect current project (5-case algorithm).

        Returns project detection result with source and path.
        """
        project = _project_from_cwd()
        directory = _session_directory()

        # Determine source
        if _find_memo_config(directory):
            source = "config"
        else:
            _git_root = _find_git_root(directory)
            if _git_root and _get_git_remote(_git_root):
                source = "git_remote"
            elif _git_root:
                source = "git_root"
            elif _find_git_child(directory):
                source = "git_child"
            else:
                source = "dir_basename"

        return {
            "project": project,
            "project_source": source,
            "project_path": directory,
            "cwd": directory,
            "available_projects": [],
        }

    @server.tool()
    def mem_review(
        project: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """List observations needing review (session pattern).

        Returns observations where review_after date has passed.
        This enables periodic review cycles for memory hygiene.

        Args:
            project: Project name (auto-detected if omitted).
            limit: Maximum results.

        Returns list of observations needing review.
        """
        proj = project or _project_from_cwd()
        now = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

        _ensure_session_table(memory)

        rows = memory.store._conn.execute(
            """
            SELECT id, title, type, review_after, created
            FROM meta
            WHERE tags LIKE ? AND review_after IS NOT NULL AND review_after < ?
            ORDER BY review_after ASC
            LIMIT ?
            """,
            (f'%"project:{proj}"%', now, limit),
        ).fetchall()

        return {
            "project": proj,
            "observations": [dict(r) for r in rows],
            "count": len(rows),
        }

    @server.tool()
    def mem_doctor(
        project: str | None = None,
        check: str | None = None,
    ) -> dict[str, Any]:
        """Run read-only diagnostics (session pattern).

        Returns health checks for store, project detection,
        and sync state.

        Args:
            project: Project name (auto-detected if omitted).
            check: Specific check code.

        Returns diagnostic report.
        """
        proj = project or _project_from_cwd()

        # Check store health
        store_issues = []

        # Check FTS sync
        try:
            fts_count = memory.store._conn.execute("SELECT COUNT(*) as c FROM fts").fetchone()["c"]
            meta_count = memory.store._conn.execute("SELECT COUNT(*) as c FROM meta").fetchone()[
                "c"
            ]

            if abs(fts_count - meta_count) > 10:
                store_issues.append("fts desync detected")
        except Exception as e:
            store_issues.append(f"fts check failed: {e}")

        # Check sessions table
        try:
            sessions_exist = memory.store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            if not sessions_exist:
                store_issues.append("sessions table missing")
        except Exception:
            store_issues.append("sessions table check failed")

        # Check memory_relations table
        try:
            relations_exist = memory.store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_relations'"
            ).fetchone()
            if not relations_exist:
                store_issues.append("memory_relations table missing")
        except Exception:
            store_issues.append("memory_relations table check failed")

        return {
            "project": proj,
            "diagnostics": store_issues,
            "status": "ok" if not store_issues else "issues",
        }

    @server.tool()
    def mem_stats() -> dict[str, Any]:
        """Memory system statistics (session pattern).

        Returns counts for observations, sessions, relations, etc.
        """
        cx = memory.store._conn

        obs_count = cx.execute("SELECT COUNT(*) as c FROM meta").fetchone()["c"]
        obs_active = cx.execute(
            "SELECT COUNT(*) as c FROM meta WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        obs_deleted = cx.execute(
            "SELECT COUNT(*) as c FROM meta WHERE deleted_at IS NOT NULL"
        ).fetchone()["c"]

        # Sessions
        sessions_total = 0
        sessions_active = 0
        try:
            sessions_total = cx.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
            sessions_active = cx.execute(
                "SELECT COUNT(*) as c FROM sessions WHERE status = 'active'"
            ).fetchone()["c"]
        except Exception:
            _log.debug("sessions query failed")

        # Relations
        relations_pending = 0
        relations_judged = 0
        try:
            relations_pending = cx.execute(
                "SELECT COUNT(*) as c FROM memory_relations WHERE judgment_status = 'pending'"
            ).fetchone()["c"]
            relations_judged = cx.execute(
                "SELECT COUNT(*) as c FROM memory_relations WHERE judgment_status = 'judged'"
            ).fetchone()["c"]
        except Exception:
            _log.debug("memory_relations query failed")

        return {
            "observations": {
                "total": obs_count,
                "active": obs_active,
                "deleted": obs_deleted,
            },
            "sessions": {
                "total": sessions_total,
                "active": sessions_active,
            },
            "relations": {
                "pending": relations_pending,
                "judged": relations_judged,
            },
        }
