"""Episode indexer + semantic search over work sessions (Phase 1).

The derived semantic index behind `memo resume`'s meaning-based search. One
embedding per session of its **prompt-arc** (the user-prompts + summary), so the
picker can find a session from weeks ago by what it was *about* — not just by
recency. See `docs/superpowers/specs/2026-06-27-semantic-resume-design.md`.

Documents (session arcs) are embedded RAW (no query prefix); the picker query is
embedded with the asymmetric prefix via `embedder_client.embed_query` — the MLX
asymmetric-retrieval invariant. The picker only embeds when the recall daemon is
warm (`ping`); cold ⇒ it degrades to substring, never cold-loads MLX on a keystroke.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._parsers import _extract_user_text
from ._types import ResumeCandidate
from ._utils import _clip, _resolve_cwd

if TYPE_CHECKING:
    from memo.config import Config
    from memo.store.episode_store import EpisodeStore

_PROMPT_ARC_CHARS = 2000
_MAX_PROMPTS = 40

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def open_store(cfg: Config) -> EpisodeStore | None:
    """Open the episode index, or None when episodic memory is disabled."""
    from memo.flags import flag_bool

    if not flag_bool("MEMO_EPISODIC_ENABLED"):
        return None
    from memo.store.episode_store import EpisodeStore

    return EpisodeStore(cfg.episode_db, cfg.embedder_dims)


def _gather_user_prompts(path: Path, *, max_prompts: int = _MAX_PROMPTS, max_lines: int = 6000) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    prompts: list[str] = []
    for raw in lines[-max_lines:]:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        text = _extract_user_text(item)
        if text and text.strip():
            prompts.append(" ".join(text.split()))
    return prompts[-max_prompts:]


def prompt_arc(candidate: ResumeCandidate) -> str:
    """Representative text for a session: its summary + the arc of user prompts.

    Uniform across agents: memo sessions contribute their LLM `running_summary`,
    native sessions contribute the gathered user prompts from the transcript.
    """
    parts: list[str] = []
    if candidate.summary:
        parts.append(candidate.summary)
    path = str(candidate.metadata.get("path") or "")
    if path:
        parts.extend(_gather_user_prompts(Path(path)))
    arc = "\n".join(p for p in parts if p).strip()
    return _clip(arc, _PROMPT_ARC_CHARS)


def _content_hash(arc: str) -> str:
    return hashlib.sha256(arc.encode("utf-8")).hexdigest()[:16]


def index_candidate(store: EpisodeStore, candidate: ResumeCandidate, *, embed_fn: EmbedFn) -> bool:
    """Embed + upsert one session. Returns True if (re)embedded, False if skipped
    (empty arc or unchanged content_hash)."""
    arc = prompt_arc(candidate)
    if not arc:
        return False
    digest = _content_hash(arc)
    if store.content_hash_for(candidate.agent, candidate.session_id) == digest:
        return False
    vec = embed_fn([arc])[0]
    turn_raw = candidate.metadata.get("turn_count")
    turn_count = int(turn_raw) if isinstance(turn_raw, (int, float)) else 0
    store.upsert(
        agent=candidate.agent,
        session_id=candidate.session_id,
        content_hash=digest,
        embedding=vec,
        cwd=candidate.cwd,
        updated_at=candidate.updated_at,
        summary=candidate.summary or candidate.title,
        resume_command=list(candidate.resume_command),
        turn_count=turn_count,
    )
    return True


def _row_to_candidate(row: dict[str, Any]) -> ResumeCandidate:
    agent = str(row["agent"])
    sid = str(row["session_id"])
    cmd = [str(p) for p in (row.get("resume_command") or [])]
    summary = str(row.get("summary") or "")
    uri = f"memo://episode/{agent}/{sid}"
    return ResumeCandidate(
        agent=agent,
        provider="episode",
        uri=uri,
        session_id=sid,
        title=_clip(summary or sid, 240),
        updated_at=str(row.get("updated_at") or ""),
        cwd=_resolve_cwd(str(row.get("cwd") or "")),
        summary=_clip(summary, 1000),
        resume_mode="native_resume" if cmd else "context_resume",
        resume_command=cmd,
        provenance=[uri],
        metadata={"score": row.get("score"), "episode": True},
    )


def semantic_search(cfg: Config, query: str, *, k: int | None = None) -> list[ResumeCandidate]:
    """Top-k sessions by meaning. Returns ``[]`` (degrade to substring) when the
    recall daemon is cold, the index is empty, or episodic memory is disabled —
    never cold-loads MLX on the picker's hot path."""
    q = (query or "").strip()
    if not q:
        return []
    from memo.embedder_client import embed_query, ping

    if ping(state_dir=cfg.state_dir) is None:
        return []  # cold embedder → caller stays on substring
    store = open_store(cfg)
    if store is None or store.count() == 0:
        return []
    from memo.flags import flag_int

    topk = k or flag_int("MEMO_RESUME_SEMANTIC_K") or 50
    try:
        qvec = embed_query(q, state_dir=cfg.state_dir)
        rows = store.search(qvec, topk)
    except Exception:
        return []
    return [_row_to_candidate(r) for r in rows]


def backfill(cfg: Config, *, agent: str = "all", rebuild: bool = False) -> dict[str, Any]:
    """Index the newest unindexed sessions, bounded by ``MEMO_RESUME_INDEX_BATCH``.

    Bypasses the per-provider mtime cap so the full history is enumerable; the
    content_hash skip keeps successive runs cheap (only new/changed sessions
    embed), so the whole backlog is covered over a few nightly passes.
    """
    from memo.embedder_client import embed
    from memo.flags import flag_int

    from ._orchestration import discover_resume_candidates
    from ._utils import _scan_cap_override

    store = open_store(cfg)
    if store is None:
        return {"enabled": False}
    if rebuild:
        store.clear()
    batch = flag_int("MEMO_RESUME_INDEX_BATCH") or 500
    embed_fn = partial(embed, state_dir=cfg.state_dir)

    # Lift the per-provider parse cap so the FULL history is enumerable (the
    # content_hash skip keeps re-embeds cheap). ContextVar, not os.environ —
    # MEMO_* flags must never be written through the environment.
    token = _scan_cap_override.set(1_000_000)
    try:
        report = discover_resume_candidates(agent=agent, include_all_cwd=True, limit=1_000_000)
    finally:
        _scan_cap_override.reset(token)

    indexed = 0
    skipped = 0
    for cand in report.candidates:
        if indexed >= batch:
            break
        try:
            if index_candidate(store, cand, embed_fn=embed_fn):
                indexed += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    return {"enabled": True, "indexed": indexed, "skipped": skipped, "total": store.count()}


def index_memo_session(cfg: Config, session_id: str, transcript_path: str | None) -> bool:
    """Best-effort index of the just-finished memo (Claude) session — called from
    the Stop hook after `running_summary` is computed. Never raises."""
    if not session_id:
        return False
    try:
        store = open_store(cfg)
        if store is None:
            return False
        from memo.embedder_client import embed
        from memo.session import get_session

        snap = get_session(cfg.state_dir, session_id) or {}
        summary = str(
            snap.get("running_summary") or snap.get("summary") or snap.get("last_user_msg") or ""
        )
        path = transcript_path or str(snap.get("transcript_path") or "")
        candidate = ResumeCandidate(
            agent="claude",
            provider="memo",
            uri=f"memo://session/{session_id}",
            session_id=session_id,
            title=_clip(summary or session_id, 240),
            updated_at=str(snap.get("updated") or ""),
            cwd=_resolve_cwd(str(snap.get("cwd") or "")),
            summary=summary,
            resume_mode="native_resume",
            resume_command=["claude", "--resume", session_id],
            metadata={"path": path, "turn_count": snap.get("turn_count") or 0},
        )
        return index_candidate(store, candidate, embed_fn=partial(embed, state_dir=cfg.state_dir))
    except Exception:
        return False
