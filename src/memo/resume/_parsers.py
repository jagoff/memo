from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._types import ResumeCandidate
from ._utils import (
    _clip,
    _file_updated_at,
    _resolve_cwd,
    _strip_ansi,
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks).strip()
    if isinstance(content, dict):
        text = content.get("text")
        return str(text).strip() if text is not None else ""
    return ""


def _strip_image_placeholders(text: str) -> str:
    clean = re.sub(r"<image\b[^>]*>\s*</image>", " ", text)
    clean = re.sub(r"\[Image #[0-9]+\]", " ", clean)
    return " ".join(clean.split())


def _is_prompt_noise(text: str) -> bool:
    low = text.lower()
    noise_prefixes = (
        "# agents.md instructions",
        "<instructions>",
        "<environment_context>",
        "<subagent_notification>",
        "<turn_aborted>",
        "<developer_message>",
        "<session_context>",
        "memflow startup context:",
    )
    if low.startswith(noise_prefixes):
        return True
    if low.startswith("<command-"):
        return True
    if low.startswith("<local-command-"):
        return True
    if low.startswith("[request interrupted"):
        return True
    if low.startswith("base directory for this skill:"):
        return True
    return low.startswith("⚠ mcp client") or low.startswith("warning: mcp client")


def _is_low_signal_prompt(text: str) -> bool:
    low = text.strip().lower()
    low = re.sub(r"\s+", " ", low)
    return low in {
        "implement the plan.",
        "implement the plan",
        "implementa el plan",
        "implementa los 5",
        "commit y push",
        "hace el commit y el push",
    }


def _extract_labeled_segment(text: str, label: str, stop_markers: Sequence[str]) -> str:
    start = text.find(label)
    if start < 0:
        return ""
    start += len(label)
    end = len(text)
    for marker in stop_markers:
        marker_index = text.find(marker, start)
        if marker_index >= 0:
            end = min(end, marker_index)
    segment = text[start:end].strip(" .")
    return _clip(segment, 600)


def _representative_user_text(text: str) -> tuple[str, int]:
    clean = _strip_ansi(" ".join(str(text or "").split()))
    clean = _strip_image_placeholders(clean).strip()
    if not clean or _is_prompt_noise(clean):
        return "", 0

    if clean.startswith("Repo:"):
        goal = _extract_labeled_segment(
            clean,
            "Goal:",
            [
                " Preserve ",
                " Use existing ",
                " Run focused ",
                " Run ",
                " Edit files ",
                " Final response:",
            ],
        )
        if goal:
            return goal, 95
        task = _extract_labeled_segment(
            clean,
            "Task:",
            [" Context:", " Ownership:", " Goal:", " Preserve ", " Run ", " Final response:"],
        )
        if task:
            return task, 90
        ownership = _extract_labeled_segment(
            clean,
            "Ownership:",
            [" Goal:", " Preserve ", " Use existing ", " Run ", " Final response:"],
        )
        if ownership:
            return f"Work on {ownership}", 65
        return "", 0

    plan_prefix = "Plan Mode read-only task."
    if clean.startswith(plan_prefix):
        return clean[len(plan_prefix) :].strip(), 75

    if _is_low_signal_prompt(clean):
        return clean, 20
    if len(clean) >= 40:
        return clean, 80
    return clean, 50


def _extract_user_text(item: dict[str, Any]) -> str:
    if item.get("type") == "event_msg":
        payload = item.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "user_message":
            return str(payload.get("message") or "").strip()
    if item.get("type") == "response_item":
        payload = item.get("payload")
        if (
            isinstance(payload, dict)
            and payload.get("type") == "message"
            and payload.get("role") == "user"
        ):
            return _content_text(payload.get("content"))
    if item.get("type") == "user":
        message = item.get("message")
        if isinstance(message, dict):
            return _content_text(message.get("content"))
        return _content_text(item.get("content"))
    role = item.get("role")
    if role == "user":
        return _content_text(item.get("content"))
    return ""


def _jsonl_latest_user_text(path: Path, *, max_lines: int = 4000) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    best_text = ""
    best_score = -1
    for raw in reversed(lines[-max_lines:]):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        text = _extract_user_text(item)
        if not text:
            continue
        representative, score = _representative_user_text(text)
        if score > best_score:
            best_text = representative
            best_score = score
        if score >= 80:
            return representative
    return best_text


def _read_devin_session_meta(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "session_meta":
                    payload = item.get("payload")
                    return payload if isinstance(payload, dict) else {}
    except OSError:
        return None
    return None


def _read_codex_session_meta(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "session_meta":
                    payload = item.get("payload")
                    return payload if isinstance(payload, dict) else {}
    except OSError:
        return None
    return None


def _read_claude_session_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle):
                if index > 80:
                    break
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                session_id = item.get("sessionId") or item.get("session_id")
                if session_id and "session_id" not in out:
                    out["session_id"] = str(session_id)
                cwd = item.get("cwd")
                if cwd and "cwd" not in out:
                    out["cwd"] = str(cwd)
                version = item.get("version")
                if version and "version" not in out:
                    out["version"] = str(version)
                if out.get("session_id") and out.get("cwd"):
                    break
    except OSError:
        return {}
    return out


def _devin_candidate(path: Path) -> ResumeCandidate | None:
    # Devin stores transcripts as JSON files with session name as filename
    session_id = path.stem  # e.g., "able-artichoke" from "able-artichoke.json"
    if not session_id:
        return None

    # Read transcript to extract metadata
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    # Extract info from transcript
    cwd = ""
    # Try to find cwd from system messages
    if isinstance(data.get("steps"), list):
        for step in data["steps"]:
            msg = str(step.get("message") or "")
            if "Current workspace directories:" in msg:
                # Extract cwd from message
                for line in msg.split("\n"):
                    if "(cwd)" in line:
                        cwd = line.split("(cwd)")[0].strip()
                        break
                break
    cwd = _resolve_cwd(cwd)

    created = str(data.get("created_at") or "")
    # For Devin, use file modification time as updated_at since Devin doesn't store timestamps
    # This ensures recently-accessed sessions appear at the top
    updated = _file_updated_at(path)

    # Get latest user step as summary
    summary = ""
    if isinstance(data.get("steps"), list):
        for step in reversed(data["steps"]):
            if step.get("source") == "user":
                summary = str(step.get("message") or "").strip()
                break

    title = summary or f"Devin session {session_id[:8]}"
    uri = f"devin://session/{session_id}"
    return ResumeCandidate(
        agent="devin",
        provider="devin-native",
        uri=uri,
        session_id=session_id,
        title=_clip(title, 240),
        updated_at=updated or created,
        cwd=cwd,
        summary=_clip(summary, 1000),
        resume_mode="native_resume",
        resume_command=["devin", "-r", session_id],
        provenance=[uri],
        metadata={
            "path": str(path),
            "created_at": created,
        },
    )


def _codex_candidate(path: Path) -> ResumeCandidate | None:
    meta = _read_codex_session_meta(path)
    if meta is None:
        return None
    session_id = str(meta.get("id") or "").strip()
    if not session_id:
        return None
    cwd = _resolve_cwd(str(meta.get("cwd") or ""))
    created = str(meta.get("timestamp") or "")
    updated = _file_updated_at(path)
    summary = _jsonl_latest_user_text(path)
    title = summary or f"Codex session {session_id[:8]}"
    uri = f"codex://session/{session_id}"
    return ResumeCandidate(
        agent="codex",
        provider="codex-native",
        uri=uri,
        session_id=session_id,
        title=_clip(title, 240),
        updated_at=updated or created,
        cwd=cwd,
        summary=_clip(summary, 1000),
        resume_mode="native_resume",
        resume_command=["codex", "resume", session_id],
        provenance=[uri],
        metadata={
            "path": str(path),
            "created_at": created,
            "originator": str(meta.get("originator") or ""),
            "thread_source": str(meta.get("thread_source") or ""),
            "agent_nickname": str(meta.get("agent_nickname") or ""),
            "agent_role": str(meta.get("agent_role") or ""),
        },
    )


def _is_claude_subagent_transcript(path: Path) -> bool:
    # Subagent/sidechain transcripts carry the *parent* session's sessionId, so they collapse
    # into the parent during merge and are never independently resumable
    # (`claude --resume agent-xxx` is meaningless). Drop them so real sessions are not hidden.
    if path.stem.startswith("agent-"):
        return True
    return "subagents" in path.parts


def _claude_candidate(path: Path) -> ResumeCandidate | None:
    meta = _read_claude_session_meta(path)
    # The filename stem is the canonical `claude --resume` target; for real top-level sessions
    # it equals the internal sessionId. Prefer it, falling back to the parsed meta.
    session_id = str(path.stem or meta.get("session_id") or "").strip()
    if not session_id:
        return None
    cwd = _resolve_cwd(str(meta.get("cwd") or ""))
    updated = _file_updated_at(path)
    summary = _jsonl_latest_user_text(path)
    title = summary or f"Claude session {session_id[:8]}"
    uri = f"claude://session/{session_id}"
    return ResumeCandidate(
        agent="claude",
        provider="claude-native",
        uri=uri,
        session_id=session_id,
        title=_clip(title, 240),
        updated_at=updated,
        cwd=cwd,
        summary=_clip(summary, 1000),
        resume_mode="native_resume",
        resume_command=["claude", "--resume", session_id],
        provenance=[uri],
        metadata={"path": str(path), **meta},
    )


def _gemini_project_cwd(path: Path) -> str:
    # path = <home>/tmp/<project>/chats/session-*.jsonl → project dir is two levels up.
    project_dir = path.parent.parent
    marker = project_dir / ".project_root"
    with contextlib.suppress(OSError):
        return _resolve_cwd(marker.read_text(encoding="utf-8").strip())
    return ""


def _read_gemini_session_header(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return {}
    try:
        item = json.loads(first)
    except json.JSONDecodeError:
        return {}
    return item if isinstance(item, dict) and item.get("sessionId") else {}


def _gemini_latest_user_text(path: Path, *, max_lines: int = 4000) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    best_text = ""
    best_score = -1
    for raw in reversed(lines[-max_lines:]):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        # Messages appear both as standalone {"type": "user", ...} lines and seeded
        # inside an initial {"$set": {"messages": [...]}} mutation.
        messages: list[Any]
        if isinstance(item.get("$set"), dict) and isinstance(item["$set"].get("messages"), list):
            messages = list(item["$set"]["messages"])
        else:
            messages = [item]
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("type") != "user":
                continue
            text = _content_text(message.get("content"))
            if not text:
                continue
            representative, score = _representative_user_text(text)
            if score > best_score:
                best_text = representative
                best_score = score
            if score >= 80:
                return representative
    return best_text


def _gemini_candidate(path: Path) -> ResumeCandidate | None:
    header = _read_gemini_session_header(path)
    session_id = str(header.get("sessionId") or "").strip()
    if not session_id:
        return None
    cwd = _gemini_project_cwd(path)
    created = str(header.get("startTime") or "")
    updated = str(header.get("lastUpdated") or "") or _file_updated_at(path)
    summary = _gemini_latest_user_text(path)
    title = summary or f"Gemini session {session_id[:8]}"
    uri = f"gemini://session/{session_id}"
    return ResumeCandidate(
        agent="gemini",
        provider="gemini-native",
        uri=uri,
        session_id=session_id,
        title=_clip(title, 240),
        updated_at=updated or created,
        cwd=cwd,
        summary=_clip(summary, 1000),
        resume_mode="native_resume",
        resume_command=["gemini", "--session-file", str(path)],
        provenance=[uri],
        metadata={
            "path": str(path),
            "created_at": created,
            "kind": str(header.get("kind") or ""),
        },
    )
