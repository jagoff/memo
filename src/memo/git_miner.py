"""`memo mine-git` — seed failure_pattern memories from a repo's git history.

Fix/revert commits are a free, curated stream of failure knowledge: each one
names a mistake that actually happened and the change that corrected it —
covering pre-memo history and non-agent sessions that transcripts never saw
(2026-07-03 ecosystem survey, Tier2 #13 / hippo-memory git mining).

Deterministic — no LLM, no MLX: commit messages are already human-distilled.
Resumable: mined commit SHAs are tracked per-repo in `state_dir/mine-git.json`
(exact, rebase-safe — unlike a date watermark). Bodies follow the
failure_pattern `Pattern/Context` template from `capture._EXTRACT_SYSTEM_PROMPT`.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

_log = logging.getLogger("memo.git_miner")

_FIX_SUBJECT_RE = re.compile(r"^(revert\b|fix(\(|!?:|\b)|hotfix\b|bugfix\b)", re.I)
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"
_MAX_TRACKED_SHAS = 2000


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return out.stdout


def _state_file(state_dir: Path) -> Path:
    return state_dir / "mine-git.json"


def _load_state(state_dir: Path) -> dict[str, Any]:
    f = _state_file(state_dir)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    dest = _state_file(state_dir)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(dest)


def _matched_commits(repo: Path, since_days: int | None) -> list[dict[str, str]]:
    args = [
        "log",
        "--no-merges",
        f"--pretty=format:%H{_FIELD_SEP}%aI{_FIELD_SEP}%s{_FIELD_SEP}%b{_RECORD_SEP}",
    ]
    if since_days:
        args.append(f"--since={since_days} days ago")
    raw = _git(repo, *args)
    commits: list[dict[str, str]] = []
    for rec in raw.split(_RECORD_SEP):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        parts = rec.split(_FIELD_SEP, 3)
        if len(parts) < 3:
            continue
        sha, iso, subject = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not _FIX_SUBJECT_RE.match(subject):
            continue
        commits.append(
            {
                "sha": sha,
                "iso": iso,
                "subject": subject,
                "body": parts[3].strip() if len(parts) > 3 else "",
            }
        )
    return commits


def mine_git_history(
    repo: Path | str | None = None,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Walk `git log` for fix/revert commits, save each as `failure_pattern`.

    Returns a summary dict; ``status`` is ``ok`` or ``not_a_repo``.
    """
    from memo.capture import is_near_duplicate
    from memo.config import Config
    from memo.memory import Memory
    from memo.project import slugify_project

    cfg = Config.from_env()
    repo_path = Path(repo or ".").resolve()
    try:
        toplevel = Path(_git(repo_path, "rev-parse", "--show-toplevel").strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return {"status": "not_a_repo", "repo": str(repo_path)}

    state = _load_state(cfg.state_dir)
    repo_key = str(toplevel)
    seen_shas: list[str] = list(state.get(repo_key, {}).get("shas") or [])
    seen_set = set(seen_shas)

    commits = _matched_commits(toplevel, since_days)
    if limit is not None and limit > 0:
        commits = commits[:limit]

    slug = slugify_project(toplevel.name)
    saved: list[str] = []
    skipped_dup = 0
    skipped_seen = 0

    mem = Memory(cfg)
    try:
        for c in commits:
            if c["sha"] in seen_set:
                skipped_seen += 1
                continue
            try:
                files = [
                    ln
                    for ln in _git(
                        toplevel, "show", "--name-only", "--pretty=format:", c["sha"]
                    ).splitlines()
                    if ln.strip()
                ]
            except subprocess.CalledProcessError:
                files = []
            date = c["iso"][:10]
            body_lines = [
                f"Pattern: {c['subject']}",
                f"Context: repo {toplevel.name}, commit {c['sha'][:8]} ({date})",
            ]
            if c["body"]:
                body_lines += ["", c["body"]]
            if files:
                body_lines += ["", "Files: " + ", ".join(files[:10])]
            cand: dict[str, Any] = {
                "title": c["subject"][:80],
                "body": "\n".join(body_lines),
                "type": "failure_pattern",
                "tags": ["git-mined", f"project:{slug}"],
            }
            if is_near_duplicate(mem, cand):
                skipped_dup += 1
                seen_set.add(c["sha"])
                seen_shas.append(c["sha"])
                continue
            if dry_run:
                saved.append("<dry-run>")
                continue
            try:
                rec = mem.save(
                    content=cand["body"],
                    title=cand["title"],
                    type_="failure_pattern",
                    tags=cand["tags"],
                    auto_project=False,  # tag derived from the mined repo, not cwd
                    created=c["iso"],  # back-date to the fix's real event time
                    extra={"commit_sha": c["sha"], "repo": repo_key, "source": "mine-git"},
                )
                saved.append(rec.id)
                seen_set.add(c["sha"])
                seen_shas.append(c["sha"])
            except Exception as exc:
                _log.warning("mine-git: save failed for %s: %s", c["sha"][:8], exc)
    finally:
        mem.close()

    if not dry_run:
        state[repo_key] = {"shas": seen_shas[-_MAX_TRACKED_SHAS:]}
        _save_state(cfg.state_dir, state)

    return {
        "status": "ok",
        "repo": repo_key,
        "commits_matched": len(commits),
        "saved": saved,
        "skipped_dup": skipped_dup,
        "skipped_seen": skipped_seen,
        "dry_run": dry_run,
    }
