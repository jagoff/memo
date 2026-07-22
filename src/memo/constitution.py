"""GC-10 Dynamic Constraints — project durable decisions/preferences into each
client's project-local instruction file as concrete, self-syncing rules.

`memo mandate` (see ``cli_mandate.py``) drops a FIXED "consult memo first"
text into AGENTS.md / .cursor / … so non-hook clients read memo at all. This
module adds the DYNAMIC half: it distills your durable, load-bearing decisions
and preferences into concrete rules and writes them into the same files as a
delimited, regeneratable block — so a client treats your accumulated wisdom as
standing constraints, not a suggestion it forgets.

The rule set is not reinvented here: it reuses the exact "standing rules" motor
that ``dream_profile`` already graduates from grounding.log (memories cited in
>= K distinct sessions) and retires on resolved contradictions. ``gather_rules``
is a thin delegate to that seam so the two surfaces never drift.

Design mirrors ``cli_mandate``: project-local only (never touches global
config), marker-delimited and idempotent. The block is deterministic (no
timestamp) so re-running with the same rules is a genuine no-op, and
``--sync`` regenerates only files that already opted in — retired rules
disappear, superseding ones appear, on their own.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from memo.cli_mandate import _CLIENT_FILES

RULES_START = "<!-- memo-rules -->"
RULES_END = "<!-- /memo-rules -->"

_RULES_HEADING = "## Standing rules (memo — auto-generated, do not edit by hand)"


# --- pure rendering ----------------------------------------------------------


def render_rules_block(rules: list[tuple[str, str]]) -> str:
    """Render the marker-delimited rules block. Empty rules → ``""``.

    Deterministic (no timestamp): identical rules render byte-for-byte the
    same, which is what makes ``write_rules_to_file`` a real no-op on re-run
    and lets ``--sync`` skip unchanged files. Each rule carries its 8-char
    memory-id so provenance stays visible and a human can trace it back.
    """
    if not rules:
        return ""
    lines = [RULES_START, _RULES_HEADING, ""]
    lines += [f"- {text} `[{rid[:8]}]`" for rid, text in rules]
    lines.append(RULES_END)
    return "\n".join(lines)


def upsert_rules_block(existing: str, block: str) -> str:
    """Return ``existing`` with the rules region replaced by ``block``.

    - marker present → replace the whole ``START..END`` region in place
      (dropping retired rules, no duplication);
    - marker absent + block non-empty → append the block, preserving prior
      content;
    - block empty → remove the region entirely (or no-op when absent).

    A malformed region whose END marker was lost is still fully replaced: we
    partition on END and treat everything from START onward as the old block.
    """
    block = block.strip()
    if RULES_START in existing:
        pre, _, rest = existing.partition(RULES_START)
        _, _, post = rest.partition(RULES_END)
        pre_s = pre.rstrip()
        post_s = post.lstrip()
        parts = [p for p in (pre_s, block, post_s) if p]
        merged = "\n\n".join(parts)
        return (merged.rstrip() + "\n") if merged.strip() else ""
    if not block:
        return existing
    base = existing.rstrip()
    merged = (base + "\n\n" + block) if base else block
    return merged.rstrip() + "\n"


# --- I/O writers -------------------------------------------------------------


def write_rules_to_file(
    target: Path, rules: list[tuple[str, str]], *, dry_run: bool = False
) -> str:
    """Upsert the rules block into ``target``. Returns a status string.

    ``"no rules (skip)"`` when there is nothing to write and no block to
    remove; ``"already current (skip)"`` when the file already matches (the
    idempotent path); otherwise ``"written"`` / ``"removed"`` (or the
    ``would …`` variants under ``dry_run``).
    """
    block = render_rules_block(rules)
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    if not block and RULES_START not in existing:
        return "no rules (skip)"
    new = upsert_rules_block(existing, block)
    if new == existing:
        return "already current (skip)"
    if dry_run:
        return "would write" if block else "would remove"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new, encoding="utf-8")
    return "written" if block else "removed"


def write_rules_for_clients(
    clients: Iterable[str],
    rules: list[tuple[str, str]],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Write the rules block into each client's project-local file.

    Shared targets (codex / devin / opencode all read AGENTS.md) are collapsed
    so the file is touched once, mirroring ``write_mandates_for_clients``.
    """
    root = cwd or Path.cwd()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for client in clients:
        rel = _CLIENT_FILES.get(client)
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        out.append((rel, write_rules_to_file(root / rel, rules, dry_run=dry_run)))
    return out


def resync_rules_in_repo(
    rules: list[tuple[str, str]], *, cwd: Path | None = None, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Regenerate the rules block only in files that already carry one.

    This is the self-sync path: it never creates new files — it just refreshes
    the blocks a user previously opted into (via ``--dynamic``), so superseded
    rules retire and newly-graduated ones appear without re-listing clients.
    """
    root = cwd or Path.cwd()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for rel in _CLIENT_FILES.values():
        if rel in seen:
            continue
        seen.add(rel)
        target = root / rel
        if not target.is_file():
            continue
        if RULES_START not in target.read_text(encoding="utf-8"):
            continue
        out.append((rel, write_rules_to_file(target, rules, dry_run=dry_run)))
    return out


# --- rule source (delegates to the shared motor) -----------------------------


def gather_rules(mem: Any, cfg: Any, *, k: int = 3, min_used: float = 0.5) -> list[tuple[str, str]]:
    """Load-bearing standing rules as ``(memory_id, text)`` pairs.

    Thin delegate to ``dream_profile._gather_rules`` — the same graduation
    (cited in >= ``k`` distinct sessions) and retire-on-contradiction logic
    that feeds the profile's Standing-rules block, so the mandate surface and
    the profile surface can never disagree about what the standing rules are.
    """
    from memo.dream_profile import _gather_rules

    return _gather_rules(mem, cfg, k=k, min_used=min_used)


# --- opted-in repo registry + nightly auto-sync ------------------------------
#
# The nightly dream daemon is repo-agnostic — it cannot discover the arbitrary
# project repos where a user ran `memo mandate --dynamic`. So the write path
# records each opted-in repo root here, and the (flag-gated) dream pass iterates
# that registry to refresh every block, retiring superseded rules on its own.


def _registry_path(state_dir: Any) -> Path:
    return Path(state_dir) / "mandate_repos.json"


def registered_repos(state_dir: Any) -> list[str]:
    """Repo roots that opted into dynamic mandate rules (best-effort, never raises)."""
    path = _registry_path(state_dir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def register_repo(state_dir: Any, root: Any) -> None:
    """Record ``root`` as an opted-in repo (idempotent) for nightly auto-sync."""
    resolved = str(Path(root).resolve())
    roots = registered_repos(state_dir)
    if resolved in roots:
        return
    roots.append(resolved)
    path = _registry_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roots, indent=2), encoding="utf-8")


def run_mandate_sync_pass(cfg: Any, mem: Any) -> dict[str, Any]:
    """Nightly pass: refresh the rules block in every opted-in repo that still
    exists. Never raises — the ``cli_dream`` caller records a returned
    ``status="error"`` in ``receipt["errors"]``. A removed repo is skipped; a
    repo whose block a human deleted is left alone (resync only touches files
    that still carry the marker)."""
    res: dict[str, Any] = {"status": "noop", "synced": []}
    try:
        rules = gather_rules(mem, cfg)
        for root in registered_repos(cfg.state_dir):
            if not Path(root).is_dir():
                continue
            results = resync_rules_in_repo(rules, cwd=Path(root))
            if results:
                res["synced"].append({"repo": root, "files": results})
        if res["synced"]:
            res["status"] = "done"
    except Exception as exc:  # surfaced via receipt["errors"], never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
