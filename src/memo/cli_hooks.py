"""`memo install-recall-hook` — memo-owned, self-healing recall wiring.

The recall hook (``memo recall-hook`` on ``UserPromptSubmit``) is memo's single
most important ambient integration: it injects relevant memories before every
prompt. Historically it was delivered *only* by the memo plugin's
``hooks/hooks.json``. That mechanism is fragile: a bad commit clobbered the
plugin hooks file (f5232b2) and a directory-marketplace refresh then silently
dropped the wiring, so recall stopped firing with no error — invisible to users.

This module makes recall delivery **memo-owned and self-healing**, mirroring the
statusline self-heal (`cli_statusline.py`):

- `wire_recall_hook()` idempotently installs a memo-owned ``UserPromptSubmit``
  hook in ``~/.claude/settings.json``, **coexisting** with foreign hooks
  (caveman, memflow, supacode) rather than replacing them. The command uses the
  **absolute** path to the ``memo`` binary of the running runtime, so a
  GUI-launched agent with a minimal PATH (no ``~/.local/bin``) still finds it.
- `selfheal_recall_hook()` re-asserts the wiring on every memo-mcp start (gated
  by ``MEMO_HOOK_SELFHEAL``), so a de-registered/clobbered plugin self-repairs on
  the next session — no user command required.

The wired command intentionally does **not** pin ``MEMO_RECALL_MIN_SIM``: flag
resolution is env > overlay > default, so pinning it would override the nightly
recall tuner's overlay. The budget-shaping vars (top-k, token budget) are pinned
so the hook stays within its ~5s UserPromptSubmit budget.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import click

from memo.cli_common import console

_HOOK_TIMEOUT = 12
# Marker verb that identifies memo's own recall group in a foreign hooks list.
_RECALL_VERB = "recall-hook"
# Budget-shaping env (NOT MEMO_RECALL_MIN_SIM — that stays tuner-governed).
_HOOK_ENV = (
    "MEMO_NONINTERACTIVE=1 MEMO_CONTEXTUAL_RETRIEVAL=1 "
    "MEMO_RECALL_TOKEN_BUDGET=160 MEMO_RECALL_TOP_K=1 MEMO_RECALL_FEEDBACK_HINT=0"
)


def _claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))


def _resolve_memo_bin() -> str:
    """Absolute path to the ``memo`` binary of the *running* runtime.

    Prefer the sibling of the current process's entry script (memo-mcp → memo in
    the same isolated runtime) so the wired hook can never drift to a different
    install than the MCP server — the classic "works in CLI, broken in MCP"
    mixed-runtime trap. Fall back to a PATH lookup, then a bare name.
    """
    try:
        sibling = Path(sys.argv[0]).with_name("memo")
        if sibling.exists():
            return str(sibling.resolve())
    except (OSError, ValueError):
        pass
    found = shutil.which("memo")
    if found:
        return str(Path(found).resolve())
    return "memo"


def _memo_command(memo_bin: str | None = None) -> str:
    return f"{_HOOK_ENV} {memo_bin or _resolve_memo_bin()} {_RECALL_VERB}"


def _canonical_group(memo_bin: str | None = None) -> dict[str, object]:
    return {
        "hooks": [{"type": "command", "command": _memo_command(memo_bin), "timeout": _HOOK_TIMEOUT}]
    }


_PRECOMPACT_VERB = "capture-tick"


def _precompact_command(memo_bin: str | None = None) -> str:
    return f"MEMO_NONINTERACTIVE=1 {memo_bin or _resolve_memo_bin()} capture-tick --force"


def _precompact_group(memo_bin: str | None = None) -> dict[str, object]:
    return {"hooks": [{"type": "command", "command": _precompact_command(memo_bin), "timeout": 60}]}


def wire_precompact_hook(
    claude_dir: Path | None = None, *, memo_bin: str | None = None
) -> dict[str, str]:
    """Idempotently wire the memo PreCompact force-flush into ``settings.json``.

    Parity with `wire_recall_hook`: memo-owned, self-healing, coexists with
    foreign PreCompact hooks. Double-fire with the plugin copy is a cheap
    no-op (per-session flock + watermark in `run_capture_incremental`).
    """
    claude_dir = claude_dir or _claude_dir()
    settings_path = claude_dir / "settings.json"

    settings: dict[str, object] = {}
    if settings_path.is_file():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings = loaded

    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    pre = hooks.get("PreCompact")
    if not isinstance(pre, list):
        pre = []

    canonical = _precompact_group(memo_bin)
    memo_groups = [g for g in pre if _is_memo_group(g, verb=_PRECOMPACT_VERB)]
    foreign = [g for g in pre if not _is_memo_group(g, verb=_PRECOMPACT_VERB)]
    command = str(canonical["hooks"][0]["command"])  # type: ignore[index]

    if len(memo_groups) == 1 and memo_groups[0] == canonical:
        return {"action": "already", "command": command}

    action = "added" if not memo_groups else "updated"
    hooks["PreCompact"] = [*foreign, canonical]
    settings["hooks"] = hooks

    claude_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, settings_path)
    return {"action": action, "command": command}


def _is_memo_group(group: object, verb: str = _RECALL_VERB) -> bool:
    """True if a hook group is memo's own group for `verb`.

    memo-only verbs (``recall-hook``, ``capture-tick``) uniquely mark memo-owned
    groups — foreign hooks never contain them.
    """
    if not isinstance(group, dict):
        return False
    return any(
        isinstance(h, dict) and isinstance(h.get("command"), str) and verb in h["command"]
        for h in group.get("hooks", [])
    )


def wire_recall_hook(
    claude_dir: Path | None = None, *, memo_bin: str | None = None
) -> dict[str, str]:
    """Idempotently wire the memo recall hook into ``settings.json``.

    Drops any prior memo-owned recall group and appends the canonical one,
    leaving foreign ``UserPromptSubmit`` hooks untouched. Writes only when the
    value changes, so a correct install is a true no-op (safe on every memo-mcp
    start). Returns ``{"action": added|updated|already, "command": ...}``. Raises
    ``json.JSONDecodeError`` on an unparseable settings file.
    """
    claude_dir = claude_dir or _claude_dir()
    settings_path = claude_dir / "settings.json"

    settings: dict[str, object] = {}
    if settings_path.is_file():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings = loaded

    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    ups = hooks.get("UserPromptSubmit")
    if not isinstance(ups, list):
        ups = []

    canonical = _canonical_group(memo_bin)
    memo_groups = [g for g in ups if _is_memo_group(g)]
    foreign = [g for g in ups if not _is_memo_group(g)]
    command = str(canonical["hooks"][0]["command"])  # type: ignore[index]

    if len(memo_groups) == 1 and memo_groups[0] == canonical:
        return {"action": "already", "command": command}

    action = "added" if not memo_groups else "updated"
    hooks["UserPromptSubmit"] = [*foreign, canonical]
    settings["hooks"] = hooks

    claude_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, settings_path)
    return {"action": action, "command": command}


def recall_hook_wired(claude_dir: Path | None = None) -> bool:
    """True if a memo-owned recall group is present in ``settings.json``.

    Read-only detection for ``memo doctor``: when this is False, recall has
    silently stopped firing and needs a (self-)heal.
    """
    settings_path = (claude_dir or _claude_dir()) / "settings.json"
    if not settings_path.is_file():
        return False
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(loaded, dict):
        return False
    hooks = loaded.get("hooks")
    if not isinstance(hooks, dict):
        return False
    ups = hooks.get("UserPromptSubmit")
    if not isinstance(ups, list):
        return False
    return any(_is_memo_group(g) for g in ups)


def selfheal_recall_hook() -> None:
    """Best-effort recall + precompact hook re-assert for memo-mcp start. Never raises."""
    try:
        from memo.flags import flag_bool

        if not flag_bool("MEMO_HOOK_SELFHEAL"):
            return
        wire_recall_hook()
        wire_precompact_hook()
    except Exception:  # noqa: S110 — best-effort self-heal must never break mcp start
        pass


@click.command(name="install-recall-hook")
def install_recall_hook() -> None:
    """Wire memo's recall hook (UserPromptSubmit) into Claude Code settings."""
    claude_dir = _claude_dir()
    try:
        result = wire_recall_hook(claude_dir)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"could not parse {claude_dir / 'settings.json'}: {exc}. "
            "Fix it or wire the recall hook manually."
        ) from exc

    if result["action"] == "already":
        console.print("[dim]recall hook already wired — nothing to do.[/dim]")
    else:
        verb = "wired" if result["action"] == "added" else "updated"
        console.print(f"[green]✓[/green] {verb} memo recall hook → {result['command']}")

    pre = wire_precompact_hook(claude_dir)
    if pre["action"] != "already":
        console.print(f"[green]✓[/green] wired memo PreCompact force-flush → {pre['command']}")
    console.print("[dim]Open a new Claude Code session for recall to take effect.[/dim]")
