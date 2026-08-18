"""Declarative per-command filters for `tool_result` blocks — the 92% plane.

Measured 2026-08-18: 15.15M tokens of tool-loop output against 1.25M tokens of
answer output across memo's own usage — 92% of output spend is tool results,
and nothing upstream of this transform touches them. Live zone, so unlike a
prefix transform this carries no prompt-cache risk (see `zones.py`).

Coverage is total from day one: a command with no matching filter still gets
`generic_fallback` (head + tail + an elision count), never a pass-through of
an unbounded block. The YAML catalog under `filters/` is an optimization on
top of that floor, not a prerequisite for it.

Attribution: the match/pipeline YAML shape and the pipeline-action vocabulary
below (`keep_lines`, `remove_lines`, `head`, `tail`, `dedup`, `truncate_lines`,
`aggregate`, `json_extract`, `format_template`) are ported from snip
(https://github.com/edouard-claude/snip), Copyright (c) 2026 snip
contributors, licensed MIT (verified via the upstream LICENSE file,
2026-08-18). snip is a Go CLI (`internal/filter`, `internal/engine`) with a
richer schema (`inject`, `exclude_flags`, `on_error`, embedded `tests`); no
source was copied — this module is an independent Python reimplementation of
its filter shape and action set, scoped down to what this proxy needs. See
`/NOTICE` for the full attribution.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

DEFAULT_FILTERS_DIR = Path(__file__).resolve().parent.parent / "filters"

# Below this, a tool result is already cheap enough that cutting it is not
# worth spending a recovery-cache entry on.
_FALLBACK_MAX_CHARS = 4000


@dataclass
class Filter:
    name: str
    match_command: str | None = None
    match_subcommand: str | None = None
    pipeline: list[dict] = field(default_factory=list)


# --- Pipeline actions -------------------------------------------------------
# Each is a pure str -> str function. `pattern` fields come from YAML a user
# may have hand-edited, so every regex compile is guarded individually: a
# broken pattern degrades that one action to a no-op rather than aborting the
# pipeline or raising into the caller.


def _lines(text: str) -> list[str]:
    return text.splitlines()


# ReDoS guard: keep_lines/remove_lines/aggregate run a `pattern` that comes
# from hand-edited, untrusted YAML (or, one day, a user-authored filter). An
# unlucky quantifier shape (the shipped pytest.yaml's original
# `=+ .* =+` among them) backtracks O(line_length^2) against a single long
# line with no closing structure to match -- no malice required, just a big
# diff, a printed data structure, or a stray separator line in captured
# output. Measured on this pattern: 0.03s/10k chars, 0.5s/40k, ~3.3s/100k. The
# proxy runs synchronously in the critical path of every model call, so a
# slow match doesn't fail one request, it hangs the whole session. Capping
# both axes BEFORE any regex runs bounds worst-case work to a small constant
# regardless of what pattern a filter supplies -- this is the floor; the
# shipped patterns are additionally rewritten below to not backtrack at all.
_MAX_REGEX_LINE_CHARS = 2000
_MAX_REGEX_TOTAL_CHARS = 2_000_000


def _regex_safe_lines(text: str) -> list[str]:
    """Lines to run a filter's regex against. A line over the cap is
    truncated before matching is even attempted, not matched in full."""
    return [line[:_MAX_REGEX_LINE_CHARS] for line in text[:_MAX_REGEX_TOTAL_CHARS].splitlines()]


def _compile(pattern: object) -> re.Pattern[str] | None:
    if not isinstance(pattern, str):
        return None
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _keep_lines(text: str, action: dict) -> str:
    rx = _compile(action.get("pattern"))
    if rx is None:
        return text
    return "\n".join(line for line in _regex_safe_lines(text) if rx.search(line))


def _remove_lines(text: str, action: dict) -> str:
    rx = _compile(action.get("pattern"))
    if rx is None:
        return text
    return "\n".join(line for line in _regex_safe_lines(text) if not rx.search(line))


def _head(text: str, action: dict) -> str:
    n = action.get("n")
    if not isinstance(n, int):
        return text
    n = max(0, n)
    return "\n".join(_lines(text)[:n])


def _tail(text: str, action: dict) -> str:
    n = action.get("n")
    if not isinstance(n, int):
        return text
    n = max(0, n)
    lines = _lines(text)
    return "\n".join(lines[-n:] if n > 0 else [])


def _dedup(text: str, action: dict) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for line in _lines(text):
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return "\n".join(out)


def _truncate_lines(text: str, action: dict) -> str:
    max_len = action.get("max")
    if not isinstance(max_len, int) or max_len <= 0:
        return text
    out = []
    for line in _lines(text) or [text]:
        out.append(line[:max_len] + "..." if len(line) > max_len else line)
    return "\n".join(out)


def _aggregate(text: str, action: dict) -> str:
    rx = _compile(action.get("pattern"))
    label = action.get("label")
    if rx is None or not isinstance(label, str):
        return text
    count = sum(1 for line in _regex_safe_lines(text) if rx.search(line))
    return f"{count} {label}"


def _json_extract(text: str, action: dict) -> str:
    path = action.get("path")
    if not isinstance(path, str) or not path:
        return text
    try:
        current: Any = json.loads(text)
    except Exception:
        return text
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return text
    try:
        return json.dumps(current, ensure_ascii=False)
    except Exception:
        return text


def _format_template(text: str, action: dict) -> str:
    template = action.get("template")
    if not isinstance(template, str):
        return text
    # `.replace`, not `.format` -- `text` may itself contain `{`/`}` (JSON,
    # source code) that `.format` would try to interpret as fields and raise.
    return template.replace("{text}", text)


_ACTIONS: dict[str, Callable[[str, dict], str]] = {
    "keep_lines": _keep_lines,
    "remove_lines": _remove_lines,
    "head": _head,
    "tail": _tail,
    "dedup": _dedup,
    "truncate_lines": _truncate_lines,
    "aggregate": _aggregate,
    "json_extract": _json_extract,
    "format_template": _format_template,
}


def apply_pipeline(text: str, actions: list[dict]) -> str:
    """Run `actions` in order. An unknown or failing action is a no-op for
    that step, never a crash for the whole pipeline."""
    if not isinstance(actions, list):
        return text
    result = text
    for action in actions:
        if not isinstance(action, dict):
            continue
        fn = _ACTIONS.get(action.get("action", ""))
        if fn is None:
            continue
        try:
            result = fn(result, action)
        except Exception:
            _log.debug("proxy: pipeline action %r failed; left unchanged", action, exc_info=True)
            continue
    return result


def generic_fallback(text: str, max_chars: int) -> str:
    """Keep the first and last `max_chars // 2` characters with an elision
    marker between them. Returns `text` unchanged when already short enough."""
    try:
        if not isinstance(text, str) or len(text) <= max_chars:
            return text
        half = max(0, max_chars // 2)
        head, tail = text[:half], text[len(text) - half :]
        elided = len(text) - len(head) - len(tail)
        return f"{head}\n[... {elided} chars elided ...]\n{tail}"
    except Exception:
        return text


def load_filters(dir_path: Path) -> list[Filter]:
    """Load every `*.yaml` filter in `dir_path`. A missing directory yields no
    filters; a malformed file is skipped, never fatal to the rest."""
    filters: list[Filter] = []
    try:
        paths = sorted(Path(dir_path).glob("*.yaml"))
    except Exception:
        return []
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                continue
            name = data.get("name")
            if not isinstance(name, str) or not name:
                continue
            match = data.get("match")
            match = match if isinstance(match, dict) else {}
            command = match.get("command")
            subcommand = match.get("subcommand")
            pipeline = data.get("pipeline")
            pipeline = (
                [p for p in pipeline if isinstance(p, dict)] if isinstance(pipeline, list) else []
            )
            filters.append(
                Filter(
                    name=name,
                    match_command=command if isinstance(command, str) else None,
                    match_subcommand=subcommand if isinstance(subcommand, str) else None,
                    pipeline=pipeline,
                )
            )
        except Exception:
            _log.warning("proxy: skipping malformed filter %s", path, exc_info=True)
            continue
    return filters


def _select_pipeline(command: str | None, filters: list[Filter]) -> list[dict] | None:
    if not command:
        return None
    try:
        parts = command.strip().split()
        if not parts:
            return None
        cmd = Path(parts[0]).name
        sub = parts[1] if len(parts) > 1 else None
    except Exception:
        return None
    for f in filters:
        if not f.match_command and not f.match_subcommand:
            continue
        if f.match_command and f.match_command != cmd:
            continue
        if f.match_subcommand and f.match_subcommand != sub:
            continue
        return f.pipeline
    return None


def _tool_use_commands(messages: list) -> dict[str, str]:
    """Map `tool_use_id -> shell command string` from every `tool_use` block
    whose `input` carries a `command` field (the Bash tool's shape)."""
    out: dict[str, str] = {}
    try:
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_id = block.get("id")
                block_input = block.get("input")
                if not isinstance(tool_id, str) or not isinstance(block_input, dict):
                    continue
                command = block_input.get("command")
                if isinstance(command, str) and command:
                    out[tool_id] = command
    except Exception:
        return {}
    return out


def _block_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c["text"]
            for c in content
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _set_block_text(block: dict, text: str) -> None:
    if isinstance(block.get("content"), list):
        block["content"] = [{"type": "text", "text": text}]
    else:
        block["content"] = text


class ToolResults:
    name = "toolresults"
    zone = ZONE_LIVE

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_TOOL_RESULTS"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            if not zones.live_messages:
                return 0
            filters = load_filters(DEFAULT_FILTERS_DIR)
            commands = _tool_use_commands(zones.live_messages)
            saved = 0
            for message in zones.live_messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    saved += self._rewrite_block(block, commands, filters, ctx)
            return saved
        except Exception:
            return 0

    def _rewrite_block(
        self, block: object, commands: dict[str, str], filters: list[Filter], ctx: Context
    ) -> int:
        try:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return 0
            text = _block_text(block)
            if not text:
                return 0

            pipeline = _select_pipeline(commands.get(block.get("tool_use_id", "")), filters)
            new_text = (
                apply_pipeline(text, pipeline)
                if pipeline is not None
                else generic_fallback(text, _FALLBACK_MAX_CHARS)
            )
            # Cheap pre-filter: a pipeline that didn't even shrink the raw
            # text can't possibly be worth a stash call -- the marker we'd
            # append only adds more bytes on top.
            if not isinstance(new_text, str) or len(new_text) >= len(text):
                return 0

            key = ccr.stash(ctx.state_dir, text)
            if not key:
                # No recovery path -- leave the block completely untouched.
                return 0

            marked = new_text + ccr.marker(
                key, kept_chars=len(new_text), dropped_chars=len(text) - len(new_text)
            )
            # The REAL comparison: the marker is a ~130-byte fixed recovery
            # pointer appended after the pre-filter above ran, so a pipeline
            # reduction smaller than that overhead would otherwise still get
            # written back as a "cut" that is actually net-LARGER than the
            # original -- and reported as saved=0, hiding it from the ledger.
            # Only the FINAL, marker-included text counts; anything else
            # leaves the block completely untouched.
            if len(marked) >= len(text):
                return 0

            _set_block_text(block, marked)
            return max(0, est_tokens(text) - est_tokens(marked))
        except Exception:
            return 0
