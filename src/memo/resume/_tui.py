from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ._formatting import (
    _ansi_color,
    _display_title,
    _status_badge,
)
from ._types import (
    _RESUME_AGENT_COLORS,
    _RESUME_CYAN,
    _RESUME_ESCAPE_TIMEOUT_SECONDS,
    ResumeCandidate,
    _ResumeKey,
)
from ._utils import (
    _clip,
    _sort_key,
    _strip_ansi,
)

_RICH_KEY_ARROW_LEFT = "left"
_RICH_KEY_ARROW_RIGHT = "right"
_RICH_KEY_ARROW_UP = "up"
_RICH_KEY_ARROW_DOWN = "down"
_RICH_KEY_ENTER = "enter"
_RICH_KEY_ESC = "esc"
_RICH_KEY_TAB = "tab"
_RICH_KEY_BACKSPACE = "backspace"
_RICH_KEY_CTRL_C = "ctrl_c"
_RICH_KEY_CTRL_O = "ctrl_o"
_RICH_KEY_CTRL_T = "ctrl_t"
_RICH_KEY_CTRL_E = "ctrl_e"

_RICH_CONTROL_KEYS = frozenset(
    {
        _RICH_KEY_ARROW_LEFT,
        _RICH_KEY_ARROW_RIGHT,
        _RICH_KEY_ARROW_UP,
        _RICH_KEY_ARROW_DOWN,
        _RICH_KEY_ENTER,
        _RICH_KEY_ESC,
        _RICH_KEY_TAB,
        _RICH_KEY_BACKSPACE,
        _RICH_KEY_CTRL_C,
        _RICH_KEY_CTRL_O,
        _RICH_KEY_CTRL_T,
        _RICH_KEY_CTRL_E,
    }
)

_RESUME_TUI_HEADER = "Resume a previous session"
_RESUME_TUI_HIGHLIGHT_OPEN = "\x1b[48;5;60m\x1b[97m"
_RESUME_TUI_HIGHLIGHT_CLOSE = "\x1b[0m"
_RESUME_TUI_ZEBRA_OPEN = "\x1b[48;5;236m"
_RESUME_TUI_ZEBRA_CLOSE = "\x1b[49m"
_RESUME_TUI_DIM_OPEN = "\x1b[2m"
_RESUME_TUI_DIM_CLOSE = "\x1b[22m"
_RESUME_TUI_FOCUS_OPEN = "\x1b[1;36m"
_RESUME_TUI_FOCUS_CLOSE = "\x1b[0m"
_RESUME_TUI_HEADER_RESERVED = 2
_RESUME_TUI_FOOTER_RESERVED = 5
_RESUME_TUI_START_NEW_LABEL = "[ Start a new session ]"


def _resume_key_from_sequence(sequence: str) -> _ResumeKey:
    if sequence in {"\n", "\r"}:
        return "enter"
    if sequence == "\x1b":
        return "quit"
    if sequence in {"q", "Q"}:
        return "quit"
    if sequence in {"j", "J"}:
        return "down"
    if sequence in {"k", "K"}:
        return "up"
    if sequence.startswith(("\x1b[", "\x1bO")):
        if sequence.endswith("A"):
            return "up"
        if sequence.endswith("B"):
            return "down"
    return ""


def _read_resume_key_from_fd(
    stdin_fd: int,
    *,
    escape_timeout: float = _RESUME_ESCAPE_TIMEOUT_SECONDS,
) -> _ResumeKey:
    import select

    first = os.read(stdin_fd, 1)
    if not first:
        return ""
    if first != b"\x1b":
        return _resume_key_from_sequence(first.decode("utf-8", errors="ignore"))

    sequence = bytearray(first)
    if not select.select([stdin_fd], [], [], escape_timeout)[0]:
        return "quit"
    second = os.read(stdin_fd, 1)
    if not second:
        return ""
    sequence.extend(second)
    if second not in {b"[", b"O"}:
        return _resume_key_from_sequence(sequence.decode("ascii", errors="ignore"))

    while len(sequence) < 8:
        if not select.select([stdin_fd], [], [], escape_timeout)[0]:
            break
        ch = os.read(stdin_fd, 1)
        if not ch:
            break
        sequence.extend(ch)
        if 0x40 <= ch[0] <= 0x7E:
            break
    return _resume_key_from_sequence(sequence.decode("ascii", errors="ignore"))


def _rich_key_from_sequence(sequence: str) -> str:
    if not sequence:
        return ""
    if sequence in {"\n", "\r"}:
        return _RICH_KEY_ENTER
    if sequence == "\t":
        return _RICH_KEY_TAB
    if sequence in {"\x7f", "\x08"}:
        return _RICH_KEY_BACKSPACE
    if sequence == "\x03":
        return _RICH_KEY_CTRL_C
    if sequence == "\x05":
        return _RICH_KEY_CTRL_E
    if sequence == "\x0f":
        return _RICH_KEY_CTRL_O
    if sequence == "\x14":
        return _RICH_KEY_CTRL_T
    if sequence == "\x1b":
        return _RICH_KEY_ESC
    if sequence.startswith(("\x1b[", "\x1bO")):
        last = sequence[-1]
        if last == "A":
            return _RICH_KEY_ARROW_UP
        if last == "B":
            return _RICH_KEY_ARROW_DOWN
        if last == "C":
            return _RICH_KEY_ARROW_RIGHT
        if last == "D":
            return _RICH_KEY_ARROW_LEFT
        return ""
    if len(sequence) >= 1 and sequence.isprintable():
        return sequence
    return ""


def _read_rich_key_from_fd(
    stdin_fd: int,
    *,
    escape_timeout: float = _RESUME_ESCAPE_TIMEOUT_SECONDS,
) -> str:
    import select

    first = os.read(stdin_fd, 1)
    if not first:
        return ""
    if first != b"\x1b":
        buffer = bytearray(first)
        while True:
            try:
                return _rich_key_from_sequence(buffer.decode("utf-8"))
            except UnicodeDecodeError:
                if not select.select([stdin_fd], [], [], escape_timeout)[0]:
                    return ""
                more = os.read(stdin_fd, 1)
                if not more:
                    return ""
                buffer.extend(more)
                if len(buffer) >= 8:
                    return ""

    sequence = bytearray(first)
    if not select.select([stdin_fd], [], [], escape_timeout)[0]:
        return _RICH_KEY_ESC
    second = os.read(stdin_fd, 1)
    if not second:
        return ""
    sequence.extend(second)
    if second not in {b"[", b"O"}:
        return _rich_key_from_sequence(sequence.decode("ascii", errors="ignore"))

    while len(sequence) < 16:
        if not select.select([stdin_fd], [], [], escape_timeout)[0]:
            break
        ch = os.read(stdin_fd, 1)
        if not ch:
            break
        sequence.extend(ch)
        if 0x40 <= ch[0] <= 0x7E:
            break
    return _rich_key_from_sequence(sequence.decode("ascii", errors="ignore"))


def _candidate_created_at(candidate: ResumeCandidate) -> str:
    metadata = candidate.metadata or {}
    for key in ("created_at", "created", "timestamp"):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return candidate.updated_at


def _candidate_search_text(candidate: ResumeCandidate) -> str:
    parts = [
        candidate.title or "",
        candidate.summary or "",
        candidate.agent or "",
        candidate.session_id or "",
        candidate.cwd or "",
    ]
    return " ".join(parts).lower()


def _filter_resume_candidates(
    candidates: Sequence[ResumeCandidate],
    *,
    query: str,
    filter_mode: str,
    current_cwd: str,
) -> list[ResumeCandidate]:
    from ._utils import _same_cwd
    needle = query.strip().lower()
    visible: list[ResumeCandidate] = []
    for candidate in candidates:
        if filter_mode == "cwd" and (
            not current_cwd or not _same_cwd(candidate.cwd, current_cwd)
        ):
            continue
        if needle and needle not in _candidate_search_text(candidate):
            continue
        visible.append(candidate)
    return visible


def _sort_resume_candidates(
    candidates: Sequence[ResumeCandidate],
    *,
    sort_mode: str,
) -> list[ResumeCandidate]:
    if sort_mode == "created":
        return sorted(
            candidates,
            key=lambda item: _sort_key(_candidate_created_at(item)),
            reverse=True,
        )
    return sorted(candidates, key=lambda item: _sort_key(item.updated_at), reverse=True)


def _colorize_resume_interactive_line(line: str, *, agent: str | None = None) -> str:
    colored = line
    if colored.startswith(">"):
        colored = f"{_ansi_color('>', _RESUME_CYAN)}{colored[1:]}"
    if agent:
        color = _RESUME_AGENT_COLORS.get(agent.lower())
        tag = f"[{agent}]"
        if color and tag in colored:
            colored = colored.replace(tag, _ansi_color(tag, color), 1)
    return colored


def _resume_none_line(*, width: int, selected: bool = False) -> str:
    marker = ">" if selected else " "
    line = f"{marker} 0. [None · q/ESC]"
    return line[: max(1, width - 1)]


@dataclass
class _ResumeTuiState:
    candidates: list[ResumeCandidate]
    current_cwd: str
    query: str = ""
    filter_mode: str = "cwd"
    sort_mode: str = "updated"
    focus: str = "list"
    index: int = 0
    comfortable: bool = False
    expanded: bool = False
    view: str = "list"
    transcript_offset: int = 0
    notice: str = ""  # one-line warning (e.g. provider errors) shown under the header
    # Semantic search state (episodic memory). `semantic_order` maps session_id →
    # rank (0 = best match) for `semantic_query`; non-empty ⇒ the list is ordered
    # by meaning instead of recency/substring.
    semantic_query: str = ""
    semantic_order: dict[str, int] = field(default_factory=dict)


def _resume_tui_clamp(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(index, total - 1))


def _semantic_active(state: _ResumeTuiState) -> bool:
    """True when a meaning-ranked result set is current for the typed query."""
    return bool(state.query) and state.semantic_query == state.query and bool(state.semantic_order)


def _resume_tui_visible(state: _ResumeTuiState) -> list[ResumeCandidate]:
    if _semantic_active(state):
        # Meaning mode: show the ranked episode hits (not a substring match), still
        # honouring the Cwd/All filter. Old sessions surface here even though the
        # substring needle wouldn't match them.
        from ._utils import _same_cwd

        items = [c for c in state.candidates if c.session_id in state.semantic_order]
        if state.filter_mode == "cwd" and state.current_cwd:
            items = [c for c in items if _same_cwd(c.cwd, state.current_cwd)]
        items.sort(key=lambda c: state.semantic_order.get(c.session_id, 1_000_000))
        return items
    items = _filter_resume_candidates(
        state.candidates,
        query=state.query,
        filter_mode=state.filter_mode,
        current_cwd=state.current_cwd,
    )
    return _sort_resume_candidates(items, sort_mode=state.sort_mode)


def _apply_semantic(
    state: _ResumeTuiState, query: str, hits: Sequence[ResumeCandidate]
) -> None:
    """Merge semantic hits into the candidate pool + record their ranking.

    Episode-only hits (sessions beyond the loaded recency set) are appended so the
    picker can display them; the ranking drives `_resume_tui_visible` ordering.
    """
    state.semantic_query = query
    state.semantic_order = {c.session_id: i for i, c in enumerate(hits)}
    seen = {c.session_id for c in state.candidates}
    for hit in hits:
        if hit.session_id not in seen:
            state.candidates.append(hit)
            seen.add(hit.session_id)


def _resume_tui_dispatch(
    key: str,
    state: _ResumeTuiState,
    visible: list[ResumeCandidate],
) -> str:
    if key in {_RICH_KEY_CTRL_C, _RICH_KEY_ESC}:
        if state.view == "transcript":
            state.view = "list"
            return ""
        return "quit"
    if key == _RICH_KEY_ENTER:
        if state.view == "transcript":
            state.view = "list"
            return ""
        if state.index == 0:
            return "quit"
        return "select"
    if key in {"q", "Q"} and not state.query and state.focus == "list":
        return "quit"
    if state.view == "transcript":
        if key == _RICH_KEY_ARROW_UP:
            state.transcript_offset = max(0, state.transcript_offset - 1)
        elif key == _RICH_KEY_ARROW_DOWN:
            state.transcript_offset += 1
        elif key == _RICH_KEY_CTRL_T:
            state.view = "list"
        return ""
    if key == _RICH_KEY_TAB:
        order = ["list", "filter", "sort"]
        state.focus = order[(order.index(state.focus) + 1) % len(order)]
    elif key == _RICH_KEY_ARROW_UP:
        state.index = max(0, state.index - 1)
    elif key == _RICH_KEY_ARROW_DOWN:
        state.index = min(len(visible), state.index + 1)
    elif key in {_RICH_KEY_ARROW_LEFT, _RICH_KEY_ARROW_RIGHT}:
        if state.focus == "filter":
            state.filter_mode = "all" if state.filter_mode == "cwd" else "cwd"
            state.index = 1
        elif state.focus == "sort":
            state.sort_mode = "created" if state.sort_mode == "updated" else "updated"
            state.index = 1
    elif key == _RICH_KEY_CTRL_O:
        state.comfortable = not state.comfortable
    elif key == _RICH_KEY_CTRL_E:
        state.expanded = not state.expanded
    elif key == _RICH_KEY_CTRL_T:
        if visible:
            state.view = "transcript"
            state.transcript_offset = 0
    elif key == _RICH_KEY_BACKSPACE:
        if state.query:
            state.query = state.query[:-1]
            state.index = 1
    elif key and key not in _RICH_CONTROL_KEYS and len(key) == 1 and key.isprintable():
        state.query += key
        state.index = 1
    return ""


def _resume_tui_render(state: _ResumeTuiState, visible: list[ResumeCandidate]) -> None:
    width, height = shutil.get_terminal_size((120, 30))
    width = max(60, width)
    height = max(12, height)
    notice_rows = 1 if state.notice else 0
    body_height = max(
        3, height - _RESUME_TUI_HEADER_RESERVED - _RESUME_TUI_FOOTER_RESERVED - 1 - notice_rows
    )
    rows: list[str] = ["\x1b[H\x1b[2J\x1b[H"]
    rows.append(_RESUME_TUI_HEADER)
    if state.notice:
        rows.append(_dim_text(state.notice[: max(10, width - 1)]))
    rows.append("")
    rows.append(_resume_tui_filter_bar(state, width=width))
    rows.append("")
    if state.view == "transcript":
        rows.extend(_resume_tui_transcript_body(state, visible, width=width, height=body_height))
    else:
        rows.extend(_resume_tui_list_body(state, visible, width=width, height=body_height))
    rows.append("")
    rows.append(_resume_tui_pagination(state, visible, width=width))
    rows.append(_resume_tui_footer_primary(width=width))
    rows.append(_resume_tui_footer_secondary(width=width))
    sys.stdout.write("\r\n".join(rows))
    sys.stdout.flush()


def _resume_tui_filter_bar(state: _ResumeTuiState, *, width: int) -> str:
    if state.query:
        left = f"{state.query}_" if state.focus == "list" else state.query
    elif state.focus == "list":
        left = "_"
    else:
        left = f"{_RESUME_TUI_DIM_OPEN}Type to search{_RESUME_TUI_DIM_CLOSE}"
    filter_part = _resume_tui_toggle(
        "Filter",
        [("Cwd", "cwd"), ("All", "all")],
        state.filter_mode,
        active=state.focus == "filter",
    )
    sort_part = _resume_tui_toggle(
        "Sort",
        [("Updated", "updated"), ("Created", "created")],
        state.sort_mode,
        active=state.focus == "sort",
    )
    right = f"{filter_part}   {sort_part}"
    pad = max(1, width - _visible_len(left) - _visible_len(right))
    return f"{left}{' ' * pad}{right}"


def _resume_tui_toggle(
    label: str,
    options: list[tuple[str, str]],
    current: str,
    *,
    active: bool,
) -> str:
    parts: list[str] = []
    for caption, key in options:
        if key == current:
            chunk = f"[{caption}]"
            if active:
                chunk = f"{_RESUME_TUI_FOCUS_OPEN}{chunk}{_RESUME_TUI_FOCUS_CLOSE}"
            parts.append(chunk)
        else:
            parts.append(caption)
    inner = " ".join(parts)
    label_text = f"{_RESUME_TUI_DIM_OPEN}{label}:{_RESUME_TUI_DIM_CLOSE} {inner}"
    return label_text


def _resume_tui_list_body(
    state: _ResumeTuiState,
    visible: list[ResumeCandidate],
    *,
    width: int,
    height: int,
) -> list[str]:
    total_items = len(visible) + 1
    rows_per_item = 2 if state.comfortable else 1
    capacity = max(1, height // rows_per_item)
    start = max(0, min(state.index - capacity + 1, max(0, total_items - capacity)))
    end = min(total_items, start + capacity)
    body: list[str] = []
    for absolute in range(start, end):
        selected = absolute == state.index
        if absolute == 0:
            body.append(_resume_tui_start_new_row(selected=selected, width=width))
            if state.comfortable:
                body.append("")
            continue
        candidate = visible[absolute - 1]
        zebra = absolute % 2 == 0
        body.append(
            _resume_tui_candidate_row(
                candidate, selected=selected, width=width, zebra=zebra
            )
        )
        if state.expanded and selected and candidate.summary and candidate.summary != candidate.title:
            body.append(_resume_tui_detail_row(candidate, width=width))
        elif state.comfortable:
            body.append("")
    if not visible:
        body.append(_dim_text("  No sessions match the current filters."))
    return body


def _resume_tui_start_new_row(*, selected: bool, width: int) -> str:
    cursor = "▌" if selected else " "
    raw = f" {cursor}            {_RESUME_TUI_START_NEW_LABEL}"
    line_width = max(20, width - 1)
    clipped = raw[:line_width].ljust(line_width)
    if selected:
        return f"{_RESUME_TUI_HIGHLIGHT_OPEN}{clipped}{_RESUME_TUI_HIGHLIGHT_CLOSE}"
    return f"{_RESUME_TUI_DIM_OPEN}{clipped}{_RESUME_TUI_DIM_CLOSE}"


def _resume_tui_candidate_row(
    candidate: ResumeCandidate,
    *,
    selected: bool,
    width: int,
    zebra: bool = False,
) -> str:
    from ._utils import _format_relative_time
    time_text = _format_relative_time(candidate.updated_at) or "—"
    time_col = time_text.rjust(8)
    cursor = "▌" if selected else " "
    title = _display_title(candidate)

    # Add agent tag with color
    agent_tag = f"[{candidate.agent}]"
    color = _RESUME_AGENT_COLORS.get(candidate.agent.lower())
    if color:
        agent_tag = _ansi_color(agent_tag, color)
    badge = _status_badge(candidate.status)

    raw = f" {cursor} {time_col}    {badge}{agent_tag} {title}"
    line_width = max(20, width - 1)
    # Calculate display width without ANSI codes
    raw_display = _strip_ansi(raw)

    # Build final string with ANSI codes
    if len(raw_display) <= line_width:
        clipped = raw
    else:
        # Need to clip, but preserve ANSI codes
        clipped = raw[:len(raw) - len(raw_display) + line_width]

    if selected:
        return f"{_RESUME_TUI_HIGHLIGHT_OPEN}{clipped}{_RESUME_TUI_HIGHLIGHT_CLOSE}"
    if zebra:
        return f"{_RESUME_TUI_ZEBRA_OPEN}{clipped}{_RESUME_TUI_ZEBRA_CLOSE}"
    return clipped


def _resume_tui_detail_row(candidate: ResumeCandidate, *, width: int) -> str:
    detail = _clip(candidate.summary, max(20, width - 20))
    return f"           {_RESUME_TUI_DIM_OPEN}{detail}{_RESUME_TUI_DIM_CLOSE}"


def _resume_tui_transcript_body(
    state: _ResumeTuiState,
    visible: list[ResumeCandidate],
    *,
    width: int,
    height: int,
) -> list[str]:
    if not visible:
        return [_dim_text("  No session selected.")]
    candidate = visible[state.index]
    lines: list[str] = [
        f"  Agent:   {candidate.agent}",
        f"  Session: {candidate.session_id}",
        f"  URI:     {candidate.uri}",
    ]
    if candidate.cwd:
        lines.append(f"  Cwd:     {candidate.cwd}")
    if candidate.updated_at:
        lines.append(f"  Updated: {candidate.updated_at}")
    lines.append("")
    summary = candidate.summary or candidate.title or "(no transcript captured)"
    lines.extend(f"  {chunk}" for chunk in _wrap_text(summary, width - 4))
    slice_ = lines[state.transcript_offset : state.transcript_offset + height]
    return slice_ or [""]


def _wrap_text(text: str, width: int) -> list[str]:
    target = max(10, width)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) <= target:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _resume_tui_pagination(
    state: _ResumeTuiState,
    visible: list[ResumeCandidate],
    *,
    width: int,
) -> str:
    total_candidates = len(visible)
    if state.index == 0:
        text = f"new / {total_candidates}   --"
    elif total_candidates:
        percent = round((state.index / total_candidates) * 100)
        text = f"{state.index} / {total_candidates}   {percent}%"
    else:
        text = "new / 0   --"
    pad = max(1, width - len(text))
    return f"{' ' * pad}{_RESUME_TUI_DIM_OPEN}{text}{_RESUME_TUI_DIM_CLOSE}"


def _resume_tui_footer_primary(*, width: int) -> str:
    parts = [
        "enter resume",
        "esc start new",
        "ctrl+c quit",
        "tab focus sort/filter",
        "←/→ change option",
    ]
    raw = "   ".join(parts)
    clipped = raw[: max(10, width - 1)]
    return _dim_text(clipped)


def _resume_tui_footer_secondary(*, width: int) -> str:
    parts = [
        "ctrl+o comfortable view",
        "ctrl+t transcript",
        "ctrl+e expand",
        "↑/↓ browse",
    ]
    raw = "   ".join(parts)
    clipped = raw[: max(10, width - 1)]
    return _dim_text(clipped)


def _dim_text(text: str) -> str:
    return f"{_RESUME_TUI_DIM_OPEN}{text}{_RESUME_TUI_DIM_CLOSE}"


def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


def pick_resume_candidate_interactive(
    candidates: Sequence[ResumeCandidate],
    *,
    current_cwd: str | None = None,
    start_filter: str = "cwd",
    notice: str = "",
    semantic_fn: Callable[[str], Sequence[ResumeCandidate]] | None = None,
    debounce_s: float = 0.3,
) -> ResumeCandidate | None:
    if not candidates:
        return None
    import select
    import termios
    import tty

    from ._utils import _resolve_cwd

    stdin_fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(stdin_fd)
    cwd = _resolve_cwd(current_cwd) if current_cwd else _resolve_cwd(os.getcwd())
    state = _ResumeTuiState(
        candidates=list(candidates),
        current_cwd=cwd,
        filter_mode="all" if start_filter == "all" else "cwd",
        notice=notice,
        index=1,
    )

    selected: ResumeCandidate | None = None
    try:
        tty.setcbreak(stdin_fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        while True:
            visible = _resume_tui_visible(state)
            state.index = _resume_tui_clamp(state.index, len(visible) + 1)
            _resume_tui_render(state, visible)
            # Debounced semantic re-rank: when the query has settled (no keypress
            # for `debounce_s`) and changed since the last search, ask the episode
            # index. Best-effort — a cold embedder returns [] and we stay on
            # substring. No threads: the warm-socket embed is ~50 ms.
            if (
                semantic_fn is not None
                and state.query
                and state.query != state.semantic_query
                and not select.select([stdin_fd], [], [], debounce_s)[0]
            ):
                try:
                    hits = semantic_fn(state.query)
                except Exception:
                    hits = []
                _apply_semantic(state, state.query, list(hits))
                continue
            try:
                key = _read_rich_key_from_fd(stdin_fd)
            except KeyboardInterrupt:
                break
            action = _resume_tui_dispatch(key, state, visible)
            if action == "quit":
                break
            if action == "select" and visible and state.index >= 1:
                selected = visible[state.index - 1]
                break
    except KeyboardInterrupt:
        selected = None
    finally:
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)
    return selected
