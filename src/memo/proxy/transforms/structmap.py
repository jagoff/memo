"""Reduce a code file's FIRST read to its structural signatures.

`Delta` (`delta.py`) handles every read of a path AFTER the first one in a
conversation; this module only ever touches the first. Both consult
`memo.proxy.transforms.delta.seen_files` to agree on which case a given
`tool_result` block is, so the two never compete for the same block — see
`delta.py`'s module docstring for why "seen" is scoped the way it is, and
`registry.py` for why both must run before `JsonCrush`/`ToolResults`.

Python only today, driven off the stdlib `ast` module. A wrong signature map
would mislead the model about what a file actually contains — worse than no
compression at all — so an unsupported language, or ANY parse failure (a
syntax error, an encoding problem, a `.py`-named file that is not actually
Python), returns the source completely unchanged. No regex-based
approximation is attempted as a fallback: an approximate parse is exactly the
"confidently wrong" failure mode this module exists to avoid.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.transforms.delta import _block_text, _read_tool_paths, _set_block_text, seen_files
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES = frozenset({"python"})
_ELISION = "..."

# Extended as more languages grow an `ast`-equivalent parser. A path whose
# extension isn't here — or isn't recognized above even if it is — passes
# straight through `signatures()` unchanged.
_LANGUAGE_BY_EXTENSION = {".py": "python"}


def signatures(source: str, language: str) -> str:
    """`source` reduced to import/def/class signatures for `language`.

    Returns `source` unchanged for any language other than "python", for any
    Python source `ast.parse` cannot handle, or when the reduction would come
    out empty (a file with no imports/defs/classes at all still deserves a
    real read, not nothing).
    """
    if language not in _SUPPORTED_LANGUAGES:
        return source
    try:
        tree = ast.parse(source)
        lines = source.splitlines()
        out: list[str] = []
        _emit_body(tree.body, lines, out, source)
        result = "\n".join(out)
        return result if result.strip() else source
    except Exception:
        return source


def _emit_body(stmts: list[ast.stmt], lines: list[str], out: list[str], source: str) -> None:
    for node in stmts:
        if isinstance(node, ast.Import | ast.ImportFrom):
            segment = ast.get_source_segment(source, node)
            if segment:
                out.append(segment)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.extend(_header_lines(node, lines))
            out.append(" " * (node.col_offset + 4) + _ELISION)
        elif isinstance(node, ast.ClassDef):
            out.extend(_header_lines(node, lines))
            _emit_body(node.body, lines, out, source)


def _header_lines(node: ast.stmt, lines: list[str]) -> list[str]:
    """Source lines spanning `node`'s decorators (if any) through the line
    its signature ends on — the literal original text, so wrapped
    parameters, type hints, and decorators all come through byte-for-byte
    rather than a reformatted approximation."""
    decorators = getattr(node, "decorator_list", [])
    start = min((d.lineno for d in decorators), default=node.lineno)
    start = min(start, node.lineno)
    body = getattr(node, "body", None)
    end = body[0].lineno - 1 if body else node.lineno
    end = max(end, start)
    return lines[start - 1 : end]


def _language_for(file_path: str) -> str:
    try:
        return _LANGUAGE_BY_EXTENSION.get(Path(file_path).suffix.lower(), "")
    except Exception:
        return ""


class StructMap:
    name = "structmap"
    zone = ZONE_LIVE

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_STRUCTMAP"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            if not zones.live_messages:
                return 0
            seen = seen_files(zones)
            live_paths = _read_tool_paths(zones.live_messages)
            saved = 0
            for message in zones.live_messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    saved += self._rewrite_block(block, live_paths, seen, ctx)
            return saved
        except Exception:
            return 0

    def _rewrite_block(
        self,
        block: object,
        live_paths: dict[str, str],
        seen: dict[str, str],
        ctx: Context,
    ) -> int:
        try:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return 0
            path = live_paths.get(block.get("tool_use_id", ""))
            if not path:
                return 0
            if seen.get(path):
                return 0  # already seen this path -- Delta's case, not ours

            text = _block_text(block)
            if not text:
                return 0

            new_text = signatures(text, _language_for(path))
            if not isinstance(new_text, str) or len(new_text) >= len(text):
                return 0

            key = ccr.stash(ctx.state_dir, text)
            if not key:
                # No recovery path -- leave the block completely untouched.
                return 0

            marked = new_text + ccr.marker(
                key,
                kept_chars=len(new_text),
                dropped_chars=len(text) - len(new_text),
                stashed=text,
            )
            # Only the FINAL, marker-included text counts as the saving --
            # a reduction smaller than the marker's own overhead must leave
            # the block untouched rather than write back something net
            # larger than the original while still reporting a "cut".
            if len(marked) >= len(text):
                return 0

            _set_block_text(block, marked)
            return max(0, est_tokens(text) - est_tokens(marked))
        except Exception:
            return 0
