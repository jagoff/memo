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

Detection is content-SHAPE based, not tool-name based
------------------------------------------------------
Ground truth for why: a real captured payload showed the model reading a
source file via `Bash` (`cat -n <path>`), never the `Read` tool -- so
`delta._read_tool_paths`, which used to recognize only `Read`, was blind to
it, and this module never even attempted to compress a 25k-char Python file
sitting in a `Bash` result. Two independent fixes, both content-driven:

1. `delta._extract_bash_read_path` now also recovers a real path from an
   unambiguous single-file Bash read (`cat`/`cat -n`/`head`/`tail`/`sed -n`),
   so the SAME `_language_for` + `signatures()` path used for `Read` covers
   those too. A real path with a recognized extension remains the STRONGER
   signal when it's available — it is what `Delta` needs for diff identity,
   and it is checked first in `_rewrite_block`/`_apply_sniff_pass` below.
2. `sniff_signatures()` is the fallback for everything else: `Grep -A`, a
   piped/globbed Bash command `_extract_bash_read_path` deliberately declines
   to parse, or any future tool. It runs `ast.parse` directly against the
   block's own text — the content's SHAPE is the only evidence it has, so it
   applies two extra guards a known-path read doesn't need (see its
   docstring): a minimum size, and a MATERIAL reduction, not just a
   non-empty one. A wrong signature map is worse than no compression, and
   sniffing has one less piece of corroborating evidence than a real path
   does, so it is the more conservative of the two routes.

`cat -n`-numbered output (and `Read`'s own "<n>\\t<line>" rendering) is not
valid Python as-is -- `_strip_line_numbers`/`_parseable_python` below give
`ast.parse` a de-numbered candidate to fall back to, so a file read via
`cat -n` parses exactly as if it had been read clean. This applies uniformly
whether the path was found via `Read`, recovered via `_extract_bash_read_path`,
or the content is being sniffed with no path at all.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context

try:
    import tree_sitter
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_typescript as ts_typescript

    _TS_LANG = tree_sitter.Language(ts_typescript.language_typescript())
    _JS_LANG = tree_sitter.Language(ts_javascript.language())
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False
    _TS_LANG = None
    _JS_LANG = None
from memo.proxy.transforms.delta import (
    _block_text,
    _read_tool_paths,
    _set_block_text,
    read_occurrences,
    seen_files,
)
from memo.proxy.zones import Zones, whole_history_scope

_log = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES = frozenset(
    {"python"} | ({"typescript", "javascript"} if _HAS_TREE_SITTER else set())
)
_ELISION = "..."

# Extended as more languages grow an `ast`-equivalent parser. A path whose
# extension isn't here — or isn't recognized above even if it is — passes
# straight through `signatures()` unchanged. Real captured proxy traffic
# (the ground truth for this module, /tmp/mt_payloads/*.json) is 100%
# Python -- this repo's own source, read back to itself -- so a second
# language has no evidence behind it yet; adding one speculatively would be
# exactly the "flexibility that wasn't requested" this codebase's own style
# guide warns against.
_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".mts": "typescript",
}

# A `cat -n`/Read-style numbered line: leading whitespace, digits, then
# whitespace before the real content. Deliberately loose on the exact
# width/separator (space or tab) since different renderers pad differently
# — see `_strip_line_numbers`.
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+[ \t]")

# Below this, a tool_result is cheap enough already that sniffing it without
# a real path is not worth the extra false-positive exposure (a short
# snippet is far likelier to be a coincidental parse than 25k+ chars of real
# structure — see sniff_signatures()).
_SNIFF_MIN_CHARS = 2000

# A sniffed reduction must bring the block to at most this fraction of its
# original size before it's trusted WITHOUT a real path/extension as
# corroborating evidence — stricter than the known-path route (which only
# requires ANY reduction, already gated by the marker-overhead check in
# `_finish`). A parse that technically succeeds but barely shrinks the block
# (e.g. a few real defs buried in mostly verbatim-kept imports) is exactly
# the "parses, but isn't really worth trusting blind" shape this guards
# against.
_SNIFF_MAX_RATIO = 0.7


def _strip_line_numbers(text: str) -> str | None:
    """De-number `cat -n`/Read-style `"<n> <line>"` output back to plain
    source. `None` unless EVERY line matches the numbered shape — a single
    unnumbered line is enough to call this "not actually numbered output"
    and refuse to touch it, rather than mangling arbitrary text that merely
    happens to start some OTHER line with a digit."""
    lines = text.splitlines()
    if not lines:
        return None
    out: list[str] = []
    for line in lines:
        match = _LINE_NUMBER_PREFIX_RE.match(line)
        if not match:
            return None
        out.append(line[match.end() :])
    return "\n".join(out)


def _try_parse(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except Exception:
        return None


def _parseable_python(source: str) -> tuple[ast.Module, str] | None:
    """`(tree, text)` for whichever of `source` or its de-numbered form
    `ast.parse` actually accepts — `None` if neither does. Tries `source`
    raw first (the common case) and only attempts de-numbering as a
    fallback, so this costs nothing extra for a normal clean read."""
    tree = _try_parse(source)
    if tree is not None:
        return tree, source
    stripped = _strip_line_numbers(source)
    if stripped is None:
        return None
    tree = _try_parse(stripped)
    if tree is not None:
        return tree, stripped
    return None


def signatures(source: str, language: str) -> str:
    """`source` reduced to import/def/class signatures for `language`.

    Returns `source` unchanged for any language other than "python", for any
    Python source `ast.parse` cannot handle even after de-numbering (see
    `_parseable_python`), or when the reduction would come out empty (a file
    with no imports/defs/classes at all still deserves a real read, not
    nothing).
    """
    try:
        if language not in _SUPPORTED_LANGUAGES:
            return source
        if language in ("typescript", "javascript") and _HAS_TREE_SITTER:
            lang = _TS_LANG if language == "typescript" else _JS_LANG
            return _signatures_typescript(source, lang)
        parsed = _parseable_python(source)
        if parsed is None:
            return source
        tree, parseable = parsed
        lines = parseable.splitlines()
        out: list[str] = []
        _emit_body(tree.body, lines, out, parseable)
        result = "\n".join(out)
        return result if result.strip() else source
    except Exception:
        return source


def sniff_signatures(text: str) -> str | None:
    """`text` reduced to its Python signature map when its SHAPE says it's
    source — independent of which tool produced it. Used only when NO file
    path is known for a block; when one is, a real path + extension is
    stronger evidence and `signatures()` is called directly instead (see
    `StructMap._rewrite`).

    `None` — meaning "leave the block alone" — for: non-string input; text
    shorter than `_SNIFF_MIN_CHARS`; text `signatures()` can't reduce at all
    (unparseable even after de-numbering, or parseable but structurally
    empty — a JSON object or a short config file can be valid Python syntax
    with zero imports/defs/classes, `signatures()`'s own emptiness guard
    already refuses those); or a reduction that doesn't clear
    `_SNIFF_MAX_RATIO` (see that constant's own comment).
    """
    try:
        if not isinstance(text, str) or len(text) < _SNIFF_MIN_CHARS:
            return None
        reduced = signatures(text, "python")
        if (
            isinstance(reduced, str)
            and reduced != text
            and len(reduced) <= len(text) * _SNIFF_MAX_RATIO
        ):
            return reduced
    except Exception:
        _log.debug("structmap: python sniff failed", exc_info=True)

    # Also try TypeScript/JavaScript sniffing for blocks with no known path
    if _HAS_TREE_SITTER:
        for _lang_name, lang_obj in [("typescript", _TS_LANG), ("javascript", _JS_LANG)]:
            if lang_obj is None:
                continue
            try:
                reduced = _signatures_typescript(text, lang_obj)
                if (
                    isinstance(reduced, str)
                    and reduced != text
                    and len(reduced) <= len(text) * _SNIFF_MAX_RATIO
                ):
                    return reduced
            except Exception:
                _log.debug("structmap: ts/js sniff failed for %s", _lang_name, exc_info=True)

    return None


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


def _signatures_typescript(source: str, lang: tree_sitter.Language | None) -> str:
    """Extract import/export/function/class signatures from TypeScript/JavaScript source."""
    if lang is None or not _HAS_TREE_SITTER:
        return source
    try:
        parser = tree_sitter.Parser(lang)
        tree = parser.parse(bytes(source, "utf8"))
        lines = source.splitlines()
        out: list[str] = []
        _walk_ts_nodes(tree.root_node, lines, out, source)
        result = "\n".join(out)
        return result if result.strip() else source
    except Exception:
        return source


def _walk_ts_nodes(node: Any, lines: list[str], out: list[str], source: str) -> None:
    """Walk AST and extract signatures for imports, exports, functions, classes."""
    for child in node.children:
        ctype = child.type
        if ctype in ("import_statement", "import_clause"):
            segment = _ts_source_segment(child, source)
            if segment:
                out.append(segment)
        elif ctype == "export_statement":
            # Only export declarations, not re-exports
            if child.child_by_field_name("declaration"):
                decl = child.child_by_field_name("declaration")
                if decl and decl.type in (
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                ):
                    segment = _ts_source_segment(child, source)
                    if segment:
                        out.append(segment)
        elif ctype in ("function_declaration", "function"):
            seg = _ts_header_segment(child, lines, source)
            if seg:
                out.append(seg)
                out.append(" " * (child.start_point[1] + 2) + _ELISION)
        elif ctype in ("class_declaration", "class"):
            seg = _ts_header_segment(child, lines, source)
            if seg:
                out.append(seg)
                _walk_ts_nodes(child, lines, out, source)
        elif ctype in ("lexical_declaration", "variable_declaration"):
            # const foo = ... or let foo = ...
            seg = _ts_source_segment(child, source)
            if seg and len(seg) < 200:
                out.append(seg)
        else:
            # Recurse into nested structures
            _walk_ts_nodes(child, lines, out, source)


def _ts_source_segment(node: Any, source: str) -> str:
    """Extract source text for a node."""
    start = node.start_byte
    end = node.end_byte
    try:
        return source[start:end].split("\n")[0]  # first line only
    except Exception:
        return ""


def _ts_header_segment(node: Any, lines: list[str], source: str) -> str:
    """Extract the header line(s) of a function/class declaration."""
    start_line = node.start_point[0]
    # For now, just take the first line (the declaration line)
    if start_line < len(lines):
        return lines[start_line]
    return _ts_source_segment(node, source)


def _node_start_line(node: ast.stmt) -> int:
    """The first source line `node` occupies, INCLUDING its own decorators
    (if any) — not just `node.lineno`, which for a decorated def/class
    points only at the `def`/`class` keyword itself, one or more lines after
    where the node actually starts on the page."""
    decorators = getattr(node, "decorator_list", [])
    return min((d.lineno for d in decorators), default=node.lineno)


def _header_lines(node: ast.stmt, lines: list[str]) -> list[str]:
    """Source lines spanning `node`'s decorators (if any) through the line
    its signature ends on — the literal original text, so wrapped
    parameters, type hints, and decorators all come through byte-for-byte
    rather than a reformatted approximation.

    The end boundary is derived from `node.body[0]`'s OWN start line (via
    `_node_start_line`, so a decorated first body statement's decorators are
    correctly attributed to THAT statement, never mistaken for part of
    `node`'s own header) — never from `body[0].lineno` alone, which for a
    decorated child points past its own decorators and would pull them into
    the WRONG node's header (the parent class, or an enclosing function, in
    the case of a nested decorated `def`). A one-line body (the whole
    `def f(): return 1` sitting on the same physical line as the header) has
    `body[0]`'s own start line equal to `node`'s own start line, in which
    case the header correctly extends THROUGH that shared line instead of
    being clipped one line short of the `def`/`class` line itself.
    """
    start = _node_start_line(node)
    body = getattr(node, "body", None)
    end = max(node.lineno, _node_start_line(body[0]) - 1) if body else node.lineno
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
            if whole_history_scope():
                if not zones.frozen_messages and not zones.live_messages:
                    return 0
                return self._apply_whole_history(zones, ctx)
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

    def _apply_whole_history(self, zones: Zones, ctx: Context) -> int:
        """`MEMO_PROXY_CONTENT_SCOPE=all` (default): a single ordered pass
        over the WHOLE conversation via `read_occurrences` -- see that
        function's docstring (delta.py) for why this stays deterministic and
        therefore safe to run over the frozen zone too. `read_occurrences`
        only ever visits blocks with a KNOWN path (`Read`, or a Bash command
        `_extract_bash_read_path` recognizes); `_apply_sniff_pass` below is
        the second, independent pass over everything it can't identify."""
        messages = [*zones.frozen_messages, *zones.live_messages]
        saved = 0
        for block, path, text, previous in read_occurrences(messages):
            if previous is not None:
                continue  # re-read -- Delta's case, not ours
            saved += self._rewrite(block, text, path, ctx)
        saved += self._apply_sniff_pass(messages, ctx)
        return saved

    def _apply_sniff_pass(self, messages: list, ctx: Context) -> int:
        """Every `tool_result` block with NO known path -- `Grep`, a piped or
        globbed Bash command, any future tool. Each block's own text is a
        pure, order-independent input (`sniff_signatures` looks only at the
        one block), so unlike the path-based pass above there is no "seen"/
        re-read concept to respect here: every qualifying block is sniffed
        exactly once, whether it sits in the frozen or live zone."""
        try:
            paths = _read_tool_paths(messages)
            saved = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    if paths.get(block.get("tool_use_id", "")):
                        continue  # known path -- handled above, or Delta's re-read case
                    text = _block_text(block)
                    if not text:
                        continue
                    saved += self._rewrite_sniffed(block, text, ctx)
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
            text = _block_text(block)
            if not text:
                return 0

            path = live_paths.get(block.get("tool_use_id", ""))
            if path:
                if seen.get(path):
                    return 0  # already seen this path -- Delta's case, not ours
                return self._rewrite(block, text, path, ctx)

            # No known path -- fall back to sniffing the content's own shape
            # (see the module docstring for why detection is shape-based).
            return self._rewrite_sniffed(block, text, ctx)
        except Exception:
            return 0

    def _rewrite(self, block: dict, text: str, path: str, ctx: Context) -> int:
        """Shared final step once (block, its raw text, its file path) are
        known -- identical for the tail-only and whole-history scopes; only
        how a block gets classified as a "first read" differs."""
        try:
            new_text = signatures(text, _language_for(path))
            if not isinstance(new_text, str) or len(new_text) >= len(text):
                return 0
            return self._finish(block, text, new_text, ctx)
        except Exception:
            return 0

    def _rewrite_sniffed(self, block: dict, text: str, ctx: Context) -> int:
        """Same finishing step as `_rewrite`, but the reduction comes from
        `sniff_signatures` (content shape only, no known path) rather than
        `signatures()` keyed off a real file extension."""
        try:
            new_text = sniff_signatures(text)
            if new_text is None:
                return 0
            return self._finish(block, text, new_text, ctx)
        except Exception:
            return 0

    def _finish(self, block: dict, text: str, new_text: str, ctx: Context) -> int:
        """Stash + marker + net-size check shared by both `_rewrite` and
        `_rewrite_sniffed` once each has already decided `new_text` is a
        real, worthwhile reduction of `text`."""
        try:
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
