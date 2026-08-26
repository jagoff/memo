from memo.proxy.plan import Context
from memo.proxy.transforms.structmap import (
    StructMap,
    _language_for,
    _strip_line_numbers,
    signatures,
    sniff_signatures,
)
from memo.proxy.zones import Zones

SRC = '''
import os
from pathlib import Path


def alpha(a: int, b: str = "x") -> bool:
    """Docstring."""
    total = 0
    for i in range(100):
        total += i
    return bool(total)


class Beta:
    def gamma(self) -> None:
        pass
'''


def test_signatures_keep_definitions_and_imports():
    out = signatures(SRC, "python")
    assert 'def alpha(a: int, b: str = "x") -> bool:' in out
    assert "class Beta:" in out
    assert "def gamma(self) -> None:" in out
    assert "import os" in out


def test_signatures_drop_function_bodies():
    out = signatures(SRC, "python")
    assert "total += i" not in out


def test_signatures_are_shorter_than_the_source():
    assert len(signatures(SRC, "python")) < len(SRC)


def test_an_unknown_language_returns_the_source_unchanged():
    assert signatures(SRC, "brainfuck") == SRC


# --- Beyond the brief's baseline ---------------------------------------------


def test_a_syntax_error_returns_the_source_unchanged():
    """A wrong signature map is worse than no compression: `ast.parse`
    raising must never be papered over with a regex approximation."""
    broken = "def alpha(:\n    pass\n"
    assert signatures(broken, "python") == broken


def test_empty_source_is_returned_unchanged():
    assert signatures("", "python") == ""


def test_a_null_byte_makes_ast_parse_raise_valueerror_not_syntaxerror(tmp_path):
    """`ast.parse` rejects an embedded null byte with `ValueError`, not
    `SyntaxError` -- a `.py`-named file that is not actually valid Python in
    a way that doesn't fit the more obvious syntax-error case. The broad
    `except Exception` in `signatures()` must still catch it and fall open to
    the untouched source, and the same must hold end-to-end through
    `StructMap.apply()`, not just the bare function."""
    src = "def f():\x00\n    pass\n"
    assert signatures(src, "python") == src

    zones = _read_zones("bad.py", src)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == src


def test_a_non_string_tool_result_content_never_raises(tmp_path):
    """A `tool_result` block whose `content` is neither a string nor a list
    (a malformed upstream payload) must fall open, not raise out of
    `apply()`."""
    zones = Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r1",
                        "name": "Read",
                        "input": {"file_path": "a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "r1", "content": 12345}],
            },
        ]
    )
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_a_file_with_no_defs_or_imports_is_returned_unchanged():
    """A reduction that would come out empty is not a real compression --
    the file gets a real read instead of nothing."""
    src = "x = 1\nprint(x)\n"
    assert signatures(src, "python") == src


def test_decorators_are_preserved():
    src = "@staticmethod\ndef foo():\n    return 1\n"
    out = signatures(src, "python")
    assert "@staticmethod" in out
    assert "def foo():" in out
    assert "return 1" not in out


def test_async_function_signatures_are_kept():
    src = "async def fetch(url: str) -> str:\n    return await get(url)\n"
    out = signatures(src, "python")
    assert "async def fetch(url: str) -> str:" in out
    assert "return await get" not in out


# --- Fix round 1 regressions: `_header_lines` boundary calculation -----------
#
# The original `_header_lines` derived a node's header END from the NEXT
# body statement's raw `.lineno` (`body[0].lineno - 1`). Two shapes break
# that: (1) a one-liner, where the body sits on the SAME line as the
# header, one line short of where `.lineno` actually needs to land; (2) a
# body[0] that is ITSELF decorated, where `.lineno` points past its own
# decorators -- pulling them into the WRONG node's header instead of its
# own. `ast.parse` succeeds in every one of these, so the fail-open path
# never fires -- exactly the "confidently wrong" class the brief warns is
# worse than no compression.


def test_a_decorated_one_liner_keeps_its_def_line():
    """Critical 1 repro: a decorated one-line stub/property previously lost
    its `def` line entirely."""
    src = "@overload\ndef foo(x: int) -> int: ...\n"
    out = signatures(src, "python")
    assert "@overload" in out
    assert "def foo(x: int) -> int:" in out


def test_a_decorated_one_liner_property_keeps_its_def_line():
    src = "class Widget:\n    @property\n    def name(self) -> str: return self._name\n"
    out = signatures(src, "python")
    assert "@property" in out
    assert "def name(self) -> str:" in out


def test_a_class_whose_first_member_is_decorated_does_not_duplicate_the_decorator():
    """Critical 2 repro: the class's own header previously absorbed its
    first member's decorator line -- an orphaned decorator under the class
    header with no signature under it, duplicated again under the member's
    own (correct) header. `gamma` being the class's ONLY member means a
    purely positional check ("is `@property` right after `class Beta:`?")
    can't tell the bug apart from the fix -- both put it there, since
    gamma's own (correct) header follows immediately either way. The count
    check is what actually discriminates: buggy code emits it twice (once
    wrongly absorbed into Beta's header, once correctly as gamma's own);
    the fix emits it exactly once, attributed only to gamma."""
    src = "class Beta:\n    @property\n    def gamma(self) -> int:\n        return 1\n"
    out = signatures(src, "python")
    assert out.count("@property") == 1
    assert out.splitlines() == [
        "class Beta:",
        "    @property",
        "    def gamma(self) -> int:",
        "        ...",
    ]


def test_a_class_with_a_later_decorated_member_attributes_the_decorator_correctly():
    """A second, more discriminating shape than the single-member case
    above: two methods, only the SECOND decorated. Each node's header is
    now computed purely from ITS OWN start/body, independent of any
    sibling, so this must come out identical in structure regardless of
    how many undecorated members precede the decorated one."""
    src = (
        "class Beta:\n"
        "    def first(self) -> int:\n"
        "        return 1\n"
        "    @cached_property\n"
        "    def second(self) -> int:\n"
        "        return 2\n"
    )
    out = signatures(src, "python")
    assert out.count("@cached_property") == 1
    assert out.splitlines() == [
        "class Beta:",
        "    def first(self) -> int:",
        "        ...",
        "    @cached_property",
        "    def second(self) -> int:",
        "        ...",
    ]


def test_a_nested_function_with_a_decorated_inner_def_does_not_leak_into_the_outer_header():
    """Same boundary bug, a third shape: a decorated function nested INSIDE
    another function. The outer function's header must not absorb the
    inner function's decorator line either."""
    src = "def outer():\n    @cache\n    def inner():\n        return 1\n    return inner\n"
    out = signatures(src, "python")
    assert "def outer():" in out
    lines = out.splitlines()
    outer_idx = lines.index("def outer():")
    assert lines[outer_idx + 1] != "    @cache", (
        "the nested function's decorator leaked into outer()'s own header"
    )


def test_stacked_decorators_are_all_kept_in_order():
    src = "@a\n@b\n@c\ndef foo():\n    pass\n"
    out = signatures(src, "python")
    assert "@a\n@b\n@c\ndef foo():" in out


def test_an_async_decorated_one_liner_keeps_its_def_line():
    src = "@overload\nasync def fetch(url: str) -> str: ...\n"
    out = signatures(src, "python")
    assert "@overload" in out
    assert "async def fetch(url: str) -> str:" in out


# --- _strip_line_numbers / de-numbered ast.parse fallback --------------------
#
# Ground truth: a real captured payload showed the model reading a source
# file via `Bash` (`cat -n <path>`) -- its output is "<n> <line>" per line,
# which `ast.parse` rejects outright. Stripping the numbering back off (only
# when EVERY line has it -- never a partial/coincidental match) lets a
# `cat -n` read parse exactly as if it had been read clean.


def _numbered(text: str) -> str:
    """A `cat -n`-shaped rendering of `text` (4-wide right-justified line
    number + one space + content), matching the real captured payload's own
    format byte-for-byte in spirit, though not in exact width -- the
    stripping regex is deliberately width-agnostic."""
    return "\n".join(f"{i:>4} {line}" for i, line in enumerate(text.splitlines(), start=1))


def test_strip_line_numbers_recovers_the_original_when_every_line_is_numbered():
    # `_numbered`/`_strip_line_numbers` both go through `splitlines()`, which
    # discards information about a final trailing newline on either side of
    # the round trip -- irrelevant to `ast.parse`, so compared with it
    # stripped from both sides here.
    assert _strip_line_numbers(_numbered(SRC)) == SRC.rstrip("\n")


def test_strip_line_numbers_bails_when_any_line_lacks_a_number():
    """A single non-numbered line is enough to call this "not actually
    numbered output" and refuse to touch it, rather than mangling arbitrary
    text that happens to start some OTHER line with a digit."""
    text = "def f():\n    return 1\n"
    assert _strip_line_numbers(text) is None


def test_strip_line_numbers_returns_none_for_empty_text():
    assert _strip_line_numbers("") is None


def test_signatures_parses_a_cat_n_numbered_python_file():
    """The exact shape from the real captured payload: `signatures()` must
    transparently de-number and still produce the same reduction it would
    have for a clean read."""
    out = signatures(_numbered(BIG_SRC), "python")
    assert out == signatures(BIG_SRC, "python")
    assert len(out) < len(_numbered(BIG_SRC))
    assert "total += j" not in out


def test_signatures_leaves_genuinely_unparseable_numbered_looking_text_unchanged():
    """De-numbering is a FALLBACK, not a license to mangle text that merely
    starts some lines with digits -- if even the stripped candidate doesn't
    parse, fall open exactly as the plain case does."""
    text = "1 not python(:\n2 still broken\n"
    assert signatures(text, "python") == text


# --- sniff_signatures: content-shape detection, no path required -------------
#
# `delta._read_tool_paths` only ever gives IDENTITY for a `Read` or a Bash
# command matching `_extract_bash_read_path`'s narrow grammar. Everything
# else -- `Grep -A`, a piped Bash command, any future tool -- has no path at
# all. `sniff_signatures` is the fallback: the content's own SHAPE, not the
# tool that produced it, decides whether it is source.


def test_sniff_signatures_compresses_a_large_pathless_python_blob():
    big = "import os\n\n\n" + BIG_SRC  # comfortably over the size floor
    out = sniff_signatures(big)
    assert out is not None
    assert len(out) < len(big)
    assert "total += j" not in out


def test_sniff_signatures_returns_none_below_the_size_floor():
    """A tiny Python-shaped blob is real source but not worth the extra
    false-positive risk of sniffing without a known path -- the size floor
    exists for exactly this shape, not just genuinely non-source content."""
    tiny = "def f():\n    return 1\n"
    assert sniff_signatures(tiny) is None


def test_sniff_signatures_returns_none_for_unparseable_prose():
    prose = ("This is a long line of ordinary English prose, not code. " * 50) + "\n"
    assert len(prose) > 2000
    assert sniff_signatures(prose) is None


def test_sniff_signatures_returns_none_for_a_large_json_object():
    """A JSON object is also a valid Python dict-literal EXPRESSION --
    `ast.parse` accepts it -- but it has no import/def/class, so
    `signatures()`'s own emptiness guard already refuses it. This is exactly
    the "log/config/JSON that happens to be valid Python" risk the brief
    warns about."""
    import json

    blob = json.dumps({f"key_{i}": "value " * 10 for i in range(200)})
    assert len(blob) > 2000
    assert sniff_signatures(blob) is None


def test_sniff_signatures_returns_none_when_the_reduction_is_not_material():
    """Parses, and genuinely has one real function -- but it's dwarfed by
    verbatim-kept import lines, so the overall reduction barely moves the
    needle. Without a known path/extension as corroborating evidence, that's
    too thin to trust."""
    imports = "\n".join(f"import module_{i}_with_a_fairly_long_name" for i in range(150))
    src = imports + "\n\n\ndef f():\n    return 1\n"
    assert len(src) > 2000
    assert sniff_signatures(src) is None


def test_sniff_signatures_never_raises_on_non_string_input():
    assert sniff_signatures(None) is None  # type: ignore[arg-type]
    assert sniff_signatures(12345) is None  # type: ignore[arg-type]


def test_language_for_maps_py_extension_to_python():
    assert _language_for("src/memo/foo.py") == "python"
    assert _language_for("README.md") == ""
    assert _language_for("") == ""


# --- StructMap transform: Read-scoped, first-read-only, fail-open ------------


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def _read_zones(file_path: str, content: str, tool_use_id: str = "r1") -> Zones:
    return Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Read",
                        "input": {"file_path": file_path},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
                ],
            },
        ]
    )


# A real file, big enough that the signature reduction comfortably beats the
# recovery marker's own ~130-char overhead -- SRC above is deliberately tiny
# and, correctly, does NOT clear that bar (see
# test_apply_never_stores_a_net_larger_block_than_the_original below for the
# tiny-file case this module must refuse to touch).
BIG_SRC = "import os\n\n\n" + "\n\n".join(
    f'def func_{i}(a: int, b: str = "x") -> bool:\n'
    f'    """Docstring for func_{i}."""\n'
    f"    total = 0\n"
    f"    for j in range(100):\n"
    f"        total += j * {i}\n"
    f"    return bool(total)\n"
    for i in range(40)
)


def test_apply_compresses_a_first_read_of_a_python_file(tmp_path):
    zones = _read_zones("src/pkg/mod.py", BIG_SRC)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert len(new_text) < len(BIG_SRC)
    assert "total += j" not in new_text


def test_apply_marker_lets_the_original_be_recovered(tmp_path):
    zones = _read_zones("src/pkg/mod.py", BIG_SRC)
    StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert "memo_crush_retrieve" in new_text

    from memo.proxy import ccr

    key = new_text.split('hash_marker="')[1].split('"')[0]
    assert ccr.recover(tmp_path, key) == BIG_SRC


def test_apply_leaves_a_non_python_read_untouched(tmp_path):
    zones = _read_zones("README.md", "# hello\n\nsome prose\n")
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved == 0
    assert new_text == "# hello\n\nsome prose\n"


def test_apply_leaves_a_non_read_tool_result_untouched(tmp_path):
    zones = Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "b1", "content": SRC}],
            },
        ]
    )
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == SRC


def test_apply_leaves_a_re_read_of_a_seen_file_untouched_tail_only(tmp_path, monkeypatch):
    """Tail-only scope (MEMO_PROXY_CONTENT_SCOPE=tail): a second read of a
    path already in frozen_messages is Delta's case, not StructMap's --
    StructMap must not also compress it. Under this scope StructMap never
    scans frozen_messages as a compression TARGET (only as the source of
    `seen`), so the first-read frozen block stays untouched too."""
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    zones = _read_zones("src/pkg/mod.py", SRC, tool_use_id="r2")
    zones.frozen_messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "Read",
                    "input": {"file_path": "src/pkg/mod.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r1", "content": SRC}],
        },
    ]
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == SRC
    assert zones.frozen_messages[1]["content"][0]["content"] == SRC


def test_apply_compresses_the_frozen_first_read_but_not_the_live_reread_under_whole_history(
    tmp_path, monkeypatch
):
    """Default scope (MEMO_PROXY_CONTENT_SCOPE=all): the FROZEN block is a
    legitimate first read and IS StructMap's to compress -- being in the
    frozen zone must not grandfather a block out of compression. The live
    re-read of the same path stays Delta's case, not StructMap's, exactly as
    under tail-only scope."""
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    zones = _read_zones("src/pkg/mod.py", BIG_SRC, tool_use_id="r2")
    zones.frozen_messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "Read",
                    "input": {"file_path": "src/pkg/mod.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r1", "content": BIG_SRC}],
        },
    ]
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved > 0
    frozen_text = zones.frozen_messages[1]["content"][0]["content"]
    assert len(frozen_text) < len(BIG_SRC)
    assert "total += j" not in frozen_text
    # the live re-read is untouched by StructMap -- Delta's case
    assert zones.live_messages[1]["content"][0]["content"] == BIG_SRC


def test_apply_never_cuts_without_a_recovery_path(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.proxy.transforms.structmap.ccr.stash", lambda state_dir, content: "")
    zones = _read_zones("src/pkg/mod.py", SRC)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == SRC


def test_apply_disabled_flag_skips_everything(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_STRUCTMAP", "0")
    assert StructMap().enabled() is False


def test_apply_never_raises_on_malformed_live_messages(tmp_path):
    zones = Zones(
        live_messages=[None, {"role": "user", "content": "not a list"}, {"content": [None, 42]}]
    )
    saved = StructMap().apply(zones, _ctx(tmp_path))
    assert saved == 0


# --- StructMap transform: shape-based detection, no `Read` required ----------


def _bash_cat_n_zones(file_path: str, content: str, tool_use_id: str = "b1") -> Zones:
    return Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Bash",
                        "input": {"command": f"cat -n {file_path}"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": _numbered(content),
                    }
                ],
            },
        ]
    )


def _pathless_zones(content: str, tool_use_id: str = "g1") -> Zones:
    """A tool_result with no extractable identity at all -- e.g. `Grep`, or
    a Bash command outside `_extract_bash_read_path`'s narrow grammar."""
    return Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Grep",
                        "input": {"pattern": "def ", "output_mode": "content"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
                ],
            },
        ]
    )


def test_apply_compresses_a_bash_cat_n_read_of_a_python_file_tail_scope(tmp_path, monkeypatch):
    """Ground truth: a real captured payload showed the model reading a
    source file via `cat -n <path>`, never `Read`. Path extraction
    (`delta._extract_bash_read_path`) plus de-numbered parsing
    (`_strip_line_numbers`) together must compress it exactly like a `Read`
    of the same file would have."""
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    zones = _bash_cat_n_zones("src/pkg/mod.py", BIG_SRC)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert len(new_text) < len(_numbered(BIG_SRC))
    assert "total += j" not in new_text


def test_apply_compresses_a_bash_cat_n_read_of_a_python_file_whole_history_scope(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    zones = _bash_cat_n_zones("src/pkg/mod.py", BIG_SRC)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert len(new_text) < len(_numbered(BIG_SRC))
    assert "total += j" not in new_text


def test_apply_sniffs_a_pathless_large_python_blob_tail_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    big = "import os\n\n\n" + BIG_SRC
    zones = _pathless_zones(big)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert len(new_text) < len(big)
    assert "total += j" not in new_text


def test_apply_sniffs_a_pathless_large_python_blob_whole_history_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    big = "import os\n\n\n" + BIG_SRC
    zones = _pathless_zones(big)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert len(new_text) < len(big)
    assert "total += j" not in new_text


def test_apply_leaves_a_pathless_small_python_snippet_untouched(tmp_path, monkeypatch):
    """Below the sniff size floor -- real source, but not worth the
    false-positive risk without a known path."""
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    tiny = "def f():\n    return 1\n"
    zones = _pathless_zones(tiny)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved == 0
    assert new_text == tiny


def test_apply_leaves_a_pathless_large_json_blob_untouched(tmp_path, monkeypatch):
    """A JSON object also parses as a Python dict-literal expression, but
    has no import/def/class -- must never be mistaken for source."""
    import json

    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    blob = json.dumps({f"key_{i}": "value " * 10 for i in range(200)})
    zones = _pathless_zones(blob)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved == 0
    assert new_text == blob


def test_apply_never_stores_a_net_larger_block_than_the_original(tmp_path):
    """A tiny python file where the signature map plus the recovery marker's
    own overhead (~130 bytes) would be net-larger than the original must be
    left completely untouched, never written back as a "cut" that actually
    grew the block."""
    tiny = "def f():\n    return 1\n"
    zones = _read_zones("src/pkg/tiny.py", tiny)
    saved = StructMap().apply(zones, _ctx(tmp_path))
    stored = zones.live_messages[1]["content"][0]["content"]
    assert len(stored) <= len(tiny)
    assert stored == tiny
    assert saved == 0


# ── TypeScript / JavaScript signature maps ───────────────────────────────────
#
# The tree-sitter walker shipped without tests. These cover the shapes it
# actually meets in a `Read` of a real .ts/.tsx file.

_TS_SOURCE = """\
import { useState } from "react";
import type { Foo } from "./foo";

export interface Widget {
  id: string;
  label: string;
}

type Handler = (e: Event) => void;

export class Panel extends Base implements Widget {
  private count = 0;

  constructor(private readonly name: string) {
    super();
    this.count = 1;
  }

  render(depth: number): string {
    const parts = [];
    for (let i = 0; i < depth; i++) {
      parts.push(this.name);
    }
    return parts.join("/");
  }
}

export function build(a: number, b: string): Widget {
  const id = `${a}-${b}`;
  return { id, label: b };
}

const arrow = (x: number) => x * 2;
"""


def test_typescript_keeps_declarations_and_drops_bodies():
    from memo.proxy.transforms.structmap import signatures

    out = signatures(_TS_SOURCE, "typescript")

    # Declarations survive -- including METHOD signatures. A class reduced to
    # just `class Panel {` loses the whole API surface, which is the one thing
    # a signature map exists to show; the Python side emits `def render(...)`
    # and TypeScript must match it.
    for kept in ("import", "class Panel", "function build", "render", "constructor"):
        assert kept in out, f"{kept!r} missing from:\n{out}"
    # Bodies do not.
    for dropped in ("parts.push", 'parts.join("/")', "this.count = 1"):
        assert dropped not in out, f"{dropped!r} survived in:\n{out}"
    assert len(out) < len(_TS_SOURCE)


def test_javascript_uses_the_js_grammar_not_the_ts_one():
    from memo.proxy.transforms.structmap import signatures

    src = "export function go(a) {\n  const x = a + 1;\n  return x;\n}\n"
    out = signatures(src, "javascript")
    assert "function go" in out
    assert "const x = a + 1" not in out


def test_an_unsupported_language_is_returned_untouched():
    from memo.proxy.transforms.structmap import signatures

    src = "SELECT 1;\n"
    assert signatures(src, "sql") == src


def test_syntactically_broken_typescript_still_yields_its_declaration():
    """tree-sitter is error-tolerant: a half-written file still parses to a
    partial tree, so the declaration line survives rather than the map coming
    back empty. What must NOT happen is a crash or a mangled result."""
    from memo.proxy.transforms.structmap import signatures

    out = signatures("export class {{{ broken\n", "typescript")
    assert "export class" in out


def test_language_is_chosen_by_extension():
    from memo.proxy.transforms.structmap import _language_for

    assert _language_for("/a/b/c.ts") == "typescript"
    assert _language_for("/a/b/c.tsx") == "typescript"
    assert _language_for("/a/b/c.js") == "javascript"
    assert _language_for("/a/b/c.py") == "python"
    assert _language_for("/a/b/c.md") == ""
    assert _language_for("") == ""
