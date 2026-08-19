from memo.proxy.plan import Context
from memo.proxy.transforms.structmap import StructMap, _language_for, signatures
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
    assert "memo_retrieve" in new_text

    from memo.proxy import ccr

    key = new_text.split('key="')[1].split('"')[0]
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


def test_apply_leaves_a_re_read_of_a_seen_file_untouched(tmp_path):
    """A second read of a path already in frozen_messages is Delta's case,
    not StructMap's -- StructMap must not also compress it."""
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
