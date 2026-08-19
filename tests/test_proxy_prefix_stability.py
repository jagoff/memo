"""Empirical proof that MEMO_PROXY_CONTENT_SCOPE=all keeps the emitted prefix
byte-identical across a growing multi-turn session -- the property the whole
scope widening depends on (see zones.py's module docstring: a cache read
costs 0.1x a fresh input token, so rewriting an already-cached block on a
later turn can cost more than it ever saved).

A test that checks only one turn proves nothing here: five different turns,
each appending real tool-use traffic (a file read, shell output, a re-read of
the same file with a small change, a large JSON array, a dense block), drive
the SAME session through the real `rewrite_body` + `build_registry()` path
turn after turn. After every turn, the newest output's prefix (everything up
to the length of the PREVIOUS turn's output) must equal that previous
output byte-for-byte -- proving a block already sent to the provider is never
re-emitted differently once it has entered the cached prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

from memo.proxy.plan import Context
from memo.proxy.server import rewrite_body

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_SOURCE = (_REPO_ROOT / "src/memo/proxy/meter.py").read_text(encoding="utf-8")
_CHANGED_SOURCE = _REAL_SOURCE.replace("def append(", "def append_CHANGED(", 1)

# A second, distinct real file, read the way the ground-truth captured
# payload actually showed the model reading source: `Bash cat -n`, never
# `Read` (see structmap.py's module docstring). Rendered numbered exactly
# like `_strip_line_numbers` expects to de-number.
_SNIFF_SOURCE = (_REPO_ROOT / "src/memo/proxy/tool_schema_cache.py").read_text(encoding="utf-8")


def _numbered(text: str) -> str:
    return "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(text.splitlines(), start=1))


_SYSTEM = [{"type": "text", "text": "You are a careful coding assistant."}]
_TOOLS = [
    {
        "name": "memo_search",
        "description": "search memory " * 10,
        "input_schema": {"type": "object"},
    },
    {
        "name": "memo_graph",
        "description": "graph traversal " * 10,
        "input_schema": {"type": "object"},
    },
    {"name": "Read", "description": "read a file " * 10, "input_schema": {"type": "object"}},
    {
        "name": "Bash",
        "description": "run a shell command " * 10,
        "input_schema": {"type": "object"},
    },
    {
        "name": "mcp__other__tool_a",
        "description": "an unrelated tool " * 10,
        "input_schema": {"type": "object"},
    },
    {
        "name": "mcp__other__tool_b",
        "description": "another unrelated tool " * 10,
        "input_schema": {"type": "object"},
    },
]


def _turn_1() -> list[dict]:
    """A first read of a real repo file -- StructMap's case."""
    return [
        {"role": "user", "content": "Read src/memo/proxy/meter.py and summarize it."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "Read",
                    "input": {"file_path": "src/memo/proxy/meter.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r1", "content": _REAL_SOURCE}],
        },
    ]


def _turn_2() -> list[dict]:
    """Verbose shell output -- ToolResults' generic-fallback case."""
    output = "\n".join(f"tests/test_x.py::test_{i} PASSED" for i in range(300))
    return [
        {"role": "assistant", "content": "It defines Record and append/summarize."},
        {"role": "user", "content": "Now run the test suite."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "pytest -x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "b1", "content": output}],
        },
    ]


def _turn_3() -> list[dict]:
    """A re-read of the SAME file with one changed line -- Delta's case."""
    return [
        {"role": "assistant", "content": "All tests passed."},
        {"role": "user", "content": "I renamed a function, read the file again."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r2",
                    "name": "Read",
                    "input": {"file_path": "src/memo/proxy/meter.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r2", "content": _CHANGED_SOURCE}],
        },
    ]


def _turn_4() -> list[dict]:
    """A large JSON array -- JsonCrush's case."""
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(300)])
    return [
        {"role": "assistant", "content": "Renamed to append_CHANGED."},
        {"role": "user", "content": "Fetch the data file."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "j1",
                    "name": "Bash",
                    "input": {"command": "cat data.json"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "j1", "content": big}]},
    ]


def _turn_5() -> list[dict]:
    """A dense single-line block -- Pixel's case (a deterministic no-op
    without Pillow installed; the profitability gate and stash still run
    either way, so this stays a meaningful exercise of the transform)."""
    return [
        {"role": "assistant", "content": "Row 42 is ..."},
        {"role": "user", "content": "Show me a dense repeated pattern."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "p1",
                    "name": "Bash",
                    "input": {"command": "yes x | head -c 10000"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "p1", "content": "x" * 10_000}],
        },
    ]


def _turn_6() -> list[dict]:
    """A real file read via `Bash cat -n`, never `Read` -- the ground-truth
    shape (structmap.py's module docstring) that path-extraction + shape
    sniffing exists for. No `Read` tool_use anywhere in this turn, so any
    compression here can only come from `_extract_bash_read_path` +
    `signatures()`'s de-numbering fallback."""
    return [
        {"role": "assistant", "content": "Fetched the data file."},
        {"role": "user", "content": "Now show me the tool schema cache module."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "Bash",
                    "input": {"command": "cat -n src/memo/proxy/tool_schema_cache.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": _numbered(_SNIFF_SOURCE)}
            ],
        },
    ]


_TURN_CHUNKS = [_turn_1(), _turn_2(), _turn_3(), _turn_4(), _turn_5(), _turn_6()]


def _payload(turn: int) -> bytes:
    """The FULL raw request body a client would send at `turn` -- every
    earlier chunk plus this one, exactly like a real client resending its
    whole conversation every request. Built fresh from the original fixture
    chunks each call (never from a previous turn's OUTPUT), matching how the
    real client never sees or resends the proxy's rewritten bytes."""
    messages: list[dict] = []
    for chunk in _TURN_CHUNKS[:turn]:
        messages.extend(chunk)
    return json.dumps({"system": _SYSTEM, "tools": _TOOLS, "messages": messages}).encode()


def test_the_emitted_prefix_is_byte_identical_for_every_earlier_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)  # default: whole-history

    ctx = Context(state_dir=tmp_path, session_key="stability-session", project=None)

    outputs: list[dict] = []
    for turn in range(1, len(_TURN_CHUNKS) + 1):
        out_bytes, _plan = rewrite_body(_payload(turn), ctx)
        outputs.append(json.loads(out_bytes))

    for t in range(1, len(outputs)):
        previous, current = outputs[t - 1], outputs[t]
        assert current["system"] == previous["system"], (
            f"system drifted between turn {t} and {t + 1}"
        )
        assert current["tools"] == previous["tools"], f"tools drifted between turn {t} and {t + 1}"
        n = len(previous["messages"])
        assert current["messages"][:n] == previous["messages"], (
            f"turn {t + 1}'s prefix diverged from turn {t}'s cached output over "
            f"message range [0:{n}) -- a rewrite of already-cached content would "
            "force a provider re-cache instead of a cache hit"
        )

    # Setup check: prove the transforms actually did something non-trivial
    # over the whole history, so the loop above isn't vacuously true because
    # nothing was ever rewritten. The frozen first read (turn 1's block) must
    # still be compressed by the LAST turn's output -- confirming the frozen
    # zone is genuinely in scope, not just the live tail.
    final_first_read = outputs[-1]["messages"][2]["content"][0]["content"]
    assert len(final_first_read) < len(_REAL_SOURCE)
    assert "memo_crush_retrieve" in final_first_read

    # Same check for turn 6's Bash `cat -n` read -- no `Read` tool_use
    # anywhere in that turn, so this only compresses via
    # `_extract_bash_read_path` + `signatures()`'s de-numbering fallback.
    # It becomes the live tail on the very turn it's introduced (turn 6 of
    # 6), so this doubles as proof it compresses on first sight too, not
    # only once frozen.
    bash_read = outputs[-1]["messages"][-1]["content"][0]["content"]
    assert len(bash_read) < len(_numbered(_SNIFF_SOURCE))
    assert "memo_crush_retrieve" in bash_read


def test_tail_only_scope_reverts_a_block_to_raw_once_it_leaves_the_live_window(
    tmp_path, monkeypatch
):
    """Complementary control, and the concrete mechanism behind the problem
    this whole widening exists to fix: under MEMO_PROXY_CONTENT_SCOPE=tail,
    turn 1's first read is IN `live_messages` (a 3-message turn 1, live_turns
    default 2), so StructMap DOES compress it on the wire that one turn. By
    turn 2 that same message has aged into `frozen_messages`, which
    tail-only scope never touches -- so it reverts to the RAW original on
    every later turn. That is a byte change at the live-to-frozen boundary,
    not stability; MEMO_PROXY_CONTENT_SCOPE=all (the test above) is what
    actually holds a block's bytes constant once it is part of the
    conversation the provider has already seen."""
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")

    ctx = Context(state_dir=tmp_path, session_key="tail-stability-session", project=None)

    outputs: list[dict] = []
    for turn in range(1, len(_TURN_CHUNKS) + 1):
        out_bytes, _plan = rewrite_body(_payload(turn), ctx)
        outputs.append(json.loads(out_bytes))

    turn_1_text = outputs[0]["messages"][2]["content"][0]["content"]
    assert len(turn_1_text) < len(_REAL_SOURCE), "setup check: turn 1 must have compressed it"

    # From turn 2 onward this message is permanently frozen and tail-only
    # scope never rewrites the frozen zone, so it is stable -- but at RAW
    # size, a DIFFERENT byte pattern than turn 1 emitted for the same
    # position (the instability tail-only scope has always carried at this
    # one boundary).
    for out in outputs[1:]:
        assert out["messages"][2]["content"][0]["content"] == _REAL_SOURCE
