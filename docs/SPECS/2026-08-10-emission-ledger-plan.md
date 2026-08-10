# Emission Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop memo from re-emitting memory bodies that are already in the current context window, replacing repeats with `{id, title, ref}`.

**Architecture:** An append-only JSONL ledger per Claude Code session records what memo actually emitted (id, hash of the emitted text, its length). Read tools consult it before serializing and split their hits into full bodies and digest entries. The recall hook writes to the same file, so hook and MCP emissions dedupe against each other. PreCompact resets it.

**Tech Stack:** Python 3.13, stdlib only for the ledger module, fastmcp for the MCP surface, pytest.

**Spec:** `docs/SPECS/2026-08-10-emission-ledger-design.md`

## Global Constraints

- Branch: `feat/emission-ledger` (already created off `origin/master`).
- Tests run as `uv run --no-sync pytest tests/ --color=yes`. Never the repo `.venv` for the CLI; the installed tool is a separate uv tool.
- `MEMO_EMIT_LEDGER` defaults to `0`. No behaviour change ships enabled in this plan.
- `emitted_ledger.py` is a leaf module: stdlib only, no `memo.memory`, no MLX, no flag reads beyond its own three. Same contract as `recall_dedup.py`.
- Every ledger call from the recall hook is fail-open — wrapped, swallowed, never propagates. The hook has a 5s budget and recall must never break because of this feature.
- All new flags registered in `src/memo/flags_misc.py` via `_spec(...)` from `memo.flags_base`. Never read `os.environ` directly for a `MEMO_*` flag.
- Type hints required; the repo runs mypy and ruff.
- Commit after every task. Conventional commits (`feat:`, `test:`, `docs:`).

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/memo/emitted_ledger.py` | create | Ledger I/O + the partition decision. Leaf, stdlib only. |
| `tests/test_emitted_ledger.py` | create | Units for the ledger module. |
| `tests/test_emitted_ledger_partition.py` | create | Units for `partition`, including the monotonic rule. |
| `src/memo/flags_misc.py` | modify | Register the three `MEMO_EMIT_LEDGER*` flags. |
| `src/memo/server_common.py` | modify | `apply_ledger()` — the wrapper MCP tools call. |
| `src/memo/server_core_search.py` | modify | Wire `memo_search`. |
| `src/memo/server_core_search.py`, `server_ask*.py`, `server_context_pack.py` | modify | Wire the remaining four tools. |
| `src/memo/recall_logic.py` | modify | `emitted_sink` kwarg threaded through the three renderers. |
| `src/memo/cli_recall_hook.py` | modify | Append what the renderer reported. |
| `src/memo/cli_capture.py` | modify | Reset the ledger on `capture-tick --force` (the PreCompact boundary). |
| `src/memo/server_cache.py` | modify | Expose ledger counters through `memo_cache_stats`. |
| `~/.local/share/memo/bin/memo-nightly.sh` + `src/memo/cli_ops.py` | modify | Prune ledger files older than 48h. |
| `tests/test_emitted_ledger_integration.py` | create | Tools + hook + reset, end to end. |
| `docs/eval/emission-ledger-replay.md` | create | The replay harness and how the two gates are measured. |

Ordering rationale: tasks 1–5 deliver MCP-to-MCP coverage with no changes to the recall hot path, and could ship alone. Task 6 touches the renderer, which is the riskiest edit; it lands after the core is proven.

---

### Task 1: Ledger module and flags

**Files:**
- Create: `src/memo/emitted_ledger.py`
- Modify: `src/memo/flags_misc.py`
- Test: `tests/test_emitted_ledger.py`

**Interfaces:**
- Consumes: `memo.flags_base._spec`, `memo.flags.flag_int`
- Produces:
  - `Entry` — frozen dataclass `(id: str, h: str, n: int, ref: str, t: int, src: str)`
  - `emitted_hash(text: str) -> str` — 8 hex chars
  - `mint_ref(ids: Sequence[str], t: int, *, prefix: str = "memo-r") -> str`
  - `ledger_path(state_dir: Path, session_id: str) -> Path`
  - `append(state_dir, session_id, entries: Sequence[Entry]) -> None`
  - `read(state_dir, session_id) -> dict[str, Entry]` — last entry per id wins
  - `reset(state_dir, session_id) -> bool` — True if a file was removed
  - `prune(state_dir, *, max_age_s: int) -> int` — files removed

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emitted_ledger.py`:

```python
from pathlib import Path

import pytest

from memo import emitted_ledger as el


def _entry(mid: str, text: str, ref: str = "memo-r/aaaaaa", t: int = 1000) -> el.Entry:
    return el.Entry(id=mid, h=el.emitted_hash(text), n=len(text), ref=ref, t=t, src="mcp")


def test_append_then_read_roundtrip(tmp_path: Path):
    el.append(tmp_path, "sess1", [_entry("mem_a", "hello"), _entry("mem_b", "world")])
    got = el.read(tmp_path, "sess1")
    assert set(got) == {"mem_a", "mem_b"}
    assert got["mem_a"].n == 5
    assert got["mem_a"].h == el.emitted_hash("hello")


def test_read_missing_file_is_empty(tmp_path: Path):
    assert el.read(tmp_path, "nope") == {}


def test_last_entry_per_id_wins(tmp_path: Path):
    el.append(tmp_path, "s", [_entry("mem_a", "short", t=1)])
    el.append(tmp_path, "s", [_entry("mem_a", "much longer body", t=2)])
    got = el.read(tmp_path, "s")
    assert got["mem_a"].n == len("much longer body")
    assert got["mem_a"].t == 2


def test_torn_final_line_is_skipped(tmp_path: Path):
    el.append(tmp_path, "s", [_entry("mem_a", "hello")])
    path = el.ledger_path(tmp_path, "s")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id":"mem_b","h":"dead')  # no newline, truncated JSON
    got = el.read(tmp_path, "s")
    assert set(got) == {"mem_a"}


def test_cap_is_fifo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMO_EMIT_LEDGER_MAX", "3")
    for i in range(5):
        el.append(tmp_path, "s", [_entry(f"mem_{i}", "x", t=i)])
    got = el.read(tmp_path, "s")
    assert set(got) == {"mem_2", "mem_3", "mem_4"}


def test_reset_removes_only_that_session(tmp_path: Path):
    el.append(tmp_path, "s1", [_entry("mem_a", "a")])
    el.append(tmp_path, "s2", [_entry("mem_b", "b")])
    assert el.reset(tmp_path, "s1") is True
    assert el.read(tmp_path, "s1") == {}
    assert set(el.read(tmp_path, "s2")) == {"mem_b"}


def test_reset_is_idempotent(tmp_path: Path):
    assert el.reset(tmp_path, "never-existed") is False
    el.append(tmp_path, "s", [_entry("mem_a", "a")])
    assert el.reset(tmp_path, "s") is True
    assert el.reset(tmp_path, "s") is False


def test_append_is_fail_open_on_unwritable_dir(tmp_path: Path):
    target = tmp_path / "ro"
    target.mkdir()
    target.chmod(0o500)
    try:
        el.append(target, "s", [_entry("mem_a", "a")])  # must not raise
        assert el.read(target, "s") == {}
    finally:
        target.chmod(0o700)


def test_session_id_is_sanitised(tmp_path: Path):
    el.append(tmp_path, "../escape/../../etc", [_entry("mem_a", "a")])
    written = list((tmp_path / "emitted").glob("*.jsonl"))
    assert len(written) == 1
    assert ".." not in written[0].name and "/" not in written[0].name


def test_prune_removes_only_old_files(tmp_path: Path):
    import os
    import time

    el.append(tmp_path, "old", [_entry("mem_a", "a")])
    el.append(tmp_path, "new", [_entry("mem_b", "b")])
    old_path = el.ledger_path(tmp_path, "old")
    stale = time.time() - 60 * 60 * 72
    os.utime(old_path, (stale, stale))
    assert el.prune(tmp_path, max_age_s=60 * 60 * 48) == 1
    assert not old_path.exists()
    assert el.ledger_path(tmp_path, "new").exists()


def test_mint_ref_is_stable_and_order_insensitive():
    a = el.mint_ref(["mem_b", "mem_a"], 1000)
    b = el.mint_ref(["mem_a", "mem_b"], 1000)
    assert a == b
    assert a.startswith("memo-r/") and len(a) == len("memo-r/") + 6
    assert el.mint_ref(["mem_a"], 1000, prefix="memo-h").startswith("memo-h/")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_emitted_ledger.py -v --color=yes`
Expected: collection error, `ModuleNotFoundError: No module named 'memo.emitted_ledger'`

- [ ] **Step 3: Register the flags**

In `src/memo/flags_misc.py`, add to the `SPECS` tuple (keep the file's existing ordering style — append near the other MCP/budget entries):

```python
    _spec(
        "MEMO_EMIT_LEDGER",
        "bool",
        False,
        "mcp",
        "Suppress re-emission of memory bodies already sent into this session's "
        "context window. A repeat hit returns {id, title, ref} instead of the full "
        "body. Default off until the replay gates in "
        "docs/SPECS/2026-08-10-emission-ledger-design.md are met.",
    ),
    _spec(
        "MEMO_EMIT_LEDGER_TOOLS",
        "str",
        "memo_search,memo_ask,memo_context,memo_unified_briefing,memo_evidence_pack",
        "mcp",
        "Comma-separated MCP tools that consult the emission ledger. memo_get and "
        "memo_history are deliberately absent: they mean 'give me this one, "
        "explicitly', and are the escape hatch a digest points at.",
    ),
    _spec(
        "MEMO_EMIT_LEDGER_MAX",
        "int",
        500,
        "mcp",
        "Emission-ledger entry cap per session, FIFO. Bounds both the file and the "
        "read cost on the MCP hot path.",
        min_val=0,
    ),
```

- [ ] **Step 4: Write the module**

Create `src/memo/emitted_ledger.py`:

```python
"""What memo has already put into this session's context window.

memo produces the same memory bodies from two places — the recall hook on every
``UserPromptSubmit`` and the MCP read tools — pulling from one corpus with one
embedder. Nothing tracked emissions across those surfaces, so a body could enter
the window three or four times in a session and stay there.

This module is the ledger. It records what was *emitted*, not what is stored:
the hash is over the text that actually went out, plus its length. That
distinction is the whole correctness argument — see ``partition``.

Leaf module by contract, like ``recall_dedup``: stdlib only, no store access, no
MLX, no flag reads beyond its own. The recall hook imports it inside a 5s budget
and every call from there is fail-open.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from memo.flags import flag_int

_DIRNAME = "emitted"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class Entry:
    """One emission: memory ``id``, hash ``h`` and length ``n`` of the text that
    went out, the batch ``ref`` it went out under, unix seconds ``t``, and
    ``src`` (``hook`` | ``mcp``)."""

    id: str
    h: str
    n: int
    ref: str
    t: int
    src: str


def emitted_hash(text: str) -> str:
    """8 hex chars over the emitted text. Collisions only ever matter between
    versions of the *same* id, so the space is per-memory and tiny."""
    return hashlib.blake2s(text.encode("utf-8"), digest_size=4).hexdigest()


def mint_ref(ids: Sequence[str], t: int, *, prefix: str = "memo-r") -> str:
    """A short token naming one emission batch, echoed in the payload so a
    digest can point at a specific earlier message without turn numbers.
    Order-insensitive: the same set of ids at the same second is the same ref."""
    seed = ",".join(sorted(ids)) + f"@{t}"
    return f"{prefix}/{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:6]}"


def _safe(session_id: str) -> str:
    """Session ids reach us from an env var and a hook payload — both external
    input, and both used to build a path. Sanitise rather than trust."""
    cleaned = _UNSAFE.sub("_", session_id or "unknown")
    return cleaned[:120] or "unknown"


def ledger_path(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / _DIRNAME / f"{_safe(session_id)}.jsonl"


def _cap() -> int:
    value = flag_int("MEMO_EMIT_LEDGER_MAX")
    return 500 if value is None else max(0, value)


def append(state_dir: Path, session_id: str, entries: Sequence[Entry]) -> None:
    """Append entries. Fail-open: a ledger that cannot be written costs tokens,
    never correctness, so every failure here is silent by design.

    No locking. The recall hook and the MCP server both write, but each line is
    a single short ``O_APPEND`` write, which is atomic; ``read`` tolerates a
    torn tail regardless.
    """
    if not entries:
        return
    try:
        path = ledger_path(state_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        _trim(path)
    except Exception:
        return


def _trim(path: Path) -> None:
    """FIFO the file back under the cap. Rewrites in place; a crash mid-rewrite
    loses the ledger, which costs tokens and nothing else."""
    cap = _cap()
    if cap <= 0:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= cap * 2:  # amortise: only rewrite once we are well over
            return
        path.write_text("\n".join(lines[-cap:]) + "\n", encoding="utf-8")
    except Exception:
        return


def read(state_dir: Path, session_id: str) -> dict[str, Entry]:
    """Latest entry per memory id. Unparseable lines are skipped: a concurrent
    writer can leave a torn tail, and a torn tail must not blind the reader."""
    out: dict[str, Entry] = {}
    try:
        raw = ledger_path(state_dir, session_id).read_text(encoding="utf-8")
    except Exception:
        return out
    cap = _cap()
    lines = raw.splitlines()
    if cap > 0:
        lines = lines[-cap:]
    for line in lines:
        try:
            obj = json.loads(line)
            out[str(obj["id"])] = Entry(
                id=str(obj["id"]),
                h=str(obj["h"]),
                n=int(obj["n"]),
                ref=str(obj["ref"]),
                t=int(obj["t"]),
                src=str(obj.get("src") or ""),
            )
        except Exception:
            continue
    return out


def reset(state_dir: Path, session_id: str) -> bool:
    """Drop this session's ledger. Returns whether a file was actually removed.

    Called at the compaction boundary: once the window is rewritten, memo can no
    longer claim anything is in it. Idempotent — PreCompact double-fires against
    the plugin copy.
    """
    try:
        path = ledger_path(state_dir, session_id)
        if path.is_file():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def prune(state_dir: Path, *, max_age_s: int) -> int:
    """Remove ledgers whose session ended long ago. Sessions leave no close
    signal, so age is the only available liveness proxy."""
    removed = 0
    now = time.time()
    try:
        entries = list((Path(state_dir) / _DIRNAME).glob("*.jsonl"))
    except Exception:
        return 0
    for path in entries:
        try:
            if now - os.stat(path).st_mtime > max_age_s:
                path.unlink()
                removed += 1
        except Exception:
            continue
    return removed
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_emitted_ledger.py -v --color=yes`
Expected: 11 passed

- [ ] **Step 6: Run lint and types**

Run: `uv run --no-sync ruff check src/memo/emitted_ledger.py src/memo/flags_misc.py && uv run --no-sync mypy src/memo/emitted_ledger.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/memo/emitted_ledger.py src/memo/flags_misc.py tests/test_emitted_ledger.py
git commit -m "feat: emission ledger storage layer

Append-only JSONL per session recording what memo actually emitted into the
context window: memory id, hash of the emitted text, and its length. Leaf
module, stdlib only, fail-open on every write — the recall hook imports it
inside a 5s budget.

Ships behind MEMO_EMIT_LEDGER=0; nothing reads this yet."
```

---

### Task 2: The partition decision

**Files:**
- Modify: `src/memo/emitted_ledger.py`
- Test: `tests/test_emitted_ledger_partition.py`

**Interfaces:**
- Consumes: `Entry`, `emitted_hash` from Task 1
- Produces: `partition(hits, known, *, text_of, id_of) -> Partition` where
  `Partition` is a frozen dataclass with `full: list[Any]`, `digest: list[Any]`,
  `suppressed_chars: int`. `text_of` and `id_of` are callables so the same
  function serves MCP dicts and recall-hook hit objects.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emitted_ledger_partition.py`:

```python
from memo import emitted_ledger as el


def _known(mid: str, text: str) -> el.Entry:
    return el.Entry(
        id=mid, h=el.emitted_hash(text), n=len(text), ref="memo-r/aaaaaa", t=1, src="mcp"
    )


def _hit(mid: str, body: str) -> dict[str, str]:
    return {"id": mid, "title": f"title of {mid}", "body": body}


def _partition(hits, known):
    return el.partition(
        hits, known, text_of=lambda h: h["body"], id_of=lambda h: h["id"]
    )


def test_empty_ledger_emits_everything_full():
    hits = [_hit("a", "body a"), _hit("b", "body b")]
    out = _partition(hits, {})
    assert out.full == hits
    assert out.digest == []
    assert out.suppressed_chars == 0


def test_identical_reemission_is_digested():
    hits = [_hit("a", "body a")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert out.full == []
    assert [h["id"] for h in out.digest] == ["a"]
    assert out.suppressed_chars == len("body a")


def test_changed_body_is_reemitted_full():
    hits = [_hit("a", "body a, edited")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert out.full == hits
    assert out.digest == []


def test_partial_overlap_splits():
    hits = [_hit("a", "body a"), _hit("b", "body b")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert [h["id"] for h in out.full] == ["b"]
    assert [h["id"] for h in out.digest] == ["a"]


def test_longer_emission_wins_over_recorded_shorter_one():
    # The hook emitted 400 chars at turn 2; memo_ask now has room for 900.
    # Digesting here would suppress 500 chars the model has never seen.
    short = "x" * 400
    longer = "x" * 900
    out = _partition([_hit("a", longer)], {"a": _known("a", short)})
    assert out.full and out.full[0]["body"] == longer
    assert out.digest == []


def test_shorter_emission_of_same_prefix_is_digested():
    longer = "x" * 900
    shorter = "x" * 400
    out = _partition([_hit("a", shorter)], {"a": _known("a", longer)})
    assert out.full == []
    assert [h["id"] for h in out.digest] == ["a"]


def test_title_only_prior_emission_does_not_suppress_a_body():
    # render_recall_compact emits no body; n == 0 must never digest real text.
    out = _partition([_hit("a", "a real body")], {"a": _known("a", "")})
    assert out.full and out.digest == []


def test_hit_order_is_preserved_within_each_bucket():
    hits = [_hit(x, f"body {x}") for x in ("a", "b", "c", "d")]
    known = {"b": _known("b", "body b"), "d": _known("d", "body d")}
    out = _partition(hits, known)
    assert [h["id"] for h in out.full] == ["a", "c"]
    assert [h["id"] for h in out.digest] == ["b", "d"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_partition.py -v --color=yes`
Expected: FAIL, `AttributeError: module 'memo.emitted_ledger' has no attribute 'partition'`

- [ ] **Step 3: Implement `partition`**

Append to `src/memo/emitted_ledger.py`:

```python
@dataclass(frozen=True)
class Partition:
    """``full`` keeps its hits verbatim; ``digest`` are hits the caller should
    render as {id, title, ref}. ``suppressed_chars`` is what was not sent, the
    numerator of the saving the gate measures."""

    full: list[Any]
    digest: list[Any]
    suppressed_chars: int


def partition(
    hits: Sequence[Any],
    known: dict[str, Entry],
    *,
    text_of: Callable[[Any], str],
    id_of: Callable[[Any], str],
) -> Partition:
    """Split hits into what still needs sending and what is already in the window.

    A hit is digested only when the ledger proves the model has already seen at
    least this much of it — the *monotonic-emission rule*:

        same hash          -> the identical text is already up there
        new_len <= known.n -> a shorter rendering of text already up there

    Anything else, including a *longer* rendering of the same memory, is sent in
    full and replaces the ledger entry. The asymmetry is the point: emitting
    less than the model has seen is free, but digesting past content it has
    never seen is silent data loss. The recall hook truncates to
    MEMO_RECALL_BODY_CHARS (400) while memo_ask may emit far more, so this case
    is routine, not hypothetical.
    """
    full: list[Any] = []
    digest: list[Any] = []
    suppressed = 0
    for hit in hits:
        text = text_of(hit) or ""
        prior = known.get(id_of(hit))
        if prior is not None and (emitted_hash(text) == prior.h or len(text) <= prior.n):
            digest.append(hit)
            suppressed += len(text)
        else:
            full.append(hit)
    return Partition(full=full, digest=digest, suppressed_chars=suppressed)
```

Add to the imports at the top of the module:

```python
from collections.abc import Callable, Sequence
from typing import Any
```

(replacing the existing `from collections.abc import Sequence`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_partition.py -v --color=yes`
Expected: 8 passed

- [ ] **Step 5: Run the full ledger suite and types**

Run: `uv run --no-sync pytest tests/test_emitted_ledger.py tests/test_emitted_ledger_partition.py --color=yes && uv run --no-sync mypy src/memo/emitted_ledger.py`
Expected: 19 passed, mypy clean

- [ ] **Step 6: Commit**

```bash
git add src/memo/emitted_ledger.py tests/test_emitted_ledger_partition.py
git commit -m "feat: emission ledger partition with monotonic-emission rule

Digest a hit only when the ledger proves the model already saw at least that
much of it: same hash, or the new rendering is no longer than the recorded
one. A longer rendering of the same memory is always sent in full.

The asymmetry is load-bearing. The recall hook truncates to 400 chars while
memo_ask may emit far more, so a naive same-id check would suppress text the
model never received."
```

---

### Task 3: `apply_ledger` wrapper for MCP tools

**Files:**
- Modify: `src/memo/server_common.py`
- Test: `tests/test_emitted_ledger_apply.py` (create)

**Interfaces:**
- Consumes: `emitted_ledger.partition`, `read`, `append`, `mint_ref`, `Entry`
- Produces: `apply_ledger(memory, tool: str, hits: list[dict]) -> tuple[list[dict], dict[str, Any]]`.
  Returns the hits to serialize and the extra payload keys
  (`already_in_context`, `hint`, `cache_ref`) — empty dict when the flag is off,
  the tool is not in the allowlist, or nothing was suppressed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_emitted_ledger_apply.py`:

```python
import pytest

from memo import emitted_ledger as el
from memo import server_common as sc


class _Cfg:
    def __init__(self, state_dir):
        self.state_dir = state_dir


class _Mem:
    def __init__(self, state_dir):
        self.cfg = _Cfg(state_dir)


def _hits():
    return [
        {"id": "mem_a", "title": "A", "body": "body a"},
        {"id": "mem_b", "title": "B", "body": "body b"},
    ]


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-apply")
    return _Mem(tmp_path)


def test_flag_off_is_a_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off")
    hits = _hits()
    out, extra = sc.apply_ledger(_Mem(tmp_path), "memo_search", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-off") == {}


def test_tool_not_in_allowlist_is_a_passthrough(mem, tmp_path):
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_get", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-apply") == {}


def test_first_call_emits_all_and_records(mem, tmp_path):
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert [h["id"] for h in out] == ["mem_a", "mem_b"]
    assert extra == {}  # nothing suppressed on a cold ledger
    assert set(el.read(tmp_path, "sess-apply")) == {"mem_a", "mem_b"}


def test_second_identical_call_digests(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]
    assert extra["already_in_context"][0]["title"] == "A"
    assert extra["already_in_context"][0]["ref"].startswith("memo-r/")
    assert "memo_get" in extra["hint"]


def test_partial_overlap_across_tools(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    later = [*_hits(), {"id": "mem_c", "title": "C", "body": "body c"}]
    out, extra = sc.apply_ledger(mem, "memo_ask", later)
    assert [h["id"] for h in out] == ["mem_c"]
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]


def test_changed_body_is_reemitted(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    edited = [{"id": "mem_a", "title": "A", "body": "body a, edited"}]
    out, extra = sc.apply_ledger(mem, "memo_search", edited)
    assert [h["id"] for h in out] == ["mem_a"]
    assert extra == {}


def test_unwritable_state_dir_degrades_to_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-ro")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        hits = _hits()
        out, extra = sc.apply_ledger(_Mem(ro), "memo_search", hits)
        assert out == hits and extra == {}
    finally:
        ro.chmod(0o700)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_apply.py -v --color=yes`
Expected: FAIL, `AttributeError: module 'memo.server_common' has no attribute 'apply_ledger'`

- [ ] **Step 3: Implement `apply_ledger`**

Append to `src/memo/server_common.py`:

```python
def apply_ledger(
    memory: Any,
    tool: str,
    hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop bodies this session has already put in the context window.

    Returns ``(hits_to_serialize, extra_payload_keys)``. The extra keys are
    empty whenever nothing was suppressed, so a cold session's payload is
    byte-identical to the pre-feature one.

    Fail-open in both directions: flag off, tool not allowlisted, no session
    id, or any exception -> the caller's hits pass through untouched. A ledger
    that misbehaves must cost tokens, never content.
    """
    from memo.flags import flag_bool, flag_str

    if not flag_bool("MEMO_EMIT_LEDGER"):
        return hits, {}
    allow = {t.strip() for t in (flag_str("MEMO_EMIT_LEDGER_TOOLS") or "").split(",") if t.strip()}
    if tool not in allow:
        return hits, {}

    try:
        import time

        from memo import emitted_ledger as el
        from memo.server_session_patterns import _effective_session_id

        state_dir = memory.cfg.state_dir
        session_id = _effective_session_id()
        known = el.read(state_dir, session_id)
        part = el.partition(
            hits,
            known,
            text_of=lambda h: str(h.get("body") or ""),
            id_of=lambda h: str(h.get("id") or ""),
        )

        now = int(time.time())
        ref = el.mint_ref([str(h.get("id") or "") for h in part.full], now)
        el.append(
            state_dir,
            session_id,
            [
                el.Entry(
                    id=str(h.get("id") or ""),
                    h=el.emitted_hash(str(h.get("body") or "")),
                    n=len(str(h.get("body") or "")),
                    ref=ref,
                    t=now,
                    src="mcp",
                )
                for h in part.full
            ],
        )
        if not part.digest:
            return part.full, {}

        return part.full, {
            "already_in_context": [
                {
                    "id": str(h.get("id") or ""),
                    "title": str(h.get("title") or ""),
                    "ref": known[str(h.get("id") or "")].ref,
                }
                for h in part.digest
            ],
            "hint": (
                "bodies already emitted earlier in this session under the listed "
                "ref; call memo_get(id) for any you cannot see above"
            ),
            "cache_ref": ref,
        }
    except Exception:
        return hits, {}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_apply.py -v --color=yes`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/memo/server_common.py tests/test_emitted_ledger_apply.py
git commit -m "feat: apply_ledger wrapper for MCP read tools

Returns the hits still worth serializing plus the extra payload keys. Empty
extras when nothing was suppressed, so a cold session's response is
byte-identical to the pre-feature one. Fail-open on flag, allowlist, missing
session id, and any exception."
```

---

### Task 4: Wire `memo_search`

**Files:**
- Modify: `src/memo/server_core_search.py:276-303`
- Test: `tests/test_emitted_ledger_integration.py` (create)

**Interfaces:**
- Consumes: `server_common.apply_ledger`
- Produces: `memo_search` responses carrying `already_in_context` / `hint` / `cache_ref` when repeats are suppressed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_emitted_ledger_integration.py`. Use the repo's existing MCP tool-invocation fixture — check `tests/test_mcp_budget.py` for the established way to call a registered tool and mirror it exactly rather than inventing a second pattern.

```python
import pytest

from memo import emitted_ledger as el


@pytest.fixture
def ledger_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-int")
    return tmp_path


def test_repeated_search_digests_the_second_time(memory_with_memories, call_tool, ledger_env):
    first = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"], "fixture must return at least one hit"
    assert "already_in_context" not in first

    second = call_tool("memo_search", query="chat", limit=5)
    assert second["hits"] == []
    assert {e["id"] for e in second["already_in_context"]} == {h["id"] for h in first["hits"]}
    assert second["cache_ref"].startswith("memo-r/")


def test_update_between_searches_reemits_that_memory(
    memory_with_memories, call_tool, ledger_env
):
    first = call_tool("memo_search", query="chat", limit=5)
    target = first["hits"][0]["id"]
    call_tool("memo_update", id=target, body="rewritten body for the ledger test")

    third = call_tool("memo_search", query="chat", limit=5)
    assert target in {h["id"] for h in third["hits"]}
    assert target not in {e["id"] for e in third.get("already_in_context", [])}


def test_flag_off_leaves_the_payload_untouched(
    memory_with_memories, call_tool, monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off-int")
    first = call_tool("memo_search", query="chat", limit=5)
    second = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"] == second["hits"]
    assert "already_in_context" not in second
    assert el.read(tmp_path, "sess-off-int") == {}
```

If `memory_with_memories` and `call_tool` do not already exist in `tests/conftest.py`, add them there following the pattern `tests/test_mcp_budget.py` already uses; do not create a parallel fixture style.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_integration.py -v --color=yes`
Expected: FAIL — `second["hits"]` is non-empty, `KeyError: 'already_in_context'`

- [ ] **Step 3: Wire the tool**

In `src/memo/server_core_search.py`, the `memo_search` body currently ends with the loop that builds `out`, then `log_consult`, then the return. Insert the ledger call after `log_consult` and before the presence bump, and splat the extras into the return:

```python
        log_consult(memory, tool="search", query=query, hits=out, t0_ms=t0, source=source)

        # Suppress bodies already emitted into this session's window. Runs after
        # log_consult on purpose: attribution should record what was retrieved,
        # not what survived the ledger.
        from memo.server_common import apply_ledger

        out, ledger_extra = apply_ledger(memory, "memo_search", out)

        # Cross-agent presence: reflect this recall so MCP-only agents (which
        # never run the Claude recall-hook) read honest counts. Decoration only.
        if out:
            from memo import presence

            presence.bump(memory.cfg.state_dir, recalls=len(out))

        # Read pending idle notification (best-effort, races with writer)
        notification = _read_notification(memory)

        return {
            "hits": out,
            "notification": notification,
            **ledger_extra,
            **({"note": " ".join(notes)} if notes else {}),
            **({"trace": trace} if explain else {}),
            **({"degraded": degraded} if degraded else {}),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_integration.py -v --color=yes`
Expected: 3 passed

- [ ] **Step 5: Verify no regression in the existing search suite**

Run: `uv run --no-sync pytest tests/ -k "search or mcp_budget" --color=yes`
Expected: all pass — the flag defaults off, so every existing test exercises the passthrough

- [ ] **Step 6: Commit**

```bash
git add src/memo/server_core_search.py tests/test_emitted_ledger_integration.py tests/conftest.py
git commit -m "feat: consult the emission ledger from memo_search

First tool wired. Runs after log_consult so attribution records what was
retrieved rather than what survived the ledger."
```

---

### Task 5: Wire the remaining four tools

**Files:**
- Modify: `src/memo/server_core_search.py` (`memo_context`)
- Modify: the modules registering `memo_ask`, `memo_unified_briefing`, `memo_evidence_pack` — locate with `grep -rn "def memo_ask\|def memo_unified_briefing\|def memo_evidence_pack" src/memo/server_*.py`
- Test: `tests/test_emitted_ledger_integration.py` (extend)

**Interfaces:**
- Consumes: `server_common.apply_ledger` (Task 3)
- Produces: nothing new. Same three payload keys on four more tools.

Each of these four returns hits under a different key and shape. `apply_ledger`
takes and returns a `list[dict]` with `id` / `title` / `body`, so each site needs
a small adapter rather than a copy-paste of the `memo_search` call: extract the
list, call `apply_ledger`, write the survivors back into the same slot, splat the
extras at the top level of the response.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_emitted_ledger_integration.py`:

```python
@pytest.mark.parametrize(
    ("tool", "kwargs", "hits_key"),
    [
        ("memo_ask", {"question": "chat"}, "sources"),
        ("memo_context", {"question": "chat"}, "hits"),
    ],
)
def test_cross_tool_suppression_after_search(
    memory_with_memories, call_tool, ledger_env, tool, kwargs, hits_key
):
    """A memo_search at turn N suppresses the same bodies from a different tool
    at turn N+1 — the overlap this feature exists to remove."""
    first = call_tool("memo_search", query="chat", limit=5)
    seen = {h["id"] for h in first["hits"]}
    assert seen

    second = call_tool(tool, **kwargs)
    digested = {e["id"] for e in second.get("already_in_context", [])}
    assert digested & seen, f"{tool} did not digest anything memo_search already emitted"
    remaining = {h["id"] for h in second.get(hits_key, []) or []}
    assert not (remaining & digested), "a hit was both emitted and digested"
```

Confirm the real response key for each tool before running (`hits_key` above is a
starting guess for two of them); read the tool's return statement and use the
actual key. Add the analogous case for `memo_unified_briefing` and
`memo_evidence_pack` once their hit-carrying key is confirmed.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_integration.py -k cross_tool -v --color=yes`
Expected: FAIL — `already_in_context` absent

- [ ] **Step 3: Wire each tool**

For each of the four, apply this shape at the point where the response dict is
built, substituting the real key name:

```python
        from memo.server_common import apply_ledger

        kept, ledger_extra = apply_ledger(memory, "<tool_name>", out["<hits_key>"])
        out["<hits_key>"] = kept
        out.update(ledger_extra)
```

For `memo_context`, `build_context_surface` returns a packed prompt string
alongside the hit list. Suppress only the structured hit list; leave the packed
`readonly` wrapper alone. If the pack embeds bodies inside that string, this tool
is out of scope — remove it from the `MEMO_EMIT_LEDGER_TOOLS` default in
`flags_misc.py` and note why in the flag help, rather than half-wiring it.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_integration.py -v --color=yes`
Expected: all pass

- [ ] **Step 5: Full suite**

Run: `uv run --no-sync pytest tests/ --color=yes -x -q`
Expected: no regressions

- [ ] **Step 6: Commit**

```bash
git add -A src/memo tests/test_emitted_ledger_integration.py
git commit -m "feat: consult the emission ledger from the remaining read tools

memo_ask, memo_context, memo_unified_briefing, memo_evidence_pack. Cross-tool
suppression is the point: a memo_search at turn N now stops memo_ask at turn
N+1 from re-sending the same bodies."
```

---

### Task 6: Recall hook writes to the ledger

**Files:**
- Modify: `src/memo/recall_logic.py:380-513` (`render_recall_context`), `:514-571` (`render_recall_compact`), `:572-640` (`render_recall_balanced`), `:641-687` (`render_by_format`)
- Modify: `src/memo/cli_recall_hook.py:872-884`
- Test: `tests/test_emitted_ledger_hook.py` (create)

**Interfaces:**
- Consumes: `emitted_ledger.append`, `Entry`, `emitted_hash`, `mint_ref`
- Produces: `render_by_format(..., emitted_sink: list[tuple[str, str]] | None = None)`.
  When a list is passed, each renderer appends `(hit.id, body_text_actually_emitted)`
  for every hit it renders — `("id", "")` when it emitted a title with no body.

The sink exists because the emitted body cannot be recomputed outside the
renderer. `_effective_body_chars` scales the cap by score, `MEMO_RECALL_SUMMARIZE_BODY`
switches to sentence truncation, and the `max_chars` path can drop a body
entirely partway through the loop. Any external reconstruction would drift from
what actually shipped, and a ledger that over-records length silently suppresses
content — the exact failure the monotonic rule exists to prevent.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emitted_ledger_hook.py`:

```python
from dataclasses import dataclass

from memo import recall_logic as rl


@dataclass
class _Hit:
    id: str
    title: str
    body: str
    score: float
    tags: tuple[str, ...] = ()


def _hits():
    return [
        _Hit("aaaaaaaa", "First", "x" * 900, 0.80),
        _Hit("bbbbbbbb", "Second", "short body", 0.70),
    ]


def test_sink_records_what_the_full_renderer_emitted():
    sink: list[tuple[str, str]] = []
    out = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    recorded = dict(sink)
    # truncated to the cap, not the stored 900 chars
    assert len(recorded["aaaaaaaa"]) <= 420
    assert recorded["aaaaaaaa"] in out or recorded["aaaaaaaa"].rstrip("…") in out
    assert recorded["bbbbbbbb"] == "short body"


def test_sink_records_empty_body_for_the_compact_renderer():
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "compact", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    assert all(body == "" for _, body in sink)


def test_sink_omits_hits_whose_body_was_dropped_by_the_char_budget():
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=20, emitted_sink=sink
    )
    for _id, body in sink:
        assert body == "" or len(body) < 400


def test_sink_is_optional_and_default_none_changes_nothing():
    with_sink = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=[]
    )
    without = rl.render_by_format("full", _hits(), [], turn=1, body_chars=400, token_budget=0)
    assert with_sink == without
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_hook.py -v --color=yes`
Expected: FAIL, `TypeError: render_by_format() got an unexpected keyword argument 'emitted_sink'`

- [ ] **Step 3: Thread the sink through the renderers**

Add `emitted_sink: list[tuple[str, str]] | None = None` as a keyword-only
parameter to `render_by_format`, `render_recall_context`, `render_recall_compact`,
and `render_recall_balanced`, forwarding it at each dispatch site in
`render_by_format`.

In `render_recall_context`, the loop at `:430` has two exits per hit — the
`if max_chars is None or len(_render(block)) <= max_chars:` fast path, and the
budget-trimmed path below it. Record at **both**, with the body value each path
actually put into `lines`:

```python
        block = [*prefix, *([f"> {body}"] if body else []), *code_lines_by_hit.get(i, []), ""]
        if max_chars is None or len(_render(block)) <= max_chars:
            lines.extend(block)
            if emitted_sink is not None:
                emitted_sink.append((hit.id, body))
            continue
```

and in the trimmed path, append `(hit.id, trimmed_body)` using whatever local
holds the shortened body at the point it is committed to `lines` — and
`(hit.id, "")` on the branch that keeps only the title.

`render_recall_compact` emits no bodies: append `(hit.id, "")` per rendered hit.
For `render_recall_balanced`, read what it emits and record that text; if it
emits a body slice, record the slice.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_hook.py -v --color=yes`
Expected: 4 passed

- [ ] **Step 5: Verify the renderer suite did not move**

Run: `uv run --no-sync pytest tests/ -k "recall and (render or format or dedup)" --color=yes`
Expected: all pass. Any snapshot diff means the sink changed output — it must not.

- [ ] **Step 6: Append from the hook**

In `src/memo/cli_recall_hook.py`, replace the `render_by_format(...)` call with a
sink-carrying one and append after the context string is final:

```python
    _emitted: list[tuple[str, str]] = []
    context = render_by_format(
        _recall_format,
        relevant,
        nudge,
        turn=_turn,
        body_chars=body_chars,
        token_budget=token_budget,
        omitted=omitted,
        disputed_by=_disputed_by,
        state_dir=cfg.state_dir,
        emitted_sink=_emitted,
    )
```

and immediately after `context` stops being mutated (i.e. after the verbosity
steering block, before the debug print):

```python
    # Record what the model is about to see, so the MCP read tools can skip
    # re-sending it later in this session. Fail-open by contract: the recall
    # hook has a 5s budget and must never break on a bookkeeping write.
    if flag_bool("MEMO_EMIT_LEDGER") and _emitted:
        try:
            import time as _time

            from memo import emitted_ledger as _el
            from memo.identity import _session_id

            _sid = _session_id()
            if _sid:
                _now = int(_time.time())
                _ref = _el.mint_ref([i for i, _ in _emitted], _now, prefix="memo-h")
                _el.append(
                    cfg.state_dir,
                    _sid,
                    [
                        _el.Entry(
                            id=_id,
                            h=_el.emitted_hash(_body),
                            n=len(_body),
                            ref=_ref,
                            t=_now,
                            src="hook",
                        )
                        for _id, _body in _emitted
                    ],
                )
        except Exception:
            pass
```

- [ ] **Step 7: Write and run the hook-to-MCP test**

Append to `tests/test_emitted_ledger_integration.py`:

```python
def test_hook_emission_suppresses_a_later_search(memory_with_memories, call_tool, ledger_env):
    """The largest real overlap: the recall hook injected it at turn 2, so
    memo_search must not re-send the same body at turn 3."""
    from memo import emitted_ledger as el

    first = call_tool("memo_search", query="chat", limit=5)
    target = first["hits"][0]
    el.reset(ledger_env, "sess-int")

    # Simulate the hook having injected exactly this rendering.
    body = target["body"]
    el.append(
        ledger_env,
        "sess-int",
        [
            el.Entry(
                id=target["id"],
                h=el.emitted_hash(body),
                n=len(body),
                ref="memo-h/abc123",
                t=1,
                src="hook",
            )
        ],
    )

    second = call_tool("memo_search", query="chat", limit=5)
    digested = {e["id"]: e for e in second.get("already_in_context", [])}
    assert target["id"] in digested
    assert digested[target["id"]]["ref"] == "memo-h/abc123"
```

Run: `uv run --no-sync pytest tests/test_emitted_ledger_integration.py -v --color=yes`
Expected: all pass

- [ ] **Step 8: Measure the hook latency delta (success criterion 3)**

Run the hook 30 times with the flag off, then 30 with it on, and compare the p95
in `state_dir/recall_metrics.jsonl`:

```bash
for f in 0 1; do
  for i in $(seq 30); do
    echo '{"prompt":"cómo anda el chat de memo","session_id":"bench-'$f'"}' \
      | MEMO_EMIT_LEDGER=$f uv run --no-sync memo recall-hook >/dev/null
  done
done
uv run --no-sync memo stats | grep -i recall
```

Expected: p95 delta under 20ms. Record the two numbers in the commit message.

- [ ] **Step 9: Commit**

```bash
git add src/memo/recall_logic.py src/memo/cli_recall_hook.py tests/test_emitted_ledger_hook.py tests/test_emitted_ledger_integration.py
git commit -m "feat: recall hook records its emissions in the ledger

Renderers take an optional emitted_sink and report the body text they
actually shipped. The emitted body cannot be reconstructed externally --
_effective_body_chars scales the cap by score, MEMO_RECALL_SUMMARIZE_BODY
changes the truncation, and the max_chars path can drop a body mid-loop. An
external guess that over-recorded length would silently suppress content the
model never saw.

Closes the hook->MCP overlap, the largest duplicate source in a session.
Hook p95 delta: <fill in from step 8>."
```

---

### Task 7: Reset at the compaction boundary

**Files:**
- Modify: `src/memo/cli_capture.py:232-330` (the `capture-tick` command)
- Test: `tests/test_emitted_ledger_reset.py` (create)

**Interfaces:**
- Consumes: `emitted_ledger.reset`
- Produces: nothing. Behavioural only.

No `settings.json` change is needed. PreCompact already runs
`MEMO_NONINTERACTIVE=1 memo capture-tick --force` (`cli_hooks.py:87-92`), so
`--force` *is* the compaction boundary signal. Reusing it avoids wiring a second
hook and avoids a migration for existing installs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_emitted_ledger_reset.py`:

```python
from click.testing import CliRunner

from memo import emitted_ledger as el
from memo.cli_capture import capture_tick


def _seed(state_dir, sid="sess-reset"):
    el.append(
        state_dir,
        sid,
        [el.Entry(id="mem_a", h=el.emitted_hash("a"), n=1, ref="memo-r/aaaaaa", t=1, src="mcp")],
    )


def test_force_clears_the_ledger(tmp_memo_state, monkeypatch):
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-reset")
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    _seed(tmp_memo_state)
    CliRunner().invoke(capture_tick, ["--force"])
    assert el.read(tmp_memo_state, "sess-reset") == {}


def test_non_force_leaves_the_ledger_alone(tmp_memo_state, monkeypatch):
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-reset")
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    _seed(tmp_memo_state)
    CliRunner().invoke(capture_tick, [])
    assert set(el.read(tmp_memo_state, "sess-reset")) == {"mem_a"}


def test_force_is_idempotent(tmp_memo_state, monkeypatch):
    """PreCompact double-fires against the plugin copy."""
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-reset")
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "1")
    _seed(tmp_memo_state)
    r1 = CliRunner().invoke(capture_tick, ["--force"])
    r2 = CliRunner().invoke(capture_tick, ["--force"])
    assert r1.exit_code == 0 and r2.exit_code == 0
```

Use whatever fixture the repo already provides for an isolated `state_dir` — grep
`tests/conftest.py` for the existing one and use its real name in place of
`tmp_memo_state`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_reset.py -v --color=yes`
Expected: FAIL on `test_force_clears_the_ledger` — the ledger still holds `mem_a`

- [ ] **Step 3: Implement**

In `src/memo/cli_capture.py`, inside the `capture-tick` command body, early and
before the throttle check:

```python
    # PreCompact boundary. Once the window is rewritten memo can no longer claim
    # anything is in it, so the emission ledger must not outlive the compaction.
    if force:
        try:
            from memo import emitted_ledger as _el
            from memo.identity import _session_id

            _sid = _session_id()
            if _sid:
                _el.reset(cfg.state_dir, _sid)
        except Exception:
            pass
```

Extend the `--force` help text:

```python
    help=(
        "Bypass the per-session throttle (PreCompact force-flush at the "
        "compaction boundary). Also clears this session's emission ledger, "
        "since compaction invalidates every claim about what is in the window."
    ),
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_reset.py -v --color=yes`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_capture.py tests/test_emitted_ledger_reset.py
git commit -m "feat: clear the emission ledger at the compaction boundary

PreCompact already runs 'capture-tick --force', so --force is the boundary
signal. Reusing it needs no settings.json change and no migration for
existing installs. Idempotent: PreCompact double-fires against the plugin
copy."
```

---

### Task 8: Counters and `memo_cache_stats`

**Files:**
- Modify: `src/memo/emitted_ledger.py`, `src/memo/server_common.py`, `src/memo/server_cache.py`
- Test: `tests/test_emitted_ledger_stats.py` (create)

**Interfaces:**
- Consumes: `emitted_ledger.read`, `Partition.suppressed_chars`
- Produces: `emitted_ledger.stats(state_dir, session_id) -> dict[str, int]` with keys
  `entries`, `digests_served`, `tokens_suppressed`, `tokens_digest`,
  `memo_get_after_digest`, `net_saved_est`. Surfaced under the `emit_ledger` key
  of `memo_cache_stats()`.

No new MCP tool. A new tool costs schema tokens on every request, which would
partly undo what the feature saves — the spec calls this out and the design
depends on it.

Counters live in a sibling file `state_dir/emitted/<sid>.counters.json`,
read-modify-write inside the same fail-open envelope. `memo_get_after_digest`
increments in `memo_get` when the requested id appears in the current ledger with
a recorded digest, which is the conservative attribution the spec specifies: a
`memo_get` the model would have made anyway still counts against the feature.

- [ ] **Step 1: Write the failing test**

Create `tests/test_emitted_ledger_stats.py`:

```python
from memo import emitted_ledger as el


def test_stats_on_a_cold_session_are_zero(tmp_path):
    s = el.stats(tmp_path, "cold")
    assert s["entries"] == 0
    assert s["tokens_suppressed"] == 0
    assert s["net_saved_est"] == 0


def test_suppression_moves_the_counters(tmp_path):
    body = "x" * 400
    el.append(
        tmp_path,
        "s",
        [el.Entry(id="a", h=el.emitted_hash(body), n=len(body), ref="memo-r/a", t=1, src="mcp")],
    )
    el.bump(tmp_path, "s", digests_served=1, chars_suppressed=400, chars_digest=60)
    s = el.stats(tmp_path, "s")
    assert s["entries"] == 1
    assert s["digests_served"] == 1
    assert s["tokens_suppressed"] == 100  # 400 chars / 4
    assert s["tokens_digest"] == 15
    assert s["net_saved_est"] == 85


def test_memo_get_after_digest_reduces_net(tmp_path):
    el.bump(tmp_path, "s", digests_served=1, chars_suppressed=400, chars_digest=60)
    el.bump(tmp_path, "s", get_after_digest=1, chars_recovered=460)
    s = el.stats(tmp_path, "s")
    assert s["memo_get_after_digest"] == 1
    assert s["net_saved_est"] == 85 - 115  # recovery cost exceeds the saving
    assert s["net_saved_est"] < 0


def test_bump_is_fail_open_on_unwritable_dir(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        el.bump(ro, "s", digests_served=1)  # must not raise
        assert el.stats(ro, "s")["digests_served"] == 0
    finally:
        ro.chmod(0o700)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_stats.py -v --color=yes`
Expected: FAIL, no attribute `stats`

- [ ] **Step 3: Implement `bump` and `stats`**

Append to `src/memo/emitted_ledger.py`:

```python
def _counters_path(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / _DIRNAME / f"{_safe(session_id)}.counters.json"


def bump(
    state_dir: Path,
    session_id: str,
    *,
    digests_served: int = 0,
    chars_suppressed: int = 0,
    chars_digest: int = 0,
    get_after_digest: int = 0,
    chars_recovered: int = 0,
) -> None:
    """Accumulate the numbers the promotion gate reads. Fail-open."""
    try:
        path = _counters_path(state_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        for key, delta in (
            ("digests_served", digests_served),
            ("chars_suppressed", chars_suppressed),
            ("chars_digest", chars_digest),
            ("get_after_digest", get_after_digest),
            ("chars_recovered", chars_recovered),
        ):
            cur[key] = int(cur.get(key, 0)) + delta
        path.write_text(json.dumps(cur, separators=(",", ":")), encoding="utf-8")
    except Exception:
        return


def stats(state_dir: Path, session_id: str) -> dict[str, int]:
    """The session's ledger scorecard, in tokens.

    ``net_saved_est`` is the number the promotion gate turns on: what was not
    sent, minus what the digests themselves cost, minus what recovering from
    them cost. It can go negative, and that is the whole point of measuring it
    — if the model re-fetches every digested id, the feature loses.
    """
    try:
        cur = json.loads(_counters_path(state_dir, session_id).read_text(encoding="utf-8"))
    except Exception:
        cur = {}

    def _tok(chars: int) -> int:
        return chars // 4

    suppressed = _tok(int(cur.get("chars_suppressed", 0)))
    digest_cost = _tok(int(cur.get("chars_digest", 0)))
    recovery = _tok(int(cur.get("chars_recovered", 0)))
    return {
        "entries": len(read(state_dir, session_id)),
        "digests_served": int(cur.get("digests_served", 0)),
        "tokens_suppressed": suppressed,
        "tokens_digest": digest_cost,
        "memo_get_after_digest": int(cur.get("get_after_digest", 0)),
        "net_saved_est": suppressed - digest_cost - recovery,
    }
```

- [ ] **Step 4: Call `bump` from `apply_ledger`**

In `server_common.apply_ledger`, immediately before the `return part.full, {...}`
that carries suppressions:

```python
        extra_payload = {
            "already_in_context": [...],   # as already written in Task 3
            "hint": "...",
            "cache_ref": ref,
        }
        el.bump(
            state_dir,
            session_id,
            digests_served=len(part.digest),
            chars_suppressed=part.suppressed_chars,
            chars_digest=len(json.dumps(extra_payload)),
        )
        return part.full, extra_payload
```

Add `import json` to the function's local import block.

- [ ] **Step 5: Count recoveries in `memo_get`**

In the module registering `memo_get`, after the memory is fetched and before the
return:

```python
        from memo.flags import flag_bool

        if flag_bool("MEMO_EMIT_LEDGER"):
            try:
                from memo import emitted_ledger as _el
                from memo.server_session_patterns import _effective_session_id

                _sid = _effective_session_id()
                if id in _el.read(memory.cfg.state_dir, _sid):
                    _el.bump(
                        memory.cfg.state_dir,
                        _sid,
                        get_after_digest=1,
                        chars_recovered=len(str(out.get("body") or "")),
                    )
            except Exception:
                pass
```

Substitute the real local names for `id` and `out`. This over-counts — a
`memo_get` the model would have issued regardless is indistinguishable — and the
bias is deliberately against the feature.

- [ ] **Step 6: Surface in `memo_cache_stats`**

In `src/memo/server_cache.py`, extend `memo_cache_stats`:

```python
    @annotated_tool(server, **READ_ONLY)
    def memo_cache_stats() -> dict[str, Any]:
        """Cache-tier status: mode, backend, entry count, capacity, overflow.

        When MEMO_CACHE_MODE=off (the default) `enabled` is False and memo is
        behaving as a durable store with no eviction.

        Also carries `emit_ledger`: how many memory bodies were suppressed as
        already-in-context this session, and the estimated net token saving
        after digest and recovery costs. Reported here rather than as its own
        tool — a new tool costs schema tokens on every request, which would
        undo part of what the ledger saves.
        """
        out = memory.cache.stats()
        from memo.flags import flag_bool

        if flag_bool("MEMO_EMIT_LEDGER"):
            try:
                from memo import emitted_ledger as _el
                from memo.server_session_patterns import _effective_session_id

                out["emit_ledger"] = _el.stats(
                    memory.cfg.state_dir, _effective_session_id()
                )
            except Exception:
                pass
        return out
```

- [ ] **Step 7: Run the tests**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_stats.py tests/test_emitted_ledger_apply.py -v --color=yes`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/memo/emitted_ledger.py src/memo/server_common.py src/memo/server_cache.py tests/test_emitted_ledger_stats.py
git commit -m "feat: emission ledger counters behind memo_cache_stats

net_saved_est subtracts digest cost and memo_get recovery cost from what was
suppressed, so it can go negative -- which is the point. If the model
re-fetches every digested id the feature loses, and the gate has to be able
to see that.

No new MCP tool: a tool schema costs tokens on every request and would undo
part of the saving."
```

---

### Task 9: Prune stale ledgers

**Files:**
- Modify: `src/memo/cli_ops.py` (add `gc-emitted-ledgers` alongside the existing gc verbs)
- Modify: `~/.local/share/memo/bin/memo-nightly.sh`
- Test: `tests/test_emitted_ledger_gc.py` (create)

**Interfaces:**
- Consumes: `emitted_ledger.prune`
- Produces: `memo ops gc-emitted-ledgers` printing the removed count.

Sessions leave no close signal, so ledger files accumulate. Age is the only
liveness proxy available; 48h is well past any live session.

- [ ] **Step 1: Write the failing test**

Create `tests/test_emitted_ledger_gc.py`:

```python
import os
import time

from click.testing import CliRunner

from memo import emitted_ledger as el
from memo.cli_ops import ops


def test_gc_removes_ledgers_older_than_48h(tmp_memo_state):
    el.append(
        tmp_memo_state,
        "old",
        [el.Entry(id="a", h="deadbeef", n=1, ref="memo-r/a", t=1, src="mcp")],
    )
    el.append(
        tmp_memo_state,
        "live",
        [el.Entry(id="b", h="deadbeef", n=1, ref="memo-r/b", t=1, src="mcp")],
    )
    stale = time.time() - 60 * 60 * 72
    os.utime(el.ledger_path(tmp_memo_state, "old"), (stale, stale))

    result = CliRunner().invoke(ops, ["gc-emitted-ledgers"])
    assert result.exit_code == 0
    assert "1" in result.output
    assert not el.ledger_path(tmp_memo_state, "old").exists()
    assert el.ledger_path(tmp_memo_state, "live").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_gc.py -v --color=yes`
Expected: FAIL, no such command

- [ ] **Step 3: Add the command**

In `src/memo/cli_ops.py`, following the shape of the existing `gc-memo-duplicates`
and `gc-vault-orphans` commands:

```python
@ops.command(name="gc-emitted-ledgers")
@click.option(
    "--max-age-hours",
    default=48,
    show_default=True,
    help="Remove emission ledgers untouched for longer than this.",
)
def gc_emitted_ledgers(max_age_hours: int) -> None:
    """Remove emission-ledger files from sessions that are long over.

    Sessions leave no close signal, so age is the only liveness proxy.
    """
    from memo import emitted_ledger
    from memo.config import load_config

    cfg = load_config()
    removed = emitted_ledger.prune(cfg.state_dir, max_age_s=max_age_hours * 3600)
    click.echo(f"removed {removed} emission ledger(s)")
```

Match the surrounding commands' config-loading idiom exactly rather than the
`load_config()` guess above if the file uses a different one.

- [ ] **Step 4: Add to the nightly script**

In `~/.local/share/memo/bin/memo-nightly.sh`, after the existing
`memo ops gc-vault-orphans` line:

```bash
"$MEMO_BIN" ops gc-emitted-ledgers || true
```

Mirror the error handling the neighbouring gc lines use.

- [ ] **Step 5: Run the test**

Run: `uv run --no-sync pytest tests/test_emitted_ledger_gc.py -v --color=yes`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add src/memo/cli_ops.py tests/test_emitted_ledger_gc.py
git commit -m "chore: gc stale emission ledgers in the nightly pass

Sessions leave no close signal, so age is the only liveness proxy. 48h is
well past any live session."
```

---

### Task 10: Replay harness and the promotion decision

**Files:**
- Create: `docs/eval/emission-ledger-replay.md`
- Create: `scripts/eval_emission_ledger.py`
- Modify: `docs/SPECS/2026-08-10-emission-ledger-design.md` (record the measured result)

**Interfaces:**
- Consumes: `emitted_ledger.stats`, `mcp_budget.est_tokens`
- Produces: a measured verdict on success criteria 1 and 2. Nothing imports this.

This task decides whether `MEMO_EMIT_LEDGER` ever defaults to `1`. Criteria 3
was measured in Task 6 step 8; criterion 4 is the pre-push gate.

- [ ] **Step 1: Write the harness**

Create `scripts/eval_emission_ledger.py`. It replays a real Claude Code
transcript's memo tool calls twice — flag off, then flag on — against the same
corpus, and reports the delta:

```python
"""Replay a session's memo read calls with the emission ledger off and on.

Criterion 1 wants the ratio of tokens memo put into one window, not tokens
overall: the denominator is recall-hook injections plus participating tool
results with MEMO_EMIT_LEDGER=0. Criterion 2 wants the memo_get-after-digest
rate, which only the counters can supply.

Usage:
    uv run --no-sync python scripts/eval_emission_ledger.py \
        ~/.claude/projects/-Users-fer/<session>.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from memo import emitted_ledger as el
from memo.mcp_budget import est_tokens

_PARTICIPATING = {
    "memo_search",
    "memo_ask",
    "memo_context",
    "memo_unified_briefing",
    "memo_evidence_pack",
}


def _memo_calls(transcript: Path) -> list[tuple[str, dict]]:
    """Every memo read-tool call in the transcript, in order."""
    calls: list[tuple[str, dict]] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        for block in row.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name", "")).removeprefix("mcp__memo__")
            if name in _PARTICIPATING:
                calls.append((name, block.get("input") or {}))
    return calls


def _run(calls: list[tuple[str, dict]], *, enabled: bool, session: str) -> int:
    from memo.memory import Memory

    os.environ["MEMO_EMIT_LEDGER"] = "1" if enabled else "0"
    os.environ["MEMO_SESSION_ID"] = session
    memory = Memory()
    el.reset(memory.cfg.state_dir, session)

    from memo.server_common import apply_ledger

    total = 0
    for name, args in calls:
        if name != "memo_search":
            continue  # search-only replay; see the doc for why
        hits = [r.to_dict() for r in memory.search(str(args.get("query") or ""), limit=10)]
        kept, extra = apply_ledger(memory, name, hits)
        total += est_tokens(json.dumps({"hits": kept, **extra}))
    return total


def main() -> int:
    transcript = Path(sys.argv[1]).expanduser()
    calls = _memo_calls(transcript)
    if not calls:
        print("no participating memo calls in this transcript")
        return 1

    baseline = _run(calls, enabled=False, session="eval-off")
    treated = _run(calls, enabled=True, session="eval-on")

    from memo.memory import Memory

    counters = el.stats(Memory().cfg.state_dir, "eval-on")
    served = counters["digests_served"]
    recovered = counters["memo_get_after_digest"]
    reduction = 0.0 if not baseline else (baseline - treated) / baseline

    print(f"calls replayed:        {len(calls)}")
    print(f"baseline tokens:       {baseline}")
    print(f"with ledger:           {treated}")
    print(f"reduction:             {reduction:.1%}   (criterion 1: >= 25%)")
    print(f"digests served:        {served}")
    print(f"memo_get after digest: {recovered}")
    if served:
        print(f"recovery rate:         {recovered / served:.1%}   (criterion 2: < 20%)")
    print(f"net_saved_est:         {counters['net_saved_est']} tokens")

    ok = reduction >= 0.25 and (not served or recovered / served < 0.20)
    print("\nVERDICT:", "PROMOTE" if ok else "KEEP AT MEMO_EMIT_LEDGER=0")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it against a real transcript**

```bash
uv run --no-sync python scripts/eval_emission_ledger.py \
  "$(ls -t ~/.claude/projects/-Users-fer/*.jsonl | head -1)"
```

Expected: a printed verdict. Criterion 2 is only meaningful when replaying a
transcript that actually contains `memo_get` calls — a synthetic replay cannot
produce them, since the model is not in the loop. Note that limitation in the doc
rather than claiming the criterion passed on synthetic data.

- [ ] **Step 3: Write up the harness**

Create `docs/eval/emission-ledger-replay.md` covering: what the denominator is
and why it is not whole-session tokens; that criterion 2 needs a live session
rather than a replay, because a replay has no model to decide whether to recover;
how to read `net_saved_est`; and the measured numbers from step 2.

- [ ] **Step 4: Record the verdict in the spec**

Add a `## Measured result` section to
`docs/SPECS/2026-08-10-emission-ledger-design.md` with the four criteria, the
number measured for each, and the promotion decision. If criterion 1 or 2 fails,
say so plainly and leave the flag at `0` — the spec commits to that outcome.

- [ ] **Step 5: Run the full suite and the pre-push gate (criterion 4)**

```bash
uv run --no-sync pytest tests/ --color=yes -q
git worktree add /tmp/memo-master origin/master   # gate measures corpus drift, not code
```

Run the gate in the master worktree first, then on this branch, and compare. A
gate move that reproduces on master is corpus drift and is not this diff's
problem.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_emission_ledger.py docs/eval/emission-ledger-replay.md docs/SPECS/2026-08-10-emission-ledger-design.md
git commit -m "test: emission ledger replay harness and measured verdict

Replays a real transcript's memo read calls with the flag off and on, and
prints a PROMOTE / KEEP-AT-0 verdict against criteria 1 and 2.

Criterion 2 needs a live session: a replay has no model in the loop to decide
whether to recover a digested id, so a synthetic run cannot produce the rate.
Documented rather than claimed."
```

- [ ] **Step 7: Open the PR**

`master` is protected. Push and open a PR whose body carries the four criteria
and their measured numbers.

```bash
git push -u origin feat/emission-ledger
gh pr create --title "feat: emission ledger — stop re-emitting bodies already in context" --body "..."
```

---

## Self-Review

**Spec coverage.** Every section maps to a task: storage and session key (1),
partition and the monotonic rule (2), the MCP wrapper (3), tool scope (4, 5),
recall-hook writes (6), invalidation and PreCompact (7), the metric and
`memo_cache_stats` (8), orphan pruning, hazard 4 (9), success criteria and the
promotion gate (10). Hazards 1 (subagents), 3 (clients with no session id) and 6
(eval gate) are accepted behaviours documented in the spec, not code — 1 and 3
are covered by the fail-open paths in Tasks 3 and 6, and 6 is Task 10 step 5.
Hazard 5 (hook fail-open) is Task 6 step 6 plus the Task 1 unwritable-dir test.

**Placeholders.** Three tasks deliberately end in "read the real name and use it"
rather than a guess: the conftest fixture names (Task 4), the per-tool hit keys
(Task 5), and the trimmed-body local in `render_recall_context` (Task 6). These
are lookups a fresh engineer must perform because guessing them would produce a
plausible-but-wrong edit — the file locations and the exact criterion for each
are given. Task 5 additionally specifies the fallback if `memo_context` embeds
bodies in its packed string: drop the tool from the default allowlist rather than
half-wire it.

**Type consistency.** `Entry(id, h, n, ref, t, src)` is constructed identically in
Tasks 1, 3, 6, 8 and 9. `partition(hits, known, *, text_of, id_of) -> Partition`
matches its Task 2 definition at both Task 3 call sites. `apply_ledger(memory,
tool, hits) -> tuple[list[dict], dict]` matches at every call site in Tasks 4, 5
and 8. `mint_ref(ids, t, *, prefix)` is called with `prefix="memo-h"` only from
the hook. `stats()` keys match between Task 8's implementation, its test, and the
`emit_ledger` block in the spec.

One fix applied during review: Task 8's `apply_ledger` change required naming the
extra-payload dict before returning it, so `chars_digest` could measure the real
serialized digest rather than an estimate. Task 3's version returns the literal
inline; Task 8 step 4 shows the restructured form explicitly instead of leaving
the reader to infer it.
