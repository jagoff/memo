# Proxy Context Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put memo in the Anthropic API request path as a local proxy that rewrites outbound payloads to cost fewer tokens, recovers anything it cut on demand, and reports savings from the provider's own `usage` fields instead of a hardcoded estimate.

**Architecture:** A loopback ASGI service on port 8768 that Claude Code reaches via `ANTHROPIC_BASE_URL`. It splits each request payload into a cache-stable prefix and a live zone, applies independent transforms to each zone, forwards to `api.anthropic.com` with headers intact, relays the response stream byte-for-byte, and records the provider's four `usage` counters against a holdout control arm. Every transform is fail-open: an exception forwards the original body unmodified.

**Tech Stack:** Python 3.13, FastAPI + uvicorn (existing `[http]` extra), httpx (new, added to `[http]`), Pillow (new, added to `[http]`, pixel transform only), Click, pytest.

**Spec:** `docs/SPECS/2026-08-18-token-savings-proxy-context-compression-design.md`

## Global Constraints

- memo is **MIT**. Code may be ported only from snip (MIT), ccusage (MIT), headroom (Apache-2.0), entroly (Apache-2.0), each with attribution; Apache-2.0 sources also require a `NOTICE` entry. caveman's `engine/`, `proxy/`, `cacheengine/`, `rewriter/`, `browse/`, `mcp/`, `shrink/`, `shared/platform/` are **BSL-1.1**, and token-optimizer, jcodemunch-mcp (Dual-Use), and mcp-server-code-execution-mode (GPL-3.0) are **idea-only** — never copy a line from any of them.
- **Forward `anthropic-beta` verbatim.** All request headers pass upstream unmodified except `content-length`, which is recomputed. Dropping `anthropic-beta` breaks claude.ai subscription auth.
- **Never buffer the response body.** Claude Code's byte-level watchdog aborts a stream after 180s of silence on the direct API. Relay SSE bytes as they arrive, keep-alive pings included.
- **Credentials never reach a log sink.** No header value may appear in a log line, a ledger row, or an error message.
- **Fail-open everywhere.** Any exception in planning or applying a transform forwards the original body.
- Port default **8768** (8765 is `~/repos/rag`, 8767 is `com.memo.chat` on this machine).
- New flags go in a new `src/memo/flags_proxy.py`, registered in `src/memo/flags.py` — never `os.environ.get("MEMO_...")` inline.
- The proxy package must be **import-safe without the `[http]` extra**: FastAPI/uvicorn/httpx imports live inside functions, mirroring `src/memo/chat/http.py`.
- **Shared working tree.** Stage explicit paths only. Never `git add -A`, `git commit -a`, or `ruff format src/`. Lint and format only the files you touched.
- Test isolation per `tests/conftest.py`: `tmp_cfg` or an isolated `Config`; `CliRunner` invocations set `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`, `MEMO_STATE_DIR` in `env=`; never touch the developer's real vault.
- Run `uv run --no-sync pytest tests/ -m "not slow"`, `uv run --no-sync mypy src/memo/`, and `uv run --no-sync ruff check src/memo/proxy/ tests/test_proxy_*.py` before each commit.

---

## Phase A — A transparent, measuring proxy (Tasks 1–8)

Tasks 1–8 ship working software on their own: a proxy that changes no payload but measures the truth memo cannot see today, closing spec findings 4, 5, and 6. Do not start Phase B until Phase A is installed and has recorded real traffic.

---

### Task 1: Verify the load-bearing assumption

The entire design rests on Claude Code honouring `ANTHROPIC_BASE_URL` against a loopback listener while keeping subscription auth. Prove it before writing any production code. This task commits no source.

**Files:**
- Create: `/private/tmp/claude-501/-Users-fer-repos-memo/0942354a-e2af-46ad-a29d-84525e98094c/scratchpad/probe_baseurl.py` (throwaway, never committed)

**Interfaces:**
- Consumes: nothing
- Produces: a recorded yes/no that gates every later task

- [ ] **Step 1: Write a throwaway recording proxy**

```python
# scratchpad/probe_baseurl.py — throwaway. Records what Claude Code sends.
import http.server, json, pathlib, socketserver

OUT = pathlib.Path(__file__).with_name("probe_capture.jsonl")

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n)
        OUT.open("a").write(json.dumps({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()
                        if k.lower() not in ("authorization", "x-api-key")},
            "has_auth": any(k.lower() in ("authorization", "x-api-key")
                            for k in self.headers),
            "anthropic_beta": self.headers.get("anthropic-beta"),
            "body_bytes": len(body),
        }) + "\n")
        payload = b'{"error":{"type":"probe","message":"probe only"}}'
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a): pass

with socketserver.TCPServer(("127.0.0.1", 8768), H) as s:
    s.serve_forever()
```

- [ ] **Step 2: Run the probe and point one Claude Code call at it**

```bash
cd "$SCRATCHPAD" && python3 probe_baseurl.py &
sleep 1
ANTHROPIC_BASE_URL=http://127.0.0.1:8768 claude -p "say ok" --max-turns 1 ; true
kill %1
cat "$SCRATCHPAD/probe_capture.jsonl"
```

- [ ] **Step 3: Confirm the three facts the design depends on**

Expected in `probe_capture.jsonl`:
1. `"path": "/v1/messages"` — the request arrived at all.
2. `"has_auth": true` — a credential was attached.
3. `"anthropic_beta"` is non-null — the header the design must forward exists.

If `path` never appears, **stop and report**: the proxy approach does not work on this Claude Code version and the spec needs revision before any further task.

- [x] **Step 4: Record the result in the plan**

**Observed result (2026-08-18, macOS, Claude Code / `rtk` CLI v2.1.226, `claude-cli/2.1.226 (external, sdk-cli)`, port 8768):** the assumption holds — all three facts confirmed.

1. `"path": "/v1/messages?beta=true"` — request arrived (query suffix `?beta=true`, not a bare `/v1/messages`).
2. `"has_auth": true` — credential attached on every request.
3. `"anthropic_beta"` non-null — `claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advisor-tool-2026-03-01,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11` (12 beta flags; `oauth-2025-04-20` confirms subscription auth, not an API key). The proxy's synthetic 400 triggered one client-side retry (2nd request identical minus `fallback-credit-2026-06-01`); the design must tolerate retries and the `?beta=true` suffix. Full transcript: `.superpowers/sdd/2026-08-18-token-savings-proxy-context-compression-plan/task-1-report.md`.

Edit this file, replacing this line with the observed values. Commit only this plan file:

```bash
git add docs/SPECS/2026-08-18-token-savings-proxy-context-compression-plan.md
git commit -m "docs: record ANTHROPIC_BASE_URL probe result"
```

---

### Task 2: Flag registry

**Files:**
- Create: `src/memo/flags_proxy.py`
- Modify: `src/memo/flags.py:34-56` (import + `_SPECS` tuple)
- Test: `tests/test_proxy_flags.py`

**Interfaces:**
- Consumes: `memo.flags_base.FlagSpec`, `_spec`
- Produces: `memo.flags_proxy.SPECS`; accessors `flag_bool("MEMO_PROXY_ENABLED")`, `flag_int("MEMO_PROXY_PORT")`, `flag_float("MEMO_PROXY_HOLDOUT_FRAC")` resolve through the existing `memo.flags.flag_*` API

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_flags.py
from memo.flags import REGISTRY, flag_bool, flag_float, flag_int
from memo.flags_proxy import SPECS


def test_every_proxy_flag_is_registered_globally():
    for spec in SPECS:
        assert spec.name in REGISTRY, f"{spec.name} missing from memo.flags.REGISTRY"


def test_proxy_defaults(monkeypatch):
    for spec in SPECS:
        monkeypatch.delenv(spec.name, raising=False)
    assert flag_bool("MEMO_PROXY_ENABLED") is True
    assert flag_int("MEMO_PROXY_PORT") == 8768
    assert flag_float("MEMO_PROXY_HOLDOUT_FRAC") == 0.05
    assert flag_int("MEMO_PROXY_TOOL_WINDOW_SESSIONS") == 20


def test_holdout_fraction_is_bounded():
    spec = next(s for s in SPECS if s.name == "MEMO_PROXY_HOLDOUT_FRAC")
    assert spec.min_val == 0.0
    assert spec.max_val == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_flags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.flags_proxy'`

- [ ] **Step 3: Write the flag module**

```python
# src/memo/flags_proxy.py
"""Flags for the context-compression proxy (`memo proxy serve`).

The proxy sits between Claude Code and the provider. Every transform is ON by
default and self-limiting; `MEMO_PROXY_ENABLED=0` (or `memo proxy off`) turns
the whole thing into pure passthrough.
"""

from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    _spec(
        "MEMO_PROXY_ENABLED",
        "bool",
        True,
        "proxy",
        "Master switch for payload rewriting. When off the proxy still forwards "
        "and still measures, but changes nothing — the honest passthrough mode.",
    ),
    _spec(
        "MEMO_PROXY_PORT",
        "int",
        8768,
        "proxy",
        "Loopback port the proxy listens on. 8765 and 8767 are commonly taken "
        "by other local services; install fails loudly on a bound port.",
        min_val=1024,
        max_val=65535,
    ),
    _spec(
        "MEMO_PROXY_HOLDOUT_FRAC",
        "float",
        0.05,
        "proxy",
        "Fraction of requests forwarded uncompressed as the control arm. This is "
        "what every savings claim is measured against; 0 disables measurement "
        "of savings, not the transforms.",
        min_val=0.0,
        max_val=0.5,
    ),
    _spec(
        "MEMO_PROXY_TOOL_SCHEMAS",
        "bool",
        True,
        "proxy",
        "Prune MCP tool definitions to those used recently in this project.",
    ),
    _spec(
        "MEMO_PROXY_TOOL_WINDOW_SESSIONS",
        "int",
        20,
        "proxy",
        "Sessions of usage history that decide which tool schemas survive pruning.",
        min_val=1,
    ),
    _spec(
        "MEMO_PROXY_TOOL_RESULTS",
        "bool",
        True,
        "proxy",
        "Apply declarative per-command filters to tool_result blocks.",
    ),
    _spec(
        "MEMO_PROXY_STRUCTMAP",
        "bool",
        True,
        "proxy",
        "Replace code-file reads with signatures, and re-reads with a diff.",
    ),
    _spec(
        "MEMO_PROXY_JSONCRUSH",
        "bool",
        True,
        "proxy",
        "Run the existing L1 JSON crusher over large JSON tool results.",
    ),
    _spec(
        "MEMO_PROXY_PIXEL",
        "bool",
        True,
        "proxy",
        "Render dense text blocks to PNG when the per-block profitability gate "
        "says the image costs fewer tokens. No-op without the [http] extra.",
    ),
    _spec(
        "MEMO_PROXY_RETRIEVE_ALARM_FRAC",
        "float",
        0.05,
        "proxy",
        "Retrieval rate above which a transform is reported as over-cutting — a "
        "recovered original costs its tokens twice.",
        min_val=0.0,
        max_val=1.0,
    ),
)
```

- [ ] **Step 4: Register the module in the aggregate registry**

In `src/memo/flags.py`, add the import beside the other domain imports (they are alphabetical):

```python
from memo.flags_proxy import SPECS as _proxy_specs
```

and add `*_proxy_specs,` to the `_SPECS` tuple at line 46.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_flags.py -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Verify the config validator accepts them**

Run: `uv run --no-sync memo config validate`
Expected: no unknown-variable complaint about any `MEMO_PROXY_*` name

- [ ] **Step 7: Commit**

```bash
git add src/memo/flags_proxy.py src/memo/flags.py tests/test_proxy_flags.py
git commit -m "feat(proxy): register MEMO_PROXY_* flags"
```

---

### Task 3: Zone splitting and the cache rule

**Files:**
- Create: `src/memo/proxy/__init__.py`, `src/memo/proxy/zones.py`
- Test: `tests/test_proxy_zones.py`

**Interfaces:**
- Consumes: nothing outside stdlib
- Produces:
  - `Zones` dataclass with fields `system: list[dict]`, `tools: list[dict]`, `frozen_messages: list[dict]`, `live_messages: list[dict]`
  - `split(payload: dict, *, live_turns: int = 2) -> Zones`
  - `prefix_fingerprint(zones: Zones) -> str` — sha256 hex of the stable prefix
  - `LIVE_TURNS_DEFAULT: int = 2`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_zones.py
from memo.proxy.zones import Zones, prefix_fingerprint, split


def _payload(n_messages: int) -> dict:
    return {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": "you are helpful"}],
        "tools": [{"name": "memo_search", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n_messages)
        ],
    }


def test_split_puts_the_last_turns_in_the_live_zone():
    z = split(_payload(10), live_turns=2)
    assert len(z.live_messages) == 2
    assert len(z.frozen_messages) == 8
    assert z.live_messages[-1]["content"] == "m9"


def test_short_conversation_is_all_live():
    z = split(_payload(1), live_turns=2)
    assert z.frozen_messages == []
    assert len(z.live_messages) == 1


def test_missing_system_and_tools_are_empty_not_none():
    z = split({"messages": []})
    assert z.system == []
    assert z.tools == []


def test_prefix_fingerprint_ignores_the_live_zone():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.live_messages[-1]["content"] = "totally different"
    assert prefix_fingerprint(a) == prefix_fingerprint(b)


def test_prefix_fingerprint_changes_when_tools_change():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.tools.append({"name": "memo_get", "input_schema": {"type": "object"}})
    assert prefix_fingerprint(a) != prefix_fingerprint(b)


def test_zones_reassemble_into_an_equivalent_payload():
    original = _payload(6)
    z = split(original, live_turns=2)
    assert z.to_payload(original)["messages"] == original["messages"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_zones.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy'`

- [ ] **Step 3: Write the implementation**

```python
# src/memo/proxy/__init__.py
"""Context-compression proxy. Import-safe without the [http] extra."""
```

```python
# src/memo/proxy/zones.py
"""Splits a Messages-API payload into a cache-stable prefix and a live zone.

The economics of this whole package hinge on one rule: a cache read costs 0.1x
a fresh input token, so a transform that rewrites the cached prefix on every
turn can easily cost more than it saves. Transforms that touch the prefix must
be deterministic and session-stable; `prefix_fingerprint` is what proves it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

LIVE_TURNS_DEFAULT = 2


@dataclass
class Zones:
    system: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    frozen_messages: list[dict] = field(default_factory=list)
    live_messages: list[dict] = field(default_factory=list)

    def to_payload(self, original: dict) -> dict:
        """Reassemble, preserving every key the proxy does not own."""
        out = dict(original)
        if self.system or "system" in original:
            out["system"] = self.system
        if self.tools or "tools" in original:
            out["tools"] = self.tools
        out["messages"] = [*self.frozen_messages, *self.live_messages]
        return out


def _as_list(value: object) -> list[dict]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return []


def split(payload: dict, *, live_turns: int = LIVE_TURNS_DEFAULT) -> Zones:
    """Partition a request payload. Never raises on a malformed payload."""
    messages = payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    cut = max(0, len(messages) - live_turns)
    return Zones(
        system=_as_list(payload.get("system")),
        tools=_as_list(payload.get("tools")),
        frozen_messages=list(messages[:cut]),
        live_messages=list(messages[cut:]),
    )


def prefix_fingerprint(zones: Zones) -> str:
    """sha256 of everything the provider will cache. Live zone excluded."""
    blob = json.dumps(
        [zones.system, zones.tools, zones.frozen_messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_zones.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/__init__.py src/memo/proxy/zones.py tests/test_proxy_zones.py
git commit -m "feat(proxy): split payloads into cache-stable prefix and live zone"
```

---

### Task 4: Content-addressed recovery

**Files:**
- Create: `src/memo/proxy/ccr.py`
- Test: `tests/test_proxy_ccr.py`

**Interfaces:**
- Consumes: `memo.store.crush_cache.CrushCache` — `CrushCache(state_dir).cache(hash_val: str, content: str) -> None` and `.retrieve(hash_val: str, ttl_days: int = 30) -> str | None`
- Produces:
  - `stash(state_dir: Path, content: str) -> str` — returns the sha256 hex key
  - `recover(state_dir: Path, key: str) -> str | None`
  - `marker(key: str, kept_chars: int, dropped_chars: int) -> str` — the text left in the payload

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_ccr.py
from memo.proxy import ccr


def test_stash_then_recover_roundtrips(tmp_path):
    key = ccr.stash(tmp_path, "the original content")
    assert ccr.recover(tmp_path, key) == "the original content"


def test_stash_is_content_addressed(tmp_path):
    assert ccr.stash(tmp_path, "same") == ccr.stash(tmp_path, "same")
    assert ccr.stash(tmp_path, "same") != ccr.stash(tmp_path, "other")


def test_recover_returns_none_for_unknown_key(tmp_path):
    assert ccr.recover(tmp_path, "a" * 64) is None


def test_recover_never_touches_the_filesystem_for_a_non_hex_key(tmp_path):
    assert ccr.recover(tmp_path, "../../etc/passwd") is None


def test_marker_names_the_key_and_what_was_dropped():
    m = ccr.marker("abc123", kept_chars=100, dropped_chars=900)
    assert "abc123" in m
    assert "900" in m
    assert "memo_retrieve" in m


def test_stash_returns_empty_key_when_the_cache_is_unwritable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("memo.store.crush_cache.CrushCache.cache", boom)
    assert ccr.stash(tmp_path, "content") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_ccr.py -v`
Expected: FAIL with `ImportError: cannot import name 'ccr'`

- [ ] **Step 3: Write the implementation**

```python
# src/memo/proxy/ccr.py
"""Content-addressed recovery: nothing is cut without being recoverable.

Reuses the crush cache the capture crusher already writes to, so `memo
retrieve` keeps working unchanged and there is exactly one recovery path.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def stash(state_dir: Path, content: str) -> str:
    """Store `content` and return its key. Returns "" if it could not be stored.

    An empty key is the caller's signal to skip the lossy edit entirely: cutting
    without a recovery path is not a trade this package makes.
    """
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        from memo.store.crush_cache import CrushCache

        CrushCache(Path(state_dir)).cache(key, content)
    except Exception:
        _log.warning("proxy: crush cache unwritable; skipping lossy edit")
        return ""
    return key


def recover(state_dir: Path, key: str) -> str | None:
    try:
        from memo.store.crush_cache import CrushCache

        return CrushCache(Path(state_dir)).retrieve(key)
    except Exception:
        return None


def marker(key: str, *, kept_chars: int, dropped_chars: int) -> str:
    """The text that replaces what was cut. Tells the model how to get it back."""
    return (
        f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
        f"Full original: memo_retrieve(key=\"{key}\")]"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_ccr.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/ccr.py tests/test_proxy_ccr.py
git commit -m "feat(proxy): content-addressed recovery over the crush cache"
```

---

### Task 5: Measurement ledger and holdout

**Files:**
- Create: `src/memo/proxy/meter.py`
- Test: `tests/test_proxy_meter.py`

**Interfaces:**
- Consumes: `memo.mcp_budget.est_tokens(text: str) -> int`; `memo.flags.flag_float`
- Produces:
  - `LEDGER_SCHEMA: str = "memo.proxy.requests.v1"`
  - `is_holdout(request_key: str, frac: float) -> bool`
  - `Record` dataclass: `request_key`, `holdout`, `transforms` (`list[str]`), `est_saved_tokens`, `input_tokens`, `output_tokens`, `cache_creation_tokens`, `cache_read_tokens`, `retrieved` (`int`)
  - `append(state_dir: Path, record: Record) -> None`
  - `usage_from_response(body: dict) -> dict[str, int]`
  - `summarize(state_dir: Path) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_meter.py
import json

from memo.proxy.meter import (
    Record,
    append,
    is_holdout,
    summarize,
    usage_from_response,
)


def test_holdout_assignment_is_stable_for_a_key():
    assert is_holdout("abc", 0.5) == is_holdout("abc", 0.5)


def test_holdout_fraction_zero_holds_nothing_out():
    assert not any(is_holdout(str(i), 0.0) for i in range(200))


def test_holdout_fraction_one_holds_everything_out():
    assert all(is_holdout(str(i), 1.0) for i in range(200))


def test_holdout_fraction_is_roughly_honoured():
    n = sum(is_holdout(str(i), 0.1) for i in range(2000))
    assert 120 < n < 280  # 10% of 2000, generous band


def test_usage_reads_all_four_provider_fields():
    body = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 40,
        }
    }
    assert usage_from_response(body) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_tokens": 30,
        "cache_read_tokens": 40,
    }


def test_usage_of_a_bodyless_response_is_all_zeroes():
    assert usage_from_response({}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }


def test_append_writes_one_json_line_per_record(tmp_path):
    append(tmp_path, Record(request_key="k1", holdout=False, transforms=["toolschemas"],
                            est_saved_tokens=100, input_tokens=1, output_tokens=2,
                            cache_creation_tokens=3, cache_read_tokens=4, retrieved=0))
    lines = (tmp_path / "proxy" / "requests.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_key"] == "k1"


def test_summarize_compares_treated_against_holdout(tmp_path):
    for i in range(10):
        append(tmp_path, Record(request_key=f"t{i}", holdout=False, transforms=["x"],
                                est_saved_tokens=50, input_tokens=500, output_tokens=10,
                                cache_creation_tokens=0, cache_read_tokens=0, retrieved=0))
    for i in range(10):
        append(tmp_path, Record(request_key=f"h{i}", holdout=True, transforms=[],
                                est_saved_tokens=0, input_tokens=1000, output_tokens=10,
                                cache_creation_tokens=0, cache_read_tokens=0, retrieved=0))
    s = summarize(tmp_path)
    assert s["n_treated"] == 10
    assert s["n_holdout"] == 10
    assert s["mean_input_treated"] == 500
    assert s["mean_input_holdout"] == 1000
    assert s["measured_saving_frac"] == 0.5


def test_summarize_reports_no_data_rather_than_a_zero(tmp_path):
    assert summarize(tmp_path)["n_treated"] == 0
    assert summarize(tmp_path)["measured_saving_frac"] is None


def test_summarize_survives_a_corrupt_line(tmp_path):
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "requests.jsonl").write_text("{not json\n")
    assert summarize(tmp_path)["skipped"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_meter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy.meter'`

- [ ] **Step 3: Write the implementation**

```python
# src/memo/proxy/meter.py
"""Per-request measurement against a real control arm.

memo's existing token meter reads `output_tokens` alone, which is why it cannot
see its own input cost or its effect on the prompt cache. The proxy sits where
the provider's own `usage` is visible, so this module records all four counters
and compares treated requests against an uncompressed holdout.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEDGER_SCHEMA = "memo.proxy.requests.v1"

_log = logging.getLogger(__name__)
_HOLDOUT_BUCKETS = 10_000


@dataclass
class Record:
    request_key: str
    holdout: bool
    transforms: list[str] = field(default_factory=list)
    est_saved_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    retrieved: int = 0


def is_holdout(request_key: str, frac: float) -> bool:
    """Stable, unbiased assignment: the same request is always on the same arm."""
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    digest = hashlib.sha256(request_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _HOLDOUT_BUCKETS
    return bucket < int(frac * _HOLDOUT_BUCKETS)


def usage_from_response(body: dict) -> dict[str, int]:
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def _int(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) else 0

    return {
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cache_creation_tokens": _int("cache_creation_input_tokens"),
        "cache_read_tokens": _int("cache_read_input_tokens"),
    }


def ledger_path(state_dir: Path) -> Path:
    return Path(state_dir) / "proxy" / "requests.jsonl"


def append(state_dir: Path, record: Record) -> None:
    """Append one row. A measurement failure never propagates to a request."""
    path = ledger_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"schema": LEDGER_SCHEMA, **asdict(record)}
        with path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        _log.warning("proxy: could not append measurement row")


def summarize(state_dir: Path) -> dict:
    """Treated vs holdout on real provider counters. None means 'no data yet'."""
    treated: list[dict] = []
    holdout: list[dict] = []
    skipped = 0
    path = ledger_path(state_dir)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                skipped += 1
                continue
            (holdout if row.get("holdout") else treated).append(row)

    def _mean(rows: list[dict], key: str) -> float | None:
        if not rows:
            return None
        return sum(int(r.get(key) or 0) for r in rows) / len(rows)

    mean_t = _mean(treated, "input_tokens")
    mean_h = _mean(holdout, "input_tokens")
    saving = None
    if mean_t is not None and mean_h not in (None, 0):
        saving = round((mean_h - mean_t) / mean_h, 6)

    by_transform: dict[str, int] = {}
    for row in treated:
        for name in row.get("transforms") or []:
            by_transform[name] = by_transform.get(name, 0) + 1

    return {
        "n_treated": len(treated),
        "n_holdout": len(holdout),
        "mean_input_treated": mean_t,
        "mean_input_holdout": mean_h,
        "measured_saving_frac": saving,
        "by_transform": by_transform,
        "retrieved": sum(int(r.get("retrieved") or 0) for r in treated),
        "skipped": skipped,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_meter.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/meter.py tests/test_proxy_meter.py
git commit -m "feat(proxy): measure all four usage counters against a real holdout"
```

---

### Task 6: The transform interface and planner

**Files:**
- Create: `src/memo/proxy/plan.py`
- Test: `tests/test_proxy_plan.py`

**Interfaces:**
- Consumes: `memo.proxy.zones.Zones`, `split`, `prefix_fingerprint`
- Produces:
  - `Zone` string literals `"prefix"` and `"live"`
  - `Transform` protocol: `name: str`, `zone: str`, `enabled() -> bool`, `apply(zones: Zones, ctx: Context) -> int` (returns estimated tokens saved, mutates `zones` in place)
  - `Context` dataclass: `state_dir: Path`, `session_key: str`, `project: str | None`
  - `TransformPlan` dataclass: `applied: list[str]`, `est_saved_tokens: int`
  - `apply_all(zones, ctx, transforms) -> TransformPlan`
  - `REGISTRY: list[Transform]` — populated by later tasks

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_plan.py
from pathlib import Path

from memo.proxy.plan import Context, apply_all
from memo.proxy.zones import split


class _Good:
    name = "good"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx) -> int:
        zones.live_messages.clear()
        return 42


class _Boom:
    name = "boom"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx) -> int:
        raise RuntimeError("transform exploded")


class _Off:
    name = "off"
    zone = "live"

    def enabled(self) -> bool:
        return False

    def apply(self, zones, ctx) -> int:
        raise AssertionError("must not run")


def _ctx(tmp_path: Path) -> Context:
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def test_applied_transform_is_reported_with_its_saving(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Good()])
    assert result.applied == ["good"]
    assert result.est_saved_tokens == 42


def test_a_raising_transform_is_skipped_not_propagated(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Boom(), _Good()])
    assert result.applied == ["good"]
    assert "boom" not in result.applied


def test_a_disabled_transform_never_runs(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Off()])
    assert result.applied == []
    assert result.est_saved_tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy.plan'`

- [ ] **Step 3: Write the implementation**

```python
# src/memo/proxy/plan.py
"""Composes transforms over a split payload.

`plan` knows nothing about what any transform does — only its zone, whether it
is enabled, and that it must never be allowed to fail a request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

ZONE_PREFIX = "prefix"
ZONE_LIVE = "live"


@dataclass
class Context:
    state_dir: Path
    session_key: str
    project: str | None = None


@runtime_checkable
class Transform(Protocol):
    name: str
    zone: str

    def enabled(self) -> bool: ...

    def apply(self, zones: Zones, ctx: Context) -> int: ...


@dataclass
class TransformPlan:
    applied: list[str] = field(default_factory=list)
    est_saved_tokens: int = 0


REGISTRY: list[Transform] = []


def apply_all(zones: Zones, ctx: Context, transforms: list[Transform]) -> TransformPlan:
    """Run every enabled transform. One that raises is skipped, never fatal."""
    plan = TransformPlan()
    for transform in transforms:
        try:
            if not transform.enabled():
                continue
            saved = transform.apply(zones, ctx)
        except Exception:
            _log.warning("proxy: transform %s failed; skipped", transform.name)
            continue
        plan.applied.append(transform.name)
        plan.est_saved_tokens += int(saved or 0)
    return plan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_plan.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/plan.py tests/test_proxy_plan.py
git commit -m "feat(proxy): transform protocol and fail-open planner"
```

---

### Task 7: The proxy server

**Files:**
- Create: `src/memo/proxy/server.py`
- Modify: `pyproject.toml:114-117` (add `httpx`, `pillow` to the `[http]` extra)
- Test: `tests/test_proxy_server.py`

**Interfaces:**
- Consumes: `memo.proxy.zones.split`, `prefix_fingerprint`; `memo.proxy.plan.apply_all`, `Context`, `REGISTRY`; `memo.proxy.meter.Record`, `append`, `is_holdout`, `usage_from_response`
- Produces:
  - `HOP_BY_HOP: frozenset[str]`
  - `rewrite_body(raw: bytes, ctx: Context, transforms: list) -> tuple[bytes, TransformPlan]`
  - `forward_headers(headers: dict[str, str]) -> dict[str, str]`
  - `build_app(upstream: str = "https://api.anthropic.com") -> "FastAPI"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_server.py
import json

import pytest

from memo.proxy.plan import Context
from memo.proxy.server import forward_headers, rewrite_body


class _Clear:
    name = "clear"
    zone = "live"

    def enabled(self):
        return True

    def apply(self, zones, ctx):
        zones.live_messages.clear()
        return 7


class _Boom:
    name = "boom"
    zone = "live"

    def enabled(self):
        return True

    def apply(self, zones, ctx):
        raise RuntimeError("nope")


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s", project=None)


def test_anthropic_beta_is_forwarded_verbatim():
    out = forward_headers({"anthropic-beta": "oauth-2025-04-20,foo", "host": "x"})
    assert out["anthropic-beta"] == "oauth-2025-04-20,foo"


def test_auth_headers_are_forwarded():
    out = forward_headers({"authorization": "Bearer tok", "x-api-key": "k"})
    assert out["authorization"] == "Bearer tok"
    assert out["x-api-key"] == "k"


def test_hop_by_hop_headers_are_dropped():
    out = forward_headers({"host": "127.0.0.1:8768", "content-length": "12",
                           "connection": "keep-alive", "x-api-key": "k"})
    assert "host" not in out
    assert "content-length" not in out
    assert "connection" not in out


def test_rewrite_applies_a_transform(tmp_path):
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Clear()])
    assert json.loads(out)["messages"] == []
    assert plan.applied == ["clear"]


def test_a_failing_transform_leaves_the_body_untouched(tmp_path):
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Boom()])
    assert json.loads(out)["messages"] == [{"role": "user", "content": "hi"}]
    assert plan.applied == []


def test_a_non_json_body_is_forwarded_byte_identical(tmp_path):
    raw = b"not json at all"
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Clear()])
    assert out == raw
    assert plan.applied == []


def test_no_header_value_reaches_the_log(tmp_path, caplog):
    with caplog.at_level("DEBUG"):
        forward_headers({"x-api-key": "SUPERSECRET", "authorization": "Bearer SECRET2"})
    assert "SUPERSECRET" not in caplog.text
    assert "SECRET2" not in caplog.text


@pytest.mark.asyncio
async def test_streams_reach_the_client_incrementally(tmp_path):
    """The byte watchdog aborts after 180s of silence; buffering would trip it."""
    from memo.proxy.server import _relay_chunks

    async def source():
        for chunk in (b"event: a\n", b"data: 1\n\n", b"event: b\n"):
            yield chunk

    seen = [c async for c in _relay_chunks(source())]
    assert seen == [b"event: a\n", b"data: 1\n\n", b"event: b\n"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy.server'`

- [ ] **Step 3: Add the runtime dependencies**

In `pyproject.toml`, replace the `http` extra at line 114:

```toml
http = [
    "fastapi>=0.115",
    "uvicorn>=0.34",
    # Upstream client for `memo proxy serve`. Streaming-capable; the proxy must
    # never buffer a response body or Claude Code's byte watchdog aborts it.
    "httpx>=0.27",
    # Pixel transform only. Absent => that transform no-ops, nothing else breaks.
    "pillow>=11.0",
]
```

- [ ] **Step 4: Write the implementation**

```python
# src/memo/proxy/server.py
"""The proxy itself. Import-safe without the [http] extra.

Contract with Claude Code, from its gateway documentation:
  * `anthropic-beta` must be forwarded verbatim or subscription auth breaks.
  * The response body must never be buffered: the byte-level watchdog aborts a
    stream after 180s of silence on the direct API.
  * Any failure forwards the original body rather than failing the request.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from memo.proxy.plan import REGISTRY, Context, TransformPlan, apply_all
from memo.proxy.zones import split

_log = logging.getLogger(__name__)

UPSTREAM_DEFAULT = "https://api.anthropic.com"

# Dropped because they describe *this* connection, not the upstream one.
HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
    }
)


def forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Everything the client sent, minus hop-by-hop. Never logged."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def rewrite_body(
    raw: bytes, ctx: Context, transforms: list[Any] | None = None
) -> tuple[bytes, TransformPlan]:
    """Apply transforms to a Messages payload. Returns the original on any doubt."""
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except Exception:
        return raw, TransformPlan()

    zones = split(payload)
    plan = apply_all(zones, ctx, transforms if transforms is not None else REGISTRY)
    if not plan.applied:
        return raw, plan
    try:
        return json.dumps(
            zones.to_payload(payload), ensure_ascii=False
        ).encode("utf-8"), plan
    except Exception:
        _log.warning("proxy: could not re-encode rewritten payload; forwarding original")
        return raw, TransformPlan()


async def _relay_chunks(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Pass bytes straight through. Deliberately does no accumulation."""
    async for chunk in source:
        yield chunk


def sniff_usage(chunk: bytes, into: dict[str, int]) -> None:
    """Pick provider usage counters out of a passing SSE chunk.

    Streaming responses carry `usage` on `message_start` and again on
    `message_delta`. We must not buffer the body, so each chunk is scanned as it
    goes by and the counters are merged with max() — later events carry the
    final totals. A malformed chunk is ignored.
    """
    if b'"usage"' not in chunk:
        return
    for line in chunk.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        try:
            event = json.loads(line[6:])
        except (json.JSONDecodeError, ValueError):
            continue
        source = event.get("usage")
        if not isinstance(source, dict):
            message = event.get("message")
            source = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(source, dict):
            continue
        from memo.proxy.meter import usage_from_response

        for key, value in usage_from_response({"usage": source}).items():
            into[key] = max(into.get(key, 0), value)


def build_app(upstream: str = UPSTREAM_DEFAULT) -> Any:
    """Construct the ASGI app. Imports the [http] extra lazily."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from memo.config import Config
    from memo.flags import flag_bool, flag_float
    from memo.proxy import meter

    app = FastAPI(title="memo proxy", docs_url=None, redoc_url=None)
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(600.0))

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        raw = await request.body()
        headers = forward_headers(dict(request.headers))
        cfg = Config.from_env()
        state_dir = cfg.state_dir

        request_key = _request_key(raw)
        holdout = meter.is_holdout(request_key, flag_float("MEMO_PROXY_HOLDOUT_FRAC") or 0.0)

        plan = TransformPlan()
        body = raw
        if not holdout and flag_bool("MEMO_PROXY_ENABLED"):
            ctx = Context(state_dir=state_dir, session_key=request_key, project=None)
            body, plan = rewrite_body(raw, ctx)

        upstream_req = client.build_request(
            "POST", "/v1/messages", content=body, headers=headers
        )
        response = await client.send(upstream_req, stream=True)

        captured: dict[str, int] = {}

        async def _body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _relay_chunks(response.aiter_raw()):
                    sniff_usage(chunk, captured)
                    yield chunk
            finally:
                await response.aclose()
                usage = meter.usage_from_response({})
                usage.update(captured)
                meter.append(
                    state_dir,
                    meter.Record(
                        request_key=request_key,
                        holdout=holdout,
                        transforms=plan.applied,
                        est_saved_tokens=plan.est_saved_tokens,
                        **usage,
                    ),
                )

        return StreamingResponse(
            _body(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-encoding")
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _request_key(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()[:32]
```

- [ ] **Step 5: Add the usage-sniffing test**

```python
def test_usage_is_sniffed_out_of_a_streaming_response():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(
        b'event: message_start\ndata: {"message":{"usage":{"input_tokens":100}}}\n\n',
        captured,
    )
    sniff_usage(
        b'event: message_delta\ndata: {"usage":{"output_tokens":42}}\n\n', captured
    )
    assert captured["input_tokens"] == 100
    assert captured["output_tokens"] == 42


def test_sniffing_a_malformed_chunk_does_not_raise():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(b'data: {"usage": not json\n', captured)
    assert captured == {}
```

Add both to `tests/test_proxy_server.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv sync --extra dev --extra http && uv run --no-sync pytest tests/test_proxy_server.py -v`
Expected: PASS, 10 tests

- [ ] **Step 7: Commit**

```bash
git add src/memo/proxy/server.py tests/test_proxy_server.py pyproject.toml
git commit -m "feat(proxy): streaming passthrough server with header fidelity"
```

---

### Task 8: CLI, LaunchAgent, and doctor check

**Files:**
- Create: `src/memo/cli_proxy.py`
- Modify: `src/memo/cli.py` (register `proxy_group`), `src/memo/ops_launchd.py:10` (add proxy label + renderer + install/uninstall), `src/memo/cli_doctor.py` (proxy health row)
- Test: `tests/test_proxy_cli.py`

**Interfaces:**
- Consumes: `memo.proxy.server.build_app`; `memo.ops_launchd.render_chat_plist` as the pattern to mirror
- Produces:
  - `memo.cli_proxy.proxy_group` — Click group with `serve`, `off`, `on`, `status`
  - `memo.ops_launchd.PROXY_LABEL = "com.memo.proxy"`, `render_proxy_plist(memo_bin, home, *, port=8768) -> str`, `install_proxy(memo_bin, home, *, port=8768) -> Path`, `uninstall_proxy(home) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_cli.py
from click.testing import CliRunner

from memo.cli_proxy import proxy_group
from memo.ops_launchd import PROXY_LABEL, render_proxy_plist


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_plist_uses_the_proxy_label_and_port():
    xml = render_proxy_plist("/usr/local/bin/memo", "/Users/x", port=8768)
    assert f"<string>{PROXY_LABEL}</string>" in xml
    assert "<string>8768</string>" in xml
    assert "<key>KeepAlive</key>" in xml


def test_plist_never_embeds_an_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-appear")
    xml = render_proxy_plist("/usr/local/bin/memo", "/Users/x")
    assert "sk-should-not-appear" not in xml


def test_status_reports_not_running_without_a_daemon(tmp_path):
    result = CliRunner().invoke(proxy_group, ["status"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_off_writes_the_markdown_config_not_just_the_env(tmp_path):
    runner = CliRunner()
    result = runner.invoke(proxy_group, ["off"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "proxy.enabled" in result.output or "MEMO_PROXY_ENABLED" in result.output


def test_serve_without_the_http_extra_is_a_clean_cli_error(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("fastapi", "uvicorn", "httpx"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(proxy_group, ["serve"], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "pip install" in result.output or "extra" in result.output.lower()
    assert "Traceback" not in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.cli_proxy'`

- [ ] **Step 3: Add the LaunchAgent renderer**

In `src/memo/ops_launchd.py`, beside `CHAT_LABEL`, mirroring `render_chat_plist` exactly (including the `MEMO_*` env forwarding, since launchd agents inherit no shell environment):

```python
PROXY_LABEL = "com.memo.proxy"


def render_proxy_plist(memo_bin: str, home: str, *, port: int = 8768) -> str:
    args = [memo_bin, "proxy", "serve", "--host", "127.0.0.1", "--port", str(port)]
    args_xml = "\n".join(f"      <string>{escape(a)}</string>" for a in args)
    log = escape(f"{home}/Library/Logs/memo/proxy.log")
    path_env = escape(f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    # MEMO_* only — never ANTHROPIC_API_KEY or any other credential.
    memo_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MEMO_")}
    memo_env_xml = "".join(
        f"      <key>{escape(k)}</key>\n      <string>{escape(v)}</string>\n"
        for k, v in memo_env.items()
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>{PROXY_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>{path_env}</string>
{memo_env_xml}    </dict>
  </dict>
</plist>
"""
```

Then add `install_proxy` and `uninstall_proxy` as byte-for-byte analogues of `install_chat` (line 79) and `uninstall_chat` (line 101), substituting `PROXY_LABEL` and `render_proxy_plist`, and raising a `click.ClickException` naming the conflicting process when the port is already listening.

- [ ] **Step 4: Write the CLI group**

```python
# src/memo/cli_proxy.py
"""`memo proxy` — the context-compression proxy.

Point Claude Code at it with ANTHROPIC_BASE_URL. Put that variable in the `env`
block of ~/.claude/settings.json, not a shell export: the background-agent
supervisor inherits only the environment of whichever shell cold-started it, so
an export reaches background sessions unpredictably.
"""

from __future__ import annotations

import socket

import click

from memo.cli_common import console


@click.group(name="proxy")
def proxy_group() -> None:
    """Context-compression proxy commands."""


@proxy_group.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8768, show_default=True, type=int)
@click.option("--upstream", default="https://api.anthropic.com", show_default=True)
def proxy_serve(host: str, port: int, upstream: str) -> None:
    """Run the proxy in the foreground."""
    try:
        import uvicorn

        from memo.proxy.server import build_app
    except ImportError as exc:  # missing [http] extra — a clean CLI error
        raise click.ClickException(
            "memo proxy needs the [http] extra: pip install 'mlx-memo[http]'"
        ) from exc
    uvicorn.run(build_app(upstream), host=host, port=port, log_level="warning")


@proxy_group.command("off")
def proxy_off() -> None:
    """Turn payload rewriting off everywhere (daemon included)."""
    from memo.config_md import set_value

    set_value("proxy.enabled", "false")
    console.print("proxy.enabled = false  (rewriting off; still forwards and measures)")


@proxy_group.command("on")
def proxy_on() -> None:
    """Turn payload rewriting back on."""
    from memo.config_md import set_value

    set_value("proxy.enabled", "true")
    console.print("proxy.enabled = true")


@proxy_group.command("status")
@click.option("--port", default=8768, show_default=True, type=int)
def proxy_status(port: int) -> None:
    """Report whether the proxy is listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        listening = sock.connect_ex(("127.0.0.1", port)) == 0
    console.print(
        f"proxy on 127.0.0.1:{port}: {'listening' if listening else 'not running'}"
    )
```

Register it in `src/memo/cli.py` beside the other groups:

```python
from memo.cli_proxy import proxy_group
cli.add_command(proxy_group)
```

Confirm the exact helper name `set_value` against `src/memo/config_md.py` before writing; if the module exposes a different setter, use that one — the requirement is that `off` writes the Markdown config so the setting reaches the LaunchAgent, not just the current terminal.

- [ ] **Step 5: Add the doctor check**

In `src/memo/cli_doctor.py`, add a row that reports: whether `com.memo.proxy` is loaded, whether the port is listening, and — the failure mode that otherwise looks like a dead network — a **warning when `ANTHROPIC_BASE_URL` is set to a proxy address that is not listening**.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_cli.py -v`
Expected: PASS, 5 tests

- [ ] **Step 7: Commit**

```bash
git add src/memo/cli_proxy.py src/memo/cli.py src/memo/ops_launchd.py src/memo/cli_doctor.py tests/test_proxy_cli.py
git commit -m "feat(proxy): memo proxy CLI, LaunchAgent, and doctor check"
```

- [ ] **Step 8: Install and record real traffic before Phase B**

```bash
uv run --no-sync memo ops install proxy --port 8768
# then add to the `env` block of ~/.claude/settings.json:
#   "ANTHROPIC_BASE_URL": "http://127.0.0.1:8768"
```

Use Claude Code normally for at least one full session, then confirm the ledger filled:

```bash
uv run --no-sync python -c "
from pathlib import Path
from memo.config import Config
from memo.proxy.meter import summarize
print(summarize(Config.from_env().state_dir))"
```

Expected: `n_treated` and `n_holdout` both non-zero. **Phase A is done when this prints real numbers.** Every saving claimed in Phase B is measured against this baseline.

---

## Phase B — Transforms (Tasks 9–14)

Each transform is one task, added to `REGISTRY`, measured against the holdout Phase A established.

---

### Task 9: Tool-schema pruning

The measured 11,640 tokens per request. Prefix zone, so session-stability is mandatory.

**Files:**
- Create: `src/memo/proxy/transforms/__init__.py`, `src/memo/proxy/transforms/toolschemas.py`
- Modify: `src/memo/proxy/plan.py` (append to `REGISTRY`)
- Test: `tests/test_proxy_toolschemas.py`

**Interfaces:**
- Consumes: `memo.proxy.plan.Context`, `ZONE_PREFIX`; `memo.proxy.zones.Zones`; `memo.mcp_budget.est_tokens`
- Produces: `ToolSchemas` class with `name = "toolschemas"`, `zone = ZONE_PREFIX`; module function `recent_tool_names(state_dir: Path, window: int) -> set[str]`; `DOCS_TOOL_NAME = "memo_tool_docs"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_toolschemas.py
from memo.proxy.plan import Context
from memo.proxy.toolschemas_testkit import make_zones  # defined in step 3
from memo.proxy.transforms.toolschemas import DOCS_TOOL_NAME, ToolSchemas


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def test_unused_tools_are_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"memo_search"},
    )
    zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
    saved = ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert "memo_search" in names
    assert "memo_graph" not in names
    assert saved > 0


def test_the_docs_tool_is_always_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["memo_search", DOCS_TOOL_NAME])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    assert DOCS_TOOL_NAME in {t["name"] for t in zones.tools}


def test_non_memo_tools_are_never_pruned(tmp_path, monkeypatch):
    """Pruning another server's schema would break tools memo does not own."""
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["Read", "Bash", "memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert {"Read", "Bash"} <= names
    assert "memo_rename" not in names


def test_pruning_is_stable_across_turns_in_a_session(tmp_path, monkeypatch):
    """A prefix that changes every turn costs a re-cache and inverts the saving."""
    from memo.proxy.zones import prefix_fingerprint

    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"memo_search"},
    )
    fingerprints = set()
    for _ in range(5):
        zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
        ToolSchemas().apply(zones, _ctx(tmp_path))
        fingerprints.add(prefix_fingerprint(zones))
    assert len(fingerprints) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_toolschemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy.transforms'`

- [ ] **Step 3: Write the test helper**

```python
# src/memo/proxy/toolschemas_testkit.py
"""Fixture builder shared by the tool-schema tests."""

from __future__ import annotations

from memo.proxy.zones import Zones


def make_zones(tool_names: list[str]) -> Zones:
    return Zones(
        tools=[
            {
                "name": name,
                "description": f"description of {name} " * 10,
                "input_schema": {"type": "object", "properties": {}},
            }
            for name in tool_names
        ],
        live_messages=[{"role": "user", "content": "hi"}],
    )
```

- [ ] **Step 4: Write the implementation**

```python
# src/memo/proxy/transforms/__init__.py
"""Payload transforms. Each is independently testable without a running proxy."""
```

```python
# src/memo/proxy/transforms/toolschemas.py
"""Prune memo's own MCP tool schemas to the ones this project actually uses.

Measured 2026-08-18: 41 memo tools cost 46,562 B ~= 11,640 tokens in *every*
request, paid whether or not a tool is called. Only memo's own tools are pruned
— pruning another server's schema would break a tool memo does not own.

The retained set is derived from usage history and is stable for a whole
session, so the cached prefix changes once rather than every turn.
"""

from __future__ import annotations

import json
from pathlib import Path

from memo.flags import flag_bool, flag_int
from memo.mcp_budget import est_tokens
from memo.proxy.plan import ZONE_PREFIX, Context
from memo.proxy.zones import Zones

DOCS_TOOL_NAME = "memo_tool_docs"
_OWNED_PREFIX = "memo_"
# Kept regardless of usage: without these the model cannot reach memo at all.
_ALWAYS_KEEP = frozenset({DOCS_TOOL_NAME, "memo_search", "memo_save"})


def recent_tool_names(state_dir: Path, window: int) -> set[str]:
    """memo tool names called in the last `window` sessions, from the MCP log."""
    path = Path(state_dir) / "recall.log"
    if not path.is_file():
        return set()
    names: set[str] = set()
    sessions: list[str] = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        session = str(row.get("session_id") or "")
        if session and session not in sessions:
            sessions.append(session)
            if len(sessions) > window:
                break
        tool = row.get("tool")
        if isinstance(tool, str) and tool.startswith(_OWNED_PREFIX):
            names.add(tool)
    return names


class ToolSchemas:
    name = "toolschemas"
    zone = ZONE_PREFIX

    def enabled(self) -> bool:
        return bool(flag_bool("MEMO_PROXY_TOOL_SCHEMAS"))

    def apply(self, zones: Zones, ctx: Context) -> int:
        if not zones.tools:
            return 0
        window = flag_int("MEMO_PROXY_TOOL_WINDOW_SESSIONS") or 20
        keep = recent_tool_names(ctx.state_dir, window) | _ALWAYS_KEEP
        before = est_tokens(json.dumps(zones.tools, separators=(",", ":")))
        kept = [
            tool
            for tool in zones.tools
            if not str(tool.get("name", "")).startswith(_OWNED_PREFIX)
            or tool.get("name") in keep
        ]
        if len(kept) == len(zones.tools):
            return 0
        zones.tools[:] = kept
        after = est_tokens(json.dumps(zones.tools, separators=(",", ":")))
        return max(0, before - after)
```

- [ ] **Step 5: Register the transform**

At the bottom of `src/memo/proxy/plan.py`:

```python
def _load_registry() -> None:
    from memo.proxy.transforms.toolschemas import ToolSchemas

    REGISTRY.append(ToolSchemas())


_load_registry()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_toolschemas.py tests/test_proxy_plan.py -v`
Expected: PASS

- [ ] **Step 7: Add the `memo_tool_docs` MCP tool**

Create `src/memo/server_tool_docs.py` exporting `register(server, memory)` that adds `memo_tool_docs(name: str) -> dict` returning the full schema of a pruned tool, and add the one `register` call in `src/memo/server.py`'s `build_server()`. Without it, a pruned tool is unreachable rather than merely hidden.

- [ ] **Step 8: Commit**

```bash
git add src/memo/proxy/transforms/ src/memo/proxy/toolschemas_testkit.py src/memo/proxy/plan.py src/memo/server_tool_docs.py src/memo/server.py tests/test_proxy_toolschemas.py
git commit -m "feat(proxy): prune unused memo tool schemas from the prefix"
```

- [ ] **Step 9: Measure before continuing**

Run a session, then compare `summarize()` output against the Phase A baseline. Record the observed `measured_saving_frac` in the CHANGELOG entry for this transform. If it is not positive, **stop and report** rather than stacking another transform on an unproven one.

---

### Task 10: Tool-result filtering

The 92% plane. Live zone, so no cache risk.

**Files:**
- Create: `src/memo/proxy/transforms/toolresults.py`, `src/memo/proxy/filters/git-status.yaml`, `src/memo/proxy/filters/pytest.yaml`, `src/memo/proxy/filters/npm-install.yaml`, `NOTICE`
- Modify: `src/memo/proxy/plan.py` (`_load_registry`), `pyproject.toml` (package the `filters/*.yaml` data files)
- Test: `tests/test_proxy_toolresults.py`

**Interfaces:**
- Consumes: `memo.proxy.ccr.stash`, `marker`; `memo.mcp_budget.est_tokens`
- Produces: `ToolResults` class (`name = "toolresults"`, `zone = ZONE_LIVE`); `load_filters(dir: Path) -> list[Filter]`; `Filter` dataclass with `name`, `match_command`, `match_subcommand`, `pipeline`; `apply_pipeline(text: str, actions: list[dict]) -> str`; `generic_fallback(text: str, max_chars: int) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_toolresults.py
from memo.proxy.transforms.toolresults import (
    apply_pipeline,
    generic_fallback,
    load_filters,
)


def test_keep_lines_retains_only_matching_lines():
    out = apply_pipeline("a: ok\nb: FAIL\nc: ok\n", [{"action": "keep_lines", "pattern": "FAIL"}])
    assert out == "b: FAIL"


def test_remove_lines_drops_matching_lines():
    out = apply_pipeline("keep\nnoise\n", [{"action": "remove_lines", "pattern": "noise"}])
    assert out == "keep"


def test_aggregate_counts_instead_of_listing():
    text = "\n".join("PASS test_%d" % i for i in range(500))
    out = apply_pipeline(text, [{"action": "aggregate", "pattern": "^PASS", "label": "passed"}])
    assert out == "500 passed"


def test_head_and_tail_compose():
    text = "\n".join(str(i) for i in range(100))
    out = apply_pipeline(text, [{"action": "head", "n": 2}])
    assert out == "0\n1"


def test_truncate_lines_caps_line_width():
    out = apply_pipeline("x" * 200, [{"action": "truncate_lines", "max": 10}])
    assert len(out) <= 13  # 10 chars plus an ellipsis marker


def test_an_unknown_action_is_a_no_op_not_a_crash():
    assert apply_pipeline("text", [{"action": "does_not_exist"}]) == "text"


def test_generic_fallback_keeps_head_and_tail_and_says_what_it_dropped():
    text = "\n".join(str(i) for i in range(1000))
    out = generic_fallback(text, max_chars=200)
    assert out.startswith("0\n1")
    assert "999" in out
    assert "elided" in out
    assert len(out) < len(text)


def test_short_output_passes_through_the_fallback_untouched():
    assert generic_fallback("short", max_chars=200) == "short"


def test_filters_load_from_yaml(tmp_path):
    (tmp_path / "f.yaml").write_text(
        "name: demo\n"
        "match:\n"
        "  command: git\n"
        "  subcommand: status\n"
        "pipeline:\n"
        "  - action: head\n"
        "    n: 5\n"
    )
    filters = load_filters(tmp_path)
    assert filters[0].name == "demo"
    assert filters[0].match_command == "git"


def test_a_malformed_filter_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "bad.yaml").write_text("{{{not yaml")
    assert load_filters(tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_toolresults.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memo.proxy.transforms.toolresults'`

- [ ] **Step 3: Write the implementation**

Port the pipeline-action vocabulary from snip (**MIT** — the file must carry snip's copyright and license header, and `NOTICE` must record it). Implement `keep_lines`, `remove_lines`, `head`, `tail`, `dedup`, `truncate_lines`, `aggregate`, `json_extract`, and `format_template`, each a pure `str -> str` function dispatched from a dict so an unknown action degrades to identity. `generic_fallback` keeps the first and last `max_chars // 2` characters with an `[N chars elided]` marker between them, and returns the input unchanged when it is already short enough.

`ToolResults.apply` walks `zones.live_messages`, finds `tool_result` content blocks, matches the originating command against the loaded filters, applies the pipeline (or the fallback when nothing matches), stashes the original via `ccr.stash` and appends `ccr.marker` when the cut is lossy, and returns `est_tokens(before) - est_tokens(after)`. When `ccr.stash` returns `""` the block is left untouched — no cut without a recovery path.

- [ ] **Step 4: Write three starter filters**

`git-status.yaml`, `pytest.yaml`, and `npm-install.yaml`. Example:

```yaml
# src/memo/proxy/filters/pytest.yaml
name: pytest
match:
  command: pytest
pipeline:
  - action: keep_lines
    pattern: "(FAILED|ERROR|error:|assert|=+ .* =+)"
  - action: truncate_lines
    max: 200
  - action: tail
    n: 40
```

- [ ] **Step 5: Package the YAML files**

Add to `pyproject.toml` so the filters ship in the wheel:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/memo/proxy/filters" = "memo/proxy/filters"
```

Verify the existing build config first — if the project already includes package data by another mechanism, follow that instead of adding a second one.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_toolresults.py -v`
Expected: PASS, 10 tests

- [ ] **Step 7: Commit**

```bash
git add src/memo/proxy/transforms/toolresults.py src/memo/proxy/filters/ src/memo/proxy/plan.py NOTICE pyproject.toml tests/test_proxy_toolresults.py
git commit -m "feat(proxy): declarative tool-result filters (pipeline ported from snip, MIT)"
```

- [ ] **Step 8: Measure before continuing**

Same gate as Task 9 Step 9: run a session, compare against the baseline, record the number, stop if it is not positive.

---

### Task 11: JSON crush

**Files:**
- Create: `src/memo/proxy/transforms/jsoncrush.py`
- Modify: `src/memo/proxy/plan.py`
- Test: `tests/test_proxy_jsoncrush.py`

**Interfaces:**
- Consumes: `memo.capture_core.maybe_crush_json_capture(content: str, context: str, config: Config) -> tuple[str, str | None]`
- Produces: `JsonCrush` class (`name = "jsoncrush"`, `zone = ZONE_LIVE`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_jsoncrush.py
import json

from memo.proxy.plan import Context
from memo.proxy.transforms.jsoncrush import JsonCrush
from memo.proxy.zones import Zones


def _zones(text: str) -> Zones:
    return Zones(live_messages=[{
        "role": "user",
        "content": [{"type": "tool_result", "content": text}],
    }])


def test_a_large_json_array_is_crushed(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved > 0
    assert len(zones.live_messages[0]["content"][0]["content"]) < len(big)


def test_non_json_content_is_left_alone(tmp_path):
    zones = _zones("just some prose, definitely not json")
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved == 0
    assert zones.live_messages[0]["content"][0]["content"] == "just some prose, definitely not json"


def test_a_small_array_is_not_worth_crushing(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    small = json.dumps([{"id": 1}, {"id": 2}])
    zones = _zones(small)
    assert JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s")) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_jsoncrush.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Wrap `maybe_crush_json_capture` over each `tool_result` block in the live zone, building the `Config` from `Config.from_env()`. The crusher already writes originals to the crush cache and returns a hash, so `ccr` is not needed here — `memo retrieve` already recovers these. Return `est_tokens(before) - est_tokens(after)`. Set `MEMO_CRUSHER_ENABLED` in the transform's own process environment only if `MEMO_PROXY_JSONCRUSH` is on and the capture flag is unset, so the proxy does not silently change ingest behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_jsoncrush.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/transforms/jsoncrush.py src/memo/proxy/plan.py tests/test_proxy_jsoncrush.py
git commit -m "feat(proxy): run the measured L1 JSON crusher on tool results"
```

---

### Task 12: Structure map and delta

**Files:**
- Create: `src/memo/proxy/transforms/structmap.py`, `src/memo/proxy/transforms/delta.py`
- Modify: `src/memo/proxy/plan.py`
- Test: `tests/test_proxy_structmap.py`, `tests/test_proxy_delta.py`

**Interfaces:**
- Consumes: `memo.proxy.ccr.stash`, `marker`; `difflib` from stdlib
- Produces: `StructMap` class (`name = "structmap"`, `zone = ZONE_LIVE`); `signatures(source: str, language: str) -> str`; `Delta` class (`name = "delta"`, `zone = ZONE_LIVE`); `seen_files(zones: Zones) -> dict[str, str]`; `diff_against(previous: str, current: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_proxy_structmap.py
from memo.proxy.transforms.structmap import signatures

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
    assert "def alpha(a: int, b: str = \"x\") -> bool:" in out
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
```

```python
# tests/test_proxy_delta.py
from memo.proxy.transforms.delta import diff_against


def test_an_unchanged_reread_collapses_to_a_notice():
    text = "line1\nline2\nline3\n"
    out = diff_against(text, text)
    assert "unchanged" in out.lower()
    assert len(out) < len(text)


def test_a_changed_reread_shows_only_the_diff():
    before = "line1\nline2\nline3\n"
    after = "line1\nCHANGED\nline3\n"
    out = diff_against(before, after)
    assert "CHANGED" in out
    assert "line1" not in out or len(out) < len(after)


def test_a_first_read_with_no_previous_copy_is_untouched():
    assert diff_against("", "fresh content") == "fresh content"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_proxy_structmap.py tests/test_proxy_delta.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

`signatures` uses the stdlib `ast` module for Python (parse, walk, emit `import`/`def`/`class` lines with their decorators and a one-line body elision) and returns the source unchanged for any language it cannot parse — a wrong signature map is worse than no compression. `diff_against` uses `difflib.unified_diff` with `n=1`, returns `"[memo: file unchanged since last read]"` on an exact match, and returns `current` unchanged when `previous` is empty.

`StructMap.apply` and `Delta.apply` scan the live zone for `tool_result` blocks whose originating tool was `Read`, stash the original with `ccr.stash`, and append `ccr.marker`. When `stash` returns `""`, leave the block untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_structmap.py tests/test_proxy_delta.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/transforms/structmap.py src/memo/proxy/transforms/delta.py src/memo/proxy/plan.py tests/test_proxy_structmap.py tests/test_proxy_delta.py
git commit -m "feat(proxy): structure maps and re-read deltas for file reads"
```

---

### Task 13: Pixel mode

**Files:**
- Create: `src/memo/proxy/transforms/pixel.py`
- Modify: `src/memo/proxy/plan.py`
- Test: `tests/test_proxy_pixel.py`

**Interfaces:**
- Consumes: `PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont` (lazily; absent means no-op)
- Produces: `Pixel` class (`name = "pixel"`, `zone = ZONE_LIVE`); `est_image_tokens(width: int, height: int) -> int`; `is_profitable(text: str) -> bool`; `render(text: str) -> bytes | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_pixel.py
import pytest

from memo.proxy.transforms.pixel import est_image_tokens, is_profitable, render


def test_image_token_estimate_follows_the_documented_formula():
    # Anthropic's documented approximation: tokens ~= (w * h) / 750
    assert est_image_tokens(1000, 1000) == pytest.approx(1333, abs=2)


def test_short_text_is_never_profitable_to_render():
    assert not is_profitable("just a short line")


def test_a_large_dense_block_is_profitable():
    assert is_profitable("x" * 200_000)


def test_render_returns_png_bytes_or_none_without_pillow():
    out = render("hello\nworld")
    assert out is None or out[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_of_empty_text_is_none():
    assert render("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_pixel.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`est_image_tokens(w, h)` returns `round(w * h / 750)`. `is_profitable(text)` renders nothing — it computes the page geometry the renderer would use, compares `est_image_tokens` against `est_tokens(text)`, and returns `True` only when the image is under 80% of the text. `render` imports Pillow inside the function and returns `None` on `ImportError`, on empty input, or on any rendering failure. `Pixel.apply` replaces a qualifying `tool_result` block with an `image` content block carrying a base64 `data:image/png` source, after stashing the original with `ccr.stash`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_proxy_pixel.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/memo/proxy/transforms/pixel.py src/memo/proxy/plan.py tests/test_proxy_pixel.py
git commit -m "feat(proxy): pixel mode behind a per-block profitability gate"
```

---

### Task 14: Honest reporting

Closes spec finding 4: two contradicting numbers on one screen.

**Files:**
- Modify: `src/memo/cli_tokens.py`, `src/memo/token_ledger.py`, `src/memo/token_meter.py:57-72`, `src/memo/flags_misc.py:880-910` (delete `MEMO_ROI_TOKENS_PER_GROUNDED`, `_PER_REASK`, `_PER_CONSULT`), `CHANGELOG.md`
- Test: `tests/test_proxy_reporting.py`, and update any existing test that asserts the estimated panel

**Interfaces:**
- Consumes: `memo.proxy.meter.summarize`
- Produces: `memo tokens` reporting one measured number; `memo tokens --by-transform`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy_reporting.py
from click.testing import CliRunner

from memo.cli_tokens import tokens_cmd


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_no_data_says_so_instead_of_printing_a_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no measured data" in result.output.lower()


def test_the_estimated_panel_is_gone(tmp_path):
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert "estimated" not in result.output.lower()


def test_the_roi_constants_are_no_longer_registered():
    from memo.flags import REGISTRY

    assert "MEMO_ROI_TOKENS_PER_GROUNDED" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_CONSULT" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_REASK" not in REGISTRY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proxy_reporting.py -v`
Expected: FAIL — the estimated panel is still printed and the flags are still registered

- [ ] **Step 3: Extend the token meter to all four counters**

In `src/memo/token_meter.py`, change `_assistant_out` (line 57) to return the full usage dict rather than `output_tokens` alone, and carry `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens` through `TurnUsage` and `SessionUsage`. Update `tests/` wherever those dataclasses are constructed.

- [ ] **Step 4: Delete the estimate**

Remove the three `MEMO_ROI_TOKENS_PER_*` specs from `src/memo/flags_misc.py`, delete `_tokens_per_grounded` and `_tokens_per_consult` from `src/memo/token_ledger.py` along with the estimated-savings arithmetic they feed, and delete the "tokens saved (estimated)" panel from `src/memo/cli_tokens.py`. Replace it with a panel driven by `memo.proxy.meter.summarize`: treated vs holdout mean input tokens, the measured saving fraction, sample counts, and the retrieval rate per transform.

- [ ] **Step 5: Add `--by-transform`**

```python
@click.option("--by-transform", is_flag=True, help="Break the measured saving down per transform.")
```

Print one row per transform: name, requests it applied to, and its share of estimated savings, flagging any whose retrieval rate exceeds `MEMO_PROXY_RETRIEVE_ALARM_FRAC` as over-cutting.

- [ ] **Step 6: Run the full suite**

Run: `uv run --no-sync pytest tests/ -m "not slow"`
Expected: PASS. Any test asserting the old estimated panel must be updated, not deleted — the assertion moves to the measured panel.

- [ ] **Step 7: Update the CHANGELOG**

Add an `### Removed` entry recording that the estimated token-savings panel and its three `MEMO_ROI_TOKENS_PER_*` flags are gone, and why: they reported a hardcoded constant beside a measured cost, which read as a savings claim memo could not support.

- [ ] **Step 8: Commit**

```bash
git add src/memo/cli_tokens.py src/memo/token_ledger.py src/memo/token_meter.py src/memo/flags_misc.py CHANGELOG.md tests/test_proxy_reporting.py
git commit -m "refactor(tokens): report one measured number, drop the hardcoded estimate"
```

---

### Task 15: Regression gate and release

**Files:**
- Modify: `CHANGELOG.md`, `README.md`
- Test: full suite plus the two eval gates

**Interfaces:**
- Consumes: everything above
- Produces: a releasable state

- [ ] **Step 1: Run the retrieval regression gate**

Run: `PYTHONPATH=$(git rev-parse --show-toplevel)/src uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --gate`
Expected: PASS. The `PYTHONPATH` prefix is mandatory — without it the gate measures the installed binary instead of this diff, which is how a −22% precision regression shipped green once before.

- [ ] **Step 2: Run the behavior gate**

Run: `uv run --no-sync memo eval behavior --recall-only`
Expected: no regression against the last recorded result.

- [ ] **Step 3: Run mypy and ruff on the new surface**

```bash
uv run --no-sync mypy src/memo/
uv run --no-sync ruff check src/memo/proxy/ src/memo/cli_proxy.py src/memo/flags_proxy.py
uv run --no-sync ruff format src/memo/proxy/ src/memo/cli_proxy.py src/memo/flags_proxy.py
```

- [ ] **Step 4: Run the release gate**

Run: `uv run --no-sync memo release check`
Expected: PASS, including `hook-commands-resolve` — `memo proxy` must be a command the CLI actually registers.

- [ ] **Step 5: Document the surface**

Add a `## Proxy` section to `README.md` covering `memo ops install proxy`, the `ANTHROPIC_BASE_URL` setting in `~/.claude/settings.json` (not a shell export), `memo proxy off` as the revert, and the measured savings from `memo tokens`. State the numbers actually observed, not the ones the design hoped for.

- [ ] **Step 6: Commit and open the PR**

```bash
git add CHANGELOG.md README.md
git commit -m "docs: document the context-compression proxy"
gh pr create --title "feat: context-compression proxy" --body "Implements docs/SPECS/2026-08-18-token-savings-proxy-context-compression-design.md"
```

`master` is branch-protected — land via PR, never a direct push.

---

## Self-review notes

**Spec coverage.** Section 1 (architecture/deploy) → Tasks 7, 8. Section 2 (zones/cache rule) → Task 3, enforced by the stability test in Task 9. Section 3 (modules) → Tasks 3–7, 9–13, one file each. Section 4 (five transforms) → Tasks 9–13. Section 5 (recovery) → Task 4, consumed by Tasks 10, 12, 13. Section 6 (measurement) → Tasks 5, 14. Section 7 (rollout/flags/failure) → Tasks 2, 6, 8. Section 8 (licensing) → Global Constraints and Task 10 Step 3. Section 9 (error handling) → Tasks 4, 5, 6, 7 tests. Section 10 (testing) → every task's test block, with the cache-stability test in Task 9 and the streaming test in Task 7.

**Known gap, deliberate.** The spec's `memo_retrieve` MCP tool is created in Task 9 Step 7 as `memo_tool_docs`' sibling; if `memo retrieve` is not already exposed over MCP, Task 9 Step 7 must add both. Verify against `src/memo/server.py` before implementing.
