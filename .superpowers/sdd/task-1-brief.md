### Task 1: SQLite Resource Hygiene Guard

**Files:**
- Create: `tests/test_sqlite_resource_hygiene.py`
- Modify if needed: `src/memo/store/connection.py`
- Modify if needed: `tests/conftest.py`
- Modify if needed: `tests/test_runtime_isolation.py`
- Modify if needed: `tests/test_resume_episodes.py`

**Interfaces:**
- Consumes: `memo.memory.Memory.close()`, `memo.store.connection._ConnectionMixin.close()`
- Produces: targeted tests proving sqlite connections are closed or failures are reproducible under `ResourceWarning` as error

- [ ] **Step 1: Reproduce the current warning with tracebacks**

Run:

```bash
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Expected before the fix: either FAIL with `ResourceWarning: unclosed database` or PASS if the warning only appears under full-suite interleaving. If it passes, continue with Step 2 anyway so future regressions are guarded.

- [ ] **Step 2: Write a focused sqlite cleanup test file**

Create `tests/test_sqlite_resource_hygiene.py`:

```python
from __future__ import annotations

import gc
import warnings

import pytest

from memo.config import Config
from memo.memory import Memory


def _sqlite_resource_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning)
        and "unclosed database" in str(w.message).lower()
    ]


def test_memory_close_releases_sqlite_connections(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        mem.save(content="sqlite cleanup probe", title="SQLite Cleanup Probe")
        assert mem.search("sqlite cleanup probe", mode="bm25", limit=1)
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []


def test_memory_close_is_idempotent_after_lazy_connections(
    tmp_cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        _ = mem.store.get("missing-id")
        mem.close()
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []
```

- [ ] **Step 3: Run the focused tests**

Run:

```bash
uv run --no-sync pytest tests/test_sqlite_resource_hygiene.py -q -W error::ResourceWarning
```

Expected before implementation: PASS if `Memory.close()` already covers the focused lifecycle; FAIL if the leak is in the core connection lifecycle.

- [ ] **Step 4: If Step 1 failed in runtime/resume tests, apply the smallest cleanup fix**

If `test_mcp_episodes_search_tool` is the failing path, make sure the test always closes memory and forces garbage collection before returning. Patch only the test finalizer block:

```python
    finally:
        mem.close()
        import gc

        gc.collect()
```

If `test_install_slash_claude_proceeds_to_add_when_remove_fails` is the failing path, add a post-test garbage collection finalizer to the existing `_sandbox_home` autouse fixture in `tests/test_runtime_isolation.py`:

```python
@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path_factory):
    # existing setup unchanged
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(shims_mod, "_DEFAULT_BIN_DIR", home / ".memo" / "bin")

    real_write = install_mod.write_mandates_for_clients

    def _sandboxed_write(*args, **kwargs):
        if not kwargs.get("dry_run"):
            kwargs["cwd"] = home
        return real_write(*args, **kwargs)

    monkeypatch.setattr(install_mod, "write_mandates_for_clients", _sandboxed_write)
    yield home

    import gc

    gc.collect()
```

If the traceback points to `_ConnectionHolder`, keep `src/memo/store/connection.py` behavior idempotent and explicit by changing `close()` to clear references before closing, then preserve the existing `suppress(BaseException)` behavior:

```python
class _ConnectionHolder:
    """Close a thread-local connection when its owning thread exits."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn: sqlite3.Connection | None = conn

    def close(self) -> None:
        conn = self.conn
        self.conn = None
        if conn is not None:
            with suppress(BaseException):
                conn.close()
```

- [ ] **Step 5: Re-run warning-as-error checks**

Run:

```bash
uv run --no-sync pytest tests/test_sqlite_resource_hygiene.py -q -W error::ResourceWarning
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Expected: PASS with no `ResourceWarning: unclosed database`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_sqlite_resource_hygiene.py src/memo/store/connection.py tests/conftest.py tests/test_runtime_isolation.py tests/test_resume_episodes.py
git commit -m "test: guard sqlite resource cleanup"
```

Only add files that changed.

---

