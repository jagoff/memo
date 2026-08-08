Status: partially shipped — none of this plan's own deliverables (`tests/conformance/` harness, 10k-memory fixture, `conformance` pytest marker) exist on master; grep confirms `tests/conformance/` is absent. Several of the underlying defects the harness was meant to catch (raw-traceback output paths, `memo links reindex` silent data loss) were fixed ad hoc, without the formal harness, in #209. A complete implementation matching this plan (`tests/conformance/conftest.py` + 6 test modules) exists on local branch `feat/conformance-budget-deadline-admission`, which has no open PR as of 2026-08-07.

# Corpus-Scale Conformance Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest lane that seeds ~10,000 synthetic memories once per session and asserts, per surface, that reported totals are real, output paths fail cleanly, and index rebuilds preserve what they do not own.

**Architecture:** A session-scoped fixture writes N deterministic `.md` files and bulk-`upsert`s matching vectors into an isolated `VecStore`. Tests marked `conformance` run in their own CI lane, deselected from the default lanes. Two further conformance modules (MCP response budget, read-latency degradation) land with their own plans; this plan builds the fixture they consume.

**Tech Stack:** pytest (session fixtures, custom markers), `memo.store.queries.VecStore`, `python-frontmatter`, GitHub Actions.

Spec: `docs/SPECS/2026-08-06-deadline-and-corpus-conformance-design.md` (part C2).

## Global Constraints

- Tests run with `uv run --no-sync pytest`. `uv sync --extra dev` first in a fresh worktree.
- Never read or write the developer's real vault. Every path under `tmp_path_factory`.
- `MEMO_EMBEDDER_DIMS` must match the stub vector dim exactly (MLX invariant 3). This plan pins both to `64`.
- `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`, `MEMO_STATE_DIR` set for any `CliRunner` invocation.
- The working tree is shared with other agent sessions. Stage explicit paths only — never `git add -A`, never `git commit -a`. Lint only your own files.
- Corpus size is `MEMO_CONFORMANCE_CORPUS_N`, default `10000`.

---

### Task 1: Register the `conformance` marker and its CI lane

**Files:**
- Modify: `pyproject.toml` (markers list, currently ends at the `concurrency` entry)
- Modify: `.github/workflows/test.yml:57`, `.github/workflows/test.yml:117`
- Modify: `.github/workflows/release-quality.yml:42`
- Modify: `.github/workflows/test-stability.yml:31`

**Interfaces:**
- Produces: the pytest marker `conformance`, deselected from every default lane and selected by exactly one new lane.

- [ ] **Step 1: Add the marker**

In `pyproject.toml`, append to the `markers` list:

```toml
    "conformance: corpus-scale surface conformance (payload size, wall-clock, reported totals) against a ~10k synthetic corpus; own CI lane, deselected from the default lanes.",
```

- [ ] **Step 2: Verify the marker is registered**

Run: `uv run --no-sync pytest --markers | grep conformance`
Expected: the description above is printed. (`--strict-markers` is on, so an unregistered marker is an error, not a warning.)

- [ ] **Step 3: Deselect it from the default lanes**

Four edits, each changing only the `-m` expression:

- `.github/workflows/test.yml:57` — `-m "not slow"` → `-m "not slow and not conformance"`
- `.github/workflows/test.yml:117` — `-m "not slow and not float32_precision"` → `-m "not slow and not float32_precision and not conformance"`
- `.github/workflows/release-quality.yml:42` — `-m "not slow"` → `-m "not slow and not conformance"`
- `.github/workflows/test-stability.yml:31` — `-m "not slow"` → `-m "not slow and not conformance"`

- [ ] **Step 4: Add the conformance lane**

In `.github/workflows/test.yml`, after the `resource_hygiene` step (line 51), add a step with the same shape:

```yaml
      - name: Conformance (corpus-scale surfaces)
        env:
          MEMO_CONFORMANCE_CORPUS_N: "10000"
        run: .venv/bin/python -m pytest -m "conformance" -n 0 --timeout=600
```

`-n 0` because the fixture is session-scoped and expensive: parallel workers would each build their own corpus.

- [ ] **Step 5: Verify nothing is selected yet**

Run: `uv run --no-sync pytest -m "conformance" --collect-only -q`
Expected: `no tests ran` — the marker exists, nothing claims it yet. This proves the deselection edits cannot have hidden existing tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .github/workflows/test.yml .github/workflows/release-quality.yml .github/workflows/test-stability.yml
git commit -m "test: register the conformance marker and its CI lane"
```

---

### Task 2: The corpus fixture

**Files:**
- Create: `tests/conformance/conftest.py`
- Create: `tests/conformance/test_fixture_sanity.py`

**Interfaces:**
- Produces:
  - `DIMS: int` (= 64) — the stub vector dimension, also pinned into `MEMO_EMBEDDER_DIMS`.
  - `TOPICS: int` (= 20) — number of semantic clusters.
  - `corpus_size` fixture → `int`
  - `big_corpus` fixture → `Config` — an isolated config whose store and `memory_dir` hold `corpus_size` seeded memories.
  - `seeded_id(i: int) -> str` — the canonical id of the i-th seeded memory, so later tests can address known records.

- [ ] **Step 1: Write the failing sanity test**

`tests/conformance/test_fixture_sanity.py`:

```python
"""The fixture itself is load-bearing: every other conformance test trusts that
it really seeded a corpus of the requested size, on disk and in the index."""

from __future__ import annotations

import pytest

from memo.store.queries import VecStore

from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance


def test_index_holds_every_seeded_memory(big_corpus, corpus_size) -> None:
    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        rows = store.count_memories()
    finally:
        store.close()
    assert rows == corpus_size


def test_markdown_files_match_the_index(big_corpus, corpus_size) -> None:
    on_disk = list(big_corpus.memory_dir.rglob("*.md"))
    assert len(on_disk) == corpus_size


def test_seeded_ids_are_addressable(big_corpus) -> None:
    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        row = store.get(seeded_id(0))
    finally:
        store.close()
    assert row is not None
    assert row["title"].startswith("Conformance memory 0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --no-sync pytest tests/conformance/test_fixture_sanity.py -q`
Expected: collection error — `tests/conformance/conftest.py` does not exist.

- [ ] **Step 3: Write the fixture**

`tests/conformance/conftest.py`:

```python
"""Corpus-scale conformance fixtures.

Seeds a deterministic ~10k-memory corpus once per session. Twelve defects found
by hand on 2026-08-06 were invisible to 6,955 unit tests because a 3-memory
fixture cannot express a page size, an unbounded neighbor list, or a total that
disagrees with the corpus. This fixture can.

Vectors come from a hash-derived stub, NOT from MLX: `MEMO_EMBEDDER_DIMS` is
pinned to `DIMS` so the store's dims guard sees a consistent profile (MLX
invariant 3). The stub validates payload size, wall-clock and total honesty --
never semantic quality, which stays with `memo eval recall` on the live corpus.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator

import frontmatter
import pytest

from memo.config import Config
from memo.store.queries import VecStore

DIMS = 64
TOPICS = 20

_CREATED = "2026-01-01T00:00:00+00:00"
_TOPIC_NAMES = [f"topic{n:02d}" for n in range(TOPICS)]


def seeded_id(i: int) -> str:
    """Canonical id of the i-th seeded memory. 32 hex chars, stable across runs."""
    return hashlib.sha256(f"conformance-{i}".encode()).hexdigest()[:32]


def _vector(topic: int, i: int) -> list[float]:
    """Unit vector: per-topic centroid plus small per-item jitter, so same-topic
    memories cluster and a search returns a meaningful neighborhood."""
    centroid = hashlib.sha256(f"topic-{topic}".encode()).digest()
    jitter = hashlib.sha256(f"item-{i}".encode()).digest()
    out = [
        (centroid[d % len(centroid)] - 128) / 128.0 + (jitter[d % len(jitter)] - 128) / 2048.0
        for d in range(DIMS)
    ]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


@pytest.fixture(scope="session")
def corpus_size() -> int:
    return int(os.environ.get("MEMO_CONFORMANCE_CORPUS_N", "10000"))


@pytest.fixture(scope="session")
def big_corpus(tmp_path_factory, corpus_size: int) -> Iterator[Config]:
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("conformance")
    data, vault, state = root / "data", root / "vault", root / "state"
    for d in (data, vault, state):
        d.mkdir()

    mp.setenv("MEMO_EMBEDDER_DIMS", str(DIMS))
    mp.setenv("MEMO_NONINTERACTIVE", "1")
    mp.setenv("MEMO_DATA_DIR", str(data))
    mp.setenv("MEMO_STATE_DIR", str(state))
    mp.setenv("MEMO_AUTO_PROJECT_TAG", "0")
    mp.setenv("MEMO_STORE_BY_PROJECT", "0")

    cfg = Config(data_dir=data, vault_path=vault, state_dir=state, reranker_enabled=False)
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)

    store = VecStore(cfg.db_path, dims=DIMS)
    try:
        for i in range(corpus_size):
            topic = i % TOPICS
            mid = seeded_id(i)
            title = f"Conformance memory {i} about {_TOPIC_NAMES[topic]}"
            body = (
                f"Synthetic conformance record {i}. Subject: {_TOPIC_NAMES[topic]}. "
                f"This body exists so body-length gates and BM25 have real text to "
                f"work with rather than a stub token."
            )
            rel = f"{mid}.md"
            post = frontmatter.Post(
                body,
                id=mid,
                title=title,
                type="note",
                tags=[_TOPIC_NAMES[topic], "conformance"],
                created=_CREATED,
                updated=_CREATED,
                valid_at=_CREATED,
            )
            post["extra"] = {}
            post["verification_state"] = "unverified"
            (cfg.memory_dir / rel).write_text(frontmatter.dumps(post), encoding="utf-8")
            store.upsert(
                id_=mid,
                path=rel,
                title=title,
                type_="note",
                tags=[_TOPIC_NAMES[topic], "conformance"],
                created=_CREATED,
                updated=_CREATED,
                body_hash=hashlib.sha256(body.encode()).hexdigest(),
                embedding=_vector(topic, i),
                body_text=body,
            )
    finally:
        store.close()

    yield cfg
    mp.undo()
```

- [ ] **Step 4: Run the sanity test**

Run: `MEMO_CONFORMANCE_CORPUS_N=500 uv run --no-sync pytest tests/conformance/test_fixture_sanity.py -q`
Expected: 3 passed. Use 500 while iterating; the CI lane runs 10000.

If `store.count_memories()` or `store.get()` do not exist under those names, find the real accessors with `grep -n "    def " src/memo/store/queries.py` and use them — do not add wrappers to `VecStore` for the test's convenience.

- [ ] **Step 5: Measure the full-size build**

Run: `time MEMO_CONFORMANCE_CORPUS_N=10000 uv run --no-sync pytest tests/conformance/test_fixture_sanity.py -q`
Expected: passes. Record the wall-clock in the commit message. If it exceeds ~120s, wrap the seeding loop in a single `store._tx()` batch rather than 10,000 individual transactions, re-measure, and note the change.

- [ ] **Step 6: Commit**

```bash
git add tests/conformance/conftest.py tests/conformance/test_fixture_sanity.py
git commit -m "test(conformance): seed a 10k-memory corpus fixture"
```

---

### Task 3: Reported totals are real totals

**Files:**
- Create: `tests/conformance/test_reported_totals.py`
- Modify (if red): `src/memo/cli_analytics.py`, `src/memo/web_build.py`

**Interfaces:**
- Consumes: `big_corpus`, `corpus_size` from Task 2.

The defect: `memo analytics summary` computed its total as `len(list(...limit=10000))`, printing `9999` against a corpus of 11,383 and deriving a growth rate from that number. The dashboard corpus panel carried the same constant.

- [ ] **Step 1: Write the failing test**

```python
"""A total is a claim about the corpus. A page size presented as a total makes
every derived metric (growth rate, panels) quietly wrong."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from memo.cli import cli

pytestmark = pytest.mark.conformance


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
    }


def test_analytics_summary_reports_the_real_total(big_corpus, corpus_size) -> None:
    result = CliRunner().invoke(cli, ["analytics", "summary", "--json"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] == corpus_size


def test_stats_reports_the_real_total(big_corpus, corpus_size) -> None:
    result = CliRunner().invoke(cli, ["stats", "--json"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total"] == corpus_size
```

- [ ] **Step 2: Run it**

Run: `MEMO_CONFORMANCE_CORPUS_N=10001 uv run --no-sync pytest tests/conformance/test_reported_totals.py -q`

10001 is deliberate: it is above every hard-coded page size in the codebase, so a surface that reports a page reports `10000` and fails visibly.

Expected: either PASS (the fix already landed on this branch — then confirm the test is real by temporarily reverting that fix and seeing it go red) or FAIL with `10000 != 10001`.

If the actual JSON key is not `total`, read the command's output shape and assert the real key — do not add an alias.

- [ ] **Step 3: Fix any red surface**

Replace the `len(list(...limit=N))` count with the store's real count. Do not raise the limit — a larger page is the same bug with a later threshold.

- [ ] **Step 4: Re-run**

Run: `MEMO_CONFORMANCE_CORPUS_N=10001 uv run --no-sync pytest tests/conformance/test_reported_totals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/test_reported_totals.py
git commit -m "test(conformance): assert reported totals match the corpus"
```

---

### Task 4: Output paths fail cleanly

**Files:**
- Create: `tests/conformance/test_output_paths.py`

**Interfaces:**
- Consumes: `big_corpus` from Task 2.

The defect: `atomic_write_text` rejected any destination with a symlinked parent. On macOS `/tmp` **is** a symlink to `/private/tmp`, so `memo graph mindmap -o /tmp/x.html` raised a raw traceback. `memo federation export` shares the primitive; `memo backup --out` and `memo export` raw-tracebacked on a missing parent directory.

- [ ] **Step 1: Write the failing test**

```python
"""Every `-o/--out` surface: a clean error or a written file. Never a traceback.

/tmp is a symlink on macOS, which is why the symlink case is not exotic."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from memo.cli import cli

pytestmark = pytest.mark.conformance

# (argv-prefix, output-flag) for every CLI that writes a caller-named file.
OUTPUT_SURFACES = [
    (["graph", "mindmap"], "-o"),
    (["federation", "export"], None),  # positional path
    (["backup"], "--out"),
    (["export"], "--out"),
]


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
    }


@pytest.mark.parametrize("argv,flag", OUTPUT_SURFACES)
def test_symlinked_parent_is_accepted(big_corpus, tmp_path, argv, flag) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    dest = link / "out.dat"
    args = [*argv, *([flag, str(dest)] if flag else [str(dest)])]

    result = CliRunner().invoke(cli, args, env=_env(big_corpus))

    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"{argv} raised {result.exception!r} on a symlinked parent"
    )


@pytest.mark.parametrize("argv,flag", OUTPUT_SURFACES)
def test_missing_parent_gives_a_clean_error(big_corpus, tmp_path, argv, flag) -> None:
    dest = tmp_path / "does" / "not" / "exist" / "out.dat"
    args = [*argv, *([flag, str(dest)] if flag else [str(dest)])]

    result = CliRunner().invoke(cli, args, env=_env(big_corpus))

    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"{argv} raised {result.exception!r} instead of a clean error"
    )
    if result.exit_code != 0:
        assert "Traceback" not in result.output
```

- [ ] **Step 2: Run it**

Run: `MEMO_CONFORMANCE_CORPUS_N=500 uv run --no-sync pytest tests/conformance/test_output_paths.py -q`
Expected: PASS for the surfaces already fixed on this branch; FAIL loudly for any that were missed. Confirm at least one case is genuinely exercised by temporarily reverting the `atomic_write_text` symlink fix and seeing red.

Adjust `OUTPUT_SURFACES` to the real flag names if a `--out` here is actually `-o` there — read the command, do not guess twice.

- [ ] **Step 3: Fix any red surface**

Route it through the same clean-error path the fixed surfaces use.

- [ ] **Step 4: Re-run**

Run: `MEMO_CONFORMANCE_CORPUS_N=500 uv run --no-sync pytest tests/conformance/test_output_paths.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/test_output_paths.py
git commit -m "test(conformance): every output path fails cleanly, never a traceback"
```

---

### Task 5: A rebuild preserves what it does not own

**Files:**
- Create: `tests/conformance/test_index_rebuild_preserves.py`

**Interfaces:**
- Consumes: `big_corpus`, `corpus_size`, `seeded_id` from Task 2.

The defect: `memo links reindex` deleted the whole crossref index and rebuilt only the newest 10,000 rows — silent data loss above that threshold. The same class covers `memo reindex --rebuild`, which must preserve the user-signal tables (`access`, `memory_health`, `source_feedback*`) that markdown does not carry.

- [ ] **Step 1: Write the failing test**

```python
"""A rebuild is derived-data surgery. Anything it does not own must survive it,
and anything it does own must come back whole -- not the newest page of it."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.store.queries import VecStore

from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
    }


def test_links_reindex_covers_the_whole_corpus(big_corpus, corpus_size) -> None:
    result = CliRunner().invoke(cli, ["links", "reindex"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output

    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        oldest_still_indexed = store.get(seeded_id(0))
    finally:
        store.close()
    assert oldest_still_indexed is not None, (
        "the oldest seeded memory fell outside the rebuild window"
    )


def test_rebuild_preserves_user_signal(big_corpus) -> None:
    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        store.log_access(seeded_id(1))
    finally:
        store.close()

    result = CliRunner().invoke(cli, ["reindex", "--rebuild"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output

    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        assert store.access_count(seeded_id(1)) >= 1
    finally:
        store.close()
```

- [ ] **Step 2: Run it**

Run: `MEMO_CONFORMANCE_CORPUS_N=10001 uv run --no-sync pytest tests/conformance/test_index_rebuild_preserves.py -q`
Expected: FAIL if the `links reindex` truncation is unfixed on this branch; otherwise confirm by reverting that fix.

`log_access` / `access_count` are placeholders for the store's real accessors — find them with `grep -n "access" src/memo/store/queries.py` and use the actual names. If the rebuild needs an embedder, pass `--no-embed` or the equivalent; a rebuild that must call MLX does not belong in this lane.

- [ ] **Step 3: Fix any red surface**

Rebuild the whole set, not a page.

- [ ] **Step 4: Re-run and run the whole lane**

Run: `MEMO_CONFORMANCE_CORPUS_N=10001 uv run --no-sync pytest -m conformance -n 0 -q`
Expected: all pass.

- [ ] **Step 5: Verify the default lane is unaffected**

Run: `uv run --no-sync pytest -m "not slow and not conformance" -q -x`
Expected: green, and no conformance test in the selection.

- [ ] **Step 6: Commit**

```bash
git add tests/conformance/test_index_rebuild_preserves.py
git commit -m "test(conformance): a rebuild preserves user signal and the whole corpus"
```

---

## Self-review notes

- Spec sections C2 covered: fixture (Task 2), reported totals (Task 3), output paths (Task 4), index rebuild (Task 5), marker + lane (Task 1).
- `test_mcp_response_budget.py` and `test_read_latency_budget.py` are named in the spec's table but belong to the response-budget and deadline plans respectively — each needs its subject to exist first. They consume `big_corpus` from Task 2 unchanged.
- Store accessor names (`count_memories`, `get`, `log_access`, `access_count`) are marked in-step as needing confirmation against `store/queries.py`. They are the only unverified identifiers in this plan; each carries an explicit instruction to use the real name rather than add a wrapper.
