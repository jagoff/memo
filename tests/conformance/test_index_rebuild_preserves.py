"""A rebuild is derived-data surgery. Anything it does not own must survive it,
and anything it does own must come back whole -- not the newest page of it.

Two surfaces, one defect class:
  - `memo links reindex` used to wipe the whole crossref index and then
    replay only `mem.list(limit=10000)` -- on a corpus past that cap, every
    memory outside the page silently lost its crossrefs, while the capped
    number printed as if it were the corpus total (fixed by commit
    `ce740769`, which now walks `store.all_ids()` and reports how many of
    how many were actually indexed).
  - `memo reindex --rebuild` truncates the markdown-derivable tables
    (meta/vec/fts) and replays them from disk, but must never touch the
    user-signal tables (`access`, `memory_health`, `source_feedback*`) --
    markdown carries no equivalent of "how many times was this read", so a
    rebuild that dropped them would be silent, unrecoverable data loss.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from collections.abc import Iterator
from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner, Result

from memo.cli import cli
from memo.config import Config
from memo.crossref import CrossReferenceIndex
from memo.embedder import MLXEmbedder
from memo.embedder_select import active_embedder_identity
from memo.embedder_st import STEmbedder
from memo.memory.record import _compose_for_embed
from memo.store import VecStore
from memo.util import sha256_full

from .conftest import DIMS, _env, seeded_id

pytestmark = pytest.mark.conformance


# -- links reindex: covers the whole corpus, and undoes its own mutation ----
#
# `links reindex` resets and replays `mem.crossref` -- a real write to
# `crossref.db`, a sqlite sidecar file under `big_corpus.state_dir` shared
# with tests this plan hasn't written yet (the same concern `graph_seeded`
# in test_output_paths.py handles for `graph.db`). Here the mutation IS the
# command under test rather than a separate setup step, so the undo lives in
# a fixture that performs the CLI call itself and restores the sidecar
# file's bytes (main db + WAL + SHM, whichever exist) on every exit path --
# snapshot-and-restore rather than a row-level undo, since `links reindex`
# decides internally what to delete/insert and there is no smaller unit to
# reverse from the outside.


def _crossref_sidecars(crossref_db: Path) -> tuple[Path, Path, Path]:
    return (
        crossref_db,
        crossref_db.with_name(crossref_db.name + "-wal"),
        crossref_db.with_name(crossref_db.name + "-shm"),
    )


def _snapshot_sidecars(crossref_db: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in _crossref_sidecars(crossref_db) if p.exists()}


def _restore_sidecars(crossref_db: Path, snapshot: dict[Path, bytes]) -> None:
    for p in _crossref_sidecars(crossref_db):
        p.unlink(missing_ok=True)
    for p, data in snapshot.items():
        p.write_bytes(data)


@pytest.fixture
def links_reindex_result(big_corpus) -> Iterator[Result]:
    """Runs `memo links reindex --yes` against the shared `big_corpus`, then
    restores `crossref.db` (+ its WAL/SHM companions) to exactly the bytes
    they held before the call, on every exit path including an exception.
    `test_links_reindex_fixture_restores_crossref_on_teardown` below proves
    the restoration directly, the same way
    `test_graph_seeded_reverts_its_write_on_teardown` proves it for
    `graph_seeded`.
    """
    crossref_db = big_corpus.crossref_db
    snapshot = _snapshot_sidecars(crossref_db)
    try:
        yield CliRunner().invoke(cli, ["links", "reindex", "--yes"], env=_env(big_corpus))
    finally:
        _restore_sidecars(crossref_db, snapshot)


def test_links_reindex_covers_the_whole_corpus(links_reindex_result, corpus_size) -> None:
    """`links reindex` doesn't touch `VecStore`'s meta/vec rows at all -- it
    only resets and replays `mem.crossref` -- so the defect this guards is
    not "a memory's row went missing", it's "the rebuild silently walked
    fewer memories than the corpus holds and reported the smaller number as
    the truth". `mem.list(limit=10000)` (the old code) can only ever report
    `min(corpus_size, 10000)` scanned and indexed; the fix reports the real
    corpus. Assert the CLI's own accounting, the same way commit
    ce740769's unit test does at small scale
    (`test_links_reindex_covers_corpus_past_the_old_10k_cap`) -- this is
    that same assertion proven against a real store and real markdown at a
    scale past the cap, not a `_FakeMemory` stand-in.
    """
    result = links_reindex_result
    assert result.exit_code == 0, result.output

    expected = f"Reindexed {corpus_size} of {corpus_size} memories"
    assert expected in result.output, (
        f"expected the rebuild to walk and index the whole {corpus_size}-memory "
        f"corpus, not a capped page -- got:\n{result.output}"
    )


def test_links_reindex_fixture_restores_crossref_on_teardown(tmp_path) -> None:
    """`links_reindex_result` mutates a session-scoped `crossref.db` shared
    with tests this plan hasn't written yet -- a happy-path run of the
    covering test above never proves the mutation is undone, since it only
    runs *during* the fixture's lifetime. Drive the fixture generator
    directly against a throwaway `Config`, the same technique
    `test_graph_seeded_reverts_its_write_on_teardown` (test_output_paths.py)
    uses for `graph_seeded`. Seeds a REAL backlink row before driving the
    fixture, so the assertion is "the row comes back", not just "a file
    happens to exist".
    """
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)

    seed_index = CrossReferenceIndex(cfg.crossref_db)
    try:
        seed_index.index_source("preexisting-source", "see [[preexisting-target]]")
    finally:
        seed_index.close()
    before = _snapshot_sidecars(cfg.crossref_db)
    assert before, "seeding didn't create crossref.db -- nothing to prove restoration against"

    gen = links_reindex_result.__wrapped__(cfg)
    result = next(gen)  # run the CLI call through the yield
    assert result.exit_code == 0, result.output
    # `reset()` really did mutate the file -- if this held, the restoration
    # proof below would be vacuous (nothing to restore from).
    assert _snapshot_sidecars(cfg.crossref_db) != before

    with pytest.raises(StopIteration):
        next(gen)  # advance past the yield -- runs the fixture's `finally`

    assert _snapshot_sidecars(cfg.crossref_db) == before
    restored = CrossReferenceIndex(cfg.crossref_db)
    try:
        assert [w.target for w in restored.get_outlinks("preexisting-source")] == [
            "preexisting-target"
        ], "the pre-existing backlink row did not survive the fixture's teardown"
    finally:
        restored.close()


# -- reindex --rebuild: no MLX, and the user-signal tables must survive -----
#
# `--rebuild` re-embeds every memory via `Memory._embed_cached`, which is a
# content-addressed cache (`repo_embedding_cache`, keyed on
# `(embedder_model, embedder_dims, sha256(composed_text))`) in front of the
# live embedder -- a cache hit never calls `.embed()`. `big_corpus` was
# seeded with hash-derived stub vectors that were never routed through that
# cache, so priming it here (with the exact composed text `_embed_cached`
# will hash) is what keeps this lane MLX-free. If priming ever misses an
# entry, `_no_live_embedder` below turns that into a loud, immediate
# `AssertionError` instead of a silent multi-minute MLX/network stall.


def _stub_vector(seed: str) -> list[float]:
    """Deterministic unit vector for cache-priming. Never asserted against a
    real embedder's output, so any valid `DIMS`-length vector satisfies the
    rebuild -- correctness here means "present in the cache", not "semantically
    faithful"."""
    digest = hashlib.sha256(seed.encode()).digest()
    raw = [(digest[d % len(digest)] - 128) / 128.0 for d in range(DIMS)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def _prime_embedding_cache(cfg: Config, store: VecStore, corpus_size: int) -> None:
    """Populate `repo_embedding_cache` with one entry per seeded memory, keyed
    exactly like `Memory._embed_cached` will key its lookup during the
    rebuild: `(active_embedder_identity(Config.from_env()), embedder_dims,
    sha256(title + "\\n\\n" + body))`. Title/body are read back from the same
    `.md` files the rebuild itself will parse -- not reconstructed from the
    fixture's format string -- so there is no format string to drift out of
    sync.
    """
    live_cfg = Config.from_env()
    model = active_embedder_identity(live_cfg)
    dims = live_cfg.embedder_dims
    entries: list[tuple[str, list[float]]] = []
    for i in range(corpus_size):
        md_path = cfg.memory_dir / f"{seeded_id(i)}.md"
        post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
        title = str(post.metadata.get("title") or "")
        body = post.content or ""
        text = _compose_for_embed(title, body)
        entries.append((sha256_full(text), _stub_vector(text)))
    store.upsert_repo_embedding_cache(
        model=model, dims=dims, embeddings=entries, created_at="2026-01-01T00:00:00+00:00"
    )


def test_rebuild_preserves_user_signal(big_corpus, corpus_size, tmp_path, monkeypatch) -> None:
    """Runs against a COPY of `big_corpus`'s data/state dirs, not the shared
    session-scoped fixture. `--rebuild` atomically replaces meta/vec/fts and
    re-derives columns (namespace, topic_key, normalized_title, ...) the
    fixture's direct `store.upsert()` calls never set -- there is no way to
    restore true byte-equivalence afterward by re-deriving row-by-row.
    Copying sidesteps the need: `big_corpus` itself is never opened for
    write by this test, so there is nothing to restore.
    """
    data_copy = tmp_path / "data"
    state_copy = tmp_path / "state"
    shutil.copytree(big_corpus.data_dir, data_copy)
    shutil.copytree(big_corpus.state_dir, state_copy)
    cfg = Config(
        data_dir=data_copy,
        vault_path=tmp_path / "vault",
        state_dir=state_copy,
        reranker_enabled=False,
    )

    store = VecStore(cfg.db_path, dims=DIMS)
    try:
        store.touch([seeded_id(1)])
        _prime_embedding_cache(cfg, store, corpus_size)
    finally:
        store.close()

    def _no_live_embedder(self: object) -> None:
        raise AssertionError(
            "reindex --rebuild tried to load a live embedder -- the "
            "repo_embedding_cache priming in this test should have covered "
            "every seeded memory's composed text, so this lane stays MLX-free"
        )

    monkeypatch.setattr(MLXEmbedder, "_ensure_loaded", _no_live_embedder)
    monkeypatch.setattr(STEmbedder, "_ensure_loaded", _no_live_embedder)

    result = CliRunner().invoke(cli, ["reindex", "--rebuild"], env=_env(cfg))
    assert result.exit_code == 0, result.output

    store = VecStore(cfg.db_path, dims=DIMS)
    try:
        assert store.get_access(seeded_id(1))["access_count"] >= 1, (
            "the access row logged before the rebuild did not survive it"
        )
    finally:
        store.close()
