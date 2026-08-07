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

import frontmatter
import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.embedder import MLXEmbedder
from memo.embedder_select import active_embedder_identity
from memo.embedder_st import STEmbedder
from memo.memory.record import _compose_for_embed
from memo.store import VecStore
from memo.util import sha256_full

from .conftest import DIMS, _env, seeded_id

pytestmark = pytest.mark.conformance


def test_links_reindex_covers_the_whole_corpus(big_corpus, corpus_size) -> None:
    """`links reindex` doesn't touch `VecStore`'s meta/vec rows at all -- it
    only resets and replays `mem.crossref` (a separate sqlite file) -- so
    the defect this guards is not "a memory's row went missing", it's "the
    rebuild silently walked fewer memories than the corpus holds and
    reported the smaller number as the truth". `mem.list(limit=10000)`
    (the old code) can only ever report `min(corpus_size, 10000)` scanned
    and indexed; the fix reports the real corpus. Assert the CLI's own
    accounting, the same way commit ce740769's unit test does at small
    scale (`test_links_reindex_covers_corpus_past_the_old_10k_cap`) -- this
    is that same assertion proven against a real store and real markdown at
    a scale past the cap, not a `_FakeMemory` stand-in.
    """
    result = CliRunner().invoke(cli, ["links", "reindex", "--yes"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output

    expected = f"Reindexed {corpus_size} of {corpus_size} memories"
    assert expected in result.output, (
        f"expected the rebuild to walk and index the whole {corpus_size}-memory "
        f"corpus, not a capped page -- got:\n{result.output}"
    )


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
